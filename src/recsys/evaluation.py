import time
from typing import Callable, Dict, List, Set, Tuple

# En Windows, torch debe importarse primero
import torch
import numpy as np

from recsys.ranker import Ranker

# Nota de diseño metodológico:
# item_idx sigue estrictamente el orden de primera aparición en ratings.csv
# generado al mapear ratings['movieId'].unique().
# Todos los desempates deterministas priorizan el menor item_idx ante scores iguales.


def compute_user_metrics(
    recommended_items: np.ndarray,
    relevant_items: Set[int],
    top_k: int = 10,
) -> Tuple[float, float, float]:
    """Calcula precision@K, recall@K y ndcg@K para la lista de un usuario."""
    discount = 1.0 / np.log2(np.arange(2, top_k + 2))
    rel = np.array([1.0 if i in relevant_items else 0.0 for i in recommended_items])
    precision = float(rel.sum() / top_k)
    recall = float(rel.sum() / len(relevant_items)) if len(relevant_items) > 0 else 0.0
    ideal = float(discount[: min(len(relevant_items), top_k)].sum())
    ndcg = float((rel * discount).sum() / ideal) if ideal > 0 else 0.0
    return precision, recall, ndcg


def score_with_ranker(
    model: Ranker,
    user_idx: int,
    item_idx_array: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Puntúa un vector de ítems para un usuario con el ranker."""
    with torch.no_grad():
        items_t = torch.as_tensor(item_idx_array, dtype=torch.long, device=device)
        users_t = torch.full_like(items_t, int(user_idx))
        return model(users_t, items_t).cpu().numpy().copy()


def make_rec_popularity(
    item_popularity: np.ndarray,
    seen_mask_np: np.ndarray,
    top_k: int = 10,
) -> Callable[[int], np.ndarray]:
    """Recomendador por popularidad con desempate determinista por item_idx ascendente."""
    n_items = len(item_popularity)
    item_indices = np.arange(n_items)

    def rec(u: int) -> np.ndarray:
        scores = item_popularity.astype(np.float64).copy()
        scores[seen_mask_np[u]] = -np.inf
        # Desempate determinista: popularidad descendente, item_idx ascendente
        return np.lexsort((item_indices, -scores))[:top_k]

    return rec


def make_rec_retrieval_only(
    retrieval_topk: np.ndarray,
    top_k: int = 10,
) -> Callable[[int], np.ndarray]:
    """Recomendador basado únicamente en el top-K del retrieval."""
    def rec(u: int) -> np.ndarray:
        return retrieval_topk[u, :top_k]

    return rec


def make_one_stage(
    model: Ranker,
    n_items: int,
    seen_mask_np: np.ndarray,
    device: torch.device,
    top_k: int = 10,
) -> Callable[[int], np.ndarray]:
    """Recomendador Ranker en catálogo completo con desempate determinista por item_idx ascendente."""
    all_items_np = np.arange(n_items)

    def rec(u: int) -> np.ndarray:
        scores = score_with_ranker(model, u, all_items_np, device)
        scores[seen_mask_np[u]] = -np.inf
        # Desempate determinista: score descendente, item_idx ascendente
        return np.lexsort((all_items_np, -scores))[:top_k]

    return rec


def make_two_stage(
    model: Ranker,
    retrieval_topk: np.ndarray,
    n_candidates: int,
    device: torch.device,
    top_k: int = 10,
) -> Callable[[int], np.ndarray]:
    """Recomendador Two-Stage con desempate determinista por item_idx ascendente."""
    def rec(u: int) -> np.ndarray:
        cands = retrieval_topk[u, :n_candidates]
        scores = score_with_ranker(model, u, cands, device)
        # Desempate determinista entre candidatos: score descendente, item_idx (cands) ascendente
        order = np.lexsort((cands, -scores))
        return cands[order[:top_k]]

    return rec


def evaluate_model(
    name: str,
    rec_fn: Callable[[int], np.ndarray],
    eval_users: List[int],
    test_relevant_by_user: Dict[int, Set[int]],
    item_popularity: np.ndarray,
    n_items: int,
    device: torch.device,
    top_k: int = 10,
) -> Dict[str, float | str]:
    """Evalúa un recomendador calculando métricas de ranking, sesgo y latencia."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    lists = [rec_fn(u) for u in eval_users]
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms_per_user = (time.perf_counter() - t0) / len(eval_users) * 1000

    precs, recs, ndcgs = [], [], []
    for u, top in zip(eval_users, lists):
        relevant = test_relevant_by_user[u]
        p, r, n = compute_user_metrics(top, relevant, top_k=top_k)
        precs.append(p)
        recs.append(r)
        ndcgs.append(n)

    all_recs = np.concatenate(lists)
    return {
        "modelo": name,
        "precision_at_10": float(np.mean(precs)),
        "recall_at_10": float(np.mean(recs)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "cobertura_catalogo": float(len(set(all_recs.tolist())) / n_items),
        "popularidad_media_recs": float(item_popularity[all_recs].mean()),
        "ms_por_usuario": float(ms_per_user),
    }


def measure_honest_latency(
    retrieval_model,
    ranker_model,
    eval_users: List[int],
    n_items: int,
    seen_mask_np: np.ndarray,
    n_candidates_list: Tuple[int, ...] = (50, 200),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, float]]:
    """Mide honestamente por usuario la latencia del pipeline de dos etapas en CPU.
    
    Etiquetado como propio de la CPU de este equipo. Separa:
    1. Retrieval: forward de usuario, producto punto contra todo el catálogo, máscara de vistos y topk(N).
    2. Ranker: puntuación de los N candidatos con el MLP y ordenamiento top-10 con desempate determinista.
    3. Total Two-Stage: suma de ambas fases para ese usuario.
    4. Ranker catálogo completo (1 etapa): forward de los ~9724 ítems en el MLP, máscara y ordenamiento.
    """
    retrieval_model.eval()
    ranker_model.eval()

    all_items_t = torch.arange(n_items, dtype=torch.long, device=device)
    with torch.no_grad():
        all_item_embeds = retrieval_model.forward_item(all_items_t)

    results = {}

    for n_cands in n_candidates_list:
        times_retrieval = []
        times_ranker = []

        with torch.no_grad():
            for u in eval_users:
                # FASE 1: RETRIEVAL POR USUARIO
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                u_t = torch.tensor([u], dtype=torch.long, device=device)
                u_emb = retrieval_model.forward_user(u_t)
                scores = (u_emb @ all_item_embeds.T).squeeze(0)
                seen_indices = np.where(seen_mask_np[u])[0]
                if len(seen_indices) > 0:
                    scores[torch.as_tensor(seen_indices, device=device)] = -float("inf")
                cands_t = torch.topk(scores, n_cands).indices
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_retr = (time.perf_counter() - t0) * 1000
                times_retrieval.append(t_retr)

                # FASE 2: RANKER SOBRE N CANDIDATOS
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                users_t = torch.full_like(cands_t, u)
                ranker_scores = ranker_model(users_t, cands_t).cpu().numpy()
                cands_np = cands_t.cpu().numpy()
                order = np.lexsort((cands_np, -ranker_scores))
                _ = cands_np[order[:10]]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_rank = (time.perf_counter() - t1) * 1000
                times_ranker.append(t_rank)

        mean_retr = float(np.mean(times_retrieval))
        mean_rank = float(np.mean(times_ranker))
        results[f"Two-Stage N={n_cands}"] = {
            "ms_retrieval": mean_retr,
            "ms_ranker": mean_rank,
            "ms_total": mean_retr + mean_rank,
        }

    # Ranker catálogo completo (1 etapa) por usuario
    times_full_ranker = []
    with torch.no_grad():
        for u in eval_users:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            users_t = torch.full_like(all_items_t, u)
            scores = ranker_model(users_t, all_items_t).cpu().numpy()
            seen_indices = np.where(seen_mask_np[u])[0]
            if len(seen_indices) > 0:
                scores[seen_indices] = -float("inf")
            order = np.lexsort((np.arange(n_items), -scores))
            _ = order[:10]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_full = (time.perf_counter() - t0) * 1000
            times_full_ranker.append(t_full)

    results["Ranker catálogo completo (1 etapa)"] = {
        "ms_retrieval": 0.0,
        "ms_ranker": float(np.mean(times_full_ranker)),
        "ms_total": float(np.mean(times_full_ranker)),
    }

    return results
