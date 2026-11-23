"""Script de entrenamiento y evaluación comparativa para los modelos generativos G1, G2 y G3.
Soporta modo rápido (--quick), semilla fija y sensibilidad al tamaño del haz (20, 50, 100).
"""

import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from recsys.stage5.data_split import create_temporal_split, generate_sliding_windows
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender


class SequenceDataset(Dataset):
    """Dataset de ventanas deslizantes (contexto -> target)."""

    def __init__(
        self,
        windows: List[Tuple[List[int], int]],
        tokenizer: GenerativeTokenizer,
        max_context_tokens: int = 150,
    ):
        self.windows = windows
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        ctx_items, tgt_item = self.windows[idx]
        input_tokens = self.tokenizer.encode_input_sequence(
            ctx_items, max_tokens=self.max_context_tokens
        )
        target_tokens = self.tokenizer.encode_target_item(tgt_item)
        return input_tokens, target_tokens


def collate_fn(batch, pad_token_id=0):
    input_seqs, target_seqs = zip(*batch)
    max_in = max(len(s) for s in input_seqs)
    max_tgt = max(len(s) for s in target_seqs)

    B = len(batch)
    input_ids = torch.full((B, max_in), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_in), dtype=torch.long)
    labels = torch.full((B, max_tgt), -100, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(input_seqs, target_seqs)):
        input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        attention_mask[i, : len(inp)] = 1
        labels[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    return input_ids, attention_mask, labels


def compute_metrics(
    recommendations_by_user: Dict[int, List[int]],
    test_relevant_by_user: Dict[int, Set[int]],
    eval_users: List[int],
    cold_items: Set[int],
    item_buckets: Dict[int, str],
    n_items: int,
) -> Dict[str, float]:
    """Calcula NDCG@10, Recall@10, Recall@50, Cold Recall@50, Cobertura, Popularidad y Cola Larga."""
    ndcg_10_list = []
    recall_10_list = []
    recall_50_list = []

    cold_recall_50_list = []
    all_recommended_items = set()
    total_recs_count = 0
    tail_recs_count = 0
    pop_scores = []

    for u in eval_users:
        recs = recommendations_by_user.get(u, [])
        rel = test_relevant_by_user.get(u, set())
        if not rel:
            continue

        # Recall@10 y NDCG@10
        top10 = recs[:10]
        hits_10 = len(set(top10) & rel)
        recall_10_list.append(hits_10 / len(rel))

        dcg = 0.0
        for rank, item in enumerate(top10):
            if item in rel:
                dcg += 1.0 / np.log2(rank + 2)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), 10)))
        ndcg_10_list.append(dcg / idcg if idcg > 0 else 0.0)

        # Recall@50
        top50 = recs[:50]
        hits_50 = len(set(top50) & rel)
        recall_50_list.append(hits_50 / len(rel))

        # Cold items recall@50
        cold_rel = rel & cold_items
        if cold_rel:
            hits_cold = len(set(top50) & cold_rel)
            cold_recall_50_list.append(hits_cold / len(cold_rel))

        # Cobertura y popularidad sobre top-50
        for item in top50:
            all_recommended_items.add(item)
            b = item_buckets.get(item, "0")
            if b in {"0", "1-5"}:
                tail_recs_count += 1
            total_recs_count += 1

    return {
        "ndcg@10": float(np.mean(ndcg_10_list)) if ndcg_10_list else 0.0,
        "recall@10": float(np.mean(recall_10_list)) if recall_10_list else 0.0,
        "recall@50": float(np.mean(recall_50_list)) if recall_50_list else 0.0,
        "cold_recall@50": float(np.mean(cold_recall_50_list)) if cold_recall_50_list else 0.0,
        "coverage@50": float(len(all_recommended_items) / n_items),
        "long_tail_frac@50": float(tail_recs_count / total_recs_count) if total_recs_count > 0 else 0.0,
        "n_eval_users": len(recall_10_list),
        "n_cold_eval_users": len(cold_recall_50_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Modo rápido de diagnóstico (pocas épocas y usuarios)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--variants", nargs="+", default=["G1", "G2", "G3"], help="Variantes a entrenar")
    parser.add_argument("--beams", nargs="+", type=int, default=[20, 50, 100], help="Tamaños de haz a evaluar")
    parser.add_argument("--epochs", type=int, default=10, help="Épocas de entrenamiento")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    parser.add_argument("--max_len", type=int, default=50, help="Longitud de ventana de ítems")
    parser.add_argument("--w_max", type=int, default=100, help="Tope de ventanas por usuario")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ETAPA 5: MODELOS GENERATIVOS (Dispositivo: {device}, Semilla: {args.seed}) ===")
    if args.quick:
        print("[MODO RÁPIDO ACTIVO: épocas reducidas y subconjunto de usuarios para diagnóstico]")
        args.epochs = 2
        eval_sample_size = 20
    else:
        eval_sample_size = None

    # 1. Cargar datos y split temporal determinista con Regla Opción C
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=args.seed, use_event_boundary_rule=True)

    n_items = bundle.n_items
    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")
    train_indices = sorted(list(bundle.train_item_indices))
    cold_indices = sorted(list(bundle.cold_item_indices))

    # 2. Entrenar RQ-VAE solo en Train items
    print(f"\nAjustando RQ-VAE sobre los {len(train_indices)} ítems de Train...")
    train_embs = all_embeddings[train_indices]
    mu = train_embs.mean(axis=0)
    sig = train_embs.std(axis=0) + 1e-8
    X_std = (all_embeddings - mu) / sig

    rq_model = RQVAE().to(device)
    X_tr_t = torch.tensor(X_std[train_indices], dtype=torch.float32, device=device)
    rq_model.rvq.init_from_kmeans(X_tr_t, seed=args.seed)

    rq_opt = torch.optim.AdamW(rq_model.parameters(), lr=1e-3, weight_decay=1e-4)
    rq_loader = DataLoader(torch.utils.data.TensorDataset(X_tr_t), batch_size=256, shuffle=True)
    rq_model.train()
    rq_epochs = 2 if args.quick else 15
    for ep in range(rq_epochs):
        for (bx,) in rq_loader:
            rq_opt.zero_grad()
            _, _, loss, _, _ = rq_model(bx)
            loss.backward()
            rq_opt.step()

    rq_model.eval()
    with torch.no_grad():
        X_all_t = torch.tensor(X_std, dtype=torch.float32, device=device)
        _, codes_all, _, _, _ = rq_model(X_all_t)
        codes_all = codes_all.cpu().numpy()

    # 3. Asignación de Semantic IDs y Random IDs
    manager = SemanticIdManager(max_disambiguation=16)
    train_counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
    manager.fit_train_items(train_indices, codes_all, train_counts)
    manager.assign_cold_items(cold_indices, codes_all)

    semantic_id_map = manager.item_to_semantic_id
    random_id_map = build_random_id_map(semantic_id_map, seed=args.seed)

    # 4. Generar ventanas de entrenamiento desde train_full
    windows_by_user = {}
    total_train_windows = []
    for u_idx, seq in bundle.user_train_seq.items():
        w = generate_sliding_windows(seq, max_len=args.max_len, max_windows_per_user=args.w_max)
        windows_by_user[u_idx] = w
        total_train_windows.extend(w)

    print(f"Total ventanas de entrenamiento generadas (L={args.max_len}, W_max={args.w_max}): {len(total_train_windows)}")

    # Usuarios a evaluar en test
    eval_users = bundle.eval_users_test
    if eval_sample_size is not None:
        eval_users = eval_users[:eval_sample_size]
    print(f"Usuarios evaluados en Test: {len(eval_users)}")

    results_table = []

    # 5. Iterar sobre las variantes G1, G2 y G3
    for variant in args.variants:
        print(f"\n==========================================")
        print(f"ENTRENANDO VARIANTE {variant}")
        print(f"==========================================")

        if variant == "G1":
            tokenizer = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_id_map)
        elif variant == "G2":
            tokenizer = GenerativeTokenizer("G2", n_items=n_items, random_id_map=random_id_map)
        elif variant == "G3":
            tokenizer = GenerativeTokenizer("G3", n_items=n_items)
        else:
            continue

        item_trie = ItemTrie(tokenizer)
        t5_model = T5Recommender(tokenizer).to(device)

        train_ds = SequenceDataset(total_train_windows, tokenizer, max_context_tokens=150)
        train_loader = DataLoader(
            train_ds,
            batch_size=64 if not args.quick else 32,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.PAD_TOKEN_ID),
        )

        optimizer = torch.optim.AdamW(t5_model.parameters(), lr=args.lr, weight_decay=1e-4)

        t5_model.train()
        for epoch in range(1, args.epochs + 1):
            t0_ep = time.perf_counter()
            total_loss = 0.0
            for batch_in, batch_att, batch_lbl in train_loader:
                batch_in = batch_in.to(device)
                batch_att = batch_att.to(device)
                batch_lbl = batch_lbl.to(device)

                optimizer.zero_grad()
                out = t5_model(batch_in, attention_mask=batch_att, labels=batch_lbl)
                loss = out.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_in)

            t1_ep = time.perf_counter()
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {total_loss/len(train_ds):.4f} | Tiempo: {t1_ep - t0_ep:.2f}s")

        # 6. Evaluación de sensibilidad al tamaño del haz (20, 50, 100)
        for beam_size in args.beams:
            print(f"\n--- Evaluando {variant} con Haz (Beam Size) = {beam_size} ---")
            t0_eval = time.perf_counter()
            recs_by_user = {}

            for u in eval_users:
                history = bundle.user_train_seq[u]
                seen = bundle.user_seen_train.get(u, set())
                top_recs = t5_model.recommend_for_user(
                    history_item_indices=history,
                    item_trie=item_trie,
                    seen_items=seen,
                    device=device,
                    num_beams=beam_size,
                    top_k=50,
                    max_context_tokens=150,
                )
                recs_by_user[u] = top_recs

            t1_eval = time.perf_counter()
            latency_ms = (t1_eval - t0_eval) / len(eval_users) * 1000.0

            metrics = compute_metrics(
                recs_by_user,
                bundle.test_relevant_by_user,
                eval_users,
                bundle.cold_item_indices,
                bundle.item_buckets,
                n_items,
            )
            metrics["variant"] = variant
            metrics["beam_size"] = beam_size
            metrics["latency_ms_per_user"] = latency_ms

            print(f"NDCG@10: {metrics['ndcg@10']:.4f} | Recall@10: {metrics['recall@10']:.4f} | Recall@50: {metrics['recall@50']:.4f}")
            print(f"Cold Recall@50: {metrics['cold_recall@50']:.4f} | Cobertura@50: {metrics['coverage@50']:.2%} | Cola Larga: {metrics['long_tail_frac@50']:.2%}")
            print(f"Latencia por usuario: {latency_ms:.1f} ms")

            results_table.append(metrics)

    df_res = pd.DataFrame(results_table)
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO DE FASE 2")
    print("=" * 80)
    cols = ["variant", "beam_size", "ndcg@10", "recall@10", "recall@50", "cold_recall@50", "coverage@50", "long_tail_frac@50", "latency_ms_per_user"]
    print(df_res[cols].to_string(index=False))

    out_file = Path("scratch/phase2_results.json")
    with open(out_file, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nResultados guardados en {out_file}")


if __name__ == "__main__":
    main()
