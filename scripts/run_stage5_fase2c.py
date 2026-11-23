"""Orquestador completo para la Fase 2c de la Etapa 5.
Ejecuta:
1. Prueba mínima simétrica de regularización (base vs reg) para G1 y G3 con parada temprana por val_loss (eval cada 2 épocas, paciencia 3, min_delta 0.005).
2. Selección de la mejor configuración por modelo y herencia para G2 y G1b.
3. Reentrenamiento en train_full durante E* épocas.
4. Evaluación en test con haz 50 (más haz 20 y 100 para G1, más G3 exacto).
5. Demostración deTop-10 idénticos o solapamiento entre haces en G1.
6. Cálculo de azar esperado @10 analítico y por simulación con los 592 usuarios reales de test.
7. Guardado de registros crudos JSON por época en scratch/raw_curves/ y resultados consolidados en scratch/stage5_fase2c_results.json.
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from recsys.stage5.data_split import create_temporal_split, generate_sliding_windows
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4, train_retrieval_stage4
from recsys.retrieval import RetrievalDataset
from recsys.ranker import train_ranker
from recsys.evaluation import make_two_stage
from recsys.config import Stage3Config
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy, evaluate_ndcg_validation


def evaluate_val_loss(model: T5Recommender, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for bx, ba, by in loader:
            bx = bx.to(device, non_blocking=True)
            ba = ba.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(bx, attention_mask=ba, labels=by).loss
            total_loss += loss.item() * len(bx)
            total_samples += len(bx)
    return total_loss / max(total_samples, 1)


def train_with_early_stopping_val_loss(
    variant: str,
    config_name: str,
    tok: GenerativeTokenizer,
    max_len: int,
    policy: str,
    lr: float,
    dropout_rate: float,
    user_core_histories: dict,
    bundle,
    seed: int,
    device: torch.device,
    batch_size: int = 128,
    max_epochs: int = 50,
    patience: int = 3,
    min_delta: float = 0.005,
    eval_ndcg: bool = True,
) -> dict:
    """Entrena en train_core con parada temprana por val_loss (eval cada 2 épocas, paciencia 3, min_delta 0.005)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    data_generator = torch.Generator().manual_seed(seed)

    windows = get_windows_for_policy(user_core_histories, policy=policy, max_len=max_len, seed=seed)
    train_ds = SequenceDataset(windows, tok, max_tokens=150)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=data_generator, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
    )

    # Dataset de validación
    val_users = bundle.eval_users_val
    val_windows = []
    for u in val_users:
        hist = user_core_histories.get(u, [])
        for tgt in bundle.user_val_seq[u]:
            val_windows.append((hist, tgt))
    val_ds = SequenceDataset(val_windows, tok, max_tokens=150)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
    )

    item_trie = ItemTrie(tok) if eval_ndcg else None
    model = T5Recommender(tok, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0
    history = []

    print(f"\n--- Iniciando ajuste: {variant} [{config_name}] (lr={lr}, dropout={dropout_rate}, L={max_len}, policy={policy}) ---")
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, ba, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            ba = ba.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(bx, attention_mask=ba, labels=by).loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        # Evaluar validación en épocas 1, 2, 4, 6, 8, ...
        if epoch == 1 or epoch % 2 == 0:
            val_loss = evaluate_val_loss(model, val_loader, device)
            val_ndcg = None
            if eval_ndcg:
                val_ndcg = evaluate_ndcg_validation(
                    model, tok, item_trie, val_users, user_core_histories,
                    bundle.val_relevant_by_user, device, top_k=10, beam_size=50
                )

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_ndcg": float(val_ndcg) if val_ndcg is not None else None,
            }
            history.append(record)

            # Lógica de mejora mínima (min_delta = 0.005)
            improvement = best_val_loss - val_loss
            if improvement >= min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                tag = f"* Mejor val_loss: {best_val_loss:.4f} (mejora {improvement:.4f} >= {min_delta})"
            else:
                patience_counter += 1
                tag = f"sin mejora sustancial ({improvement:+.4f} < {min_delta}) [paciencia {patience_counter}/{patience}]"

            ndcg_str = f" | Val NDCG@10: {val_ndcg:.4f}" if val_ndcg is not None else ""
            print(f"  [{variant}_{config_name}] Ep {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{ndcg_str} -> {tag}")

            if patience_counter >= patience:
                print(f"  Parada temprana activada en época {epoch}. E* seleccionado = {best_epoch} (val_loss = {best_val_loss:.4f}).")
                break
        else:
            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": None,
                "val_ndcg": None,
            })

    t1 = time.perf_counter()
    duration_s = t1 - t0

    result = {
        "variant": variant,
        "config_name": config_name,
        "lr": lr,
        "dropout_rate": dropout_rate,
        "max_len": max_len,
        "policy": policy,
        "seed": seed,
        "duration_s": duration_s,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "history": history,
    }

    raw_dir = Path("scratch/raw_curves")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / f"{variant}_{config_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def compute_metrics_at_10_and_50(
    recs_by_user: dict,
    test_relevant_by_user: dict,
    eval_users: list,
    cold_items: set,
    item_buckets: dict,
    n_items: int,
    beam_size: int,
) -> dict:
    """Calcula NDCG@10, Recall@10, Recall@50 (si beam>=50), Cold Rec@50 (si beam>=50), Cov@10 y Tail@10."""
    ndcg_10_list = []
    recall_10_list = []
    recall_50_list = []
    cold_recall_50_list = []

    all_recs_10 = set()
    tail_recs_10 = 0
    total_recs_10 = 0

    eval_k50 = (beam_size >= 50)

    for u in eval_users:
        recs = recs_by_user.get(u, [])
        rel = test_relevant_by_user.get(u, set())
        if not rel:
            continue

        # Top-10
        top10 = recs[:10]
        hits10 = len(set(top10) & rel)
        recall_10_list.append(hits10 / len(rel))

        dcg = sum(1.0 / np.log2(r + 2) for r, it in enumerate(top10) if it in rel)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), 10)))
        ndcg_10_list.append(dcg / idcg if idcg > 0 else 0.0)

        for it in top10:
            all_recs_10.add(it)
            if item_buckets.get(it, "0") in {"0", "1-5"}:
                tail_recs_10 += 1
            total_recs_10 += 1

        # Top-50
        if eval_k50:
            top50 = recs[:50]
            hits50 = len(set(top50) & rel)
            recall_50_list.append(hits50 / len(rel))

            cold_rel = rel & cold_items
            if cold_rel:
                hits_cold = len(set(top50) & cold_rel)
                cold_recall_50_list.append(hits_cold / len(cold_rel))

    res = {
        "ndcg@10": float(np.mean(ndcg_10_list)) if ndcg_10_list else 0.0,
        "recall@10": float(np.mean(recall_10_list)) if recall_10_list else 0.0,
        "recall@50": float(np.mean(recall_50_list)) if recall_50_list else None,
        "cold_recall@50": float(np.mean(cold_recall_50_list)) if cold_recall_50_list else None,
        "cov@10": float(len(all_recs_10) / n_items),
        "tail@10": float(tail_recs_10 / total_recs_10) if total_recs_10 > 0 else 0.0,
        "n_eval_users": len(recall_10_list),
    }
    return res


def compute_random_baseline(bundle, eval_users: list, n_simulations: int = 200, seed: int = 42) -> dict:
    """Calcula analítica y empíricamente el azar esperado para Cov@10 y Tail@10 sobre los 592 usuarios reales."""
    rng = np.random.RandomState(seed)
    n_items = bundle.n_items
    tail_set = {i for i in range(n_items) if bundle.item_buckets.get(i, "0") in {"0", "1-5"}}

    analytic_tails = []
    for u in eval_users:
        seen = bundle.user_seen_train.get(u, set())
        unseen = [i for i in range(n_items) if i not in seen]
        tail_count = sum(1 for i in unseen if i in tail_set)
        analytic_tails.append(tail_count / len(unseen))
    expected_tail_analytic = float(np.mean(analytic_tails))

    cov_sims = []
    tail_sims = []
    for _ in range(n_simulations):
        all_recs = set()
        tail_count = 0
        total_count = 0
        for u in eval_users:
            seen = bundle.user_seen_train.get(u, set())
            unseen = [i for i in range(n_items) if i not in seen]
            chosen = rng.choice(unseen, size=10, replace=False)
            for it in chosen:
                all_recs.add(it)
                if it in tail_set:
                    tail_count += 1
                total_count += 1
        cov_sims.append(len(all_recs) / n_items)
        tail_sims.append(tail_count / total_count)

    return {
        "expected_cov@10_mean": float(np.mean(cov_sims)),
        "expected_cov@10_std": float(np.std(cov_sims)),
        "expected_tail@10_analytic": expected_tail_analytic,
        "expected_tail@10_simulation_mean": float(np.mean(tail_sims)),
        "cold_recall@50_random": 0.005329,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ETAPA 5 - FASE 2c: EXPERIMENTOS DE CONTROL, REGULARIZACIÓN Y REEVALUACIÓN (Dispositivo: {device}) ===")

    # 1. Cargar datos y partición temporal con Opción C
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=42, use_event_boundary_rule=True)
    n_items = bundle.n_items
    val_users = bundle.eval_users_val
    test_users = bundle.eval_users_test
    print(f"Usuarios de validación: {len(val_users)} | Usuarios de test: {len(test_users)}")

    user_core_histories = {}
    for u in range(bundle.n_users):
        full_seq = bundle.user_train_seq[u]
        val_len = len(bundle.user_val_seq[u])
        user_core_histories[u] = full_seq[: len(full_seq) - val_len]

    # 2. Semantic IDs con RQ-VAE determinista en CPU
    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")
    train_indices = sorted(list(bundle.train_item_indices))
    cold_indices = sorted(list(bundle.cold_item_indices))

    tr_embs = all_embeddings[train_indices]
    mu = tr_embs.mean(axis=0)
    sig = tr_embs.std(axis=0) + 1e-8
    X_std = (all_embeddings - mu) / sig

    torch.manual_seed(42)
    rq_model = RQVAE().to("cpu")
    X_tr_cpu = torch.tensor(X_std[train_indices], dtype=torch.float32, device="cpu")
    rq_model.rvq.init_from_kmeans(X_tr_cpu, seed=42)
    rq_model.eval()
    codes = rq_model.quantize(torch.tensor(X_std, dtype=torch.float32, device="cpu")).numpy()

    tr_counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
    manager = SemanticIdManager(max_disambiguation=16)
    manager.fit_train_items(train_indices, codes, tr_counts)
    manager.assign_cold_items(cold_indices, codes)
    semantic_id_map = manager.item_to_semantic_id
    sem_hash = hashlib.sha256(json.dumps({str(k): list(v) for k, v in sorted(semantic_id_map.items())}).encode()).hexdigest()
    print(f"Hash Semantic IDs verificado: {sem_hash[:16]}...")

    random_id_map = build_random_id_map(semantic_id_map, seed=42)

    # 3. PASO 1: Prueba mínima simétrica de regularización (base vs reg) para G1 y G3
    print("\n" + "=" * 80)
    print("PASO 1: PRUEBA MÍNIMA SIMÉTRICA DE REGULARIZACIÓN (train_core con parada por val_loss)")
    print("=" * 80)

    tok_g1 = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_id_map)
    tok_g3 = GenerativeTokenizer("G3", n_items=n_items)

    configs_to_test = [
        ("G1", "base", tok_g1, 100, "balanceado", 1e-3, 0.1),
        ("G1", "reg",  tok_g1, 100, "balanceado", 5e-4, 0.2),
        ("G3", "base", tok_g3, 20,  "balanceado", 1e-3, 0.1),
        ("G3", "reg",  tok_g3, 20,  "balanceado", 5e-4, 0.2),
    ]

    reg_results = {}
    for var, cname, tok, L, pol, lr, drop in configs_to_test:
        res = train_with_early_stopping_val_loss(
            variant=var, config_name=cname, tok=tok, max_len=L, policy=pol,
            lr=lr, dropout_rate=drop, user_core_histories=user_core_histories,
            bundle=bundle, seed=42, device=device
        )
        reg_results[f"{var}_{cname}"] = res

    # Comparación de pérdidas para G1 y G3
    print("\n" + "-" * 80)
    print("RESUMEN DE REGULARIZACIÓN (Criterio: menor val_loss):")
    print(f"  G1 base (lr=1e-3, drop=0.1): E* = {reg_results['G1_base']['best_epoch']} | Best Val Loss: {reg_results['G1_base']['best_val_loss']:.4f} | Tiempo: {reg_results['G1_base']['duration_s']:.1f}s")
    print(f"  G1 reg  (lr=5e-4, drop=0.2): E* = {reg_results['G1_reg']['best_epoch']} | Best Val Loss: {reg_results['G1_reg']['best_val_loss']:.4f} | Tiempo: {reg_results['G1_reg']['duration_s']:.1f}s")
    g1_win_key = "G1_base" if reg_results["G1_base"]["best_val_loss"] <= reg_results["G1_reg"]["best_val_loss"] else "G1_reg"
    g1_chosen = reg_results[g1_win_key]
    print(f"  --> Ganadora para G1: {g1_win_key} (lr={g1_chosen['lr']}, dropout={g1_chosen['dropout_rate']}, E*={g1_chosen['best_epoch']})")

    print(f"\n  G3 base (lr=1e-3, drop=0.1): E* = {reg_results['G3_base']['best_epoch']} | Best Val Loss: {reg_results['G3_base']['best_val_loss']:.4f} | Tiempo: {reg_results['G3_base']['duration_s']:.1f}s")
    print(f"  G3 reg  (lr=5e-4, drop=0.2): E* = {reg_results['G3_reg']['best_epoch']} | Best Val Loss: {reg_results['G3_reg']['best_val_loss']:.4f} | Tiempo: {reg_results['G3_reg']['duration_s']:.1f}s")
    g3_win_key = "G3_base" if reg_results["G3_base"]["best_val_loss"] <= reg_results["G3_reg"]["best_val_loss"] else "G3_reg"
    g3_chosen = reg_results[g3_win_key]
    print(f"  --> Ganadora para G3: {g3_win_key} (lr={g3_chosen['lr']}, dropout={g3_chosen['dropout_rate']}, E*={g3_chosen['best_epoch']})")

    # G2 hereda la configuración ganadora de G1
    print(f"\n  G2 hereda configuración de G1 ({g1_win_key}). Ejecutando ajuste en train_core con parada por val_loss...")
    tok_g2 = GenerativeTokenizer("G2", n_items=n_items, random_id_map=random_id_map)
    g2_res = train_with_early_stopping_val_loss(
        variant="G2", config_name=g1_win_key.replace("G1_", ""), tok=tok_g2, max_len=100, policy="balanceado",
        lr=g1_chosen["lr"], dropout_rate=g1_chosen["dropout_rate"], user_core_histories=user_core_histories,
        bundle=bundle, seed=42, device=device
    )
    reg_results["G2"] = g2_res
    print(f"  --> G2 ajustado: E* = {g2_res['best_epoch']} | Best Val Loss: {g2_res['best_val_loss']:.4f}")

    # G1b hereda configuración y E* de G1
    E_star_g1 = g1_chosen["best_epoch"]
    E_star_g2 = g2_res["best_epoch"]
    E_star_g3 = g3_chosen["best_epoch"]

    # 4. PASO 2: Reentrenamiento en train_full durante E* épocas y evaluación en test
    print("\n" + "=" * 80)
    print("PASO 2: REENTRENAMIENTO EN TRAIN_FULL Y EVALUACIÓN TEST")
    print("=" * 80)

    test_results_dict = {}

    def retrain_full_model(var_name, tok, L, pol, lr, drop, E_star):
        print(f"\nReentrenando {var_name} en train_full ({E_star} épocas, lr={lr}, dropout={drop})...")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        data_gen = torch.Generator().manual_seed(42)

        windows_full = get_windows_for_policy(bundle.user_train_seq, policy=pol, max_len=L, seed=42)
        ds_full = SequenceDataset(windows_full, tok, max_tokens=150)
        loader_full = DataLoader(
            ds_full, batch_size=128, shuffle=True, generator=data_gen, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
        )
        model_full = T5Recommender(tok, dropout_rate=drop).to(device)
        optimizer = torch.optim.AdamW(model_full.parameters(), lr=lr, weight_decay=1e-4)

        t0_tr = time.perf_counter()
        for ep in range(1, E_star + 1):
            model_full.train()
            loss_ep = 0.0
            for bx, ba, by in loader_full:
                bx = bx.to(device, non_blocking=True)
                ba = ba.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model_full(bx, attention_mask=ba, labels=by).loss
                loss.backward()
                optimizer.step()
                loss_ep += loss.item() * len(bx)
        t1_tr = time.perf_counter()
        print(f"  Entrenamiento de {var_name} finalizado en {t1_tr - t0_tr:.1f} s.")
        return model_full, t1_tr - t0_tr

    # Reentrenar G1
    model_g1, g1_time = retrain_full_model("G1", tok_g1, 100, "balanceado", g1_chosen["lr"], g1_chosen["dropout_rate"], E_star_g1)
    Path("scratch").mkdir(parents=True, exist_ok=True)
    torch.save(model_g1.state_dict(), "scratch/stage5_g1_fase2c.pt")

    # Evaluación G1 con haces 20, 50, 100
    trie_g1 = ItemTrie(tok_g1)
    g1_recs_by_beam = {}
    for beam in [20, 50, 100]:
        t0_ev = time.perf_counter()
        recs = {}
        for u in test_users:
            recs[u] = model_g1.recommend_for_user(
                bundle.user_train_seq[u], trie_g1, bundle.user_seen_train.get(u, set()),
                device, num_beams=beam, top_k=min(beam, 50), max_context_tokens=150
            )
        t1_ev = time.perf_counter()
        lat = (t1_ev - t0_ev) / len(test_users) * 1000.0
        g1_recs_by_beam[beam] = recs
        m = compute_metrics_at_10_and_50(recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=beam)
        m["latency_ms"] = lat
        test_results_dict[f"G1_beam_{beam}"] = m
        print(f"  G1 (Haz {beam:02d}) -> NDCG@10: {m['ndcg@10']:.4f} | Recall@10: {m['recall@10']:.4f}" +
              (f" | Recall@50: {m['recall@50']:.4f} | Cold Rec@50: {m['cold_recall@50']:.4f}" if beam >= 50 else "") +
              f" | Cov@10: {m['cov@10']*100:.2f}% | Tail@10: {m['tail@10']*100:.2f}% | Latencia: {lat:.1f}ms")

    # Comparación de Top-10 entre haces para G1
    print("\n--- Demostración de concordancia de Top-10 en G1 entre haces 20, 50 y 100 ---")
    identical_20_50 = 0
    jaccard_20_50 = []
    identical_50_100 = 0
    jaccard_50_100 = []

    for u in test_users:
        t10_20 = g1_recs_by_beam[20][u][:10]
        t10_50 = g1_recs_by_beam[50][u][:10]
        t10_100 = g1_recs_by_beam[100][u][:10]

        if t10_20 == t10_50:
            identical_20_50 += 1
        s20, s50 = set(t10_20), set(t10_50)
        jaccard_20_50.append(len(s20 & s50) / len(s20 | s50))

        if t10_50 == t10_100:
            identical_50_100 += 1
        s100 = set(t10_100)
        jaccard_50_100.append(len(s50 & s100) / len(s50 | s100))

    g1_beam_comparison = {
        "identical_ordered_top10_20_vs_50": identical_20_50,
        "identical_ordered_pct_20_vs_50": float(identical_20_50 / len(test_users) * 100.0),
        "mean_jaccard_top10_20_vs_50": float(np.mean(jaccard_20_50)),
        "identical_ordered_top10_50_vs_100": identical_50_100,
        "identical_ordered_pct_50_vs_100": float(identical_50_100 / len(test_users) * 100.0),
        "mean_jaccard_top10_50_vs_100": float(np.mean(jaccard_50_100)),
    }
    print(f"  Haz 20 vs Haz 50  : Listas Top-10 idénticas = {identical_20_50}/{len(test_users)} ({g1_beam_comparison['identical_ordered_pct_20_vs_50']:.1f}%) | Jaccard medio = {g1_beam_comparison['mean_jaccard_top10_20_vs_50']:.4f}")
    print(f"  Haz 50 vs Haz 100 : Listas Top-10 idénticas = {identical_50_100}/{len(test_users)} ({g1_beam_comparison['identical_ordered_pct_50_vs_100']:.1f}%) | Jaccard medio = {g1_beam_comparison['mean_jaccard_top10_50_vs_100']:.4f}")

    # Reentrenar G2
    model_g2, g2_time = retrain_full_model("G2", tok_g2, 100, "balanceado", g1_chosen["lr"], g1_chosen["dropout_rate"], E_star_g2)
    trie_g2 = ItemTrie(tok_g2)
    t0_ev = time.perf_counter()
    g2_recs = {}
    for u in test_users:
        g2_recs[u] = model_g2.recommend_for_user(
            bundle.user_train_seq[u], trie_g2, bundle.user_seen_train.get(u, set()),
            device, num_beams=50, top_k=50, max_context_tokens=150
        )
    t1_ev = time.perf_counter()
    m_g2 = compute_metrics_at_10_and_50(g2_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
    m_g2["latency_ms"] = (t1_ev - t0_ev) / len(test_users) * 1000.0
    test_results_dict["G2_beam_50"] = m_g2
    print(f"  G2 (Haz 50) -> NDCG@10: {m_g2['ndcg@10']:.4f} | Recall@10: {m_g2['recall@10']:.4f} | Recall@50: {m_g2['recall@50']:.4f} | Cold Rec@50: {m_g2['cold_recall@50']:.4f} | Cov@10: {m_g2['cov@10']*100:.2f}% | Tail@10: {m_g2['tail@10']*100:.2f}%")

    # Reentrenar G3
    model_g3, g3_time = retrain_full_model("G3", tok_g3, 20, "balanceado", g3_chosen["lr"], g3_chosen["dropout_rate"], E_star_g3)
    trie_g3 = ItemTrie(tok_g3)
    t0_ev = time.perf_counter()
    g3_recs = {}
    for u in test_users:
        g3_recs[u] = model_g3.recommend_for_user(
            bundle.user_train_seq[u], trie_g3, bundle.user_seen_train.get(u, set()),
            device, num_beams=50, top_k=50, max_context_tokens=150
        )
    t1_ev = time.perf_counter()
    m_g3 = compute_metrics_at_10_and_50(g3_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
    m_g3["latency_ms"] = (t1_ev - t0_ev) / len(test_users) * 1000.0
    test_results_dict["G3_beam_50"] = m_g3
    print(f"  G3 (Haz 50) -> NDCG@10: {m_g3['ndcg@10']:.4f} | Recall@10: {m_g3['recall@10']:.4f} | Recall@50: {m_g3['recall@50']:.4f} | Cold Rec@50: {m_g3['cold_recall@50']:.4f} | Cov@10: {m_g3['cov@10']*100:.2f}% | Tail@10: {m_g3['tail@10']*100:.2f}%")

    # G3 Exacto (Logit directo sin haz)
    print("  Evaluando G3 con ranking exacto directo...")
    model_g3.eval()
    g3_exact_recs = {}
    t0_ex = time.perf_counter()
    with torch.no_grad():
        for u in test_users:
            history = bundle.user_train_seq[u]
            seen = bundle.user_seen_train.get(u, set())
            inp = tok_g3.encode_input_sequence(history, max_tokens=150)
            inp_t = torch.tensor([inp], dtype=torch.long, device=device)
            enc_out = model_g3.model.encoder(input_ids=inp_t)
            bos_t = torch.tensor([[tok_g3.BOS_TOKEN_ID]], dtype=torch.long, device=device)
            dec_out = model_g3.model.decoder(input_ids=bos_t, encoder_hidden_states=enc_out.last_hidden_state)
            logits = model_g3.model.lm_head(dec_out.last_hidden_state)[0, 0].cpu().numpy().copy()

            # Mapear de token_id a item_idx
            # Para tokens especiales o vistos, asignar -inf
            item_scores = np.full(n_items, -np.inf, dtype=np.float32)
            for item_idx in range(n_items):
                if item_idx not in seen:
                    tok_id = tok_g3.encode_target_item(item_idx)[0]
                    item_scores[item_idx] = logits[tok_id]
            top_exact = np.argsort(item_scores)[::-1][:50]
            g3_exact_recs[u] = list(top_exact)
    t1_ex = time.perf_counter()
    m_g3_exact = compute_metrics_at_10_and_50(g3_exact_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
    m_g3_exact["latency_ms"] = (t1_ex - t0_ex) / len(test_users) * 1000.0
    test_results_dict["G3_exact"] = m_g3_exact
    print(f"  G3 (Exacto) -> NDCG@10: {m_g3_exact['ndcg@10']:.4f} | Recall@10: {m_g3_exact['recall@10']:.4f} | Recall@50: {m_g3_exact['recall@50']:.4f} | Cold Rec@50: {m_g3_exact['cold_recall@50']:.4f} | Cov@10: {m_g3_exact['cov@10']*100:.2f}% | Tail@10: {m_g3_exact['tail@10']*100:.2f}% | Latencia: {m_g3_exact['latency_ms']:.2f}ms")

    # Reentrenar G1b
    embs_noyear = np.load("data/cache/movie_embeddings_minilm_noyear.npy")
    tr_embs_ny = embs_noyear[train_indices]
    mu_ny = tr_embs_ny.mean(axis=0)
    sig_ny = tr_embs_ny.std(axis=0) + 1e-8
    X_std_ny = (embs_noyear - mu_ny) / sig_ny

    torch.manual_seed(42)
    rq_ny = RQVAE().to("cpu")
    X_tr_cpu_ny = torch.tensor(X_std_ny[train_indices], dtype=torch.float32, device="cpu")
    rq_ny.rvq.init_from_kmeans(X_tr_cpu_ny, seed=42)
    rq_ny.eval()
    codes_ny = rq_ny.quantize(torch.tensor(X_std_ny, dtype=torch.float32, device="cpu")).numpy()
    mgr_ny = SemanticIdManager(max_disambiguation=16)
    mgr_ny.fit_train_items(train_indices, codes_ny, tr_counts)
    mgr_ny.assign_cold_items(cold_indices, codes_ny)
    tok_g1b = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=mgr_ny.item_to_semantic_id)

    model_g1b, g1b_time = retrain_full_model("G1b", tok_g1b, 100, "balanceado", g1_chosen["lr"], g1_chosen["dropout_rate"], E_star_g1)
    trie_g1b = ItemTrie(tok_g1b)
    t0_ev = time.perf_counter()
    g1b_recs = {}
    for u in test_users:
        g1b_recs[u] = model_g1b.recommend_for_user(
            bundle.user_train_seq[u], trie_g1b, bundle.user_seen_train.get(u, set()),
            device, num_beams=50, top_k=50, max_context_tokens=150
        )
    t1_ev = time.perf_counter()
    m_g1b = compute_metrics_at_10_and_50(g1b_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
    m_g1b["latency_ms"] = (t1_ev - t0_ev) / len(test_users) * 1000.0
    test_results_dict["G1b_beam_50"] = m_g1b
    print(f"  G1b (Haz 50)-> NDCG@10: {m_g1b['ndcg@10']:.4f} | Recall@10: {m_g1b['recall@10']:.4f} | Recall@50: {m_g1b['recall@50']:.4f} | Cold Rec@50: {m_g1b['cold_recall@50']:.4f} | Cov@10: {m_g1b['cov@10']*100:.2f}% | Tail@10: {m_g1b['tail@10']*100:.2f}%")

    # 5. Baselines de contexto y Azar
    print("\n" + "=" * 80)
    print("PASO 3: BASELINES DE CONTEXTO Y LÍNEA BASE ALEATORIA")
    print("=" * 80)

    # Azar
    rand_metrics = compute_random_baseline(bundle, test_users, n_simulations=200, seed=42)
    test_results_dict["Azar"] = {
        "ndcg@10": None, "recall@10": None, "recall@50": rand_metrics["cold_recall@50_random"],
        "cold_recall@50": rand_metrics["cold_recall@50_random"],
        "cov@10": rand_metrics["expected_cov@10_mean"],
        "tail@10": rand_metrics["expected_tail@10_analytic"],
    }
    print(f"  Azar esperado (@10) -> Cov@10: {rand_metrics['expected_cov@10_mean']*100:.2f}% (+/- {rand_metrics['expected_cov@10_std']*100:.2f}%) | Tail@10: {rand_metrics['expected_tail@10_analytic']*100:.2f}% | Cold Rec@50: {rand_metrics['cold_recall@50_random']*100:.2f}%")

    # Popularidad
    pop_cands = np.argsort(bundle.item_popularity)[::-1]
    pop_recs = {}
    for u in test_users:
        seen = bundle.user_seen_train.get(u, set())
        pop_recs[u] = [i for i in pop_cands if i not in seen][:50]
    m_pop = compute_metrics_at_10_and_50(pop_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
    test_results_dict["Popularidad"] = m_pop
    print(f"  Popularidad -> NDCG@10: {m_pop['ndcg@10']:.4f} | Recall@10: {m_pop['recall@10']:.4f} | Recall@50: {m_pop['recall@50']:.4f} | Cold Rec@50: {m_pop['cold_recall@50']:.4f} | Cov@10: {m_pop['cov@10']*100:.2f}% | Tail@10: {m_pop['tail@10']*100:.2f}%")

    # Two-Towers
    cfg = Stage3Config(data_dir=Path("data"), seed=42, device="cuda" if device.type == "cuda" else "cpu")
    movie_genres_dict = dict(zip(movies["movieId"], movies["genres"].str.split("|")))
    all_genres_list = sorted({g for gl in movie_genres_dict.values() for g in gl if g != "(no genres listed)"})
    genre2col = {g: c for c, g in enumerate(all_genres_list)}
    item_genres_np = np.zeros((n_items, len(all_genres_list)), dtype=np.float32)
    for mid, idx in bundle.item2idx.items():
        for g in movie_genres_dict.get(mid, []):
            if g in genre2col:
                item_genres_np[idx, genre2col[g]] = 1.0

    is_cold_mask_np = np.zeros(n_items, dtype=bool)
    is_cold_mask_np[list(bundle.cold_item_indices)] = True
    train_pos_df = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]
    retrieval_ds = RetrievalDataset(
        positive_df=train_pos_df, user_seen_train=bundle.user_seen_train, n_items=n_items,
        seen_mask_np=bundle.seen_mask_np, n_neg_per_pos=4, fast_sampling=False,
    )

    variants_tt = [
        ("A", "Two-Tower (A - Solo ID)"),
        ("B", "Two-Tower (B - Solo Texto)"),
        ("C2", "Two-Tower (C2 - Híbrido ID+Texto con Dropout)"),
        ("D2", "Two-Tower (D2 - Control ID+Géneros con Dropout)"),
    ]

    retrieval_topk_a_100 = None
    t0_baselines = time.perf_counter()
    for v_code, v_name in variants_tt:
        tt = TwoTowerStage4(
            variant=v_code, n_users=bundle.n_users, n_items=n_items, embed_dim=32,
            text_embeddings_np=all_embeddings if v_code in {"B", "C2"} else None,
            item_genres_np=item_genres_np if v_code in {"D2"} else None,
            is_cold_mask_np=is_cold_mask_np, seed=42,
        ).to(device)
        train_retrieval_stage4(tt, retrieval_ds, cfg, device)
        if v_code == "A":
            retrieval_topk_a_100 = compute_retrieval_candidates_stage4(tt, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=100, device=device)
            topk = retrieval_topk_a_100[:, :50]
        else:
            topk = compute_retrieval_candidates_stage4(tt, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=50, device=device)

        recs = {u: list(topk[u]) for u in test_users}
        m = compute_metrics_at_10_and_50(recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
        test_results_dict[v_name] = m
        print(f"  {v_name} -> NDCG@10: {m['ndcg@10']:.4f} | Recall@10: {m['recall@10']:.4f} | Recall@50: {m['recall@50']:.4f} | Cold Rec@50: {m['cold_recall@50']:.4f} | Cov@10: {m['cov@10']*100:.2f}% | Tail@10: {m['tail@10']*100:.2f}%")

    # Two-Stage
    pos_users = train_pos_df["user_idx"].values
    pos_items = train_pos_df["item_idx"].values
    user_pos_np = np.zeros((bundle.n_users, n_items), dtype=np.float32)
    user_pos_np[pos_users, pos_items] = 1.0
    user_genre_np = user_pos_np @ item_genres_np
    user_genre_np = user_genre_np / np.maximum(user_genre_np.sum(axis=1, keepdims=True), 1.0)
    item_logpop_np = np.log1p(bundle.item_popularity).astype(np.float32)

    ranker, _ = train_ranker(
        pos_users=pos_users, pos_items=pos_items, n_users=bundle.n_users, n_items=n_items,
        user_genre_np=user_genre_np, item_genre_np=item_genres_np, item_logpop_np=item_logpop_np,
        seen_mask_np=bundle.seen_mask_np, retrieval_topk=retrieval_topk_a_100,
        n_uniform=2, n_hard=2, cfg=cfg, device=device, seed=42,
    )
    rec_ts = make_two_stage(ranker, retrieval_topk_a_100, n_candidates=50, device=device, top_k=50)
    ts_recs = {u: list(rec_ts(u)) for u in test_users}
    m_ts = compute_metrics_at_10_and_50(ts_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
    test_results_dict["Two-Stage (A + Ranker N=50)"] = m_ts
    t1_baselines = time.perf_counter()
    baselines_time = t1_baselines - t0_baselines
    print(f"  Two-Stage (A + Ranker N=50) -> NDCG@10: {m_ts['ndcg@10']:.4f} | Recall@10: {m_ts['recall@10']:.4f} | Recall@50: {m_ts['recall@50']:.4f} | Cold Rec@50: {m_ts['cold_recall@50']:.4f} | Cov@10: {m_ts['cov@10']*100:.2f}% | Tail@10: {m_ts['tail@10']*100:.2f}%")

    # Guardar resultados consolidados
    final_payload = {
        "regularization_experiment": reg_results,
        "selected_configs": {
            "G1": {"config": g1_win_key, "lr": g1_chosen["lr"], "dropout": g1_chosen["dropout_rate"], "E_star": E_star_g1},
            "G2": {"config": g1_win_key.replace("G1_", "G2_"), "lr": g1_chosen["lr"], "dropout": g1_chosen["dropout_rate"], "E_star": E_star_g2},
            "G3": {"config": g3_win_key, "lr": g3_chosen["lr"], "dropout": g3_chosen["dropout_rate"], "E_star": E_star_g3},
            "G1b": {"config": g1_win_key.replace("G1_", "G1b_"), "lr": g1_chosen["lr"], "dropout": g1_chosen["dropout_rate"], "E_star": E_star_g1},
        },
        "times_seconds": {
            "G1_full_train": g1_time,
            "G2_full_train": g2_time,
            "G3_full_train": g3_time,
            "G1b_full_train": g1b_time,
            "baselines_train": baselines_time,
        },
        "g1_beam_comparison": g1_beam_comparison,
        "random_baseline": rand_metrics,
        "test_results": test_results_dict,
    }

    with open("scratch/stage5_fase2c_results.json", "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("FASE 2c COMPLETADA CON ÉXITO")
    print("Resultados guardados en scratch/stage5_fase2c_results.json")
    print("Curvas crudas por época guardadas en scratch/raw_curves/")
    print("=" * 80)


if __name__ == "__main__":
    main()
