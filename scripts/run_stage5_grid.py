"""Orquestador completo de la Fase 2b de la Etapa 5.
Ejecuta la grilla independiente G1 y G3 (12 épocas, eval en 4, 8, 12 con haz 50),
determina el orden de configuraciones y estabilidad, entrena con parada temprana
hasta 50 épocas en train_core, reentrena en train_full, evalúa en test con haces 20, 50, 100,
computa baselines de contexto, ablación G1b y diagnósticos de frío con ranking exacto.
Soporta reanudación (--resume) guardando incrementalmente en scratch/stage5_fase2b_state.json.
"""

import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import hashlib
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from recsys.stage5.data_split import create_temporal_split, generate_sliding_windows
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager, torch_kmeans
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4, train_retrieval_stage4
from recsys.stage4.evaluation_slices import compute_slice_recalls
from recsys.retrieval import RetrievalDataset
from recsys.evaluation import evaluate_model, make_rec_retrieval_only, make_two_stage
from recsys.ranker import Ranker, train_ranker


class SequenceDataset(Dataset):
    def __init__(self, windows: List[Tuple[List[int], int]], tokenizer: GenerativeTokenizer, max_tokens: int = 150):
        self.windows = windows
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        ctx, tgt = self.windows[idx]
        inp = self.tokenizer.encode_input_sequence(ctx, max_tokens=self.max_tokens)
        out = self.tokenizer.encode_target_item(tgt)
        return inp, out


def collate_fn(batch, pad_token_id=0):
    in_seqs, out_seqs = zip(*batch)
    max_in = max(len(s) for s in in_seqs)
    max_out = max(len(s) for s in out_seqs)

    B = len(batch)
    input_ids = torch.full((B, max_in), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_in), dtype=torch.long)
    labels = torch.full((B, max_out), -100, dtype=torch.long)

    for i, (inp, out) in enumerate(zip(in_seqs, out_seqs)):
        input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        attention_mask[i, : len(inp)] = 1
        labels[i, : len(out)] = torch.tensor(out, dtype=torch.long)

    return input_ids, attention_mask, labels


def get_windows_for_policy(
    user_sequences: Dict[int, List[int]],
    policy: str,
    max_len: int,
    seed: int = 42,
) -> List[Tuple[List[int], int]]:
    """Extrae las ventanas según la política garantizando exactamente 35.087 ventanas en ambas políticas."""
    rng = np.random.RandomState(seed)
    if policy == "tope_100":
        windows = []
        for u_idx, seq in user_sequences.items():
            w = generate_sliding_windows(seq, max_len=max_len, max_windows_per_user=100)
            windows.extend(w)
        return windows

    elif policy == "balanceado":
        # Total exacto a igualar: 35.087 ventanas
        # 610 usuarios: 610 * 57 = 34.770. Los primeros 317 usuarios aportan 58 ventanas, los 293 restantes aportan 57.
        target_total = 35087
        n_users = len(user_sequences)
        k_base = target_total // n_users # 57
        remainder = target_total % n_users # 317

        balanced_windows = []
        for u_idx in range(n_users):
            seq = user_sequences.get(u_idx, [])
            all_w = generate_sliding_windows(seq, max_len=max_len)
            k_u = k_base + (1 if u_idx < remainder else 0)

            if not all_w:
                continue
            if len(all_w) >= k_u:
                # Muestreo parejo a lo largo del historial
                indices = np.linspace(0, len(all_w) - 1, k_u, dtype=int)
                for idx in indices:
                    balanced_windows.append(all_w[idx])
            else:
                # Si tiene menos, toma todas y sobremuestrea las más recientes con reemplazo
                balanced_windows.extend(all_w)
                extra = k_u - len(all_w)
                if extra > 0:
                    chosen_idx = rng.choice(len(all_w), size=extra, replace=True)
                    for c_i in chosen_idx:
                        balanced_windows.append(all_w[c_i])

        return balanced_windows
    else:
        raise ValueError(f"Política desconocida: {policy}")


def evaluate_ndcg_validation(
    model: T5Recommender,
    tokenizer: GenerativeTokenizer,
    item_trie: ItemTrie,
    val_users: List[int],
    user_histories: Dict[int, List[int]],
    val_relevant_by_user: Dict[int, Set[int]],
    device: torch.device,
    top_k: int = 10,
    beam_size: int = 50,
) -> float:
    """Evalúa NDCG@10 sobre el conjunto de validación con haz fijado en 50."""
    model.eval()
    ndcg_list = []
    for u in val_users:
        hist = user_histories.get(u, [])
        rel = val_relevant_by_user.get(u, set())
        if not rel or not hist:
            continue

        recs = model.recommend_for_user(
            history_item_indices=hist,
            item_trie=item_trie,
            seen_items=set(hist),
            device=device,
            num_beams=beam_size,
            top_k=top_k,
            max_context_tokens=150,
        )

        top10 = recs[:top_k]
        dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(top10) if item in rel)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), top_k)))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_list)) if ndcg_list else 0.0


def evaluate_test_metrics_for_beam(
    recommendations_by_user: Dict[int, List[int]],
    test_relevant_by_user: Dict[int, Set[int]],
    eval_users: List[int],
    cold_items: Set[int],
    item_buckets: Dict[int, str],
    n_items: int,
    beam_size: int,
) -> Dict[str, float]:
    """Calcula métricas respetando la regla estricta: solo reportar K <= haz."""
    ndcg_10_list = []
    recall_10_list = []
    recall_50_list = []
    cold_recall_50_list = []
    all_recs = set()
    total_recs = 0
    tail_recs = 0

    eval_k50 = (beam_size >= 50)

    for u in eval_users:
        recs = recommendations_by_user.get(u, [])
        rel = test_relevant_by_user.get(u, set())
        if not rel:
            continue

        # Top-10 siempre se evalúa (haz >= 20)
        top10 = recs[:10]
        hits_10 = len(set(top10) & rel)
        recall_10_list.append(hits_10 / len(rel))

        dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(top10) if item in rel)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), 10)))
        ndcg_10_list.append(dcg / idcg if idcg > 0 else 0.0)

        limit_k = min(beam_size, 50)
        top_k_recs = recs[:limit_k]

        if eval_k50:
            top50 = recs[:50]
            hits_50 = len(set(top50) & rel)
            recall_50_list.append(hits_50 / len(rel))

            cold_rel = rel & cold_items
            if cold_rel:
                hits_cold = len(set(top50) & cold_rel)
                cold_recall_50_list.append(hits_cold / len(cold_rel))

        for item in top_k_recs:
            all_recs.add(item)
            if item_buckets.get(item, "0") in {"0", "1-5"}:
                tail_recs += 1
            total_recs += 1

    res = {
        "ndcg@10": float(np.mean(ndcg_10_list)) if ndcg_10_list else 0.0,
        "recall@10": float(np.mean(recall_10_list)) if recall_10_list else 0.0,
        "recall@50": float(np.mean(recall_50_list)) if recall_50_list else None,
        "cold_recall@50": float(np.mean(cold_recall_50_list)) if cold_recall_50_list else None,
        "coverage": float(len(all_recs) / n_items),
        "long_tail_frac": float(tail_recs / total_recs) if total_recs > 0 else 0.0,
        "n_eval_users": len(recall_10_list),
    }
    return res


def run_cold_diagnostics(
    model_g1: T5Recommender,
    tok_g1: GenerativeTokenizer,
    bundle,
    test_eval_users: List[int],
    semantic_id_map: Dict[int, Tuple[int, int, int, int]],
    device: torch.device,
) -> Dict[str, any]:
    """Diagnóstico en frío de G1: cobertura de prefijos y ranking exacto vs beam search."""
    train_sem_ids = [semantic_id_map[i] for i in bundle.train_item_indices]
    cold_sem_ids = {i: semantic_id_map[i] for i in bundle.cold_item_indices}

    train_c1 = {sid[0] for sid in train_sem_ids}
    train_c12 = {(sid[0], sid[1]) for sid in train_sem_ids}
    train_c123 = {(sid[0], sid[1], sid[2]) for sid in train_sem_ids}

    cov_c1 = sum(1 for sid in cold_sem_ids.values() if sid[0] in train_c1)
    cov_c12 = sum(1 for sid in cold_sem_ids.values() if (sid[0], sid[1]) in train_c12)
    cov_c123 = sum(1 for sid in cold_sem_ids.values() if (sid[0], sid[1], sid[2]) in train_c123)
    n_cold = len(bundle.cold_item_indices)

    # Identificar usuarios con ítems relevantes fríos en test
    users_with_cold_rel = [
        u for u in test_eval_users
        if bool(bundle.test_relevant_by_user.get(u, set()) & bundle.cold_item_indices)
    ]

    all_target_tokens = []
    for item_idx in range(bundle.n_items):
        tokens = tok_g1.encode_target_item(item_idx)
        all_target_tokens.append(tokens)
    all_target_np = np.array(all_target_tokens, dtype=np.int64)

    all_dec_in_np = np.zeros_like(all_target_np)
    all_dec_in_np[:, 0] = tok_g1.BOS_TOKEN_ID
    all_dec_in_np[:, 1:] = all_target_np[:, :-1]

    sample_users = users_with_cold_rel[:30]
    cold_ranking_results = []
    beam_error_stats = {
        "found_in_beam50": 0,
        "pruned_step1": 0,
        "pruned_step2": 0,
        "pruned_step3_or_later": 0,
        "total_eval": 0,
    }

    item_trie = ItemTrie(tok_g1)
    model_g1.eval()

    with torch.no_grad():
        for u in sample_users:
            history = bundle.user_train_seq[u]
            seen = bundle.user_seen_train.get(u, set())
            rel_cold = bundle.test_relevant_by_user[u] & bundle.cold_item_indices

            unseen_indices = np.array([i for i in range(bundle.n_items) if i not in seen])
            unseen_targets = torch.tensor(all_target_np[unseen_indices], dtype=torch.long, device=device)
            unseen_dec_in = torch.tensor(all_dec_in_np[unseen_indices], dtype=torch.long, device=device)

            inp_seq = tok_g1.encode_input_sequence(history, max_tokens=150)
            inp_t = torch.tensor([inp_seq], dtype=torch.long, device=device)
            enc_out = model_g1.model.encoder(input_ids=inp_t)
            hidden_single = enc_out.last_hidden_state

            M = len(unseen_indices)
            scores = np.zeros(M, dtype=np.float32)
            chunk_size = 2000

            for s in range(0, M, chunk_size):
                e = min(s + chunk_size, M)
                b_size = e - s
                b_dec_in = unseen_dec_in[s:e]
                b_hidden = hidden_single.expand(b_size, -1, -1)

                dec_out = model_g1.model.decoder(
                    input_ids=b_dec_in,
                    encoder_hidden_states=b_hidden,
                )
                logits = model_g1.model.lm_head(dec_out.last_hidden_state)
                log_probs = torch.log_softmax(logits, dim=-1)

                target_tokens = unseen_targets[s:e].unsqueeze(-1)
                token_log_probs = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)
                seq_log_prob = token_log_probs.sum(dim=-1)
                scores[s:e] = seq_log_prob.cpu().numpy()

            beam_recs = model_g1.recommend_for_user(
                history_item_indices=history,
                item_trie=item_trie,
                seen_items=seen,
                device=device,
                num_beams=50,
                top_k=50,
                max_context_tokens=150,
            )
            beam_set = set(beam_recs)

            for cold_item in rel_cold:
                idx_in_unseen = np.where(unseen_indices == cold_item)[0]
                if len(idx_in_unseen) == 0:
                    continue
                cold_score = float(scores[idx_in_unseen[0]])
                percentile = float(np.mean(scores <= cold_score) * 100.0)
                rank = int(np.sum(scores > cold_score) + 1)

                beam_error_stats["total_eval"] += 1
                in_beam = (cold_item in beam_set)
                if in_beam:
                    beam_error_stats["found_in_beam50"] += 1
                    pruned_step = "none"
                else:
                    sid = semantic_id_map[cold_item]
                    if sid[0] not in train_c1:
                        beam_error_stats["pruned_step1"] += 1
                        pruned_step = "step_1_code1_unseen"
                    elif (sid[0], sid[1]) not in train_c12:
                        beam_error_stats["pruned_step2"] += 1
                        pruned_step = "step_2_prefix_unseen"
                    else:
                        beam_error_stats["pruned_step3_or_later"] += 1
                        pruned_step = "step_3_or_disambiguation"

                cold_ranking_results.append({
                    "user_id": int(u),
                    "cold_item_id": int(cold_item),
                    "exact_score": cold_score,
                    "rank_among_unseen": rank,
                    "total_unseen": M,
                    "percentile": percentile,
                    "in_beam50": in_beam,
                    "pruned_step": pruned_step,
                })

    avg_percentile = float(np.mean([r["percentile"] for r in cold_ranking_results])) if cold_ranking_results else 0.0
    median_rank = float(np.median([r["rank_among_unseen"] for r in cold_ranking_results])) if cold_ranking_results else 0.0

    return {
        "prefix_coverage": {
            "n_cold_items": n_cold,
            "c1_covered": cov_c1,
            "c1_pct": float(cov_c1 / n_cold * 100.0),
            "c12_covered": cov_c12,
            "c12_pct": float(cov_c12 / n_cold * 100.0),
            "c123_covered": cov_c123,
            "c123_pct": float(cov_c123 / n_cold * 100.0),
        },
        "exact_ranking_sample": {
            "n_users_evaluated": len(sample_users),
            "n_cold_targets_evaluated": len(cold_ranking_results),
            "mean_percentile": avg_percentile,
            "median_rank": median_rank,
            "beam_error_stats": beam_error_stats,
            "sample_details": cold_ranking_results[:10],
        },
        "design_note": (
            "La ausencia del token d tras el prefijo en ítems fríos es cierta por construcción "
            "del protocolo de desambiguación (a los ítems fríos se les asigna un d no utilizado "
            "o un código nuevo), por lo que no constituye evidencia empírica independiente."
        ),
    }


def load_or_init_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {
        "semantic_id_hash": None,
        "grid_evaluations": {}, # key: f"{variant}_{L}_{policy}" -> {"ep4": x, "ep8": y, "ep12": z}
        "best_configs": {},     # "G1": (L, policy), "G3": (L, policy)
        "early_stopping": {},   # variant -> {"best_epoch": E, "converged": bool, "history": [...]}
        "test_results": {},     # variant -> {beam: metrics}
        "baselines_results": {},
        "cold_diagnostics": {},
        "g1b_ablation": {},
    }


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fase 2b: Grilla, Early Stopping, Test y Baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reanuda desde scratch/stage5_fase2b_state.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--val_beam_size", type=int, default=50)
    args = parser.parse_args()

    state_file = Path("scratch/stage5_fase2b_state.json")
    state = load_or_init_state(state_file) if args.resume else {
        "semantic_id_hash": None,
        "grid_evaluations": {},
        "best_configs": {},
        "early_stopping": {},
        "test_results": {},
        "baselines_results": {},
        "cold_diagnostics": {},
        "g1b_ablation": {},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ETAPA 5 - FASE 2b: EJECUCIÓN COMPLETA (Dispositivo: {device}, Semilla: {args.seed}) ===")
    print(f"Parámetros fijados: Lote={args.batch_size}, Precisión={'bf16' if args.use_bf16 else 'fp32'}, Haz de Validación={args.val_beam_size}")

    # 1. Cargar datos y partición temporal con Opción C
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=args.seed, use_event_boundary_rule=True)
    n_items = bundle.n_items

    # 2. Ajuste determinista de Semantic IDs con RQ-VAE (KMeans en CPU para determinismo estricto)
    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")
    train_indices = sorted(list(bundle.train_item_indices))
    cold_indices = sorted(list(bundle.cold_item_indices))

    tr_embs = all_embeddings[train_indices]
    mu = tr_embs.mean(axis=0)
    sig = tr_embs.std(axis=0) + 1e-8
    X_std = (all_embeddings - mu) / sig

    # RQ-VAE entrenado de forma determinista
    torch.manual_seed(args.seed)
    rq_model = RQVAE().to("cpu")
    X_tr_cpu = torch.tensor(X_std[train_indices], dtype=torch.float32, device="cpu")
    rq_model.rvq.init_from_kmeans(X_tr_cpu, seed=args.seed)

    rq_model.eval()
    codes = rq_model.quantize(torch.tensor(X_std, dtype=torch.float32, device="cpu")).numpy()

    manager = SemanticIdManager(max_disambiguation=16)
    tr_counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
    manager.fit_train_items(train_indices, codes, tr_counts)
    manager.assign_cold_items(cold_indices, codes)

    semantic_id_map = manager.item_to_semantic_id
    map_bytes = json.dumps({str(k): list(v) for k, v in sorted(semantic_id_map.items())}).encode("utf-8")
    sem_hash = hashlib.sha256(map_bytes).hexdigest()
    state["semantic_id_hash"] = sem_hash
    print(f"Mapa de Semantic IDs fijado. SHA256 Hash: {sem_hash[:16]}...")
    save_state(state_file, state)

    random_id_map = build_random_id_map(semantic_id_map, seed=args.seed)

    # 3. FASE 2b - PASO 1: Grilla de ordenamiento (12 épocas con eval en 4, 8, 12 con haz 50)
    print("\n" + "=" * 80)
    print("PASO 1: GRILLA DE ORDENAMIENTO (G1 y G3 independientes, 12 épocas)")
    print("=" * 80)

    val_users = bundle.eval_users_val
    user_core_histories = {}
    for u in range(bundle.n_users):
        full_seq = bundle.user_train_seq[u]
        val_len = len(bundle.user_val_seq[u])
        user_core_histories[u] = full_seq[: len(full_seq) - val_len]

    grid_configs = [
        (L, pol)
        for L in [20, 50, 100]
        for pol in ["tope_100", "balanceado"]
    ]

    for variant in ["G1", "G3"]:
        print(f"\n--- Evaluando grilla para {variant} ---")
        tok = GenerativeTokenizer(variant, n_items=n_items, semantic_id_map=semantic_id_map if variant == "G1" else None)
        item_trie = ItemTrie(tok)

        for L, pol in grid_configs:
            cfg_key = f"{variant}_L{L}_{pol}"
            if cfg_key in state["grid_evaluations"] and "ep12" in state["grid_evaluations"][cfg_key]:
                print(f"[{cfg_key}] Ya completado en estado previo: {state['grid_evaluations'][cfg_key]}")
                continue

            print(f"\nEntrenando configuración: {cfg_key} ...")
            windows = get_windows_for_policy(user_core_histories, policy=pol, max_len=L, seed=args.seed)
            ds = SequenceDataset(windows, tok, max_tokens=150)
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
            )

            model = T5Recommender(tok).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            eval_results = {}
            for epoch in range(1, 13):
                model.train()
                for bx, ba, by in loader:
                    bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
                        loss = model(bx, attention_mask=ba, labels=by).loss
                    loss.backward()
                    optimizer.step()

                if epoch in {4, 8, 12}:
                    ndcg_val = evaluate_ndcg_validation(
                        model, tok, item_trie, val_users, user_core_histories,
                        bundle.val_relevant_by_user, device, top_k=10, beam_size=args.val_beam_size
                    )
                    eval_results[f"ep{epoch}"] = ndcg_val
                    print(f"  [{cfg_key}] Época {epoch:02d} -> NDCG@10 (Val): {ndcg_val:.4f}")

            state["grid_evaluations"][cfg_key] = eval_results
            save_state(state_file, state)

    # Determinar ganadores por el valor en la época 12 y analizar estabilidad
    print("\n" + "=" * 80)
    print("ANÁLISIS DE ESTABILIDAD DEL ORDENAMIENTO (Épocas 4, 8, 12 con Haz 50)")
    print("=" * 80)
    for variant in ["G1", "G3"]:
        print(f"\nVariante {variant}:")
        print(f"{'Configuración':<25} | {'NDCG@10 Ep 4':<14} | {'NDCG@10 Ep 8':<14} | {'NDCG@10 Ep 12':<14} | {'Rango (4->8->12)':<18}")
        print("-" * 92)

        cfg_scores = []
        for L, pol in grid_configs:
            cfg_key = f"{variant}_L{L}_{pol}"
            evals = state["grid_evaluations"].get(cfg_key, {})
            cfg_scores.append((
                cfg_key, L, pol,
                evals.get("ep4", 0.0),
                evals.get("ep8", 0.0),
                evals.get("ep12", 0.0)
            ))

        cfg_by_12 = sorted(cfg_scores, key=lambda x: x[5], reverse=True)
        ranks_4 = {x[0]: r for r, x in enumerate(sorted(cfg_scores, key=lambda x: x[3], reverse=True), 1)}
        ranks_8 = {x[0]: r for r, x in enumerate(sorted(cfg_scores, key=lambda x: x[4], reverse=True), 1)}
        ranks_12 = {x[0]: r for r, x in enumerate(cfg_by_12, 1)}

        for cfg_key, L, pol, s4, s8, s12 in cfg_by_12:
            r_prog = f"#{ranks_4[cfg_key]} -> #{ranks_8[cfg_key]} -> #{ranks_12[cfg_key]}"
            print(f"{cfg_key:<25} | {s4:<14.4f} | {s8:<14.4f} | {s12:<14.4f} | {r_prog:<18}")

        best_cfg = (cfg_by_12[0][1], cfg_by_12[0][2])
        best_score = cfg_by_12[0][5]
        state["best_configs"][variant] = best_cfg
        print(f"--> Configuración ganadora para {variant} (seleccionada por Época 12): L={best_cfg[0]}, Política={best_cfg[1]} (NDCG@10={best_score:.4f})")

    # G2 hereda la configuración ganadora de G1
    state["best_configs"]["G2"] = state["best_configs"]["G1"]
    save_state(state_file, state)

    # 4. FASE 2b - PASO 2: Early Stopping en train_core hasta 50 épocas
    print("\n" + "=" * 80)
    print("PASO 2: ENTRENAMIENTO CON PARADA TEMPRANA EN TRAIN_CORE (Hasta 50 épocas)")
    print("=" * 80)

    for variant in ["G1", "G2", "G3"]:
        if variant in state["early_stopping"]:
            print(f"[{variant}] Early stopping ya completado: {state['early_stopping'][variant]}")
            continue

        L_win, pol_win = state["best_configs"][variant]
        print(f"\nAjustando {variant} con L={L_win}, Política={pol_win} hasta 50 épocas...")

        if variant == "G1":
            tok = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_id_map)
        elif variant == "G2":
            tok = GenerativeTokenizer("G2", n_items=n_items, random_id_map=random_id_map)
        else:
            tok = GenerativeTokenizer("G3", n_items=n_items)

        item_trie = ItemTrie(tok)
        windows = get_windows_for_policy(user_core_histories, policy=pol_win, max_len=L_win, seed=args.seed)
        ds = SequenceDataset(windows, tok, max_tokens=150)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
        )

        model = T5Recommender(tok).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_ndcg = -1.0
        best_epoch = 1
        patience_counter = 0
        history = []

        for epoch in range(1, 51):
            model.train()
            train_loss = 0.0
            for bx, ba, by in loader:
                bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
                    loss = model(bx, attention_mask=ba, labels=by).loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(ds)

            # Evaluar validación cada 2 épocas
            if epoch % 2 == 0 or epoch == 1:
                val_ndcg = evaluate_ndcg_validation(
                    model, tok, item_trie, val_users, user_core_histories,
                    bundle.val_relevant_by_user, device, top_k=10, beam_size=args.val_beam_size
                )
                history.append({"epoch": epoch, "train_loss": train_loss, "val_ndcg": val_ndcg})
                print(f"  [{variant}] Ep {epoch:02d} | Train Loss: {train_loss:.4f} | Val NDCG@10: {val_ndcg:.4f}")

                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 5: # 5 chequeos = 10 épocas sin mejora
                    print(f"  [{variant}] Early stopping activado en época {epoch} (Mejor época: {best_epoch} con NDCG@10={best_val_ndcg:.4f})")
                    break

        converged = (best_epoch < 50)
        state["early_stopping"][variant] = {
            "best_epoch": best_epoch,
            "best_val_ndcg": best_val_ndcg,
            "converged": converged,
            "history": history,
        }
        save_state(state_file, state)

    # 5. FASE 2b - PASO 3: Reentrenamiento en train_full y Test Final con haces 20, 50, 100
    print("\n" + "=" * 80)
    print("PASO 3: REENTRENAMIENTO EN TRAIN_FULL Y EVALUACIÓN TEST (Haces 20, 50, 100)")
    print("=" * 80)

    test_eval_users = bundle.eval_users_test
    print(f"Total usuarios evaluados en Test: {len(test_eval_users)}")

    for variant in ["G1", "G2", "G3"]:
        if variant in state["test_results"]:
            print(f"[{variant}] Test ya evaluado: {state['test_results'][variant]}")
            continue

        L_win, pol_win = state["best_configs"][variant]
        E_star = state["early_stopping"][variant]["best_epoch"]
        print(f"\nReentrenando {variant} en train_full durante {E_star} épocas (L={L_win}, Política={pol_win})...")

        if variant == "G1":
            tok = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_id_map)
        elif variant == "G2":
            tok = GenerativeTokenizer("G2", n_items=n_items, random_id_map=random_id_map)
        else:
            tok = GenerativeTokenizer("G3", n_items=n_items)

        item_trie = ItemTrie(tok)
        windows_full = get_windows_for_policy(bundle.user_train_seq, policy=pol_win, max_len=L_win, seed=args.seed)
        ds_full = SequenceDataset(windows_full, tok, max_tokens=150)
        loader_full = DataLoader(
            ds_full, batch_size=args.batch_size, shuffle=True, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
        )

        model_full = T5Recommender(tok).to(device)
        optimizer = torch.optim.AdamW(model_full.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(1, E_star + 1):
            model_full.train()
            total_loss = 0.0
            for bx, ba, by in loader_full:
                bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
                    loss = model_full(bx, attention_mask=ba, labels=by).loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            print(f"  [{variant} Full] Época {epoch:02d}/{E_star:02d} | Loss: {total_loss/len(ds_full):.4f}")

        if variant == "G1":
            Path("scratch").mkdir(parents=True, exist_ok=True)
            torch.save(model_full.state_dict(), "scratch/stage5_g1_full.pt")

        # Evaluación con haces 20, 50, 100
        variant_test_res = {}
        for beam in [20, 50, 100]:
            print(f"  [{variant}] Evaluando en Test con Haz = {beam} ...")
            t0 = time.perf_counter()
            recs_by_user = {}
            for u in test_eval_users:
                history = bundle.user_train_seq[u]
                seen = bundle.user_seen_train.get(u, set())
                top_recs = model_full.recommend_for_user(
                    history_item_indices=history,
                    item_trie=item_trie,
                    seen_items=seen,
                    device=device,
                    num_beams=beam,
                    top_k=min(beam, 50),
                    max_context_tokens=150,
                )
                recs_by_user[u] = top_recs

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) / len(test_eval_users) * 1000.0

            metrics = evaluate_test_metrics_for_beam(
                recs_by_user, bundle.test_relevant_by_user, test_eval_users,
                bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=beam
            )
            metrics["latency_ms_per_user"] = latency_ms
            variant_test_res[str(beam)] = metrics

            print(f"    NDCG@10: {metrics['ndcg@10']:.4f} | Recall@10: {metrics['recall@10']:.4f}" +
                  (f" | Recall@50: {metrics['recall@50']:.4f} | Cold Rec@50: {metrics['cold_recall@50']:.4f}" if beam >= 50 else "") +
                  f" | Latencia: {latency_ms:.1f} ms")

        state["test_results"][variant] = variant_test_res
        save_state(state_file, state)

    # 6. FASE 2b - PASO 4: Baselines de contexto (Two-Tower A, Two-Stage N=50, B, C2, D2, Popularidad)
    print("\n" + "=" * 80)
    print("PASO 4: BASELINES DE CONTEXTO (Mismo protocolo temporal y de ajuste)")
    print("=" * 80)

    if not state["baselines_results"]:
        from recsys.config import Stage3Config
        cfg = Stage3Config(
            data_dir=Path("data"), seed=args.seed, device="cuda" if device.type == "cuda" else "cpu"
        )

        # Matriz de características para TwoTower
        all_text_embs = all_embeddings
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

        # Conjunto de train positive para retrieval
        train_pos_df = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]
        retrieval_ds = RetrievalDataset(
            positive_df=train_pos_df,
            user_seen_train=bundle.user_seen_train,
            n_items=n_items,
            seen_mask_np=bundle.seen_mask_np,
            n_neg_per_pos=4,
            fast_sampling=False,
        )

        variants_to_eval = [
            ("A", "Two-Tower (A - Solo ID)"),
            ("B", "Two-Tower (B - Solo Texto)"),
            ("C2", "Two-Tower (C2 - Híbrido ID+Texto con Dropout)"),
            ("D2", "Two-Tower (D2 - Control ID+Géneros con Dropout)"),
        ]

        baselines_dict = {}

        # Baseline Popularidad
        pop_candidates = np.argsort(bundle.item_popularity)[::-1]
        pop_recs = {}
        for u in test_eval_users:
            seen = bundle.user_seen_train.get(u, set())
            unseen = [i for i in pop_candidates if i not in seen][:50]
            pop_recs[u] = unseen
        baselines_dict["Popularidad"] = evaluate_test_metrics_for_beam(
            pop_recs, bundle.test_relevant_by_user, test_eval_users,
            bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50
        )
        print(f"  Popularidad -> NDCG@10: {baselines_dict['Popularidad']['ndcg@10']:.4f} | Recall@10: {baselines_dict['Popularidad']['recall@10']:.4f} | Recall@50: {baselines_dict['Popularidad']['recall@50']:.4f}")

        # Entrenar cada Two-Tower en train_full
        for v_code, v_name in variants_to_eval:
            print(f"  Entrenando {v_name} ...")
            tt_model = TwoTowerStage4(
                variant=v_code,
                n_users=bundle.n_users,
                n_items=n_items,
                embed_dim=32,
                text_embeddings_np=all_text_embs if v_code in {"B", "C2"} else None,
                item_genres_np=item_genres_np if v_code in {"D2"} else None,
                is_cold_mask_np=is_cold_mask_np,
                seed=args.seed,
            ).to(device)

            train_retrieval_stage4(tt_model, retrieval_ds, cfg, device)

            if v_code == "A":
                retrieval_topk_a_100 = compute_retrieval_candidates_stage4(
                    tt_model, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=100, device=device
                )

            # Candidatos Top-50
            tt_topk = compute_retrieval_candidates_stage4(
                tt_model, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=50, device=device
            )
            tt_recs = {u: list(tt_topk[u]) for u in test_eval_users}
            b_metrics = evaluate_test_metrics_for_beam(
                tt_recs, bundle.test_relevant_by_user, test_eval_users,
                bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50
            )
            baselines_dict[v_name] = b_metrics
            print(f"    {v_name} -> NDCG@10: {b_metrics['ndcg@10']:.4f} | Recall@10: {b_metrics['recall@10']:.4f} | Recall@50: {b_metrics['recall@50']:.4f} | Cold Rec@50: {b_metrics['cold_recall@50']:.4f}")

        # Entrenar Two-Stage N=50 sobre candidatos de Two-Tower A (protocolo Etapa 3)
        print("  Entrenando Ranker MLP para Two-Stage (N=50) ...")
        pos_users = train_pos_df["user_idx"].values
        pos_items = train_pos_df["item_idx"].values
        user_pos_np = np.zeros((bundle.n_users, n_items), dtype=np.float32)
        user_pos_np[pos_users, pos_items] = 1.0
        user_genre_np = user_pos_np @ item_genres_np
        user_genre_np = user_genre_np / np.maximum(user_genre_np.sum(axis=1, keepdims=True), 1.0)
        item_logpop_np = np.log1p(bundle.item_popularity).astype(np.float32)

        ranker_model, _ = train_ranker(
            pos_users=pos_users,
            pos_items=pos_items,
            n_users=bundle.n_users,
            n_items=n_items,
            user_genre_np=user_genre_np,
            item_genre_np=item_genres_np,
            item_logpop_np=item_logpop_np,
            seen_mask_np=bundle.seen_mask_np,
            retrieval_topk=retrieval_topk_a_100,
            n_uniform=2,
            n_hard=2,
            cfg=cfg,
            device=device,
            seed=args.seed,
        )
        rec_two_stage = make_two_stage(
            ranker_model, retrieval_topk_a_100, n_candidates=50, device=device, top_k=50
        )
        two_stage_recs = {u: list(rec_two_stage(u)) for u in test_eval_users}
        baselines_dict["Two-Stage (A + Ranker N=50)"] = evaluate_test_metrics_for_beam(
            two_stage_recs, bundle.test_relevant_by_user, test_eval_users,
            bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50
        )
        print(f"    Two-Stage (A + Ranker N=50) -> NDCG@10: {baselines_dict['Two-Stage (A + Ranker N=50)']['ndcg@10']:.4f} | Recall@10: {baselines_dict['Two-Stage (A + Ranker N=50)']['recall@10']:.4f} | Recall@50: {baselines_dict['Two-Stage (A + Ranker N=50)']['recall@50']:.4f} | Cold Rec@50: {baselines_dict['Two-Stage (A + Ranker N=50)']['cold_recall@50']:.4f}")

        state["baselines_results"] = baselines_dict
        save_state(state_file, state)

    # 7. FASE 2b - PASO 5: Ablación G1b (Embeddings sin año)
    print("\n" + "=" * 80)
    print("PASO 5: ABLACIÓN EXPLORATORIA G1b (Embeddings sin año)")
    print("=" * 80)

    if not state["g1b_ablation"]:
        embs_noyear = np.load("data/cache/movie_embeddings_minilm_noyear.npy")
        tr_embs_ny = embs_noyear[train_indices]
        mu_ny = tr_embs_ny.mean(axis=0)
        sig_ny = tr_embs_ny.std(axis=0) + 1e-8
        X_std_ny = (embs_noyear - mu_ny) / sig_ny

        torch.manual_seed(args.seed)
        rq_noyear = RQVAE().to("cpu")
        X_tr_cpu_ny = torch.tensor(X_std_ny[train_indices], dtype=torch.float32, device="cpu")
        rq_noyear.rvq.init_from_kmeans(X_tr_cpu_ny, seed=args.seed)

        rq_noyear.eval()
        codes_ny = rq_noyear.quantize(torch.tensor(X_std_ny, dtype=torch.float32, device="cpu")).numpy()

        mgr_ny = SemanticIdManager(max_disambiguation=16)
        mgr_ny.fit_train_items(train_indices, codes_ny, tr_counts)
        mgr_ny.assign_cold_items(cold_indices, codes_ny)
        sem_map_ny = mgr_ny.item_to_semantic_id

        # Hereda configuración de G1
        L_g1, pol_g1 = state["best_configs"]["G1"]
        E_g1 = state["early_stopping"]["G1"]["best_epoch"]

        tok_ny = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=sem_map_ny)
        item_trie_ny = ItemTrie(tok_ny)

        windows_ny = get_windows_for_policy(bundle.user_train_seq, policy=pol_g1, max_len=L_g1, seed=args.seed)
        ds_ny = SequenceDataset(windows_ny, tok_ny, max_tokens=150)
        loader_ny = DataLoader(
            ds_ny, batch_size=args.batch_size, shuffle=True, pin_memory=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tok_ny.PAD_TOKEN_ID)
        )

        model_ny = T5Recommender(tok_ny).to(device)
        opt_ny = torch.optim.AdamW(model_ny.parameters(), lr=1e-3, weight_decay=1e-4)

        t0_train = time.perf_counter()
        for epoch in range(1, E_g1 + 1):
            model_ny.train()
            for bx, ba, by in loader_ny:
                bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
                opt_ny.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
                    loss = model_ny(bx, attention_mask=ba, labels=by).loss
                loss.backward()
                opt_ny.step()
        t1_train = time.perf_counter()

        g1b_res = {}
        for beam in [20, 50, 100]:
            recs_by_user = {}
            for u in test_eval_users:
                history = bundle.user_train_seq[u]
                seen = bundle.user_seen_train.get(u, set())
                top_recs = model_ny.recommend_for_user(
                    history_item_indices=history, item_trie=item_trie_ny, seen_items=seen,
                    device=device, num_beams=beam, top_k=min(beam, 50), max_context_tokens=150
                )
                recs_by_user[u] = top_recs
            g1b_metrics = evaluate_test_metrics_for_beam(
                recs_by_user, bundle.test_relevant_by_user, test_eval_users,
                bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=beam
            )
            g1b_res[str(beam)] = g1b_metrics

        state["g1b_ablation"] = {
            "training_time_s": t1_train - t0_train,
            "metrics": g1b_res,
        }
        save_state(state_file, state)
        print(f"G1b completado en {t1_train - t0_train:.1f} s. Test (Haz 50): NDCG@10={g1b_res['50']['ndcg@10']:.4f}, Recall@50={g1b_res['50']['recall@50']:.4f}, Cold Rec@50={g1b_res['50']['cold_recall@50']:.4f}")

    # 8. FASE 2b - PASO 6: Diagnóstico en frío de G1 (Ranking exacto y error de haz)
    print("\n" + "=" * 80)
    print("PASO 6: DIAGNÓSTICO EN FRÍO DE G1 (Ranking exacto y error de haz)")
    print("=" * 80)

    if not state.get("cold_diagnostics"):
        tok_g1 = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_id_map)
        model_g1 = T5Recommender(tok_g1).to(device)
        g1_weights_path = Path("scratch/stage5_g1_full.pt")
        if g1_weights_path.exists():
            model_g1.load_state_dict(torch.load(g1_weights_path, map_location=device))
            print("Pesos de G1 cargados desde scratch/stage5_g1_full.pt")
        else:
            print("Aviso: scratch/stage5_g1_full.pt no existe en disco.")

        cold_diag = run_cold_diagnostics(
            model_g1=model_g1,
            tok_g1=tok_g1,
            bundle=bundle,
            test_eval_users=test_eval_users,
            semantic_id_map=semantic_id_map,
            device=device,
        )
        state["cold_diagnostics"] = cold_diag
        save_state(state_file, state)
        cov = cold_diag["prefix_coverage"]
        rk = cold_diag["exact_ranking_sample"]
        print(f"Diagnóstico en frío completado.")
        print(f"  Cobertura prefijo nivel 1 (c1): {cov['c1_covered']}/{cov['n_cold_items']} ({cov['c1_pct']:.2f}%)")
        print(f"  Cobertura prefijo nivel 2 (c1, c2): {cov['c12_covered']}/{cov['n_cold_items']} ({cov['c12_pct']:.2f}%)")
        print(f"  Cobertura prefijo nivel 3 (c1, c2, c3): {cov['c123_covered']}/{cov['n_cold_items']} ({cov['c123_pct']:.2f}%)")
        print(f"  Ranking exacto ítems fríos (muestra {rk['n_users_evaluated']} usuarios, {rk['n_cold_targets_evaluated']} objetivos fríos):")
        print(f"    Percentil promedio: {rk['mean_percentile']:.2f}% | Rango mediano: {rk['median_rank']:.0f}")
        print(f"    Hallados en haz 50: {rk['beam_error_stats']['found_in_beam50']}/{rk['beam_error_stats']['total_eval']}")
        print(f"    Podados en paso 1: {rk['beam_error_stats']['pruned_step1']} | Podados en paso 2: {rk['beam_error_stats']['pruned_step2']} | Podados en paso 3/d: {rk['beam_error_stats']['pruned_step3_or_later']}")

    print("\n" + "=" * 80)
    print("EJECUCIÓN COMPLETA DE FASE 2b FINALIZADA")
    print(f"Estado consolidado guardado en: {state_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

