"""Orquestador de Semillas Múltiples para la Etapa 5.
Ejecuta:
- 10 semillas (42 a 51) para G1, G2, G3 y baselines de contexto (Two-Tower A, B, C2, D2, Two-Stage).
- 5 semillas (42 a 46) para G1b.
- G3 incluye haz 50 y ranking exacto (7.6 s/semilla).
- G1 exacto queda excluido de la corrida multi-semilla (6.46 min/semilla > 3 min).
- Soporte para reanudación (--resume) guardando resultados modulares por semilla en results/stage5/per_seed/.
"""

import sys
import os
import json
import time
import argparse
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4
from recsys.retrieval import RetrievalDataset
from recsys.ranker import train_ranker
from recsys.evaluation import make_two_stage
from recsys.config import Stage3Config
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50
from scripts.run_baselines_with_val_es import train_two_tower_with_early_stopping


def retrain_and_eval_generative(
    variant: str,
    tok: GenerativeTokenizer,
    max_len: int,
    policy: str,
    lr: float,
    dropout_rate: float,
    epochs: int,
    bundle,
    seed: int,
    device: torch.device,
    beam_sizes: list = [50],
):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    data_gen = torch.Generator().manual_seed(seed)

    windows_full = get_windows_for_policy(bundle.user_train_seq, policy=policy, max_len=max_len, seed=seed)
    ds_full = SequenceDataset(windows_full, tok, max_tokens=150)
    loader_full = DataLoader(
        ds_full, batch_size=128, shuffle=True, generator=data_gen, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
    )
    model_full = T5Recommender(tok, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.AdamW(model_full.parameters(), lr=lr, weight_decay=1e-4)

    t0_tr = time.perf_counter()
    for ep in range(1, epochs + 1):
        model_full.train()
        for bx, ba, by in loader_full:
            bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model_full(bx, attention_mask=ba, labels=by).loss
            loss.backward()
            optimizer.step()
    t1_tr = time.perf_counter()

    trie = ItemTrie(tok)
    results = {}
    test_users = bundle.eval_users_test

    for beam in beam_sizes:
        t0_ev = time.perf_counter()
        recs = {}
        for u in test_users:
            recs[u] = model_full.recommend_for_user(
                bundle.user_train_seq[u], trie, bundle.user_seen_train.get(u, set()),
                device, num_beams=beam, top_k=min(beam, 50), max_context_tokens=150
            )
        t1_ev = time.perf_counter()
        m = compute_metrics_at_10_and_50(recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, bundle.n_items, beam_size=beam)
        m["latency_ms"] = (t1_ev - t0_ev) / len(test_users) * 1000.0
        m["train_time_s"] = t1_tr - t0_tr
        results[f"beam_{beam}"] = m

    # Si es G3, también evaluar exacto
    if variant == "G3":
        t0_ex = time.perf_counter()
        model_full.eval()
        g3_exact_recs = {}
        with torch.no_grad():
            for u in test_users:
                history = bundle.user_train_seq[u]
                seen = bundle.user_seen_train.get(u, set())
                inp = tok.encode_input_sequence(history, max_tokens=150)
                inp_t = torch.tensor([inp], dtype=torch.long, device=device)
                enc_out = model_full.model.encoder(input_ids=inp_t)
                bos_t = torch.tensor([[tok.BOS_TOKEN_ID]], dtype=torch.long, device=device)
                dec_out = model_full.model.decoder(input_ids=bos_t, encoder_hidden_states=enc_out.last_hidden_state)
                logits = model_full.model.lm_head(dec_out.last_hidden_state)[0, 0].cpu().numpy().copy()
                item_scores = np.full(bundle.n_items, -1e9, dtype=np.float32)
                for item_idx in range(bundle.n_items):
                    if item_idx not in seen:
                        tok_id = tok.encode_target_item(item_idx)[0]
                        item_scores[item_idx] = logits[tok_id]
                top_exact = np.argsort(item_scores)[::-1][:50]
                g3_exact_recs[u] = list(top_exact)
        t1_ex = time.perf_counter()
        m_ex = compute_metrics_at_10_and_50(g3_exact_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, bundle.n_items, beam_size=50)
        m_ex["latency_ms"] = (t1_ex - t0_ex) / len(test_users) * 1000.0
        results["exact"] = m_ex

    return results


def run_multi_seed():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--g1b_seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando orquestador multi-semilla en dispositivo: {device}")

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results" / "stage5"
    per_seed_dir = results_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")

    movie_genres_dict = dict(zip(movies["movieId"], movies["genres"].str.split("|")))
    all_genres_list = sorted({g for gl in movie_genres_dict.values() for g in gl if g != "(no genres listed)"})
    genre2col = {g: c for c, g in enumerate(all_genres_list)}

    for seed in args.seeds:
        print(f"\n" + "="*80)
        print(f"PROCESANDO SEMILLA {seed}")
        print("="*80)

        bundle = create_temporal_split(ratings, movies, seed=seed)
        n_items = bundle.n_items
        test_users = bundle.eval_users_test
        train_full_pos = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]

        item_genres_np = np.zeros((n_items, len(all_genres_list)), dtype=np.float32)
        for mid, idx in bundle.item2idx.items():
            for g in movie_genres_dict.get(mid, []):
                if g in genre2col:
                    item_genres_np[idx, genre2col[g]] = 1.0

        is_cold_mask_np = np.zeros(n_items, dtype=bool)
        is_cold_mask_np[list(bundle.cold_item_indices)] = True
        cfg = Stage3Config(data_dir=Path("data"), seed=seed, device="cuda" if device.type == "cuda" else "cpu")

        train_indices = sorted(list(bundle.train_item_indices))
        cold_indices = sorted(list(bundle.cold_item_indices))
        tr_embs = all_embeddings[train_indices]
        mu = tr_embs.mean(axis=0)
        sig = tr_embs.std(axis=0) + 1e-8
        X_std = (all_embeddings - mu) / sig

        torch.manual_seed(seed)
        rq_model = RQVAE().to("cpu")
        X_tr_cpu = torch.tensor(X_std[train_indices], dtype=torch.float32, device="cpu")
        rq_model.rvq.init_from_kmeans(X_tr_cpu, seed=seed)
        rq_model.eval()
        codes = rq_model.quantize(torch.tensor(X_std, dtype=torch.float32, device="cpu")).numpy()

        tr_counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
        manager = SemanticIdManager(max_disambiguation=16)
        manager.fit_train_items(train_indices, codes, tr_counts)
        manager.assign_cold_items(cold_indices, codes)
        semantic_id_map = manager.item_to_semantic_id
        code_hash = hashlib.sha256(json.dumps({str(k): list(v) for k, v in sorted(semantic_id_map.items())}).encode()).hexdigest()[:16]

        tok_g1 = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_id_map)
        rand_map = build_random_id_map(semantic_id_map, seed=seed)
        tok_g2 = GenerativeTokenizer("G2", n_items=n_items, random_id_map=rand_map)
        tok_g3 = GenerativeTokenizer("G3", n_items=n_items)

        # G1b tokenizer (solo si se necesita entrenar G1b)
        tok_g1b = None
        code_hash_g1b = None
        if seed in args.g1b_seeds:
            g1b_file = per_seed_dir / f"G1b_seed_{seed}.json"
            if not (args.resume and g1b_file.exists()):
                embs_noyear = np.load("data/cache/movie_embeddings_minilm_noyear.npy")
                tr_embs_ny = embs_noyear[train_indices]
                mu_ny = tr_embs_ny.mean(axis=0)
                sig_ny = tr_embs_ny.std(axis=0) + 1e-8
                X_std_ny = (embs_noyear - mu_ny) / sig_ny
                torch.manual_seed(seed)
                rq_ny = RQVAE().to("cpu")
                X_tr_cpu_ny = torch.tensor(X_std_ny[train_indices], dtype=torch.float32, device="cpu")
                rq_ny.rvq.init_from_kmeans(X_tr_cpu_ny, seed=seed)
                rq_ny.eval()
                codes_ny = rq_ny.quantize(torch.tensor(X_std_ny, dtype=torch.float32, device="cpu")).numpy()
                mgr_ny = SemanticIdManager(max_disambiguation=16)
                mgr_ny.fit_train_items(train_indices, codes_ny, tr_counts)
                mgr_ny.assign_cold_items(cold_indices, codes_ny)
                code_hash_g1b = hashlib.sha256(json.dumps({str(k): list(v) for k, v in sorted(mgr_ny.item_to_semantic_id.items())}).encode()).hexdigest()[:16]
                tok_g1b = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=mgr_ny.item_to_semantic_id)

        # G1 (E*=18, lr=1e-3, drop=0.1, L=100)
        g1_file = per_seed_dir / f"G1_seed_{seed}.json"
        if args.resume and g1_file.exists():
            print(f"[Skip] G1 semilla {seed} ya existe.")
        else:
            print(f"Entrenando G1 en train_full (E*=18, seed={seed})...")
            res_g1 = retrain_and_eval_generative("G1", tok_g1, 100, "balanceado", 1e-3, 0.1, 18, bundle, seed, device, [50])
            res_g1["code_hash"] = code_hash
            with open(g1_file, "w") as f:
                json.dump(res_g1["beam_50"], f, indent=2)

        # G2 (E*=18, lr=1e-3, drop=0.1, L=100)
        g2_file = per_seed_dir / f"G2_seed_{seed}.json"
        if args.resume and g2_file.exists():
            print(f"[Skip] G2 semilla {seed} ya existe.")
        else:
            print(f"Entrenando G2 en train_full (E*=18, seed={seed})...")
            res_g2 = retrain_and_eval_generative("G2", tok_g2, 100, "balanceado", 1e-3, 0.1, 18, bundle, seed, device, [50])
            with open(g2_file, "w") as f:
                json.dump(res_g2["beam_50"], f, indent=2)

        # G3 (E*=16, lr=5e-4, drop=0.2, L=20)
        g3_file = per_seed_dir / f"G3_seed_{seed}.json"
        if args.resume and g3_file.exists():
            print(f"[Skip] G3 semilla {seed} ya existe.")
        else:
            print(f"Entrenando G3 en train_full (E*=16, seed={seed})...")
            res_g3 = retrain_and_eval_generative("G3", tok_g3, 20, "balanceado", 5e-4, 0.2, 16, bundle, seed, device, [50])
            with open(g3_file, "w") as f:
                json.dump(res_g3, f, indent=2)

        # G1b (E*=18, lr=1e-3, drop=0.1, L=100)
        if seed in args.g1b_seeds:
            g1b_file = per_seed_dir / f"G1b_seed_{seed}.json"
            if args.resume and g1b_file.exists():
                print(f"[Skip] G1b semilla {seed} ya existe.")
            else:
                print(f"Entrenando G1b en train_full (E*=18, seed={seed})...")
                res_g1b = retrain_and_eval_generative("G1b", tok_g1b, 100, "balanceado", 1e-3, 0.1, 18, bundle, seed, device, [50])
                res_g1b["code_hash"] = code_hash
                with open(g1b_file, "w") as f:
                    json.dump(res_g1b["beam_50"], f, indent=2)

        # BASELINES DE CONTEXTO
        variants = [
            ("A", "Two-Tower_A"),
            ("B", "Two-Tower_B"),
            ("C2", "Two-Tower_C2"),
            ("D2", "Two-Tower_D2"),
        ]
        retrieval_topk_a_100 = None

        for v_code, v_name in variants:
            b_file = per_seed_dir / f"{v_name}_seed_{seed}.json"
            ts_file = per_seed_dir / f"Two-Stage_seed_{seed}.json"
            need_a_model = (v_code == "A" and not (args.resume and ts_file.exists()))

            if args.resume and b_file.exists() and not need_a_model:
                print(f"[Skip] {v_name} semilla {seed} ya existe.")
            else:
                print(f"Ajustando {v_name} con early stopping (seed={seed})...")
                model_full, best_ep, best_loss, hist = train_two_tower_with_early_stopping(
                    v_code, bundle, all_embeddings, item_genres_np, is_cold_mask_np, cfg, device, seed=seed
                )
                if v_code == "A":
                    retrieval_topk_a_100 = compute_retrieval_candidates_stage4(model_full, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=100, device=device)
                    topk = retrieval_topk_a_100[:, :50]
                else:
                    topk = compute_retrieval_candidates_stage4(model_full, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=50, device=device)

                if not (args.resume and b_file.exists()):
                    recs = {u: list(topk[u]) for u in test_users}
                    m = compute_metrics_at_10_and_50(recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
                    m["E_star"] = best_ep
                    m["best_val_loss"] = float(best_loss)
                    with open(b_file, "w") as f:
                        json.dump(m, f, indent=2)

        # Two-Stage
        if args.resume and ts_file.exists():
            print(f"[Skip] Two-Stage semilla {seed} ya existe.")
        else:
            print(f"Ajustando Two-Stage Ranker MLP (seed={seed})...")
            pos_users = train_full_pos["user_idx"].values
            pos_items = train_full_pos["item_idx"].values
            user_pos_np = np.zeros((bundle.n_users, n_items), dtype=np.float32)
            user_pos_np[pos_users, pos_items] = 1.0
            user_genre_np = user_pos_np @ item_genres_np
            user_genre_np = user_genre_np / np.maximum(user_genre_np.sum(axis=1, keepdims=True), 1.0)
            item_logpop_np = np.log1p(bundle.item_popularity).astype(np.float32)

            ranker, _ = train_ranker(
                pos_users=pos_users, pos_items=pos_items, n_users=bundle.n_users, n_items=n_items,
                user_genre_np=user_genre_np, item_genre_np=item_genres_np, item_logpop_np=item_logpop_np,
                seen_mask_np=bundle.seen_mask_np, retrieval_topk=retrieval_topk_a_100,
                n_uniform=2, n_hard=2, cfg=cfg, device=device, seed=seed,
            )
            rec_ts = make_two_stage(ranker, retrieval_topk_a_100, n_candidates=50, device=device, top_k=50)
            ts_recs = {u: list(rec_ts(u)) for u in test_users}
            m_ts = compute_metrics_at_10_and_50(ts_recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
            with open(ts_file, "w") as f:
                json.dump(m_ts, f, indent=2)

    print("\n" + "="*80)
    print("TODAS LAS SEMILLAS COMPLETADAS")
    print("="*80)

if __name__ == "__main__":
    run_multi_seed()
