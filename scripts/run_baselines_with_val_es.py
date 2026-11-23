import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from recsys.stage5.data_split import create_temporal_split
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4
from recsys.retrieval import RetrievalDataset
from recsys.ranker import train_ranker
from recsys.evaluation import make_two_stage
from recsys.config import Stage3Config
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50

def train_two_tower_with_early_stopping(
    variant: str,
    bundle,
    all_text_embs,
    item_genres_np,
    is_cold_mask_np,
    cfg: Stage3Config,
    device: torch.device,
    seed: int,
    patience: int = 3,
    min_delta: float = 0.005,
    max_epochs: int = 30,
):
    n_items = bundle.n_items
    train_core_pos = bundle.train_core_df[bundle.train_core_df["rating"] >= 3.5]
    val_pos = bundle.val_df[bundle.val_df["rating"] >= 3.5]

    user_seen_core = {}
    for u, seq in bundle.user_train_seq.items():
        v_len = len(bundle.user_val_seq[u])
        user_seen_core[u] = set(seq[:len(seq)-v_len])
    seen_mask_core = np.zeros((bundle.n_users, n_items), dtype=bool)
    for u, s in user_seen_core.items():
        seen_mask_core[u, list(s)] = True

    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    ds_tr = RetrievalDataset(train_core_pos, user_seen_core, n_items, seen_mask_core, n_neg_per_pos=4)
    ds_val = RetrievalDataset(val_pos, user_seen_core, n_items, seen_mask_core, n_neg_per_pos=4)

    loader_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=256, shuffle=False)

    model = TwoTowerStage4(
        variant=variant, n_users=bundle.n_users, n_items=n_items, embed_dim=32,
        text_embeddings_np=all_text_embs if variant in {"B", "C2"} else None,
        item_genres_np=item_genres_np if variant in {"D2"} else None,
        is_cold_mask_np=is_cold_mask_np, seed=seed,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
    crit = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_ep = 1
    pat = 0
    history = []

    for ep in range(1, max_epochs + 1):
        ds_tr.resample_negatives()
        model.train()
        tr_loss = 0.0
        for u, p, negs in loader_tr:
            u, p, negs = u.to(device), p.to(device), negs.to(device)
            opt.zero_grad()
            pos_scores = model.score(u, p)
            pos_labels = torch.ones_like(pos_scores)
            u_exp = u.unsqueeze(1).expand(-1, negs.size(1)).reshape(-1)
            negs_flat = negs.reshape(-1)
            neg_scores = model.score(u_exp, negs_flat)
            neg_labels = torch.zeros_like(neg_scores)
            loss = crit(torch.cat([pos_scores, neg_scores]), torch.cat([pos_labels, neg_labels]))
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(u)
        tr_loss /= len(ds_tr)

        if ep % 2 == 0 or ep == 1:
            model.eval()
            v_loss = 0.0
            cnt = 0
            with torch.no_grad():
                for u, p, negs in loader_val:
                    u, p, negs = u.to(device), p.to(device), negs.to(device)
                    pos_scores = model.score(u, p)
                    pos_labels = torch.ones_like(pos_scores)
                    u_exp = u.unsqueeze(1).expand(-1, negs.size(1)).reshape(-1)
                    negs_flat = negs.reshape(-1)
                    neg_scores = model.score(u_exp, negs_flat)
                    neg_labels = torch.zeros_like(neg_scores)
                    l = crit(torch.cat([pos_scores, neg_scores]), torch.cat([pos_labels, neg_labels]))
                    v_loss += l.item() * len(u)
                    cnt += len(u)
            v_loss /= cnt

            diff = best_loss - v_loss
            if diff >= min_delta:
                best_loss = v_loss
                best_ep = ep
                pat = 0
            else:
                pat += 1
            history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": v_loss})
            if pat >= patience:
                break
        else:
            history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": None})

    # Reentrenar en train_full durante best_ep épocas
    train_full_pos = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    ds_full = RetrievalDataset(train_full_pos, bundle.user_seen_train, n_items, bundle.seen_mask_np, n_neg_per_pos=4)
    loader_full = DataLoader(ds_full, batch_size=256, shuffle=True)

    model_full = TwoTowerStage4(
        variant=variant, n_users=bundle.n_users, n_items=n_items, embed_dim=32,
        text_embeddings_np=all_text_embs if variant in {"B", "C2"} else None,
        item_genres_np=item_genres_np if variant in {"D2"} else None,
        is_cold_mask_np=is_cold_mask_np, seed=seed,
    ).to(device)
    opt_full = optim.Adam(model_full.parameters(), lr=5e-3, weight_decay=1e-6)

    for ep in range(1, best_ep + 1):
        ds_full.resample_negatives()
        model_full.train()
        for u, p, negs in loader_full:
            u, p, negs = u.to(device), p.to(device), negs.to(device)
            opt_full.zero_grad()
            pos_scores = model_full.score(u, p)
            pos_labels = torch.ones_like(pos_scores)
            u_exp = u.unsqueeze(1).expand(-1, negs.size(1)).reshape(-1)
            negs_flat = negs.reshape(-1)
            neg_scores = model_full.score(u_exp, negs_flat)
            neg_labels = torch.zeros_like(neg_scores)
            loss = crit(torch.cat([pos_scores, neg_scores]), torch.cat([pos_labels, neg_labels]))
            loss.backward()
            opt_full.step()

    return model_full, best_ep, best_loss, history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=42)
    n_items = bundle.n_items
    test_users = bundle.eval_users_test

    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")
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
    cfg = Stage3Config(data_dir=Path("data"), seed=42, device="cuda" if device.type == "cuda" else "cpu")

    results_dir = Path("results/stage5")
    curves_dir = results_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("A", "Two-Tower (A - Solo ID)"),
        ("B", "Two-Tower (B - Solo Texto)"),
        ("C2", "Two-Tower (C2 - Híbrido ID+Texto con Dropout)"),
        ("D2", "Two-Tower (D2 - Control ID+Géneros con Dropout)"),
    ]

    baselines_out = {}
    retrieval_topk_a_100 = None

    for v_code, v_name in variants:
        print(f"\nAjustando {v_name} con parada temprana por val_loss...")
        model_full, best_ep, best_loss, hist = train_two_tower_with_early_stopping(
            v_code, bundle, all_embeddings, item_genres_np, is_cold_mask_np, cfg, device, seed=42
        )
        print(f"  {v_name}: E* = {best_ep} (Best Val Loss: {best_loss:.4f})")
        with open(curves_dir / f"TwoTower_{v_code}_seed42.json", "w") as f:
            json.dump({
                "model": v_name, "variant": v_code, "seed": 42, "best_epoch": best_ep,
                "best_val_loss": float(best_loss), "history": hist, "timestamp": time.time()
            }, f, indent=2)

        if v_code == "A":
            retrieval_topk_a_100 = compute_retrieval_candidates_stage4(model_full, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=100, device=device)
            topk = retrieval_topk_a_100[:, :50]
        else:
            topk = compute_retrieval_candidates_stage4(model_full, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=50, device=device)

        recs = {u: list(topk[u]) for u in test_users}
        m = compute_metrics_at_10_and_50(recs, bundle.test_relevant_by_user, test_users, bundle.cold_item_indices, bundle.item_buckets, n_items, beam_size=50)
        m["E_star"] = best_ep
        m["best_val_loss"] = float(best_loss)
        baselines_out[v_name] = m
        print(f"  {v_name} Test -> NDCG@10: {m['ndcg@10']:.4f} | Recall@10: {m['recall@10']:.4f} | Recall@50: {m['recall@50']:.4f} | Cold Rec@50: {m['cold_recall@50']:.4f} | Cov@10: {m['cov@10']*100:.2f}% | Tail@10: {m['tail@10']*100:.2f}%")

    # Two-Stage
    print("\nAjustando Ranker MLP para Two-Stage...")
    train_pos_df = bundle.train_full_df[bundle.train_full_df["rating"] >= 3.5]
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
    baselines_out["Two-Stage (A + Ranker N=50)"] = m_ts
    print(f"  Two-Stage Test -> NDCG@10: {m_ts['ndcg@10']:.4f} | Recall@10: {m_ts['recall@10']:.4f} | Recall@50: {m_ts['recall@50']:.4f} | Cold Rec@50: {m_ts['cold_recall@50']:.4f} | Cov@10: {m_ts['cov@10']*100:.2f}% | Tail@10: {m_ts['tail@10']*100:.2f}%")

    with open(results_dir / "baselines_early_stopping_results.json", "w") as f:
        json.dump(baselines_out, f, indent=2)

if __name__ == "__main__":
    main()
