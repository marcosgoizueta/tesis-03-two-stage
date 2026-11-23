import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import torch

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from recsys.stage4.models import TwoTowerStage4, compute_retrieval_candidates_stage4, train_retrieval_stage4
from recsys.retrieval import RetrievalDataset
from recsys.ranker import train_ranker
from recsys.evaluation import make_two_stage
from recsys.config import Stage3Config
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy

def compute_k10_coverage_tail(recs_by_user, item_buckets, n_items, eval_users):
    all_recs_10 = set()
    tail_recs_10 = 0
    total_recs_10 = 0
    for u in eval_users:
        recs10 = recs_by_user[u][:10]
        for it in recs10:
            all_recs_10.add(it)
            if item_buckets.get(it, "0") in {"0", "1-5"}:
                tail_recs_10 += 1
            total_recs_10 += 1
    cov10 = len(all_recs_10) / n_items
    tail10 = tail_recs_10 / total_recs_10 if total_recs_10 > 0 else 0.0
    return cov10, tail10

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=42)
    n_items = bundle.n_items
    eval_users = bundle.eval_users_test

    all_embeddings = np.load("data/cache/movie_embeddings_minilm.npy")
    train_indices = sorted(list(bundle.train_item_indices))
    cold_indices = sorted(list(bundle.cold_item_indices))
    tr_embs = all_embeddings[train_indices]
    mu = tr_embs.mean(axis=0)
    sig = tr_embs.std(axis=0) + 1e-8
    X_std = (all_embeddings - mu) / sig

    torch.manual_seed(42)
    rq = RQVAE().to("cpu")
    X_tr = torch.tensor(X_std[train_indices], dtype=torch.float32, device="cpu")
    rq.rvq.init_from_kmeans(X_tr, seed=42)
    rq.eval()
    codes = rq.quantize(torch.tensor(X_std, dtype=torch.float32, device="cpu")).numpy()
    tr_counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
    mgr = SemanticIdManager(max_disambiguation=16)
    mgr.fit_train_items(train_indices, codes, tr_counts)
    mgr.assign_cold_items(cold_indices, codes)
    sem_map = mgr.item_to_semantic_id
    rand_map = build_random_id_map(sem_map, seed=42)

    results_k10 = {}

    # 1. Popularidad
    pop_cands = np.argsort(bundle.item_popularity)[::-1]
    pop_recs = {}
    for u in eval_users:
        seen = bundle.user_seen_train.get(u, set())
        pop_recs[u] = [i for i in pop_cands if i not in seen][:10]
    cov, tail = compute_k10_coverage_tail(pop_recs, bundle.item_buckets, n_items, eval_users)
    results_k10["Popularidad"] = {"cov@10": cov, "tail@10": tail}
    print(f"Popularidad: Cov@10={cov*100:.2f}%, Tail@10={tail*100:.2f}%")

    # 2. Two-Tower y Two-Stage
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
        positive_df=train_pos_df,
        user_seen_train=bundle.user_seen_train,
        n_items=n_items,
        seen_mask_np=bundle.seen_mask_np,
        n_neg_per_pos=4,
        fast_sampling=False,
    )

    variants_tt = [
        ("A", "Two-Tower (A - Solo ID)"),
        ("B", "Two-Tower (B - Solo Texto)"),
        ("C2", "Two-Tower (C2 - Híbrido ID+Texto con Dropout)"),
        ("D2", "Two-Tower (D2 - Control ID+Géneros con Dropout)"),
    ]

    retrieval_topk_a = None
    for v_code, v_name in variants_tt:
        tt = TwoTowerStage4(
            variant=v_code, n_users=bundle.n_users, n_items=n_items, embed_dim=32,
            text_embeddings_np=all_embeddings if v_code in {"B", "C2"} else None,
            item_genres_np=item_genres_np if v_code in {"D2"} else None,
            is_cold_mask_np=is_cold_mask_np, seed=42,
        ).to(device)
        train_retrieval_stage4(tt, retrieval_ds, cfg, device)
        topk = compute_retrieval_candidates_stage4(tt, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=50, device=device)
        if v_code == "A":
            retrieval_topk_a = compute_retrieval_candidates_stage4(tt, bundle.n_users, n_items, bundle.seen_mask_np, max_candidates=100, device=device)
        recs = {u: list(topk[u]) for u in eval_users}
        cov, tail = compute_k10_coverage_tail(recs, bundle.item_buckets, n_items, eval_users)
        results_k10[v_name] = {"cov@10": cov, "tail@10": tail}
        print(f"{v_name}: Cov@10={cov*100:.2f}%, Tail@10={tail*100:.2f}%")

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
        seen_mask_np=bundle.seen_mask_np, retrieval_topk=retrieval_topk_a,
        n_uniform=2, n_hard=2, cfg=cfg, device=device, seed=42,
    )
    rec_ts = make_two_stage(ranker, retrieval_topk_a, n_candidates=50, device=device, top_k=10)
    ts_recs = {u: list(rec_ts(u)) for u in eval_users}
    cov, tail = compute_k10_coverage_tail(ts_recs, bundle.item_buckets, n_items, eval_users)
    results_k10["Two-Stage (A + Ranker N=50)"] = {"cov@10": cov, "tail@10": tail}
    print(f"Two-Stage: Cov@10={cov*100:.2f}%, Tail@10={tail*100:.2f}%")

    # 3. G1 con haz 50
    tok_g1 = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=sem_map)
    model_g1 = T5Recommender(tok_g1).to(device)
    model_g1.load_state_dict(torch.load("scratch/stage5_g1_full.pt", map_location=device))
    trie_g1 = ItemTrie(tok_g1)
    g1_recs = {}
    for u in eval_users:
        g1_recs[u] = model_g1.recommend_for_user(
            bundle.user_train_seq[u], trie_g1, bundle.user_seen_train.get(u, set()),
            device, num_beams=50, top_k=10, max_context_tokens=150
        )
    cov, tail = compute_k10_coverage_tail(g1_recs, bundle.item_buckets, n_items, eval_users)
    results_k10["G1 (Semantic IDs, Haz 50)"] = {"cov@10": cov, "tail@10": tail}
    print(f"G1 (Haz 50): Cov@10={cov*100:.2f}%, Tail@10={tail*100:.2f}%")

    with open("scratch/k10_metrics.json", "w") as f:
        json.dump(results_k10, f, indent=2)

if __name__ == "__main__":
    main()
