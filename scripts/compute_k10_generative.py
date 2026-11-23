import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy
from scripts.compute_k10_metrics import compute_k10_coverage_tail

def train_and_eval_generative(variant, tok, pol, max_len, epochs, seed, bundle, eval_users, device):
    windows = get_windows_for_policy(bundle.user_train_seq, policy=pol, max_len=max_len, seed=seed)
    ds = SequenceDataset(windows, tok, max_tokens=150)
    loader = DataLoader(
        ds, batch_size=128, shuffle=True, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = T5Recommender(tok).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        for bx, ba, by in loader:
            bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(bx, attention_mask=ba, labels=by).loss
            loss.backward()
            optimizer.step()

    item_trie = ItemTrie(tok)
    recs = {}
    for u in eval_users:
        recs[u] = model.recommend_for_user(
            bundle.user_train_seq[u], item_trie, bundle.user_seen_train.get(u, set()),
            device, num_beams=50, top_k=10, max_context_tokens=150
        )
    cov, tail = compute_k10_coverage_tail(recs, bundle.item_buckets, bundle.n_items, eval_users)
    return cov, tail

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

    with open("scratch/k10_metrics.json", "r") as f:
        res = json.load(f)

    # G2: random ids, L=100, bal, 2 epochs
    tok_g2 = GenerativeTokenizer("G2", n_items=n_items, random_id_map=rand_map)
    cov_g2, tail_g2 = train_and_eval_generative("G2", tok_g2, "balanceado", 100, 2, 42, bundle, eval_users, device)
    res["G2 (Control IDs Aleatorios, Haz 50)"] = {"cov@10": cov_g2, "tail@10": tail_g2}
    print(f"G2 (Haz 50): Cov@10={cov_g2*100:.2f}%, Tail@10={tail_g2*100:.2f}%")

    # G3: atomic, L=20, bal, 38 epochs
    tok_g3 = GenerativeTokenizer("G3", n_items=n_items)
    cov_g3, tail_g3 = train_and_eval_generative("G3", tok_g3, "balanceado", 20, 38, 42, bundle, eval_users, device)
    res["G3 (Tokens Atómicos, Haz 50)"] = {"cov@10": cov_g3, "tail@10": tail_g3}
    print(f"G3 (Haz 50): Cov@10={cov_g3*100:.2f}%, Tail@10={tail_g3*100:.2f}%")

    # G1b: no-year, L=100, bal, 16 epochs
    embs_ny = np.load("data/cache/movie_embeddings_minilm_noyear.npy")
    tr_embs_ny = embs_ny[train_indices]
    mu_ny = tr_embs_ny.mean(axis=0)
    sig_ny = tr_embs_ny.std(axis=0) + 1e-8
    X_std_ny = (embs_ny - mu_ny) / sig_ny
    torch.manual_seed(42)
    rq_ny = RQVAE().to("cpu")
    rq_ny.rvq.init_from_kmeans(torch.tensor(X_std_ny[train_indices], dtype=torch.float32, device="cpu"), seed=42)
    rq_ny.eval()
    codes_ny = rq_ny.quantize(torch.tensor(X_std_ny, dtype=torch.float32, device="cpu")).numpy()
    mgr_ny = SemanticIdManager(max_disambiguation=16)
    mgr_ny.fit_train_items(train_indices, codes_ny, tr_counts)
    mgr_ny.assign_cold_items(cold_indices, codes_ny)
    tok_g1b = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=mgr_ny.item_to_semantic_id)
    cov_g1b, tail_g1b = train_and_eval_generative("G1b", tok_g1b, "balanceado", 100, 16, 42, bundle, eval_users, device)
    res["G1b (Ablación Sin Año, Haz 50)"] = {"cov@10": cov_g1b, "tail@10": tail_g1b}
    print(f"G1b (Haz 50): Cov@10={cov_g1b*100:.2f}%, Tail@10={tail_g1b*100:.2f}%")

    with open("scratch/k10_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Métricas K=10 completadas.")

if __name__ == "__main__":
    main()
