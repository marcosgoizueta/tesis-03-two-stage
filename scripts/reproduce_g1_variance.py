import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy, evaluate_ndcg_validation

def evaluate_val_loss(model, loader, device, use_bf16=True):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for bx, ba, by in loader:
            bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(bx, attention_mask=ba, labels=by).loss
            total_loss += loss.item() * len(bx)
            total_count += len(bx)
    return total_loss / max(total_count, 1)

def run_experiment(run_name, seed_for_model, seed_for_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    bundle = create_temporal_split(ratings, movies, seed=42)
    n_items = bundle.n_items

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

    val_users = bundle.eval_users_val
    user_core_histories = {}
    for u in range(bundle.n_users):
        full_seq = bundle.user_train_seq[u]
        val_len = len(bundle.user_val_seq[u])
        user_core_histories[u] = full_seq[: len(full_seq) - val_len]

    # Dataset de validación para calcular val_loss
    tok = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=sem_map)
    val_windows = []
    for u in val_users:
        hist = user_core_histories.get(u, [])
        val_items = bundle.user_val_seq[u]
        for tgt in val_items:
            val_windows.append((hist, tgt))
    val_ds = SequenceDataset(val_windows, tok, max_tokens=150)
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
    )

    item_trie = ItemTrie(tok)
    windows = get_windows_for_policy(user_core_histories, policy="balanceado", max_len=100, seed=seed_for_data)
    train_ds = SequenceDataset(windows, tok, max_tokens=150)

    # Fijar semilla para el modelo y dataloader
    torch.manual_seed(seed_for_model)
    torch.cuda.manual_seed_all(seed_for_model)
    np.random.seed(seed_for_model)
    generator = torch.Generator().manual_seed(seed_for_data)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, generator=generator, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID)
    )

    model = T5Recommender(tok).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    print(f"\n--- Iniciando {run_name} (Model seed: {seed_for_model}, Data seed: {seed_for_data}) ---")
    for epoch in range(1, 13):
        model.train()
        train_loss = 0.0
        for bx, ba, by in train_loader:
            bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(bx, attention_mask=ba, labels=by).loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        val_loss = evaluate_val_loss(model, val_loader, device, use_bf16=True)
        ndcg_val = None
        if epoch in {4, 8, 12}:
            ndcg_val = evaluate_ndcg_validation(
                model, tok, item_trie, val_users, user_core_histories,
                bundle.val_relevant_by_user, device, top_k=10, beam_size=50
            )
            print(f"  [{run_name}] Ep {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val NDCG@10: {ndcg_val:.4f}")
        else:
            print(f"  [{run_name}] Ep {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ndcg": ndcg_val})

    return history

if __name__ == "__main__":
    # Corrida A: Semilla 42
    h_a = run_experiment("Corrida_A (seed 42)", 42, 42)
    # Corrida B: Idéntica semilla 42 (para medir determinismo estricto / ruido GPU)
    h_b = run_experiment("Corrida_B (seed 42 repetida)", 42, 42)
    # Corrida C: Semilla de datos/pesos 43 (para medir varianza entre semillas)
    h_c = run_experiment("Corrida_C (seed 43)", 43, 43)

    out = {"A": h_a, "B": h_b, "C": h_c}
    with open("scratch/diagnose_g1_runs.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nExperimento completado y guardado en scratch/diagnose_g1_runs.json")
