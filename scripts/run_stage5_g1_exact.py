"""
run_stage5_g1_exact.py — Tarea T4: error del haz en G1.

Para cada semilla (por defecto 42-44) reentrena G1 con la receta exacta del
orquestador multi-semilla (split, RQ-VAE, tokenizador e hiperparámetros iguales:
L=100, balanceado, lr 1e-3, dropout 0.1, E*=18), y sobre ese mismo modelo compara:
  - búsqueda en haz restringida (haz 50), y
  - ranking exacto de todos los ítems válidos no vistos (log-prob de los 4 tokens
    [c1, c2, c3, d], factorizado por el árbol de prefijos).

Reporta métricas de ambos, Jaccard@10 entre los top-10 y diferencia contra el JSON
de haz ya guardado en per_seed/ (para saber si el reentrenamiento reproduce la
corrida original; en GPU puede haber pequeñas diferencias no deterministas).

Salida: results/stage5/g1_exact/G1_exact_seed_{s}.json y g1_exact_summary.json.
Checkpoints en scratch/ (ignorado por git).  Soporta reanudación por semilla.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import GenerativeTokenizer
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender
from scripts.run_stage5_grid import SequenceDataset, collate_fn, get_windows_for_policy
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50
from scripts.test_exact_g1_ranking import build_prefix_factorized_scorer, score_unseen_items_factorized

OUT_DIR = ROOT / "results" / "stage5" / "g1_exact"
PER_SEED_DIR = ROOT / "results" / "stage5" / "per_seed"
CKPT_DIR = ROOT / "scratch"

G1_CFG = dict(max_len=100, policy="balanceado", lr=1e-3, dropout=0.1, epochs=18)


def build_g1(bundle, all_embeddings, seed):
    """Replica el preámbulo del orquestador multi-semilla para G1."""
    n_items = bundle.n_items
    train_indices = sorted(bundle.train_item_indices)
    cold_indices = sorted(bundle.cold_item_indices)
    mu = all_embeddings[train_indices].mean(axis=0)
    sig = all_embeddings[train_indices].std(axis=0) + 1e-8
    X_std = (all_embeddings - mu) / sig

    torch.manual_seed(seed)
    rq = RQVAE().to("cpu")
    rq.rvq.init_from_kmeans(torch.tensor(X_std[train_indices], dtype=torch.float32), seed=seed)
    rq.eval()
    codes = rq.quantize(torch.tensor(X_std, dtype=torch.float32)).numpy()
    counts = bundle.train_full_df["item_idx"].value_counts().to_dict()
    mgr = SemanticIdManager(max_disambiguation=16)
    mgr.fit_train_items(train_indices, codes, counts)
    mgr.assign_cold_items(cold_indices, codes)
    code_hash = hashlib.sha256(json.dumps(
        {str(k): list(v) for k, v in sorted(mgr.item_to_semantic_id.items())}
    ).encode()).hexdigest()[:16]
    tok = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=mgr.item_to_semantic_id)
    return tok, code_hash


def train_g1(tok, bundle, seed, device):
    c = G1_CFG
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    windows = get_windows_for_policy(bundle.user_train_seq, policy=c["policy"], max_len=c["max_len"], seed=seed)
    ds = SequenceDataset(windows, tok, max_tokens=150)
    loader = DataLoader(ds, batch_size=128, shuffle=True, generator=gen, pin_memory=True,
                        collate_fn=lambda b: collate_fn(b, pad_token_id=tok.PAD_TOKEN_ID))
    model = T5Recommender(tok, dropout_rate=c["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(c["epochs"]):
        model.train()
        for bx, ba, by in loader:
            bx, ba, by = bx.to(device, non_blocking=True), ba.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(bx, attention_mask=ba, labels=by).loss
            loss.backward()
            opt.step()
    return model, time.perf_counter() - t0


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def run_seed(seed, ratings, movies, all_embeddings, device):
    bundle = create_temporal_split(ratings, movies, seed=seed)
    tok, code_hash = build_g1(bundle, all_embeddings, seed)
    model, train_s = train_g1(tok, bundle, seed, device)
    torch.save(model.state_dict(), CKPT_DIR / f"g1_seed{seed}_exact_run.pt")
    model.eval()

    users = bundle.eval_users_test
    n_items = bundle.n_items
    trie = ItemTrie(tok)

    t0 = time.perf_counter()
    beam = {u: model.recommend_for_user(bundle.user_train_seq[u], trie, bundle.user_seen_train.get(u, set()),
                                        device, num_beams=50, top_k=50, max_context_tokens=150)
            for u in users}
    beam_ms = (time.perf_counter() - t0) / len(users) * 1000

    item_tokens = build_prefix_factorized_scorer(tok, n_items)
    t0 = time.perf_counter()
    exact = {u: score_unseen_items_factorized(model, tok, bundle.user_train_seq[u],
                                              bundle.user_seen_train.get(u, set()), n_items,
                                              item_tokens, device, chunk_size=256)
             for u in users}
    exact_ms = (time.perf_counter() - t0) / len(users) * 1000

    args = (bundle.test_relevant_by_user, users, bundle.cold_item_indices, bundle.item_buckets, n_items)
    m_beam = compute_metrics_at_10_and_50(beam, *args, beam_size=50)
    m_exact = compute_metrics_at_10_and_50(exact, *args, beam_size=50)
    m_beam["latency_ms"], m_exact["latency_ms"] = beam_ms, exact_ms

    j10 = [jaccard(beam[u][:10], exact[u][:10]) for u in users]
    j50 = [jaccard(beam[u][:50], exact[u][:50]) for u in users]
    top1_same = float(np.mean([beam[u][0] == exact[u][0] for u in users]))

    saved = json.loads((PER_SEED_DIR / f"G1_seed_{seed}.json").read_text())
    return {
        "seed": seed, "code_hash": code_hash, "n_eval_users": len(users), "train_time_s": train_s,
        "config": G1_CFG,
        "beam_50": m_beam, "exact": m_exact,
        "jaccard_top10_mean": float(np.mean(j10)), "jaccard_top50_mean": float(np.mean(j50)),
        "top1_agreement": top1_same,
        "delta_ndcg10_exact_minus_beam": m_exact["ndcg@10"] - m_beam["ndcg@10"],
        "saved_per_seed_beam_ndcg10": saved["ndcg@10"],
        "reproduction_diff_beam_ndcg10": m_beam["ndcg@10"] - saved["ndcg@10"],
        "device": str(device), "torch": torch.__version__,
        "note_exact": "log-prob de 4 tokens [c1,c2,c3,d] sobre ítems válidos no vistos; sin token EOS.",
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    movies = pd.read_csv(ROOT / "data/ml-latest-small/movies.csv")
    emb = np.load(ROOT / "data/cache/movie_embeddings_minilm.npy")

    for seed in args.seeds:
        f = OUT_DIR / f"G1_exact_seed_{seed}.json"
        if f.exists():
            print(f"[Skip] semilla {seed}")
            continue
        print(f"\n=== G1 exacto vs haz, semilla {seed} ===", flush=True)
        r = run_seed(seed, ratings, movies, emb, device)
        f.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"  NDCG@10 haz={r['beam_50']['ndcg@10']:.4f} exacto={r['exact']['ndcg@10']:.4f} "
              f"Jaccard@10={r['jaccard_top10_mean']:.3f} | repro vs guardado: {r['reproduction_diff_beam_ndcg10']:+.5f}",
              flush=True)

    rs = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("G1_exact_seed_*.json"))]
    keys = ["ndcg@10", "recall@10", "recall@50", "cov@10", "tail@10"]
    summary = {"seeds": [r["seed"] for r in rs],
               "beam_50": {k: [r["beam_50"][k] for r in rs] for k in keys},
               "exact": {k: [r["exact"][k] for r in rs] for k in keys},
               "jaccard_top10": [r["jaccard_top10_mean"] for r in rs],
               "top1_agreement": [r["top1_agreement"] for r in rs],
               "reproduction_diff_beam_ndcg10": [r["reproduction_diff_beam_ndcg10"] for r in rs]}
    (OUT_DIR / "g1_exact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResumen: {OUT_DIR / 'g1_exact_summary.json'}")


if __name__ == "__main__":
    main()
