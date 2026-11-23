"""
audit_stage5_seed_dependence.py — Auditoría: ¿qué cambia con la semilla en la Etapa 5?

Para cada semilla recalcula, sin entrenar ningún modelo y con la misma lógica del
orquestador multi-semilla:
  - hash del mapa de Semantic IDs de G1 (RQ-VAE + KMeans, seed=seed),
  - hash del mapa aleatorio de G2 (build_random_id_map(seed=seed)),
  - hash del mapa de G1b (embeddings sin año; solo semillas 42-46),
  - tamaños del split y NDCG@10 de popularidad con el split de esa semilla.
Guarda results/stage5/seed_dependence_audit.json.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch
from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.rq_vae import RQVAE, SemanticIdManager
from recsys.stage5.tokenizer import build_random_id_map
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50
from scripts.run_stage5_g1_exact import build_g1


def h(m):
    return hashlib.sha256(json.dumps({str(k): list(v) for k, v in sorted(m.items())}).encode()).hexdigest()[:16]


def g1b_hash(bundle, emb_ny, seed):
    tr = sorted(bundle.train_item_indices); cold = sorted(bundle.cold_item_indices)
    X = (emb_ny - emb_ny[tr].mean(0)) / (emb_ny[tr].std(0) + 1e-8)
    torch.manual_seed(seed)
    rq = RQVAE().to("cpu")
    rq.rvq.init_from_kmeans(torch.tensor(X[tr], dtype=torch.float32), seed=seed)
    rq.eval()
    codes = rq.quantize(torch.tensor(X, dtype=torch.float32)).numpy()
    mgr = SemanticIdManager(max_disambiguation=16)
    mgr.fit_train_items(tr, codes, bundle.train_full_df["item_idx"].value_counts().to_dict())
    mgr.assign_cold_items(cold, codes)
    return h(mgr.item_to_semantic_id)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ratings = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    movies = pd.read_csv(ROOT / "data/ml-latest-small/movies.csv")
    emb = np.load(ROOT / "data/cache/movie_embeddings_minilm.npy")
    emb_ny = np.load(ROOT / "data/cache/movie_embeddings_minilm_noyear.npy")
    out = {}
    for seed in range(42, 52):
        b = create_temporal_split(ratings, movies, seed=seed)
        tok, g1h = build_g1(b, emb, seed)
        g2h = h(build_random_id_map(tok.id_map, seed=seed))
        users = b.eval_users_test
        pop = np.argsort(b.item_popularity)[::-1]
        recs = {u: [i for i in pop if i not in b.user_seen_train.get(u, set())][:50] for u in users}
        m = compute_metrics_at_10_and_50(recs, b.test_relevant_by_user, users, b.cold_item_indices,
                                         b.item_buckets, b.n_items, beam_size=50)
        out[seed] = {"G1_hash": g1h, "G2_hash": g2h,
                     "G1b_hash": g1b_hash(b, emb_ny, seed) if seed <= 46 else None,
                     "n_train_full": int(len(b.train_full_df)), "n_test_users": len(users),
                     "popularity_ndcg10": m["ndcg@10"], "popularity_cov10": m["cov@10"]}
        print(seed, out[seed], flush=True)
    p = ROOT / "results" / "stage5" / "seed_dependence_audit.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Guardado:", p)


if __name__ == "__main__":
    main()
