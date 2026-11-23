"""
run_stage5_explicit_baselines.py — Etapas 1 y 2 (SVD, NCF) evaluadas con el protocolo temporal full-ranking.

--gate : reproduce las cifras pointwise de los notebooks de Colab sobre el split aleatorio original
         (SVD RMSE 0.8807 / MAE 0.6766; NCF RMSE 0.9645 / MAE 0.7388). Guarda results/stage5/explicit_gate.json.
Principal: por semilla (42-51), split temporal de la Etapa 5; entrena con TODAS las valoraciones de train_full
         (regresión de rating, como en los notebooks), puntúa todo el catálogo, enmascara lo visto en train_full y
         los ítems sin interacciones en train (no tienen señal; Two-Tower A tampoco los recomienda: cold recall 0),
         y evalúa con compute_metrics_at_10_and_50. Variantes:
           SVD      Funk/Surprise SVD, 100 factores, 20 épocas (a priori, config del notebook).
           NCF      NeuMF, 15 épocas fijas (config del notebook).
           NCF_val  sensibilidad: NeuMF con épocas elegidas por RMSE de validación (train_core) y reentrenado
                    en train_full durante E* épocas.
Salida: results/stage5/per_seed_explicit/{modelo}_seed_{s}.json y results/stage5/recs/EXPL_{modelo}_seed_{s}.npz
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split

from recsys.stage5.data_split import create_temporal_split
from recsys.stage5.explicit_baselines import (FunkSVD, train_neumf, rmse_neumf, score_all_items_neumf)
from scripts.run_stage5_fase2c import compute_metrics_at_10_and_50

OUT = ROOT / "results" / "stage5" / "per_seed_explicit"
RECS = ROOT / "results" / "stage5" / "recs"


def gate(device):
    r = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    out = {}
    # SVD: split de Surprise (ShuffleSplit sobre las valoraciones, random_state=42), ids internos por orden de aparición.
    uid = {u: k for k, u in enumerate(r["userId"].unique())}; iid = {m: k for k, m in enumerate(r["movieId"].unique())}
    r["u"] = r["userId"].map(uid); r["i"] = r["movieId"].map(iid)
    tr, te = train_test_split(r, test_size=0.2, random_state=42)
    nu, ni = len(uid), len(iid)
    t0 = time.perf_counter()
    svd = FunkSVD(seed=42).fit(tr["u"].values, tr["i"].values, tr["rating"].values, nu, ni)
    p = np.clip(svd.predict_pairs(te["u"].values, te["i"].values), 0.5, 5.0)
    e = te["rating"].values - p
    out["SVD"] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
                  "colab_rmse": 0.8807462819979623, "colab_mae": 0.6765729095860605, "fit_s": time.perf_counter() - t0}
    # NCF: split sklearn del notebook (misma llamada), 15 épocas.
    model, hist = train_neumf(tr["u"].values, tr["i"].values, tr["rating"].values, nu, ni, device, seed=42, epochs=15)
    rm = rmse_neumf(model, te["u"].values, te["i"].values, te["rating"].values, device)
    with torch.no_grad():
        pr = model(torch.tensor(te["u"].values, dtype=torch.long, device=device),
                   torch.tensor(te["i"].values, dtype=torch.long, device=device)).cpu().numpy()
    out["NCF"] = {"rmse": rm, "mae": float(np.abs(te["rating"].values - pr).mean()),
                  "colab_rmse": 0.9645499936943835, "colab_mae": 0.7388425987599531,
                  "final_train_mse": hist[-1]["train_mse"], "colab_final_train_mse": 0.4375}
    out["tolerance_note"] = "El notebook de NCF corrió en GPU de Colab; el RNG de la GPU difiere (ver CLAUDE.md sección 3)."
    (ROOT / "results/stage5/explicit_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def topk_from_scores(scores, eval_users, bundle, cold_mask, k=50):
    s = scores.copy()
    s[bundle.seen_mask_np[eval_users]] = -np.inf
    s[:, cold_mask] = -np.inf
    top = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def evaluate_and_save(name, seed, top, bundle, extra, device):
    users = bundle.eval_users_test
    recs = {u: [int(x) for x in top[k]] for k, u in enumerate(users)}
    m = compute_metrics_at_10_and_50(recs, bundle.test_relevant_by_user, users, bundle.cold_item_indices,
                                     bundle.item_buckets, bundle.n_items, beam_size=50)
    m.update(extra); m.update({"seed": seed, "device": str(device), "torch": torch.__version__,
                               "protocol": "temporal_full_ranking, entrenado con todas las valoraciones de train_full"})
    OUT.mkdir(parents=True, exist_ok=True); RECS.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_seed_{seed}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    np.savez_compressed(RECS / f"EXPL_{name}_seed_{seed}.npz", users=np.array(users, dtype=np.int32),
                        recs=top.astype(np.int32))
    print(f"  {name} seed {seed}: NDCG@10={m['ndcg@10']:.4f} Recall@10={m['recall@10']:.4f} "
          f"Cov@10={m['cov@10']*100:.2f}% | RMSE test={extra.get('rmse_test_pointwise'):.4f}", flush=True)


def run(seeds, device):
    ratings = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    movies = pd.read_csv(ROOT / "data/ml-latest-small/movies.csv")
    for seed in seeds:
        todo = [n for n in ("SVD", "NCF", "NCF_val") if not (OUT / f"{n}_seed_{seed}.json").exists()]
        if not todo:
            print(f"[Skip] semilla {seed}"); continue
        print(f"\n=== SEMILLA {seed} ===", flush=True)
        b = create_temporal_split(ratings, movies, seed=seed)
        n_u, n_i = b.n_users, b.n_items
        users = b.eval_users_test
        cold = np.zeros(n_i, dtype=bool); cold[list(b.cold_item_indices)] = True
        tf, te = b.train_full_df, b.test_df
        tu, ti, tr_ = tf["user_idx"].values, tf["item_idx"].values, tf["rating"].values.astype(np.float64)
        te_ok = te[~cold[te["item_idx"].values]]          # RMSE puntual solo sobre ítems con historial

        if "SVD" in todo:
            t0 = time.perf_counter()
            svd = FunkSVD(seed=seed).fit(tu, ti, tr_, n_u, n_i)
            sc = svd.predict_matrix(users)
            rm = float(np.sqrt(np.mean((te_ok["rating"].values - np.clip(svd.predict_pairs(te_ok["user_idx"].values, te_ok["item_idx"].values), 0.5, 5)) ** 2)))
            evaluate_and_save("SVD", seed, topk_from_scores(sc, users, b, cold), b,
                              {"rmse_test_pointwise": rm, "n_factors": 100, "n_epochs": 20, "fit_s": time.perf_counter() - t0}, device)
        if "NCF" in todo:
            t0 = time.perf_counter()
            model, hist = train_neumf(tu, ti, tr_, n_u, n_i, device, seed=seed, epochs=15)
            rm = rmse_neumf(model, te_ok["user_idx"].values, te_ok["item_idx"].values, te_ok["rating"].values, device)
            evaluate_and_save("NCF", seed, topk_from_scores(score_all_items_neumf(model, users, n_i, device), users, b, cold), b,
                              {"rmse_test_pointwise": rm, "epochs": 15, "final_train_mse": hist[-1]["train_mse"],
                               "fit_s": time.perf_counter() - t0}, device)
        if "NCF_val" in todo:
            t0 = time.perf_counter()
            core, vdf = b.train_core_df, b.val_df
            seen_core = np.zeros(n_i, dtype=bool); seen_core[core["item_idx"].values] = True
            vdf = vdf[seen_core[vdf["item_idx"].values]]
            _, hist = train_neumf(core["user_idx"].values, core["item_idx"].values, core["rating"].values.astype(np.float64),
                                  n_u, n_i, device, seed=seed, epochs=30,
                                  val=(vdf["user_idx"].values, vdf["item_idx"].values, vdf["rating"].values))
            e_star = int(min(hist, key=lambda h: h["val_rmse"])["epoch"])
            model, _ = train_neumf(tu, ti, tr_, n_u, n_i, device, seed=seed, epochs=e_star)
            rm = rmse_neumf(model, te_ok["user_idx"].values, te_ok["item_idx"].values, te_ok["rating"].values, device)
            evaluate_and_save("NCF_val", seed, topk_from_scores(score_all_items_neumf(model, users, n_i, device), users, b, cold), b,
                              {"rmse_test_pointwise": rm, "E_star": e_star, "best_val_rmse": min(h["val_rmse"] for h in hist),
                               "val_curve": hist, "fit_s": time.perf_counter() - t0}, device)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--out-dir", default=None,
                    help="Carpeta base de salida (crea per_seed_explicit/ y recs/ adentro). Default: results/stage5.")
    a = ap.parse_args()
    global OUT, RECS
    if a.out_dir:
        base = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
        OUT, RECS = base / "per_seed_explicit", base / "recs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate(device) if a.gate else run(a.seeds, device)


if __name__ == "__main__":
    main()
