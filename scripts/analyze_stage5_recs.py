"""
analyze_stage5_recs.py — Sesgo y diversidad desde las listas guardadas por run_stage5_save_recs.py.

Por modelo y semilla (top-10, 592 usuarios de test):
  NDCG@10 por usuario (misma fórmula que compute_metrics_at_10_and_50), popularidad media de lo
  recomendado (interacciones en train_full), Gini de la exposición sobre todo el catálogo,
  diversidad intra-lista (1 - similitud coseno media entre pares, embeddings MiniLM) y Cov@10.
Inferencia: bootstrap por usuarios (NDCG por usuario promediado sobre semillas, diferencia pareada por
usuario) y bootstrap por semillas, con Holm para los 3 contrastes primarios.
Salida: results/stage5/bias_diversity.json y .md
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from recsys.stage5.data_split import create_temporal_split
from scripts.compute_stats_stage5 import holm_correction, sign_test_pvalue

RECS = ROOT / "results/stage5/recs"
MODELS = ["FP_Two-Tower_A", "FP_Two-Tower_B", "FP_Two-Tower_C2", "FP_Two-Tower_D2", "FP_Two-Stage",
          "ES_Two-Tower_A", "ES_Two-Tower_B", "ES_Two-Tower_C2", "ES_Two-Tower_D2", "ES_Two-Stage",
          "G1", "G2", "G3_beam", "G3_exact", "G1b", "EXPL_SVD", "EXPL_NCF", "EXPL_NCF_val", "Popularidad"]
NB = 10000


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n, s = len(x), x.sum()
    return 0.0 if s == 0 else float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def ndcg_user(top10, rel):
    dcg = sum(1.0 / np.log2(r + 2) for r, it in enumerate(top10) if it in rel)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), 10)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = ap.parse_args()
    ratings = pd.read_csv(ROOT / "data/ml-latest-small/ratings.csv")
    movies = pd.read_csv(ROOT / "data/ml-latest-small/movies.csv")
    emb = np.load(ROOT / "data/cache/movie_embeddings_minilm.npy")
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    per = {m: {} for m in MODELS}      # per[m][seed] = dict(metrics..., ndcg_users=array)
    users_ref = None
    for seed in args.seeds:
        b = create_temporal_split(ratings, movies, seed=seed)
        users = list(b.eval_users_test)
        if users_ref is None:
            users_ref = users
        assert users == users_ref, f"Usuarios de test distintos en semilla {seed}"
        pop = np.asarray(b.item_popularity, dtype=np.float64)
        n_items = b.n_items
        cold_rank = np.argsort(pop)[::-1]
        for m in MODELS:
            if m == "Popularidad":
                recs = np.array([[i for i in cold_rank if i not in b.user_seen_train.get(u, set())][:10] for u in users])
            else:
                f = RECS / f"{m}_seed_{seed}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                assert list(z["users"]) == users
                recs = z["recs"][:, :10]
            assert (recs >= 0).all(), f"{m} semilla {seed}: lista corta"
            seen_viol = sum(len(set(recs[i]) & b.user_seen_train.get(u, set())) for i, u in enumerate(users))
            nd = np.array([ndcg_user(list(recs[i]), b.test_relevant_by_user.get(u, set())) for i, u in enumerate(users)])
            counts = np.bincount(recs.ravel(), minlength=n_items)
            div = []
            for r in recs:
                E = emb[r]; S = E @ E.T
                div.append(1.0 - (S.sum() - np.trace(S)) / (len(r) * (len(r) - 1)))
            per[m][seed] = {"ndcg10": float(nd.mean()), "ndcg_users": nd, "mean_pop": float(pop[recs].mean()),
                            "gini": gini(counts), "intra_div": float(np.mean(div)),
                            "cov10": float((counts > 0).sum() / n_items), "seen_violations": int(seen_viol)}
        print(f"semilla {seed} procesada", flush=True)

    rng = np.random.default_rng(0)
    summary = {}
    for m in MODELS:
        if not per[m]:
            continue
        ss = sorted(per[m])
        row = {"n_seeds": len(ss), "seeds": ss}
        for k in ["ndcg10", "mean_pop", "gini", "intra_div", "cov10", "seen_violations"]:
            v = np.array([per[m][s][k] for s in ss], dtype=float)
            row[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None}
        summary[m] = row

    def user_paired(a, bm):
        common = sorted(set(per[a]) & set(per[bm]))
        ua = np.mean([per[a][s]["ndcg_users"] for s in common], axis=0)
        ub = np.mean([per[bm][s]["ndcg_users"] for s in common], axis=0)
        d = ua - ub
        idx = rng.integers(0, len(d), size=(NB, len(d)))
        bs = d[idx].mean(axis=1)
        sa = np.array([per[a][s]["ndcg10"] - per[bm][s]["ndcg10"] for s in common])
        return {"comparison": f"{a} - {bm}", "n_seeds": len(common), "n_users": len(d), "mean_delta": float(d.mean()),
                "ci95_users": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "pvalue_sign_test": sign_test_pvalue(sa), "n_positive_seeds": int((sa > 0).sum()),
                "ci95_users_note": "bootstrap por usuarios sobre NDCG por usuario promediado entre semillas"}

    best = max([m for m in MODELS if m.startswith("FP_") and m != "FP_Two-Stage" and m in summary],
               key=lambda m: summary[m]["ndcg10"]["mean"])
    primary = holm_correction([user_paired("G1", "G3_beam"), user_paired("G1", "G2"), user_paired("G1", best)])
    second = [user_paired(a, c) for a, c in [("G3_beam", best), ("G1", "Popularidad"), ("G2", "Popularidad"),
                                              ("G3_beam", "Popularidad"), ("FP_Two-Stage", "FP_Two-Tower_A"),
                                              ("EXPL_SVD", "Popularidad"), ("EXPL_NCF", "Popularidad"),
                                              ("EXPL_NCF_val", "Popularidad"), ("EXPL_NCF", "EXPL_SVD"),
                                              ("FP_Two-Tower_A", "EXPL_SVD"), ("FP_Two-Tower_A", "EXPL_NCF"),
                                              ("G3_beam", "EXPL_SVD"), ("G1", "EXPL_SVD")]
              if a in summary and c in summary]
    out = {"best_30ep_posthoc": best, "summary": summary, "primary_holm": primary, "secondary": second}
    (ROOT / "results/stage5/bias_diversity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Sesgo y diversidad — Etapa 5 (top-10, 592 usuarios, media ± desvío entre semillas)", "",
         "Fuente: `results/stage5/recs/*.npz`, procesado por `scripts/analyze_stage5_recs.py`.",
         "Popularidad media = interacciones en train_full del ítem recomendado. Gini: desigualdad de la exposición sobre los "
         f"{n_items} ítems (0 = pareja, 1 = todo en un ítem). Diversidad intra-lista = 1 − coseno medio entre pares (MiniLM).", "",
         "| Modelo | n sem. | NDCG@10 | Cov@10 | Pop. media | Gini | Div. intra-lista |", "|---|---|---|---|---|---|---|"]
    fmt = lambda r, k, p=False: (f"{r[k]['mean']*100:.2f}%" if p else f"{r[k]['mean']:.3f}" if k != "ndcg10" else f"{r[k]['mean']:.4f}") + \
        (f" ± {r[k]['std']*(100 if p else 1):.{2 if p else (4 if k=='ndcg10' else 3)}f}" if r[k]["std"] is not None else "")
    for m, r in summary.items():
        L.append(f"| {m} | {r['n_seeds']} | {fmt(r,'ndcg10')} | {fmt(r,'cov10',True)} | {r['mean_pop']['mean']:.1f} | {fmt(r,'gini')} | {fmt(r,'intra_div')} |")
    L += ["", f"## Contrastes primarios con bootstrap por usuarios (Holm). Mejor con 30 épocas: {best} (post hoc)", "",
          "| Comparación | Δ NDCG@10 | IC95% (usuarios) | semillas a favor | p signo (semillas) | Umbral Holm | ¿Rechaza? |", "|---|---|---|---|---|---|---|"]
    for r in sorted(primary, key=lambda x: x["holm_rank"]):
        L.append(f"| {r['comparison']} | {r['mean_delta']:+.4f} | [{r['ci95_users'][0]:+.4f}, {r['ci95_users'][1]:+.4f}] | "
                 f"{r['n_positive_seeds']}/{r['n_seeds']} | {r['pvalue_sign_test']:.4f} | {r['holm_threshold']:.4f} | {'sí' if r['reject_holm'] else 'no'} |")
    L += ["", "## Contrastes secundarios (exploratorios)", "", "| Comparación | Δ NDCG@10 | IC95% (usuarios) | semillas a favor |", "|---|---|---|---|"]
    for r in second:
        L.append(f"| {r['comparison']} | {r['mean_delta']:+.4f} | [{r['ci95_users'][0]:+.4f}, {r['ci95_users'][1]:+.4f}] | {r['n_positive_seeds']}/{r['n_seeds']} |")
    L += ["", "Nota: el IC por usuarios mide la incertidumbre por la muestra de 592 usuarios; no incluye la variabilidad entre semillas "
          "(promedia las semillas dentro de cada usuario). Ambas se reportan por separado."]
    (ROOT / "results/stage5/bias_diversity.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
