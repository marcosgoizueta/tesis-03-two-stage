"""
build_final_table_stage5.py — Tabla comparativa unificada de la Etapa 5 (protocolo temporal).

Lee solo JSON de disco:
  results/stage5/per_seed/        (parada por validación + generativos)
  results/stage5/per_seed_30ep/   (Two-Tower y Two-Stage con 30 épocas fijas)
  results/stage5/fase2c_reference.json  (filas de Azar y Popularidad; semilla 42; copia versionada del registro de la Fase 2c)
Salida: results/stage5/final_comparison.json y final_comparison.md

Métricas disponibles por semilla: NDCG@10, Recall@10, Recall@50, Recall@50 en frío,
Cov@10, Tail@10.  Popularidad media recomendada, Gini y diversidad intra-lista NO se
pueden calcular desde estos JSON (no se guardaron las recomendaciones): pendientes.
IC95%: bootstrap percentil sobre semillas (no hay datos por usuario).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.compute_stats_stage5 import (bootstrap_ci, bootstrap_ci_paired, sign_test_pvalue,
                                          holm_correction, N_BOOTSTRAP)
from scipy import stats as sp

ES = ROOT / "results/stage5/per_seed"
FP = ROOT / "results/stage5/per_seed_30ep"
EX = ROOT / "results/stage5/per_seed_explicit"
S10, S5 = list(range(42, 52)), list(range(42, 47))
METRICS = ["ndcg@10", "recall@10", "recall@50", "cold_recall@50", "cov@10", "tail@10"]

# (etiqueta, directorio, prefijo, subclave, semillas, grupo)
MODELS = [
    ("Two-Tower A (30 ép.)",  FP, "Two-Tower_A",  None,      S10, "Discriminativo, 30 épocas fijas"),
    ("Two-Tower B (30 ép.)",  FP, "Two-Tower_B",  None,      S10, "Discriminativo, 30 épocas fijas"),
    ("Two-Tower C2 (30 ép.)", FP, "Two-Tower_C2", None,      S10, "Discriminativo, 30 épocas fijas"),
    ("Two-Tower D2 (30 ép.)", FP, "Two-Tower_D2", None,      S10, "Discriminativo, 30 épocas fijas"),
    ("Two-Stage (30 ép.)",    FP, "Two-Stage",    None,      S10, "Discriminativo, 30 épocas fijas"),
    ("Two-Tower A (val.)",    ES, "Two-Tower_A",  None,      S10, "Discriminativo, parada por validación"),
    ("Two-Tower B (val.)",    ES, "Two-Tower_B",  None,      S10, "Discriminativo, parada por validación"),
    ("Two-Tower C2 (val.)",   ES, "Two-Tower_C2", None,      S10, "Discriminativo, parada por validación"),
    ("Two-Tower D2 (val.)",   ES, "Two-Tower_D2", None,      S10, "Discriminativo, parada por validación"),
    ("Two-Stage (val.)",      ES, "Two-Stage",    None,      S10, "Discriminativo, parada por validación"),
    ("G1 (haz 50)",           ES, "G1",           None,      S10, "Generativo, E* fijo"),
    ("G2 (haz 50)",           ES, "G2",           None,      S10, "Generativo, E* fijo"),
    ("G3 (haz 50)",           ES, "G3",           "beam_50", S10, "Generativo, E* fijo"),
    ("G3 (exacto)",           ES, "G3",           "exact",   S10, "Generativo, E* fijo"),
    ("G1b (haz 50, 5 sem.)",  ES, "G1b",          None,      S5,  "Generativo, E* fijo"),
    ("SVD (Etapa 1)",         EX, "SVD",          None,      S10, "Regresión de rating, Etapas 1 y 2"),
    ("NCF (Etapa 2, 15 ép.)", EX, "NCF",          None,      S10, "Regresión de rating, Etapas 1 y 2"),
    ("NCF (épocas por val.)", EX, "NCF_val",      None,      S10, "Regresión de rating, Etapas 1 y 2"),
]
LABEL = {m[0]: m for m in MODELS}


def load(label, metric):
    _, d, pre, sub, seeds, _ = LABEL[label]
    out = []
    for s in seeds:
        j = json.loads((d / f"{pre}_seed_{s}.json").read_text(encoding="utf-8"))
        out.append(float((j[sub] if sub else j)[metric]))
    return np.array(out)


def paired(a, b, metric, rng):
    sa, sb = LABEL[a][4], LABEL[b][4]
    common = [s for s in sa if s in sb]
    va = load(a, metric)[[sa.index(s) for s in common]]
    vb = load(b, metric)[[sb.index(s) for s in common]]
    d = va - vb
    lo, hi = bootstrap_ci_paired(d, rng=rng)
    try:
        w = float(sp.wilcoxon(d).pvalue)
    except ValueError:
        w = None
    return {"comparison": f"{a} - {b}", "metric": metric, "n_seeds": len(common),
            "mean_delta": float(d.mean()), "ci95_lo": lo, "ci95_hi": hi,
            "n_positive": int((d > 0).sum()), "pvalue_sign_test": sign_test_pvalue(d), "pvalue_wilcoxon": w}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    ref = json.loads((ROOT / "results/stage5/fase2c_reference.json").read_text(encoding="utf-8"))
    pop, rnd = ref["test_results"]["Popularidad"], ref["random_baseline"]

    table = {}
    for label, *_ in MODELS:
        row = {}
        for m in METRICS:
            v = load(label, m)
            lo, hi = bootstrap_ci(v, rng=rng)
            row[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "ci95": [lo, hi], "n": len(v)}
        table[label] = row

    # Mejor discriminativo con 30 épocas por NDCG@10 medio (selección sobre el agregado de test: post hoc).
    disc30 = [m[0] for m in MODELS if m[1] is FP and m[0] != "Two-Stage (30 ép.)"] + ["Two-Stage (30 ép.)"]
    best = max(disc30, key=lambda l: table[l]["ndcg@10"]["mean"])

    primary = [paired("G1 (haz 50)", "G3 (haz 50)", "ndcg@10", rng),
               paired("G1 (haz 50)", "G2 (haz 50)", "ndcg@10", rng),
               paired("G1 (haz 50)", best, "ndcg@10", rng)]
    primary = holm_correction(primary)
    secondary = [paired(a, b, "ndcg@10", rng) for a, b in [
        ("G3 (haz 50)", best), ("G3 (haz 50)", "Two-Stage (30 ép.)"),
        ("Two-Stage (30 ép.)", "Two-Tower A (30 ép.)"), ("Two-Stage (val.)", "Two-Tower A (val.)"),
        ("Two-Tower C2 (30 ép.)", "Two-Tower D2 (30 ép.)"), ("Two-Tower C2 (30 ép.)", "Two-Tower A (30 ép.)"),
        ("G3 (exacto)", "G3 (haz 50)"), ("NCF (Etapa 2, 15 ép.)", "SVD (Etapa 1)"),
        ("NCF (épocas por val.)", "NCF (Etapa 2, 15 ép.)"), ("Two-Tower A (30 ép.)", "SVD (Etapa 1)"),
        ("G3 (haz 50)", "SVD (Etapa 1)"), ("G1 (haz 50)", "SVD (Etapa 1)")]]
    cov = [paired(a, b, "cov@10", rng) for a, b in [
        ("G1 (haz 50)", "G2 (haz 50)"), ("G3 (haz 50)", best), ("G1 (haz 50)", best)]]

    out = {"best_discriminative_30ep_by_mean_ndcg10_posthoc": best, "n_bootstrap": N_BOOTSTRAP,
           "references": {"popularity_seed42": pop, "random_expected": rnd,
                          "note": "Popularidad: NDCG@10 y Cov@10 idénticos en las 10 semillas (seed_dependence_audit.json); "
                                  "fila leída de results/stage5/fase2c_reference.json (semilla 42)."},
           "table": table, "primary_holm": primary, "secondary_paired_ndcg10": secondary,
           "paired_cov10": cov,
           "pending_metrics": ["popularidad media recomendada", "Gini", "diversidad intra-lista",
                               "bootstrap por usuarios de test"]}
    (ROOT / "results/stage5/final_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    f = lambda r, m, pct=False: (f"{r[m]['mean']*100:.2f}%" if pct else f"{r[m]['mean']:.4f}")
    L = ["# Tabla comparativa unificada — Etapa 5 (split temporal, 592 usuarios de test)", "",
         "Fuente: `results/stage5/final_comparison.json`, generado por `scripts/build_final_table_stage5.py` "
         "desde `per_seed/` y `per_seed_30ep/`. 10 semillas (G1b: 5). Media ± desvío muestral; entre corchetes, "
         "IC95% bootstrap sobre semillas. Cov@10 y Tail@10 con K=10 para todos los modelos.", "",
         "| Modelo | NDCG@10 | Recall@10 | Recall@50 | Cold Rec@50 | Cov@10 | Tail@10 |", "|---|---|---|---|---|---|---|"]
    L.append(f"| Azar (esperado) | - | - | - | {rnd['cold_recall@50_random']*100:.2f}% | {rnd['expected_cov@10_mean']*100:.2f}% | {rnd['expected_tail@10_analytic']*100:.2f}% |")
    L.append(f"| Popularidad | {pop['ndcg@10']:.4f} | {pop['recall@10']:.4f} | {pop['recall@50']:.4f} | {pop['cold_recall@50']:.4f} | {pop['cov@10']*100:.2f}% | {pop['tail@10']*100:.2f}% |")
    grp = None
    for label, *_rest in MODELS:
        g = LABEL[label][5]
        if g != grp:
            L.append(f"| *{g}* | | | | | | |"); grp = g
        r = table[label]
        c = r["ndcg@10"]["ci95"]
        L.append(f"| {label} | {r['ndcg@10']['mean']:.4f} ± {r['ndcg@10']['std']:.4f} [{c[0]:.4f}, {c[1]:.4f}] | "
                 f"{f(r,'recall@10')} | {f(r,'recall@50')} | {f(r,'cold_recall@50')} | {f(r,'cov@10',True)} | {f(r,'tail@10',True)} |")
    L += ["", f"## Contrastes primarios (NDCG@10, Holm α=0.05). Mejor discriminativo con 30 épocas: **{best}** "
          "(elegido por NDCG@10 medio sobre test: selección post hoc; la brecha con G1 es grande, pero queda declarado).", "",
          "| Comparación | Δ medio | IC95% | n+ | p signo | p Wilcoxon | Umbral Holm | ¿Rechaza? |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(primary, key=lambda x: x["holm_rank"]):
        L.append(f"| {r['comparison']} | {r['mean_delta']:+.4f} | [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {r['n_positive']}/{r['n_seeds']} | "
                 f"{r['pvalue_sign_test']:.4f} | {r['pvalue_wilcoxon']:.4f} | {r['holm_threshold']:.4f} | {'sí' if r['reject_holm'] else 'no'} |")
    L += ["", "## Contrastes secundarios (NDCG@10, sin corrección; exploratorios)", "",
          "| Comparación | Δ medio | IC95% | n+ | p signo | p Wilcoxon |", "|---|---|---|---|---|---|"]
    for r in secondary:
        w = f"{r['pvalue_wilcoxon']:.4f}" if r["pvalue_wilcoxon"] is not None else "n/a"
        L.append(f"| {r['comparison']} | {r['mean_delta']:+.4f} | [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {r['n_positive']}/{r['n_seeds']} | {r['pvalue_sign_test']:.4f} | {w} |")
    L += ["", "## Contrastes de cobertura @10 (exploratorios)", "",
          "| Comparación | Δ medio (puntos de cobertura) | IC95% | n+ | p Wilcoxon |", "|---|---|---|---|---|"]
    for r in cov:
        L.append(f"| {r['comparison']} | {r['mean_delta']*100:+.2f} | [{r['ci95_lo']*100:+.2f}, {r['ci95_hi']*100:+.2f}] | {r['n_positive']}/{r['n_seeds']} | {r['pvalue_wilcoxon']:.4f} |")
    L += ["", "## Límites y pendientes", "",
          "- Con 10 semillas el p mínimo del test de signo es 0.002; ese valor aparece siempre que las 10 diferencias tienen el mismo signo.",
          "- Los IC95% son bootstrap sobre semillas; no hay bootstrap por usuarios porque los JSON no guardan métricas por usuario.",
          "- Faltan popularidad media, Gini y diversidad intra-lista: requieren guardar las listas recomendadas (re-evaluar modelos).",
          "- Popularidad y azar son referencias de la semilla 42 (popularidad idéntica en las 10 semillas; azar es esperado analítico/simulado).",
          "- Asimetría de protocolo: Two-Tower con épocas elegidas por validación o fijas a 30; generativos con E* fijo elegido en validación (G1/G2: 18, G3: 16)."]
    (ROOT / "results/stage5/final_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
