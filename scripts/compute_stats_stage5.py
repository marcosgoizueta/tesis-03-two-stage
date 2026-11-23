"""
compute_stats_stage5.py — Estadística inferencial para los resultados de Etapa 5.

Calcula para NDCG@10:
  - Diferencias pareadas por semilla entre todos los pares relevantes.
  - Intervalo de confianza del 95% por bootstrap sobre semillas (10 000 resamples).
  - p-valor exacto por test de signo (binomial bilateral) sobre las 10/5 semillas.
  - Corrección de Holm para las comparaciones primarias:
      P1: G1 vs G3_beam
      P2: G1 vs G2
      P3: G1 vs Two-Stage
  - Comparación de G1, G2, G3_beam contra la línea de base de popularidad
    (valor fijo, test de signo contra 0 con la hipótesis nula de que la diferencia es 0).

Limitación: no hay datos por usuario almacenados en los per_seed/*.json.
El bootstrap aquí es sobre semillas (n=10 ó 5), no sobre usuarios de test.
Para bootstrap por usuario sería necesario re-ejecutar la evaluación guardando
per_user_ndcg en cada JSON; se propone eso como mejora futura.

Uso:
    python scripts/compute_stats_stage5.py
    python scripts/compute_stats_stage5.py --metric recall@10
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
PER_SEED_DIR = ROOT / "results" / "stage5" / "per_seed"
OUTPUT_PATH  = ROOT / "results" / "stage5" / "stats_report.json"

# NDCG@10 de popularidad: idéntico en las 10 semillas (results/stage5/seed_dependence_audit.json,
# generado por scripts/audit_stage5_seed_dependence.py).
POPULARITY_NDCG10 = 0.08205159442744218

N_BOOTSTRAP = 10_000
ALPHA = 0.05
RNG_SEED = 0  # semilla del bootstrap (independiente de las semillas de entrenamiento)

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

FLAT_MODELS = [
    "Two-Tower_A", "Two-Tower_B", "Two-Tower_C2", "Two-Tower_D2",
    "Two-Stage", "G1", "G2", "G1b",
]
SEEDS_10 = list(range(42, 52))
SEEDS_5  = list(range(42, 47))

MODEL_CFG = {
    "Two-Tower_A":  dict(prefix="Two-Tower_A",  seeds=SEEDS_10, extractor="flat"),
    "Two-Tower_B":  dict(prefix="Two-Tower_B",  seeds=SEEDS_10, extractor="flat"),
    "Two-Tower_C2": dict(prefix="Two-Tower_C2", seeds=SEEDS_10, extractor="flat"),
    "Two-Tower_D2": dict(prefix="Two-Tower_D2", seeds=SEEDS_10, extractor="flat"),
    "Two-Stage":    dict(prefix="Two-Stage",    seeds=SEEDS_10, extractor="flat"),
    "G1":           dict(prefix="G1",           seeds=SEEDS_10, extractor="flat"),
    "G2":           dict(prefix="G2",           seeds=SEEDS_10, extractor="flat"),
    "G3_beam":      dict(prefix="G3",           seeds=SEEDS_10, extractor="beam_50"),
    "G3_exact":     dict(prefix="G3",           seeds=SEEDS_10, extractor="exact"),
    "G1b":          dict(prefix="G1b",          seeds=SEEDS_5,  extractor="flat"),
}


def load_metric(prefix: str, seed: int, extractor: str, metric: str) -> float:
    path = PER_SEED_DIR / f"{prefix}_seed_{seed}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    if extractor == "flat":
        return float(d[metric])
    elif extractor == "beam_50":
        return float(d["beam_50"][metric])
    elif extractor == "exact":
        return float(d["exact"][metric])
    raise ValueError(f"Extractor desconocido: {extractor}")


def load_vals(model: str, metric: str) -> np.ndarray:
    cfg = MODEL_CFG[model]
    return np.array([
        load_metric(cfg["prefix"], s, cfg["extractor"], metric)
        for s in cfg["seeds"]
    ])


# ---------------------------------------------------------------------------
# Estadísticos
# ---------------------------------------------------------------------------

def bootstrap_ci(vals: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = ALPHA,
                 rng: np.random.Generator = None) -> tuple[float, float]:
    """IC percentil bootstrap bilateral sobre la media."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    resampled = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    lo = float(np.percentile(resampled, 100 * alpha / 2))
    hi = float(np.percentile(resampled, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci_paired(deltas: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = ALPHA,
                        rng: np.random.Generator = None) -> tuple[float, float]:
    """IC percentil bootstrap bilateral sobre la media de diferencias pareadas."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    resampled = rng.choice(deltas, size=(n, len(deltas)), replace=True).mean(axis=1)
    lo = float(np.percentile(resampled, 100 * alpha / 2))
    hi = float(np.percentile(resampled, 100 * (1 - alpha / 2)))
    return lo, hi


def sign_test_pvalue(deltas: np.ndarray) -> float:
    """
    Test de signo bilateral (hipótesis nula: mediana = 0).
    p = 2 * P(X >= k) donde X ~ Binomial(n, 0.5) y k = max(n_pos, n_neg).
    Con empates se usa n_pos + n_neg como n efectivo.
    """
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_eff = n_pos + n_neg
    if n_eff == 0:
        return 1.0
    k = max(n_pos, n_neg)
    # P(X >= k) = sum_{i=k}^{n} C(n,i) * 0.5^n
    p_one = sum(
        math.comb(n_eff, i) * (0.5 ** n_eff)
        for i in range(k, n_eff + 1)
    )
    return min(1.0, 2.0 * p_one)


def paired_stats(model_a: str, model_b: str, metric: str,
                 rng: np.random.Generator) -> dict:
    """Calcula estadísticos de A - B sobre semillas en común."""
    cfg_a = MODEL_CFG[model_a]
    cfg_b = MODEL_CFG[model_b]
    seeds = sorted(set(cfg_a["seeds"]) & set(cfg_b["seeds"]))
    vals_a = np.array([load_metric(cfg_a["prefix"], s, cfg_a["extractor"], metric) for s in seeds])
    vals_b = np.array([load_metric(cfg_b["prefix"], s, cfg_b["extractor"], metric) for s in seeds])
    deltas = vals_a - vals_b

    n_pos = int(np.sum(deltas > 0))
    ci_lo, ci_hi = bootstrap_ci_paired(deltas, rng=rng)
    pval_sign = sign_test_pvalue(deltas)

    # Wilcoxon signed-rank (non-parametric, uses magnitudes; None if all ties)
    try:
        _, pval_wilcoxon = sp_stats.wilcoxon(deltas, alternative="two-sided")
        pval_wilcoxon = float(pval_wilcoxon)
    except ValueError:
        pval_wilcoxon = None

    return {
        "comparison": f"{model_a} - {model_b}",
        "metric": metric,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=0)),
        "n_positive": n_pos,
        "n_negative": int(np.sum(deltas < 0)),
        "n_tie": int(np.sum(deltas == 0)),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "pvalue_sign_test": pval_sign,
        "pvalue_wilcoxon": pval_wilcoxon,
        "deltas": deltas.tolist(),
    }


def compare_vs_popularity(model: str, metric: str, pop_val: float,
                          rng: np.random.Generator) -> dict:
    """
    Compara un modelo contra una línea de base fija (popularidad).
    El test de signo se aplica sobre las diferencias val_seed - pop_val.
    """
    vals = load_vals(model, metric)
    deltas = vals - pop_val
    n_pos = int(np.sum(deltas > 0))
    ci_lo, ci_hi = bootstrap_ci_paired(deltas, rng=rng)
    pval_sign = sign_test_pvalue(deltas)
    try:
        _, pval_wilcoxon = sp_stats.wilcoxon(deltas, alternative="two-sided")
        pval_wilcoxon = float(pval_wilcoxon)
    except ValueError:
        pval_wilcoxon = None
    return {
        "comparison": f"{model} - Popularidad",
        "metric": metric,
        "n_seeds": len(vals),
        "mean_model": float(np.mean(vals)),
        "popularity_fixed": pop_val,
        "mean_delta": float(np.mean(deltas)),
        "n_positive": n_pos,
        "n_negative": int(np.sum(deltas < 0)),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "pvalue_sign_test": pval_sign,
        "pvalue_wilcoxon": pval_wilcoxon,
        "deltas": deltas.tolist(),
    }


def holm_correction(comparisons: list[dict]) -> list[dict]:
    """
    Aplica la corrección de Holm (stepdown Bonferroni) a una lista de
    comparaciones con clave 'pvalue_sign_test'.
    Agrega 'holm_threshold' y 'reject_holm' a cada una.
    """
    m = len(comparisons)
    indexed = sorted(enumerate(comparisons), key=lambda x: x[1]["pvalue_sign_test"])
    result = [None] * m
    rejected = True
    for rank, (orig_idx, comp) in enumerate(indexed):
        threshold = ALPHA / (m - rank)
        reject = rejected and (comp["pvalue_sign_test"] <= threshold)
        if not reject:
            rejected = False
        result[orig_idx] = {**comp,
                            "holm_rank": rank + 1,
                            "holm_threshold": threshold,
                            "reject_holm": reject}
    return result


# ---------------------------------------------------------------------------
# Resumen de distribución por modelo
# ---------------------------------------------------------------------------

def model_summary(model: str, metric: str, rng: np.random.Generator) -> dict:
    vals = load_vals(model, metric)
    ci_lo, ci_hi = bootstrap_ci(vals, rng=rng)
    return {
        "model": model,
        "metric": metric,
        "n_seeds": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "vals": vals.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="ndcg@10",
                        choices=["ndcg@10", "recall@10", "recall@50", "cov@10", "tail@10"],
                        help="Métrica principal para los contrastes (default: ndcg@10)")
    args = parser.parse_args()
    metric = args.metric

    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print(f"ESTADÍSTICA ETAPA 5 — métrica: {metric}")
    print("=" * 70)

    # 1. Distribución por modelo
    print("\n[1] Resumen por modelo")
    print(f"{'Modelo':<15} {'Media':>8} {'Std':>8} {'IC95%':>22}  {'n'}")
    print("-" * 65)
    summaries = {}
    for m in MODEL_CFG:
        try:
            s = model_summary(m, metric, rng)
        except (KeyError, FileNotFoundError) as e:
            print(f"  SKIP {m}: {e}")
            continue
        summaries[m] = s
        print(f"  {m:<13} {s['mean']:>8.4f} {s['std']:>8.4f}  "
              f"[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]  n={s['n_seeds']}")

    pop_val = POPULARITY_NDCG10
    print(f"  {'Popularidad':<13} {pop_val:>8.4f}  (valor fijo, semilla 42)")

    # 2. Comparaciones pareadas (todos los pares con semillas en común)
    print(f"\n[2] Diferencias pareadas (A - B, {metric})")
    all_pairs = [
        ("G1",       "G2"),
        ("G1",       "G3_beam"),
        ("G1",       "Two-Stage"),
        ("G1",       "Two-Tower_C2"),
        ("G3_beam",  "Two-Tower_C2"),
        ("G3_beam",  "Two-Stage"),
        ("G3_beam",  "G2"),
        ("G3_exact", "G3_beam"),
        ("Two-Stage","Two-Tower_C2"),
        ("Two-Tower_C2", "Two-Tower_A"),
        ("G1b",      "G1"),   # G1b solo tiene 5 semillas; intersección con G1 = 42-46
    ]
    pair_results = []
    for a, b in all_pairs:
        try:
            r = paired_stats(a, b, metric, rng)
        except (KeyError, FileNotFoundError) as e:
            print(f"  SKIP {a}-{b}: {e}")
            continue
        pair_results.append(r)
        sign = "+" if r["mean_delta"] > 0 else ""
        w_str = f"W={r['pvalue_wilcoxon']:.4f}" if r.get("pvalue_wilcoxon") is not None else "W=n/a"
        print(f"  {a:<14} - {b:<14}  "
              f"delta={sign}{r['mean_delta']:.4f}  "
              f"IC95%=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}]  "
              f"n+={r['n_positive']}/{r['n_seeds']}  "
              f"sign={r['pvalue_sign_test']:.4f}  {w_str}")

    # 3. Comparaciones primarias con corrección de Holm
    print(f"\n[3] Comparaciones primarias con corrección de Holm (α={ALPHA})")
    primary_keys = {("G1", "G3_beam"), ("G1", "G2"), ("G1", "Two-Stage")}
    primary = [r for r in pair_results
               if (r["comparison"].split(" - ")[0].strip(),
                   r["comparison"].split(" - ")[1].strip()) in primary_keys]

    # Verificar que las tres comparaciones primarias estén presentes
    if len(primary) < 3:
        missing = primary_keys - {
            (r["comparison"].split(" - ")[0].strip(),
             r["comparison"].split(" - ")[1].strip())
            for r in primary
        }
        print(f"  ADVERTENCIA: faltan comparaciones primarias: {missing}")

    primary_holm = holm_correction(primary)
    for r in sorted(primary_holm, key=lambda x: x["holm_rank"]):
        reject_str = "RECHAZA H0" if r["reject_holm"] else "no rechaza"
        w_str = f"W={r['pvalue_wilcoxon']:.4f}" if r.get("pvalue_wilcoxon") is not None else "W=n/a"
        print(f"  rank{r['holm_rank']} {r['comparison']:<33}  "
              f"sign={r['pvalue_sign_test']:.4f}  {w_str}  "
              f"umbral={r['holm_threshold']:.4f}  -> {reject_str}")

    # 4. Comparación contra popularidad
    print(f"\n[4] Comparación contra popularidad ({metric}={pop_val:.4f}, valor fijo)")
    pop_comparisons = {}
    for model in ["G1", "G2", "G3_beam", "G3_exact", "Two-Stage",
                  "Two-Tower_A", "Two-Tower_B", "Two-Tower_C2", "Two-Tower_D2"]:
        if model not in summaries:
            continue
        try:
            r = compare_vs_popularity(model, metric, pop_val, rng)
        except (KeyError, FileNotFoundError) as e:
            print(f"  SKIP {model}: {e}")
            continue
        pop_comparisons[model] = r
        sign = "+" if r["mean_delta"] > 0 else ""
        w_str = f"W={r['pvalue_wilcoxon']:.4f}" if r.get("pvalue_wilcoxon") is not None else "W=n/a"
        print(f"  {model:<14}  "
              f"media={r['mean_model']:.4f}  "
              f"delta={sign}{r['mean_delta']:.4f}  "
              f"n+={r['n_positive']}/{r['n_seeds']}  "
              f"sign={r['pvalue_sign_test']:.4f}  {w_str}")

    # 5. Aviso sobre bootstrap por usuarios
    print(
        "\n[NOTA] Bootstrap por usuarios de test: los per_seed/*.json almacenan solo "
        "métricas agregadas, no NDCG por usuario. Para hacer bootstrap por usuarios "
        "habría que re-ejecutar la evaluación guardando 'per_user_ndcg' en cada JSON "
        "(cambio en run_stage5_multi_seed.py). Los intervalos de confianza reportados "
        "aquí son bootstrap sobre las 10 (ó 5) semillas."
    )

    # 6. Guardar
    report = {
        "metric": metric,
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "bootstrap_rng_seed": RNG_SEED,
        "popularity_ndcg10": pop_val,
        "model_summaries": summaries,
        "paired_comparisons": pair_results,
        "primary_comparisons_holm": primary_holm,
        "vs_popularity": pop_comparisons,
        "note_per_user_bootstrap": (
            "No disponible: per_seed/*.json no almacenan NDCG por usuario. "
            "El bootstrap es sobre semillas (n=10 ó 5)."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
