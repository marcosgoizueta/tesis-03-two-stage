import json
import time
from pathlib import Path
import numpy as np
from scipy import stats

def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results" / "stage5"
    per_seed_dir = results_dir / "per_seed"

    seeds = list(range(42, 52))
    g1b_seeds = list(range(42, 47))

    models = [
        ("Two-Tower_A", "Two-Tower (A - Solo ID)", seeds),
        ("Two-Tower_B", "Two-Tower (B - Solo Texto)", seeds),
        ("Two-Tower_C2", "Two-Tower (C2 - Híbrido ID+Texto con Dropout)", seeds),
        ("Two-Tower_D2", "Two-Tower (D2 - Control ID+Géneros con Dropout)", seeds),
        ("Two-Stage", "Two-Stage (A + Ranker N=50)", seeds),
        ("G1", "G1 (Semantic IDs, haz 50)", seeds),
        ("G2", "G2 (Random IDs control, haz 50)", seeds),
        ("G3_beam", "G3 (Atomic IDs, haz 50)", seeds),
        ("G3_exact", "G3 (Atomic IDs, exacto)", seeds),
        ("G1b", "G1b (Semantic IDs sin año, haz 50)", g1b_seeds),
    ]

    metrics_list = ["ndcg@10", "recall@10", "recall@50", "cold_recall@50", "cov@10", "tail@10"]

    summary = {}
    table_data = []

    for mod_key, mod_name, mod_seeds in models:
        raw_metrics = {m: [] for m in metrics_list}
        e_stars = []
        for s in mod_seeds:
            if mod_key == "G3_beam":
                fpath = per_seed_dir / f"G3_seed_{s}.json"
                with open(fpath) as f:
                    data = json.load(f)["beam_50"]
            elif mod_key == "G3_exact":
                fpath = per_seed_dir / f"G3_seed_{s}.json"
                with open(fpath) as f:
                    data = json.load(f)["exact"]
            else:
                fpath = per_seed_dir / f"{mod_key}_seed_{s}.json"
                with open(fpath) as f:
                    data = json.load(f)

            for m in metrics_list:
                raw_metrics[m].append(data.get(m, 0.0))
            if "E_star" in data:
                e_stars.append(data["E_star"])

        model_stat = {
            "n_seeds": len(mod_seeds),
            "seeds": mod_seeds,
        }
        for m in metrics_list:
            vals = np.array(raw_metrics[m], dtype=float)
            model_stat[f"{m}_mean"] = float(np.mean(vals))
            model_stat[f"{m}_std"] = float(np.std(vals, ddof=1))
            model_stat[f"{m}_vals"] = list(vals)

        if e_stars:
            model_stat["E_star_mean"] = float(np.mean(e_stars))
            model_stat["E_star_vals"] = e_stars
        elif mod_key.startswith("G1"):
            model_stat["E_star_mean"] = 18.0
        elif mod_key.startswith("G2"):
            model_stat["E_star_mean"] = 18.0
        elif mod_key.startswith("G3"):
            model_stat["E_star_mean"] = 16.0

        summary[mod_key] = model_stat

        e_str = f"{model_stat.get('E_star_mean', '-'):.1f}" if "E_star_mean" in model_stat else "-"
        table_data.append({
            "Modelo": mod_name,
            "Semillas": f"{len(mod_seeds)}",
            "E*": e_str,
            "NDCG@10": f"{model_stat['ndcg@10_mean']:.4f} ± {model_stat['ndcg@10_std']:.4f}",
            "Recall@10": f"{model_stat['recall@10_mean']:.4f} ± {model_stat['recall@10_std']:.4f}",
            "Recall@50": f"{model_stat['recall@50_mean']:.4f} ± {model_stat['recall@50_std']:.4f}",
            "Cold Rec@50": f"{model_stat['cold_recall@50_mean']:.4f} ± {model_stat['cold_recall@50_std']:.4f}",
            "Cov@10": f"{model_stat['cov@10_mean']*100:.2f}% ± {model_stat['cov@10_std']*100:.2f}%",
            "Tail@10": f"{model_stat['tail@10_mean']*100:.2f}% ± {model_stat['tail@10_std']*100:.2f}%",
        })

    # Cargar azar y popularidad de semilla 42 para contexto de tabla
    seed42_path = results_dir / "stage5_seed42_summary.json"
    with open(seed42_path) as f:
        s42 = json.load(f)
    rand = s42["generatives_full"]["random_baseline"]
    pop = s42["generatives_full"]["test_results"]["Popularidad"]

    print("\n" + "="*95)
    print("TABLA MULTI-SEMILLA: RENDIMIENTO EN TEST (10 SEMILLAS / 5 EN G1b)")
    print("="*95)
    headers = ["Modelo", "Semillas", "E*", "NDCG@10", "Recall@10", "Recall@50", "Cold Rec@50", "Cov@10", "Tail@10"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    # Imprimir Azar
    print(f"| Azar (esperado / simulado) | - | - | - | - | {rand['cold_recall@50_random']:.4f} | {rand['cold_recall@50_random']:.4f} | {rand['expected_cov@10_mean']*100:.2f}% ± {rand['expected_cov@10_std']*100:.2f}% | {rand['expected_tail@10_analytic']*100:.2f}% |")
    # Imprimir Popularidad
    print(f"| Popularidad (control determinista) | - | - | {pop['ndcg@10']:.4f} | {pop['recall@10']:.4f} | {pop['recall@50']:.4f} | {pop['cold_recall@50']:.4f} | {pop['cov@10']*100:.2f}% | {pop['tail@10']*100:.2f}% |")

    for r in table_data:
        print(f"| {r['Modelo']} | {r['Semillas']} | {r['E*']} | {r['NDCG@10']} | {r['Recall@10']} | {r['Recall@50']} | {r['Cold Rec@50']} | {r['Cov@10']} | {r['Tail@10']} |")

    # Pruebas de hipótesis estadísticas (t-test pareado por semilla en ndcg@10)
    print("\n" + "="*95)
    print("TESTS ESTADÍSTICOS PAREADOS (NDCG@10)")
    print("="*95)
    comparisons = [
        ("G1", "G2", "G1 vs G2 (Efecto Semantic IDs vs Random IDs)"),
        ("G3_beam", "G1", "G3 (Atomic) vs G1 (Semantic IDs)"),
        ("G3_exact", "G3_beam", "G3 Exacto vs G3 Beam 50"),
        ("Two-Stage", "Two-Tower_A", "Two-Stage vs Two-Tower A"),
        ("Two-Tower_C2", "Two-Tower_A", "Two-Tower C2 vs Two-Tower A"),
        ("Two-Tower_C2", "G1", "Two-Tower C2 vs G1"),
        ("Two-Tower_C2", "G3_beam", "Two-Tower C2 vs G3"),
    ]
    stat_results = {}
    for k1, k2, label in comparisons:
        v1 = np.array(summary[k1]["ndcg@10_vals"])
        v2 = np.array(summary[k2]["ndcg@10_vals"])
        diff = v1 - v2
        t_stat, p_val = stats.ttest_rel(v1, v2)
        stat_results[label] = {
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant_p05": bool(p_val < 0.05),
            "significant_p01": bool(p_val < 0.01),
        }
        sig_str = "*** (p < 0.001)" if p_val < 0.001 else ("** (p < 0.01)" if p_val < 0.01 else ("* (p < 0.05)" if p_val < 0.05 else "n.s."))
        print(f"{label:55s} | Δ = {np.mean(diff):+.4f} | t = {t_stat:+.3f} | p = {p_val:.4e} | {sig_str}")

    # G1 vs G1b (5 semillas)
    v_g1_5 = np.array(summary["G1"]["ndcg@10_vals"][:5])
    v_g1b_5 = np.array(summary["G1b"]["ndcg@10_vals"])
    diff_b = v_g1_5 - v_g1b_5
    t_b, p_b = stats.ttest_rel(v_g1_5, v_g1b_5)
    sig_b = "*** (p < 0.001)" if p_b < 0.001 else ("** (p < 0.01)" if p_b < 0.01 else ("* (p < 0.05)" if p_b < 0.05 else "n.s."))
    print(f"{'G1 vs G1b (Efecto del año en Semantic IDs, 5 seeds)':55s} | Δ = {np.mean(diff_b):+.4f} | t = {t_b:+.3f} | p = {p_b:.4e} | {sig_b}")
    stat_results["G1 vs G1b"] = {
        "mean_diff": float(np.mean(diff_b)),
        "t_stat": float(t_b),
        "p_value": float(p_b),
        "significant_p05": bool(p_b < 0.05),
    }

    # Guardar resumen final en results/stage5
    out_file = results_dir / "multi_seed_summary.json"
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "n_users_test": 592,
            "models_summary": summary,
            "statistical_tests": stat_results,
            "table_data": table_data
        }, f, indent=2)
    print(f"\nResumen multi-semilla guardado en: {out_file}")

if __name__ == "__main__":
    main()
