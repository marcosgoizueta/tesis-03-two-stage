import json
import shutil
import time
from pathlib import Path
import numpy as np

def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results" / "stage5"
    curves_dir = results_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copiar curvas de scratch/raw_curves a results/stage5/curves
    scratch_curves = root / "scratch" / "raw_curves"
    if scratch_curves.exists():
        for f in scratch_curves.glob("*.json"):
            dest = curves_dir / f.name
            if not dest.exists():
                shutil.copy(f, dest)

    # 2. Cargar resultados
    gen_results_path = results_dir / "fase2c_reference.json"
    if not gen_results_path.exists():
        gen_results_path = root / "scratch" / "stage5_fase2c_results.json"
    baselines_results_path = results_dir / "baselines_early_stopping_results.json"

    with open(gen_results_path, "r") as f:
        gen_data = json.load(f)

    with open(baselines_results_path, "r") as f:
        baselines_data = json.load(f)

    # 3. TABLA 1: Regularización y Parada Temprana por val_loss
    print("\n" + "="*85)
    print("TABLA 1: Regularización y Parada Temprana por val_loss (Semilla 42)")
    print("="*85)
    table1_rows = []
    for curve_file in ["G1_base.json", "G1_reg.json", "G2_base.json", "G3_base.json", "G3_reg.json"]:
        c_path = curves_dir / curve_file
        if c_path.exists():
            with open(c_path, "r") as f:
                cd = json.load(f)
            name = f"{cd['variant']} ({cd['config_name']}: lr={cd['lr']}, drop={cd['dropout_rate']})"
            best_ep = cd["best_epoch"]
            best_vl = cd["best_val_loss"]
            tr_loss = None
            for h in cd["history"]:
                if h["epoch"] == best_ep:
                    tr_loss = h["train_loss"]
                    break
            diff = best_vl - tr_loss if (best_vl and tr_loss) else 0.0
            table1_rows.append({
                "Configuración": name,
                "E*": best_ep,
                "Train Loss (E*)": f"{tr_loss:.4f}" if tr_loss else "N/A",
                "Val Loss (E*)": f"{best_vl:.4f}" if best_vl else "N/A",
                "Brecha Val-Train": f"{diff:.4f}" if diff else "N/A",
            })

    headers1 = ["Configuración", "E*", "Train Loss (E*)", "Val Loss (E*)", "Brecha Val-Train"]
    print("| " + " | ".join(headers1) + " |")
    print("| " + " | ".join(["---"] * len(headers1)) + " |")
    for r in table1_rows:
        print(f"| {r['Configuración']} | {r['E*']} | {r['Train Loss (E*)']} | {r['Val Loss (E*)']} | {r['Brecha Val-Train']} |")

    # 4. TABLA 2: Resultados en Test (Semilla 42, 592 usuarios)
    print("\n" + "="*85)
    print("TABLA 2: Rendimiento en Test (Semilla 42, 592 usuarios evaluables)")
    print("="*85)
    headers2 = ["Modelo", "E*", "NDCG@10", "Recall@10", "Recall@50", "Cold Rec@50", "Cov@10", "Tail@10"]
    print("| " + " | ".join(headers2) + " |")
    print("| " + " | ".join(["---"] * len(headers2)) + " |")

    rows2 = []
    # Azar
    rand = gen_data["random_baseline"]
    rows2.append({
        "Modelo": "Azar (esperado / simulado)",
        "E*": "-",
        "NDCG@10": "-",
        "Recall@10": "-",
        "Recall@50": f"{rand['cold_recall@50_random']:.4f}",
        "Cold Rec@50": f"{rand['cold_recall@50_random']:.4f}",
        "Cov@10": f"{rand['expected_cov@10_mean']*100:.2f}%",
        "Tail@10": f"{rand['expected_tail@10_analytic']*100:.2f}%",
    })

    # Popularidad
    pop = gen_data["test_results"]["Popularidad"]
    rows2.append({
        "Modelo": "Popularidad",
        "E*": "-",
        "NDCG@10": f"{pop['ndcg@10']:.4f}",
        "Recall@10": f"{pop['recall@10']:.4f}",
        "Recall@50": f"{pop['recall@50']:.4f}",
        "Cold Rec@50": f"{pop['cold_recall@50']:.4f}",
        "Cov@10": f"{pop['cov@10']*100:.2f}%",
        "Tail@10": f"{pop['tail@10']*100:.2f}%",
    })

    # Baselines
    for b_name, b_m in baselines_data.items():
        rows2.append({
            "Modelo": b_name,
            "E*": str(b_m.get("E_star", "-")),
            "NDCG@10": f"{b_m['ndcg@10']:.4f}",
            "Recall@10": f"{b_m['recall@10']:.4f}",
            "Recall@50": f"{b_m['recall@50']:.4f}",
            "Cold Rec@50": f"{b_m['cold_recall@50']:.4f}",
            "Cov@10": f"{b_m['cov@10']*100:.2f}%",
            "Tail@10": f"{b_m['tail@10']*100:.2f}%",
        })

    # Generativos (haz 50)
    gen_map = [
        ("G1 (base, haz 50)", gen_data["test_results"]["G1_beam_50"], 18),
        ("G2 (base, haz 50)", gen_data["test_results"]["G2_beam_50"], 18),
        ("G3 (reg, haz 50)", gen_data["test_results"]["G3_beam_50"], 16),
        ("G1b (base, haz 50)", gen_data["test_results"]["G1b_beam_50"], 18),
    ]
    for g_label, g_m, e_star in gen_map:
        rows2.append({
            "Modelo": g_label,
            "E*": str(e_star),
            "NDCG@10": f"{g_m['ndcg@10']:.4f}",
            "Recall@10": f"{g_m['recall@10']:.4f}",
            "Recall@50": f"{g_m['recall@50']:.4f}",
            "Cold Rec@50": f"{g_m['cold_recall@50']:.4f}",
            "Cov@10": f"{g_m['cov@10']*100:.2f}%",
            "Tail@10": f"{g_m['tail@10']*100:.2f}%",
        })

    for r in rows2:
        print(f"| {r['Modelo']} | {r['E*']} | {r['NDCG@10']} | {r['Recall@10']} | {r['Recall@50']} | {r['Cold Rec@50']} | {r['Cov@10']} | {r['Tail@10']} |")

    # 5. TABLA 3: Sensibilidad al haz en G1, G2 y G3
    print("\n" + "="*85)
    print("TABLA 3: Sensibilidad al Haz en G1, G2 y G3 (K <= haz)")
    print("="*85)
    headers3 = ["Modelo", "Haz", "NDCG@10", "Recall@10", "Recall@50", "Cov@10", "Tail@10", "Latencia (ms/u)"]
    print("| " + " | ".join(headers3) + " |")
    print("| " + " | ".join(["---"] * len(headers3)) + " |")
    rows3 = []
    # G1
    for b in [20, 50, 100]:
        m = gen_data["test_results"][f"G1_beam_{b}"]
        rec50_str = f"{m['recall@50']:.4f}" if b >= 50 else "-"
        rows3.append({
            "Modelo": "G1", "Haz": str(b), "NDCG@10": f"{m['ndcg@10']:.4f}",
            "Recall@10": f"{m['recall@10']:.4f}", "Recall@50": rec50_str,
            "Cov@10": f"{m['cov@10']*100:.2f}%", "Tail@10": f"{m['tail@10']*100:.2f}%",
            "Latencia (ms/u)": f"{m['latency_ms']:.1f}"
        })
    # G2 (haz 50)
    m = gen_data["test_results"]["G2_beam_50"]
    rows3.append({
        "Modelo": "G2", "Haz": "50", "NDCG@10": f"{m['ndcg@10']:.4f}",
        "Recall@10": f"{m['recall@10']:.4f}", "Recall@50": f"{m['recall@50']:.4f}",
        "Cov@10": f"{m['cov@10']*100:.2f}%", "Tail@10": f"{m['tail@10']*100:.2f}%",
        "Latencia (ms/u)": f"{m['latency_ms']:.1f}"
    })
    # G3 (haz 50)
    m = gen_data["test_results"]["G3_beam_50"]
    rows3.append({
        "Modelo": "G3", "Haz": "50", "NDCG@10": f"{m['ndcg@10']:.4f}",
        "Recall@10": f"{m['recall@10']:.4f}", "Recall@50": f"{m['recall@50']:.4f}",
        "Cov@10": f"{m['cov@10']*100:.2f}%", "Tail@10": f"{m['tail@10']*100:.2f}%",
        "Latencia (ms/u)": f"{m['latency_ms']:.1f}"
    })

    for r in rows3:
        print(f"| {r['Modelo']} | {r['Haz']} | {r['NDCG@10']} | {r['Recall@10']} | {r['Recall@50']} | {r['Cov@10']} | {r['Tail@10']} | {r['Latencia (ms/u)']} |")

    # 6. Guardar consolidado final
    summary_path = results_dir / "stage5_seed42_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "seed": 42,
            "n_eval_val": 573,
            "n_eval_test": 592,
            "table1_regularization": table1_rows,
            "table2_test_results": rows2,
            "table3_beam_sensitivity": rows3,
            "generatives_full": gen_data,
            "baselines_full": baselines_data
        }, f, indent=2)
    print(f"\nResumen consolidado guardado en: {summary_path}")

if __name__ == "__main__":
    main()
