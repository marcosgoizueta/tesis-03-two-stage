#!/usr/bin/env python
"""Script de ejecución multi-semilla para análisis de estabilidad y diferencias pareadas.

Genera:
- results/stage3_per_seed.csv (formato largo: modelo, semilla y todas las métricas)
- results/stage3_summary.csv (media, desvío, mín, máx, y n por modelo y métrica)
- results/stage3_paired_differences.csv (diferencias pareadas por semilla para contrastes clave)
- Medición de latencia honesta por usuario en CPU separando retrieval y ranker.

Nota importante: en modo multi-semilla NO se modifica results/comparativa_modelos.csv.
"""

import argparse
import random
import sys
import time
from pathlib import Path

# En Windows, torch debe importarse primero
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.config import Stage3Config
from recsys.data import load_and_prepare_data
from recsys.evaluation import (
    evaluate_model,
    make_one_stage,
    make_rec_popularity,
    make_rec_retrieval_only,
    make_two_stage,
    measure_honest_latency,
)
from recsys.ranker import train_ranker
from recsys.retrieval import (
    RetrievalDataset,
    TwoTower,
    compute_retrieval_candidates,
    train_retrieval,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación multi-semilla para análisis de estabilidad de la Etapa 3"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Lista de semillas a evaluar (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo de cómputo (default: auto)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Ruta a datos (default: data)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Ruta a resultados (default: results)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Sufijo para los archivos CSV generados (ej: _10seeds)",
    )
    return parser.parse_args()


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def plot_and_save_losses(
    retrieval_losses: list,
    trivial_bce: float,
    figures_dir: Path,
    epochs: int,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "loss_retrieval.png"

    plt.figure(figsize=(6, 4))
    if epochs == 1:
        plt.plot(range(1, epochs + 1), retrieval_losses, label="Retrieval", marker="o")
    else:
        plt.plot(range(1, epochs + 1), retrieval_losses, label="Retrieval")
    plt.axhline(trivial_bce, color="gray", linestyle="--", label="Modelo trivial")
    plt.legend()
    plt.title("Curva de entrenamiento — Retrieval (BCE)")
    plt.xlabel("Época")
    plt.ylabel("BCE loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Figura de pérdida guardada en {out_path}")
    return out_path


def run_pipeline_for_seed(seed: int, cfg: Stage3Config, data, device: torch.device):
    print(f"\n==========================================")
    print(f"EJECUTANDO SEMILLA: {seed}")
    print(f"==========================================")
    set_seed(seed, device)

    # 1. Retrieval
    retrieval_dataset = RetrievalDataset(
        positive_df=data.train_positive,
        user_seen_train=data.user_seen_train,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
        fast_sampling=False,
    )
    retrieval_model = TwoTower(
        n_users=data.n_users,
        n_items=data.n_items,
        embed_dim=cfg.embed_dim_retrieval,
    ).to(device)

    retrieval_losses = train_retrieval(retrieval_model, retrieval_dataset, cfg, device)

    # Guardar curva de entrenamiento para la semilla base
    if seed == cfg.seed:
        p_pos = 1 / (1 + cfg.n_neg_per_pos_retrieval)
        trivial_bce = -(p_pos * np.log(p_pos) + (1 - p_pos) * np.log(1 - p_pos))
        plot_and_save_losses(
            retrieval_losses,
            trivial_bce,
            cfg.figures_dir,
            cfg.n_epochs_retrieval,
        )

    retrieval_topk = compute_retrieval_candidates(
        model=retrieval_model,
        n_users=data.n_users,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        max_candidates=cfg.max_candidates,
        device=device,
    )

    # 2. Rankers
    pos_users = data.train_positive["user_idx"].values
    pos_items = data.train_positive["item_idx"].values

    rankers = {}
    for variant, (n_uni, n_hard) in {"uniforme": (4, 0), "mixto": (2, 2)}.items():
        print(f"Entrenando Ranker [{variant}] (seed {seed})...")
        model, _ = train_ranker(
            pos_users=pos_users,
            pos_items=pos_items,
            n_users=data.n_users,
            n_items=data.n_items,
            user_genre_np=data.user_genre_np,
            item_genre_np=data.item_genre_np,
            item_logpop_np=data.item_logpop_np,
            seen_mask_np=data.seen_mask_np,
            retrieval_topk=retrieval_topk,
            n_uniform=n_uni,
            n_hard=n_hard,
            cfg=cfg,
            device=device,
            seed=seed,
        )
        rankers[variant] = model

    # 3. Evaluación
    eval_users = data.eval_users
    results = [
        evaluate_model(
            name="Popularidad (baseline)",
            rec_fn=make_rec_popularity(data.item_popularity, data.seen_mask_np, cfg.top_k),
            eval_users=eval_users,
            test_relevant_by_user=data.test_relevant_by_user,
            item_popularity=data.item_popularity,
            n_items=data.n_items,
            device=device,
            top_k=cfg.top_k,
        ),
        evaluate_model(
            name="Retrieval (two-tower) solo",
            rec_fn=make_rec_retrieval_only(retrieval_topk, cfg.top_k),
            eval_users=eval_users,
            test_relevant_by_user=data.test_relevant_by_user,
            item_popularity=data.item_popularity,
            n_items=data.n_items,
            device=device,
            top_k=cfg.top_k,
        ),
    ]

    for variant, model in rankers.items():
        results.append(
            evaluate_model(
                name=f"Ranker[{variant}] catálogo completo (1 etapa)",
                rec_fn=make_one_stage(model, data.n_items, data.seen_mask_np, device, cfg.top_k),
                eval_users=eval_users,
                test_relevant_by_user=data.test_relevant_by_user,
                item_popularity=data.item_popularity,
                n_items=data.n_items,
                device=device,
                top_k=cfg.top_k,
            )
        )
        for n in cfg.candidate_ns_to_eval:
            results.append(
                evaluate_model(
                    name=f"Two-Stage[{variant}] N={n}",
                    rec_fn=make_two_stage(model, retrieval_topk, n, device, cfg.top_k),
                    eval_users=eval_users,
                    test_relevant_by_user=data.test_relevant_by_user,
                    item_popularity=data.item_popularity,
                    n_items=data.n_items,
                    device=device,
                    top_k=cfg.top_k,
                )
            )

    for r in results:
        r["semilla"] = seed

    # Medir latencia honesta en la primera semilla
    honest_latencies = None
    if seed == cfg.seed:
        print("\n--- Midiendo Latencia Honesta del Pipeline Two-Stage en CPU ---")
        honest_latencies = measure_honest_latency(
            retrieval_model=retrieval_model,
            ranker_model=rankers["uniforme"],
            eval_users=eval_users,
            n_items=data.n_items,
            seen_mask_np=data.seen_mask_np,
            n_candidates_list=(50, 200),
            device=device,
        )

    return results, honest_latencies


def compute_paired_differences(df_long: pd.DataFrame, metric: str = "ndcg_at_10"):
    """Calcula diferencias pareadas por semilla para los contrastes requeridos."""
    contrasts = [
        (
            "Two-Stage[uniforme] N=50",
            "Retrieval (two-tower) solo",
            "Two-Stage[uniforme] N=50 menos Retrieval solo",
        ),
        (
            "Two-Stage[uniforme] N=50",
            "Ranker[uniforme] catálogo completo (1 etapa)",
            "Two-Stage[uniforme] N=50 menos Ranker[uniforme] catálogo completo",
        ),
        (
            "Retrieval (two-tower) solo",
            "Popularidad (baseline)",
            "Retrieval solo menos Popularidad",
        ),
        (
            "Ranker[uniforme] catálogo completo (1 etapa)",
            "Retrieval (two-tower) solo",
            "Ranker[uniforme] catálogo completo menos Retrieval solo",
        ),
    ]

    piv = df_long.pivot(index="semilla", columns="modelo", values=metric)
    contrast_rows = []

    for m_a, m_b, label in contrasts:
        diffs = piv[m_a] - piv[m_b]
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std()) if len(diffs) > 1 else 0.0
        n_pos = int((diffs > 0).sum())
        total = len(diffs)
        contrast_rows.append({
            "contraste": label,
            "metrica": metric,
            "media_diferencia": mean_diff,
            "desvio_diferencia": std_diff,
            "semillas_positivas": f"{n_pos}/{total} ({n_pos/total*100:.1f}%)",
        })

    return pd.DataFrame(contrast_rows)


def main():
    start_time = time.perf_counter()
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    
    # Separar almacenamiento por dispositivo para no mezclar ni sobrescribir
    target_results_dir = args.results_dir / ("gpu" if device.type == "cuda" else "cpu")
    target_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ANÁLISIS MULTI-SEMILLA (ETAPA 3)")
    print(f"Semillas: {args.seeds} | Dispositivo: {device} ({dev_name})")
    print(f"Directorio de resultados: {target_results_dir}")
    print(f"Nota: NO se modifica results/comparativa_modelos.csv")
    print("=" * 70)

    cfg = Stage3Config(
        data_dir=args.data_dir,
        results_dir=target_results_dir,
        seed=args.seeds[0],
        device=args.device,
    )

    # El split se realiza una sola vez con random_state=42 (fijo e independiente de la semilla)
    data = load_and_prepare_data(cfg)

    all_results = []
    first_latency_report = None

    for s in args.seeds:
        seed_results, honest_lat = run_pipeline_for_seed(s, cfg, data, device)
        all_results.extend(seed_results)
        if honest_lat is not None and first_latency_report is None:
            first_latency_report = honest_lat

    total_time = time.perf_counter() - start_time
    df_long = pd.DataFrame(all_results)

    # 1. Guardar results/<device>/stage3_per_seed<suffix>.csv
    per_seed_csv = target_results_dir / f"stage3_per_seed{args.suffix}.csv"
    cols_order = [
        "modelo",
        "semilla",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "cobertura_catalogo",
        "popularidad_media_recs",
        "ms_por_usuario",
    ]
    df_long[cols_order].to_csv(per_seed_csv, index=False)
    print(f"\nResultados por semilla guardados en {per_seed_csv}")

    # 2. Generar results/<device>/stage3_summary<suffix>.csv (media, desvío, mín, máx, y n)
    metric_cols = [
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "cobertura_catalogo",
        "popularidad_media_recs",
        "ms_por_usuario",
    ]

    summary_rows = []
    for model_name, grp in df_long.groupby("modelo"):
        n_samples = len(grp)
        for m in metric_cols:
            summary_rows.append({
                "modelo": model_name,
                "metrica": m,
                "n": n_samples,
                "media": float(grp[m].mean()),
                "desvio": float(grp[m].std()) if n_samples > 1 else 0.0,
                "min": float(grp[m].min()),
                "max": float(grp[m].max()),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = target_results_dir / f"stage3_summary{args.suffix}.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Resumen estadístico guardado en {summary_csv}")

    # 3. Diferencias pareadas por semilla
    paired_ndcg = compute_paired_differences(df_long, metric="ndcg_at_10")
    paired_rec = compute_paired_differences(df_long, metric="recall_at_10")
    paired_prec = compute_paired_differences(df_long, metric="precision_at_10")
    paired_df = pd.concat([paired_ndcg, paired_rec, paired_prec], ignore_index=True)

    paired_csv = target_results_dir / f"stage3_paired_differences{args.suffix}.csv"
    paired_df.to_csv(paired_csv, index=False)
    print(f"Diferencias pareadas guardadas en {paired_csv}")

    # 4. Imprimir Reportes Finales
    print("\n" + "=" * 100)
    print("RESUMEN DE MÉTRICAS (MEDIA ± DESVÍO ESTÁNDAR)")
    print(f"Semillas evaluadas: {args.seeds} | Tiempo total: {total_time:.2f} s ({total_time/60:.2f} min)")
    print("=" * 100)

    table_rows = []
    order_pref = [
        "Popularidad (baseline)",
        "Retrieval (two-tower) solo",
        "Ranker[uniforme] catálogo completo (1 etapa)",
        "Two-Stage[uniforme] N=50",
        "Two-Stage[uniforme] N=200",
        "Ranker[mixto] catálogo completo (1 etapa)",
        "Two-Stage[mixto] N=50",
        "Two-Stage[mixto] N=200",
    ]

    for model_name, grp in df_long.groupby("modelo"):
        r = {"modelo": model_name}
        for m in [
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "cobertura_catalogo",
            "popularidad_media_recs",
        ]:
            r[m] = f"{grp[m].mean():.4f} ± {grp[m].std():.4f}"
        table_rows.append(r)

    df_table = pd.DataFrame(table_rows)
    df_table["_order"] = df_table["modelo"].map(
        lambda x: order_pref.index(x) if x in order_pref else 99
    )
    df_table = df_table.sort_values("_order").drop(columns=["_order"])
    print(df_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("DIFERENCIAS PAREADAS POR SEMILLA (EVALUACIÓN DE CONTRASTES)")
    print("=" * 100)
    print(paired_df.to_string(index=False))

    if first_latency_report:
        print("\n" + "=" * 100)
        print("LATENCIA HONESTA POR USUARIO EN CPU (DESGLOSE PASO A PASO)")
        print("Etiquetado como propio de la CPU de este equipo local")
        print("=" * 100)
        lat_rows = []
        for arch, times in first_latency_report.items():
            lat_rows.append({
                "Arquitectura / Pipeline": arch,
                "Retrieval (ms)": f"{times['ms_retrieval']:.3f} ms",
                "Ranker (ms)": f"{times['ms_ranker']:.3f} ms",
                "Total Pipeline (ms)": f"{times['ms_total']:.3f} ms",
            })
        print(pd.DataFrame(lat_rows).to_string(index=False))

    print(f"\nCorrida multi-semilla completada en {total_time:.2f} segundos.")


if __name__ == "__main__":
    main()
