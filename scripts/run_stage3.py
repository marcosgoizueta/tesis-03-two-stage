#!/usr/bin/env python
"""Script de ejecución completa de la Etapa 3: Retrieval Two-Tower + Ranker."""

import argparse
import random
import sys
import time
from pathlib import Path

# En Windows, torch debe importarse primero para inicializar sus bibliotecas DLL
import torch

# Permitir importación directa desde src/
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
)
from recsys.ranker import train_ranker
from recsys.retrieval import (
    RetrievalDataset,
    TwoTower,
    compute_candidate_recalls,
    compute_retrieval_candidates,
    train_retrieval,
)
from recsys.tracking import save_comparativa_csv, save_run_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Etapa 3 — Arquitectura de dos etapas (Candidate Generation + Ranking)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria global para random, numpy y torch (default: 42)",
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
        help="Ruta base para los datos (default: data)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Ruta base para guardar resultados (default: results/gpu si cuda, results/cpu si cpu)",
    )
    parser.add_argument(
        "--epochs-retrieval",
        type=int,
        default=30,
        help="Número de épocas de entrenamiento para el retrieval (default: 30)",
    )
    parser.add_argument(
        "--epochs-ranker",
        type=int,
        default=8,
        help="Número de épocas de entrenamiento para el ranker (default: 8)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo de prueba rápida: 1 época y subconjunto acotado de usuarios",
    )
    parser.add_argument(
        "--fast-negative-sampling",
        action="store_true",
        help="Muestreo vectorizado de negativos para retrieval (altera consumo de RNG)",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicitó --device cuda pero CUDA no está disponible.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, device: torch.device) -> None:
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


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    device = resolve_device(args.device)

    # Separar almacenamiento por dispositivo para no mezclar ni sobrescribir
    if args.results_dir is not None:
        target_results_dir = args.results_dir
    else:
        target_results_dir = Path("results/gpu") if device.type == "cuda" else Path("results/cpu")

    cfg = Stage3Config(
        data_dir=args.data_dir,
        results_dir=target_results_dir,
        seed=args.seed,
        device=args.device,
        n_epochs_retrieval=args.epochs_retrieval,
        n_epochs_ranker=args.epochs_ranker,
        quick=args.quick,
        fast_negative_sampling=args.fast_negative_sampling,
    )

    print("=" * 60)
    print("ETAPA 3 — ARQUITECTURA DE DOS ETAPAS (TWO-STAGE RECSYS)")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Dispositivo activo: {device} ({dev_name})")
    if device.type == "cuda":
        print(f"CUDA Version: {torch.version.cuda} | PyTorch: {torch.__version__}")
    print(f"Directorio de resultados: {cfg.results_dir}")
    print(f"Semilla: {cfg.seed} | Modo Quick: {cfg.quick}")
    print(f"Fast Negative Sampling: {cfg.fast_negative_sampling}")
    print("=" * 60)

    # 1. Semillas globales
    set_seed(cfg.seed, device)

    # 2. Carga de datos y split
    data = load_and_prepare_data(cfg)
    print(f"Usuarios: {data.n_users}, Ítems: {data.n_items}")
    print(f"Train: {len(data.train_df)} | Test: {len(data.test_df)}")
    print(f"Interacciones positivas en train: {len(data.train_positive)}")

    # 3. Etapa A — Retrieval (Two-Tower)
    print("\n--- Etapa A: Retrieval (Two-Tower) ---")
    retrieval_dataset = RetrievalDataset(
        positive_df=data.train_positive,
        user_seen_train=data.user_seen_train,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
        fast_sampling=cfg.fast_negative_sampling,
    )

    retrieval_model = TwoTower(
        n_users=data.n_users,
        n_items=data.n_items,
        embed_dim=cfg.embed_dim_retrieval,
    ).to(device)

    p_pos = 1 / (1 + cfg.n_neg_per_pos_retrieval)
    trivial_bce = -(p_pos * np.log(p_pos) + (1 - p_pos) * np.log(1 - p_pos))
    print(f"BCE de referencia (modelo trivial): {trivial_bce:.4f}")

    retrieval_losses = train_retrieval(
        model=retrieval_model,
        dataset=retrieval_dataset,
        cfg=cfg,
        device=device,
    )

    epochs_retrieval_run = 1 if cfg.quick else cfg.n_epochs_retrieval
    plot_and_save_losses(
        retrieval_losses, trivial_bce, cfg.figures_dir, epochs_retrieval_run
    )

    # 4. Candidatos del retrieval
    print("\n--- Generación y Evaluación de Candidatos de Retrieval ---")
    retrieval_topk = compute_retrieval_candidates(
        model=retrieval_model,
        n_users=data.n_users,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        max_candidates=cfg.max_candidates,
        device=device,
    )

    eval_users = data.eval_users
    if cfg.quick:
        eval_users = eval_users[:30]
        print(f"Modo Quick: acotando usuarios evaluables a {len(eval_users)}")
    else:
        print(f"Usuarios evaluables: {len(eval_users)}")

    cand_recalls = compute_candidate_recalls(
        retrieval_topk=retrieval_topk,
        test_relevant_by_user=data.test_relevant_by_user,
        eval_users=eval_users,
        cutoffs=cfg.candidate_recall_cutoffs,
    )
    for n, r in cand_recalls.items():
        print(f"Recall de candidatos @ {n:>3}: {r:.4f}")

    # 5. Etapa B — Ranker
    print("\n--- Etapa B: Ranker ---")
    pos_users = data.train_positive["user_idx"].values
    pos_items = data.train_positive["item_idx"].values

    rankers = {}
    ranker_losses = {}
    variants = {"uniforme": (4, 0), "mixto": (2, 2)}

    for variant, (n_uni, n_hard) in variants.items():
        print(
            f"\n=== Ranker [{variant}]: {n_uni} negativos uniformes + {n_hard} del retrieval por positivo ==="
        )
        model, losses = train_ranker(
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
            seed=cfg.seed,
        )
        rankers[variant] = model
        ranker_losses[variant] = losses

    # 6. Evaluación de estrategias
    print("\n--- Evaluación comparativa ---")
    results_rows = [
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
        results_rows.append(
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
            results_rows.append(
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

    results_df = pd.DataFrame(results_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nResultados finales:")
    print(results_df.round(4).to_string(index=False))

    total_elapsed = time.perf_counter() - total_start

    # 7. Persistencia separada
    save_comparativa_csv(results_df, cfg, device)
    save_run_metadata(
        results_list=results_rows,
        candidate_recalls=cand_recalls,
        retrieval_losses=retrieval_losses,
        ranker_losses=ranker_losses,
        cfg=cfg,
        device=device,
        elapsed_total_seconds=total_elapsed,
    )

    print(f"\nPipeline finalizado con éxito en {total_elapsed:.2f} segundos.")


if __name__ == "__main__":
    main()
