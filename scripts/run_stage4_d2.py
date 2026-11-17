#!/usr/bin/env python
"""Ejecución aislada de la Variante D2 (Control Géneros con ID Dropout) para la Etapa 4.

No modifica los resultados existentes de las variantes A, B, C, C-cero, C2, D, E, R.
Escribe sus propios archivos de resultados con prefijo stage4_d2_*.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

# Permitir importación directa desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recsys.config import Stage3Config
from recsys.data import compute_feature_matrices, load_and_prepare_data
from recsys.evaluation import evaluate_model, make_rec_retrieval_only, make_two_stage
from recsys.ranker import train_ranker
from recsys.retrieval import RetrievalDataset
from recsys.stage4.cold_start import create_cold_item_split
from recsys.stage4.evaluation_slices import (
    assign_popularity_buckets,
    compute_popularity_bias,
    compute_slice_recalls,
)
from recsys.stage4.models import (
    TwoTowerStage4,
    compute_retrieval_candidates_stage4,
    train_retrieval_stage4,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Etapa 4 — Variante D2 (Control Géneros con ID Dropout)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
        help="Semilla(s) aleatoria(s) a ejecutar (default: 42 a 51)",
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
        help="Ruta base para resultados (default: results/gpu/stage4 si cuda)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo de prueba rápida: 1 época y 30 usuarios evaluables",
    )
    parser.add_argument(
        "--epochs-retrieval",
        type=int,
        default=30,
        help="Épocas de entrenamiento para retrieval (default: 30)",
    )
    parser.add_argument(
        "--epochs-ranker",
        type=int,
        default=8,
        help="Épocas de entrenamiento para ranker (default: 8)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["standard", "cold_start"],
        choices=["standard", "cold_start"],
        help="Escenarios a evaluar (default: standard cold_start)",
    )
    parser.add_argument(
        "--candidate-ns",
        type=int,
        nargs="+",
        default=[50, 200],
        help="Valores de N candidatos para Two-Stage (default: 50 200)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar corridas: omite semillas que ya tengan su archivo de resultados completo",
    )

    return parser.parse_args()


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def compute_random_candidate_baseline(
    eval_users: List[int],
    test_relevant_by_user: Dict[int, Set[int]],
    seen_mask_np: np.ndarray,
    n_items: int,
    item_buckets: Dict[int, str],
    cold_items_set: Set[int],
    cutoffs: Tuple[int, ...] = (10, 50, 200),
    n_candidates: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    slice_names = ["0", "1-5", "6-20", ">20"]
    metrics: Dict[str, float] = {}

    for k in cutoffs:
        for s in slice_names:
            users_with_rel = [
                u
                for u in eval_users
                if any(item_buckets.get(i) == s for i in test_relevant_by_user.get(u, set()))
            ]
            if not users_with_rel:
                metrics[f"random_analytical_recall_{s}_at_{k}"] = 0.0
                continue
            user_probs = [k / (n_items - seen_mask_np[u].sum()) for u in users_with_rel]
            metrics[f"random_analytical_recall_{s}_at_{k}"] = float(np.mean(user_probs))

        if cold_items_set:
            users_with_cold = [
                u
                for u in eval_users
                if any(i in cold_items_set for i in test_relevant_by_user.get(u, set()))
            ]
            if users_with_cold:
                user_probs_cold = [k / (n_items - seen_mask_np[u].sum()) for u in users_with_cold]
                metrics[f"random_analytical_recall_cold_at_{k}"] = float(np.mean(user_probs_cold))
            else:
                metrics[f"random_analytical_recall_cold_at_{k}"] = 0.0

    for s in slice_names:
        metrics[f"random_analytical_recall_{s}"] = metrics.get(f"random_analytical_recall_{s}_at_200", 0.0)
    if cold_items_set:
        metrics["random_analytical_recall_cold"] = metrics.get("random_analytical_recall_cold_at_200", 0.0)

    return metrics


def run_experiment_seed_d2(
    seed: int,
    scenario: str,
    cfg: Stage3Config,
    base_data,
    device: torch.device,
    candidate_ns: List[int],
    out_dir: Path,
) -> Tuple[List[Dict], Dict]:
    print(f"\n{'='*75}")
    print(f"ESCENARIO: {scenario.upper()} | SEMILLA: {seed} | DISPOSITIVO: {device} | VAR: D2")
    print(f"{'='*75}")

    n_users = base_data.n_users
    n_items = base_data.n_items

    cold_items_set: Set[int] = set()
    if scenario == "standard":
        train_df = base_data.train_df
        test_df = base_data.test_df
        user_seen_train = base_data.user_seen_train
        seen_mask_np = base_data.seen_mask_np
        train_positive = base_data.train_positive
        item_genre_np = base_data.item_genre_np
        user_genre_np = base_data.user_genre_np
        item_popularity = base_data.item_popularity
        item_logpop_np = base_data.item_logpop_np
        test_relevant_by_user = base_data.test_relevant_by_user
        eval_users = base_data.eval_users
    else:
        train_df, test_df, cold_items_set, _ = create_cold_item_split(
            ratings=base_data.ratings,
            n_items=n_items,
            cold_item_ratio=0.10,
            seed=42,
            rating_threshold=cfg.rating_threshold,
        )
        user_seen_train = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        train_positive = train_df[train_df["rating"] >= cfg.rating_threshold]

        seen_mask_np = np.zeros((n_users, n_items), dtype=bool)
        for u, items in user_seen_train.items():
            seen_mask_np[u, list(items)] = True

        (
            item_genre_np,
            user_genre_np,
            item_popularity,
            item_logpop_np,
            _,
        ) = compute_feature_matrices(
            base_data.movies, base_data.item2idx, train_positive, n_users, n_items
        )

        test_relevant_by_user = (
            test_df[test_df["rating"] >= cfg.rating_threshold]
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )
        eval_users = list(test_relevant_by_user.keys())

    if cfg.quick:
        eval_users = eval_users[:30]

    train_counts_series = train_df["item_idx"].value_counts()
    item_train_counts = np.zeros(n_items, dtype=np.int64)
    item_train_counts[train_counts_series.index.values] = train_counts_series.values
    item_buckets = assign_popularity_buckets(item_train_counts, n_items)
    is_cold_mask_np = (item_train_counts == 0)

    random_baseline_metrics = compute_random_candidate_baseline(
        eval_users=eval_users,
        test_relevant_by_user=test_relevant_by_user,
        seen_mask_np=seen_mask_np,
        n_items=n_items,
        item_buckets=item_buckets,
        cold_items_set=cold_items_set,
        cutoffs=(10, 50, 200),
        seed=seed,
    )

    # 1. Entrenar el Ranker Uniforme compartido (idéntico a run_stage4.py)
    print("\n--- Entrenando Ranker Uniforme compartido (1 vez por semilla) ---")
    ranker_uniform, ranker_losses = train_ranker(
        pos_users=train_positive["user_idx"].values,
        pos_items=train_positive["item_idx"].values,
        n_users=n_users,
        n_items=n_items,
        user_genre_np=user_genre_np,
        item_genre_np=item_genre_np,
        item_logpop_np=item_logpop_np,
        seen_mask_np=seen_mask_np,
        retrieval_topk=None,
        n_uniform=4,
        n_hard=0,
        cfg=cfg,
        device=device,
        seed=seed,
    )
    final_ranker_loss = ranker_losses[-1]
    print(f"Ranker final BCE loss: {final_ranker_loss:.4f}")

    # 2. Guardia de regresión opcional para A en semilla 42 estándar
    if seed == 42 and scenario == "standard" and not cfg.quick:
        set_seed(42, device)
        ds_A = RetrievalDataset(
            positive_df=train_positive,
            user_seen_train=user_seen_train,
            n_items=n_items,
            seen_mask_np=seen_mask_np,
            n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
            fast_sampling=False,
        )
        model_A = TwoTowerStage4(
            variant="A",
            n_users=n_users,
            n_items=n_items,
            embed_dim=cfg.embed_dim_retrieval,
            seed=42,
        ).to(device)
        train_retrieval_stage4(model_A, ds_A, cfg, device)
        topk_A = compute_retrieval_candidates_stage4(
            model=model_A,
            n_users=n_users,
            n_items=n_items,
            seen_mask_np=seen_mask_np,
            max_candidates=cfg.max_candidates,
            device=device,
        )
        eval_A = evaluate_model(
            name="Guardia A",
            rec_fn=make_rec_retrieval_only(topk_A, cfg.top_k),
            eval_users=eval_users,
            test_relevant_by_user=test_relevant_by_user,
            item_popularity=item_popularity,
            n_items=n_items,
            device=device,
            top_k=cfg.top_k,
        )
        rec_fn_ts50_A = make_two_stage(
            model=ranker_uniform,
            retrieval_topk=topk_A,
            n_candidates=50,
            device=device,
            top_k=cfg.top_k,
        )
        eval_ts50_A = evaluate_model(
            name="Guardia A TS50",
            rec_fn=rec_fn_ts50_A,
            eval_users=eval_users,
            test_relevant_by_user=test_relevant_by_user,
            item_popularity=item_popularity,
            n_items=n_items,
            device=device,
            top_k=cfg.top_k,
        )
        assert abs(eval_A["ndcg_at_10"] - 0.190736) < 1e-4, f"Fallo guardia A: {eval_A['ndcg_at_10']}"
        assert abs(eval_ts50_A["ndcg_at_10"] - 0.217769) < 1e-4, f"Fallo guardia A TS50: {eval_ts50_A['ndcg_at_10']}"
        print(f"[Guardia de Regresión A] CONFIRMADA: Retrieval NDCG@10 = {eval_A['ndcg_at_10']:.4f}, TS N=50 = {eval_ts50_A['ndcg_at_10']:.4f}")

    # 3. Entrenar Variante D2
    print(f"\n--- Variante D2 ({scenario}) ---")
    set_seed(seed, device)
    retrieval_dataset = RetrievalDataset(
        positive_df=train_positive,
        user_seen_train=user_seen_train,
        n_items=n_items,
        seen_mask_np=seen_mask_np,
        n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
        fast_sampling=False,
    )

    model_D2 = TwoTowerStage4(
        variant="D2",
        n_users=n_users,
        n_items=n_items,
        embed_dim=cfg.embed_dim_retrieval,
        item_genres_np=item_genre_np,
        is_cold_mask_np=is_cold_mask_np,
        seed=seed,
    ).to(device)

    retrieval_losses = train_retrieval_stage4(
        model=model_D2,
        dataset=retrieval_dataset,
        cfg=cfg,
        device=device,
    )

    retrieval_topk = compute_retrieval_candidates_stage4(
        model=model_D2,
        n_users=n_users,
        n_items=n_items,
        seen_mask_np=seen_mask_np,
        max_candidates=cfg.max_candidates,
        device=device,
    )

    params_info = model_D2.count_parameters()
    norms_info = model_D2.compute_component_norms()

    # Guardar vectores de D2 (npz, fuera de git)
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    vec_file = vectors_dir / f"{scenario}_seed{seed}_D2_vectors.npz"
    user_vecs = model_D2.user_embedding.weight.detach().cpu().numpy()
    item_vecs = model_D2.forward_all_items().detach().cpu().numpy()
    np.savez_compressed(vec_file, user_vectors=user_vecs, item_vectors=item_vecs)

    # 4. Evaluaciones
    slice_cand_recalls = compute_slice_recalls(
        recommendations_by_user={u: retrieval_topk[u] for u in eval_users},
        test_relevant_by_user=test_relevant_by_user,
        eval_users=eval_users,
        item_buckets=item_buckets,
        cutoffs=(10, 50, 100, 200, 500),
    )

    cand_recall_cold_10 = None
    cand_recall_cold_50 = None
    cand_recall_cold_200 = None
    if scenario == "cold_start" and len(cold_items_set) > 0:
        cold_rel_by_user = {
            u: (test_relevant_by_user.get(u, set()) & cold_items_set)
            for u in eval_users
        }
        users_with_cold_rel = [u for u in eval_users if len(cold_rel_by_user[u]) > 0]
        if len(users_with_cold_rel) > 0:
            cand_recall_cold_10 = float(np.mean([
                len(set(retrieval_topk[u, :10]) & cold_rel_by_user[u]) / len(cold_rel_by_user[u])
                for u in users_with_cold_rel
            ]))
            cand_recall_cold_50 = float(np.mean([
                len(set(retrieval_topk[u, :50]) & cold_rel_by_user[u]) / len(cold_rel_by_user[u])
                for u in users_with_cold_rel
            ]))
            cand_recall_cold_200 = float(np.mean([
                len(set(retrieval_topk[u, :200]) & cold_rel_by_user[u]) / len(cold_rel_by_user[u])
                for u in users_with_cold_rel
            ]))
        else:
            cand_recall_cold_10 = 0.0
            cand_recall_cold_50 = 0.0
            cand_recall_cold_200 = 0.0

    eval_retrieval = evaluate_model(
        name="Retrieval Solo (Var D2)",
        rec_fn=make_rec_retrieval_only(retrieval_topk, cfg.top_k),
        eval_users=eval_users,
        test_relevant_by_user=test_relevant_by_user,
        item_popularity=item_popularity,
        n_items=n_items,
        device=device,
        top_k=cfg.top_k,
    )

    bias_retrieval = compute_popularity_bias(
        recommendations_by_user={u: retrieval_topk[u] for u in eval_users},
        eval_users=eval_users,
        item_buckets=item_buckets,
        top_k=cfg.top_k,
    )

    two_stage_metrics: Dict[int, Dict] = {}
    for n_cand in candidate_ns:
        rec_fn_ts = make_two_stage(
            model=ranker_uniform,
            retrieval_topk=retrieval_topk,
            n_candidates=n_cand,
            device=device,
            top_k=cfg.top_k,
        )
        eval_ts = evaluate_model(
            name=f"Two-Stage N={n_cand} (Var D2)",
            rec_fn=rec_fn_ts,
            eval_users=eval_users,
            test_relevant_by_user=test_relevant_by_user,
            item_popularity=item_popularity,
            n_items=n_items,
            device=device,
            top_k=cfg.top_k,
        )
        two_stage_metrics[n_cand] = eval_ts

    bias_ts50 = compute_popularity_bias(
        recommendations_by_user={
            u: rec_fn_ts(u) for u in eval_users
        } if 50 in two_stage_metrics else {},
        eval_users=eval_users,
        item_buckets=item_buckets,
        top_k=cfg.top_k,
    )

    row = {
        "scenario": scenario,
        "seed": seed,
        "variant": "D2",
        "trainable_params": params_info["trainable_params"],
        "ranker_loss": float(final_ranker_loss),
        "retrieval_loss_final": float(retrieval_losses[-1]),
        # PRIMARIOS
        "ndcg_at_10_retrieval_solo": float(eval_retrieval["ndcg_at_10"]),
        "ndcg_at_10_two_stage_N50": float(two_stage_metrics[50]["ndcg_at_10"]) if 50 in two_stage_metrics else None,
        "cand_recall_cut0_at_200": float(slice_cand_recalls["0"]["recall_at_200"]),
        "cand_recall_cut1_5_at_200": float(slice_cand_recalls["1-5"]["recall_at_200"]),
        "cand_recall_cold_at_200": cand_recall_cold_200,
        # NORMAS DE COMPONENTES
        "mean_norm_id": norms_info.get("mean_norm_id"),
        "mean_norm_content": norms_info.get("mean_norm_content"),
        "ratio_id_to_content": norms_info.get("ratio_id_to_content"),
        # EXPLORATORIOS - Recall por corte en K=10 y K=50
        "cand_recall_cut0_at_10": float(slice_cand_recalls["0"]["recall_at_10"]),
        "cand_recall_cut0_at_50": float(slice_cand_recalls["0"]["recall_at_50"]),
        "cand_recall_cut1_5_at_10": float(slice_cand_recalls["1-5"]["recall_at_10"]),
        "cand_recall_cut1_5_at_50": float(slice_cand_recalls["1-5"]["recall_at_50"]),
        "cand_recall_cold_at_10": cand_recall_cold_10,
        "cand_recall_cold_at_50": cand_recall_cold_50,
        # BASELINE ALEATORIO ANALÍTICO
        "random_expected_recall_cut0_at_10": random_baseline_metrics.get("random_analytical_recall_0_at_10"),
        "random_expected_recall_cut0_at_50": random_baseline_metrics.get("random_analytical_recall_0_at_50"),
        "random_expected_recall_cut0_at_200": random_baseline_metrics.get("random_analytical_recall_0_at_200"),
        "random_expected_recall_cut1_5_at_10": random_baseline_metrics.get("random_analytical_recall_1-5_at_10"),
        "random_expected_recall_cut1_5_at_50": random_baseline_metrics.get("random_analytical_recall_1-5_at_50"),
        "random_expected_recall_cut1_5_at_200": random_baseline_metrics.get("random_analytical_recall_1-5_at_200"),
        "random_expected_recall_cold_at_10": random_baseline_metrics.get("random_analytical_recall_cold_at_10"),
        "random_expected_recall_cold_at_50": random_baseline_metrics.get("random_analytical_recall_cold_at_50"),
        "random_expected_recall_cold_at_200": random_baseline_metrics.get("random_analytical_recall_cold_at_200"),
        # EXPLORATORIOS - Retrieval Solo
        "p_at_10_retrieval_solo": float(eval_retrieval["precision_at_10"]),
        "r_at_10_retrieval_solo": float(eval_retrieval["recall_at_10"]),
        "catalog_coverage_retrieval_solo": float(eval_retrieval["cobertura_catalogo"]),
        "mean_pop_retrieval_solo": float(eval_retrieval["popularidad_media_recs"]),
        # EXPLORATORIOS - Two-Stage N=50
        "p_at_10_two_stage_N50": float(two_stage_metrics[50]["precision_at_10"]) if 50 in two_stage_metrics else None,
        "r_at_10_two_stage_N50": float(two_stage_metrics[50]["recall_at_10"]) if 50 in two_stage_metrics else None,
        "catalog_coverage_two_stage_N50": float(two_stage_metrics[50]["cobertura_catalogo"]) if 50 in two_stage_metrics else None,
        "mean_pop_two_stage_N50": float(two_stage_metrics[50]["popularidad_media_recs"]) if 50 in two_stage_metrics else None,
        # EXPLORATORIOS - Two-Stage N=200
        "ndcg_at_10_two_stage_N200": float(two_stage_metrics[200]["ndcg_at_10"]) if 200 in two_stage_metrics else None,
        "p_at_10_two_stage_N200": float(two_stage_metrics[200]["precision_at_10"]) if 200 in two_stage_metrics else None,
        "r_at_10_two_stage_N200": float(two_stage_metrics[200]["recall_at_10"]) if 200 in two_stage_metrics else None,
        # EXPLORATORIOS - Sesgo de catálogo
        "bias_tail_0_5_retrieval_solo": float(bias_retrieval["bias_fraction_tail_0_5"]),
        "bias_cut0_retrieval_solo": float(bias_retrieval["bias_fraction_0"]),
        "bias_cut1_5_retrieval_solo": float(bias_retrieval["bias_fraction_1-5"]),
        "bias_tail_0_5_two_stage_N50": float(bias_ts50.get("bias_fraction_tail_0_5", 0.0)),
        "bias_cut0_two_stage_N50": float(bias_ts50.get("bias_fraction_0", 0.0)),
        "bias_cut1_5_two_stage_N50": float(bias_ts50.get("bias_fraction_1-5", 0.0)),
    }

    print(
        f"Resultados D2: NDCG@10 (Retr)={row['ndcg_at_10_retrieval_solo']:.4f} | "
        f"NDCG@10 (TS N=50)={row['ndcg_at_10_two_stage_N50']:.4f} | "
        f"CandRec@200 [0]={row['cand_recall_cut0_at_200']:.4f} | "
        f"CandRec@200 [1-5]={row['cand_recall_cut1_5_at_200']:.4f}"
        + (f" | CandRec@200 [Cold]={cand_recall_cold_200:.4f}" if cand_recall_cold_200 is not None else "")
    )

    return [row], {
        "scenario": scenario,
        "seed": seed,
        "ranker_final_loss": float(final_ranker_loss),
    }


def main():
    total_start = time.perf_counter()
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.results_dir is not None:
        target_results_dir = args.results_dir
    elif args.quick:
        target_results_dir = Path("results/quick/stage4")
    else:
        target_results_dir = Path("results/gpu/stage4") if device.type == "cuda" else Path("results/cpu/stage4")

    target_results_dir.mkdir(parents=True, exist_ok=True)

    cfg = Stage3Config(
        data_dir=args.data_dir,
        results_dir=target_results_dir,
        device=args.device,
        n_epochs_retrieval=args.epochs_retrieval,
        n_epochs_ranker=args.epochs_ranker,
        quick=args.quick,
    )

    print("=" * 80)
    print("ETAPA 4 — VARIANTE D2 (GÉNEROS CON ID DROPOUT)")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Dispositivo activo: {device} ({dev_name})")
    print(f"Directorio de salida: {target_results_dir}")
    print(f"Semillas: {args.seeds} | Quick: {args.quick} | Escenarios: {args.scenarios}")
    print("=" * 80)

    base_data = load_and_prepare_data(cfg)

    all_rows = []
    for seed in args.seeds:
        seed_suffix = f"_seed{seed}"
        if args.quick:
            seed_suffix += "_quick"
        seed_csv_path = target_results_dir / f"stage4_d2_results{seed_suffix}.csv"

        expected_rows = len(args.scenarios) * 1  # Solo D2
        if args.resume and seed_csv_path.exists():
            try:
                df_prev = pd.read_csv(seed_csv_path)
                if len(df_prev) >= expected_rows:
                    print(f"\n[Resume] Semilla {seed} D2 ya completada previamente ({len(df_prev)} filas en {seed_csv_path}). Omitiendo.")
                    all_rows.extend(df_prev.to_dict(orient="records"))
                    continue
            except Exception as e:
                print(f"[Resume] Advertencia al leer {seed_csv_path}: {e}. Se reejecutará.")

        seed_rows = []
        for scenario in args.scenarios:
            rows, _ = run_experiment_seed_d2(
                seed=seed,
                scenario=scenario,
                cfg=cfg,
                base_data=base_data,
                device=device,
                candidate_ns=args.candidate_ns,
                out_dir=target_results_dir,
            )
            seed_rows.extend(rows)

        df_seed = pd.DataFrame(seed_rows)
        df_seed.to_csv(seed_csv_path, index=False)
        print(f"\n[Checkpoint] Resultados D2 de semilla {seed} guardados en: {seed_csv_path}")
        all_rows.extend(seed_rows)

    df_results = pd.DataFrame(all_rows)

    suffix = f"_{len(args.seeds)}seeds" if len(args.seeds) > 1 else f"_seed{args.seeds[0]}"
    if args.quick:
        suffix += "_quick"

    csv_path = target_results_dir / f"stage4_d2_results{suffix}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados D2 guardados en: {csv_path}")

    # Guardar resumen de D2
    summary_rows = []
    for (scenario, var), grp in df_results.groupby(["scenario", "variant"], sort=False):
        row = {"scenario": scenario, "variant": var, "n_seeds": len(grp)}
        numeric_cols = grp.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if c in {"seed", "trainable_params"}:
                continue
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_std"] = float(grp[c].std(ddof=1)) if len(grp) > 1 else 0.0
            row[f"{c}_min"] = float(grp[c].min())
            row[f"{c}_max"] = float(grp[c].max())
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = target_results_dir / f"stage4_d2_summary{suffix}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Resumen D2 guardado en: {summary_path}")
    print(f"Ejecución D2 finalizada en {time.perf_counter() - total_start:.2f} segundos.")


if __name__ == "__main__":
    main()
