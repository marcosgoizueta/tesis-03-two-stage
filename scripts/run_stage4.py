#!/usr/bin/env python
"""Script de ejecución y orquestación de la Etapa 4: Retrieval Semántico Multimodal."""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# En Windows, torch debe importarse primero
import torch
import numpy as np
import pandas as pd

# Permitir importación directa desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recsys.config import Stage3Config
from recsys.data import compute_feature_matrices, ensure_dataset, load_and_prepare_data
from recsys.evaluation import evaluate_model, make_rec_retrieval_only, make_two_stage
from recsys.ranker import train_ranker
from recsys.retrieval import RetrievalDataset, compute_candidate_recalls
from recsys.stage4.cold_start import create_cold_item_split
from recsys.stage4.embeddings import generate_or_load_item_embeddings
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
        description="Etapa 4 — Retrieval Semántico Multimodal (Variantes A, B, C, C2, D, E y Escenarios)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Semilla(s) aleatoria(s) a ejecutar (default: 42)",
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
        help="Ruta base para resultados (default: results/gpu/stage4 si cuda, results/cpu/stage4 si cpu; results/quick/stage4 si --quick)",
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
    """Calcula el recall esperado analítico y empírico para un baseline aleatorio de candidatos en cortes K."""
    slice_names = ["0", "1-5", "6-20", ">20"]
    metrics: Dict[str, float] = {}

    # 1. Esperado analítico: E[recall@K] = (1 / |U_s|) * sum_{u in U_s} (K / unseen_u)
    for k in cutoffs:
        for s in slice_names:
            users_with_rel = [u for u in eval_users if any(item_buckets.get(i) == s for i in test_relevant_by_user.get(u, set()))]
            if not users_with_rel:
                metrics[f"random_analytical_recall_{s}_at_{k}"] = 0.0
                continue
            user_probs = [k / (n_items - seen_mask_np[u].sum()) for u in users_with_rel]
            metrics[f"random_analytical_recall_{s}_at_{k}"] = float(np.mean(user_probs))

        if cold_items_set:
            users_with_cold = [u for u in eval_users if any(i in cold_items_set for i in test_relevant_by_user.get(u, set()))]
            if users_with_cold:
                user_probs_cold = [k / (n_items - seen_mask_np[u].sum()) for u in users_with_cold]
                metrics[f"random_analytical_recall_cold_at_{k}"] = float(np.mean(user_probs_cold))
            else:
                metrics[f"random_analytical_recall_cold_at_{k}"] = 0.0

    # Mantener claves sin sufijo como alias para K=200
    for s in slice_names:
        metrics[f"random_analytical_recall_{s}"] = metrics.get(f"random_analytical_recall_{s}_at_200", 0.0)
    if cold_items_set:
        metrics["random_analytical_recall_cold"] = metrics.get("random_analytical_recall_cold_at_200", 0.0)

    # 2. Empírico con semilla fija en K=200
    rng = np.random.RandomState(seed)
    random_recs = {}
    for u in eval_users:
        unseen = np.where(~seen_mask_np[u])[0]
        chosen = rng.choice(unseen, size=min(200, len(unseen)), replace=False)
        random_recs[u] = chosen

    emp_slice = compute_slice_recalls(
        recommendations_by_user=random_recs,
        test_relevant_by_user=test_relevant_by_user,
        eval_users=eval_users,
        item_buckets=item_buckets,
        cutoffs=(200,),
    )

    for s in slice_names:
        metrics[f"random_empirical_recall_{s}"] = emp_slice[s]["recall_at_200"]

    if cold_items_set:
        users_with_cold = [u for u in eval_users if any(i in cold_items_set for i in test_relevant_by_user.get(u, set()))]
        if users_with_cold:
            emp_cold = [
                len(set(random_recs[u]) & (test_relevant_by_user[u] & cold_items_set)) / len(test_relevant_by_user[u] & cold_items_set)
                for u in users_with_cold
            ]
            metrics["random_empirical_recall_cold"] = float(np.mean(emp_cold))
        else:
            metrics["random_empirical_recall_cold"] = 0.0

    return metrics


def run_experiment_seed(
    seed: int,
    scenario: str,
    cfg: Stage3Config,
    base_data,
    text_embs: np.ndarray,
    metadata_json: Dict,
    device: torch.device,
    candidate_ns: List[int],
    out_dir: Path,
) -> Tuple[List[Dict], Dict]:
    """Ejecuta un experimento para una semilla y escenario dados."""
    print(f"\n{'='*75}")
    print(f"ESCENARIO: {scenario.upper()} | SEMILLA: {seed} | DISPOSITIVO: {device}")
    print(f"{'='*75}")

    n_users = base_data.n_users
    n_items = base_data.n_items

    # 1. Preparar datos según escenario
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
    else:  # cold_start
        train_df, test_df, cold_items_set, _ = create_cold_item_split(
            ratings=base_data.ratings,
            n_items=n_items,
            cold_item_ratio=0.10,
            seed=42,  # Semilla fija 42 para el split cold-start según especificación
            rating_threshold=cfg.rating_threshold,
        )
        user_seen_train = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        train_positive = train_df[train_df["rating"] >= cfg.rating_threshold]

        seen_mask_np = np.zeros((n_users, n_items), dtype=bool)
        seen_mask_np[train_df["user_idx"].values, train_df["item_idx"].values] = True

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

    # Calcular cortes de popularidad basados en el train_df del escenario
    train_counts_series = train_df["item_idx"].value_counts()
    item_train_counts = np.zeros(n_items, dtype=np.int64)
    item_train_counts[train_counts_series.index.values] = train_counts_series.values
    item_buckets = assign_popularity_buckets(item_train_counts, n_items)

    # Máscara de ítems con 0 interacciones en train (para variante C2)
    is_cold_mask_np = (item_train_counts == 0)

    print(f"Usuarios evaluables: {len(eval_users)} | Positivos train: {len(train_positive)}")
    if scenario == "cold_start":
        print(f"Ítems fríos en test: {len(cold_items_set)} (0 en train)")

    # Baseline Aleatorio de candidatos
    random_baseline_metrics = compute_random_candidate_baseline(
        eval_users=eval_users,
        test_relevant_by_user=test_relevant_by_user,
        seen_mask_np=seen_mask_np,
        n_items=n_items,
        item_buckets=item_buckets,
        cold_items_set=cold_items_set,
        n_candidates=200,
        seed=seed,
    )
    print(f"[Baseline Aleatorio] CandRecall@200 esperado: Corte 0 = {random_baseline_metrics['random_analytical_recall_0']:.4f} | Corte 1-5 = {random_baseline_metrics['random_analytical_recall_1-5']:.4f}"
          + (f" | Cold = {random_baseline_metrics.get('random_analytical_recall_cold', 0.0):.4f}" if cold_items_set else ""))

    # 2. Entrenar las variantes de retrieval: A, B, C, C2, D, E, R
    # REGLA CLAVE: Para cada variante, reiniciamos la semilla para que RetrievalDataset y TwoTowerStage4
    # comiencen desde el estado exacto inicial de la semilla, garantizando reproducibilidad exacta con Etapa 3.
    trained_variants = ["A", "B", "C", "C2", "D", "E", "R"]
    retrieval_models = {}
    retrieval_topks = {}
    retrieval_losses_all = {}
    component_norms_all = {}

    for var in trained_variants:
        print(f"\n--- Variante {var} ({scenario}) ---")

        set_seed(seed, device)
        retrieval_dataset = RetrievalDataset(
            positive_df=train_positive,
            user_seen_train=user_seen_train,
            n_items=n_items,
            seen_mask_np=seen_mask_np,
            n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
            fast_sampling=False,
        )

        model = TwoTowerStage4(
            variant=var,
            n_users=n_users,
            n_items=n_items,
            embed_dim=cfg.embed_dim_retrieval,
            text_embeddings_np=text_embs,
            item_genres_np=item_genre_np,
            is_cold_mask_np=is_cold_mask_np,
            seed=seed,
        ).to(device)

        retrieval_losses = train_retrieval_stage4(
            model=model,
            dataset=retrieval_dataset,
            cfg=cfg,
            device=device,
        )
        retrieval_losses_all[var] = retrieval_losses

        topk = compute_retrieval_candidates_stage4(
            model=model,
            n_users=n_users,
            n_items=n_items,
            seen_mask_np=seen_mask_np,
            max_candidates=cfg.max_candidates,
            device=device,
        )
        retrieval_topks[var] = topk
        retrieval_models[var] = model
        component_norms_all[var] = model.compute_component_norms()

    # Control de inferencia para C: "C-cero" (sin reentrenar)
    # Evalúa los pesos ya entrenados de C anulando el embedding por ID para ítems sin interacciones en train
    print(f"\n--- Variante C-cero ({scenario}) [Control inferencia sobre C] ---")
    is_cold_mask_tensor = torch.tensor(is_cold_mask_np, dtype=torch.bool, device=device)
    c_zero_item_embs = retrieval_models["C"].forward_all_items(
        zero_cold_id=True, is_cold_mask=is_cold_mask_tensor
    )
    topk_c_zero = compute_retrieval_candidates_stage4(
        model=retrieval_models["C"],
        n_users=n_users,
        n_items=n_items,
        seen_mask_np=seen_mask_np,
        max_candidates=cfg.max_candidates,
        device=device,
        override_item_embs=c_zero_item_embs,
    )
    retrieval_topks["C-cero"] = topk_c_zero
    retrieval_models["C-cero"] = retrieval_models["C"]
    component_norms_all["C-cero"] = component_norms_all["C"]
    retrieval_losses_all["C-cero"] = retrieval_losses_all["C"]

    # 3. Entrenar el Ranker Uniforme compartido (1 vez por semilla y escenario)
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
        retrieval_topk=retrieval_topks["A"],
        n_uniform=4,
        n_hard=0,
        cfg=cfg,
        device=device,
        seed=seed,
    )
    final_ranker_loss = ranker_losses[-1]
    print(f"Ranker final BCE loss: {final_ranker_loss:.4f}")

    # 4. Evaluación de métricas primarias y exploratorias para las 7 variantes + C-cero
    eval_variants = ["A", "B", "C", "C-cero", "C2", "D", "E", "R"]
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for var in eval_variants:
        model = retrieval_models[var]
        retrieval_topk = retrieval_topks[var]
        params_info = model.count_parameters()
        norms_info = component_norms_all[var]

        # Guardar vectores finales (npz, fuera de git)
        vec_file = vectors_dir / f"{scenario}_seed{seed}_{var}_vectors.npz"
        user_vecs = model.user_embedding.weight.detach().cpu().numpy()
        if var == "C-cero":
            item_vecs = c_zero_item_embs.detach().cpu().numpy()
        else:
            item_vecs = model.forward_all_items().detach().cpu().numpy()
        np.savez_compressed(vec_file, user_vectors=user_vecs, item_vectors=item_vecs)

        # a) Recall de candidatos por corte de popularidad (y sobre ítems fríos si aplica)
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

        # b) Evaluación de Retrieval Solo
        eval_retrieval = evaluate_model(
            name=f"Retrieval Solo (Var {var})",
            rec_fn=make_rec_retrieval_only(retrieval_topk, cfg.top_k),
            eval_users=eval_users,
            test_relevant_by_user=test_relevant_by_user,
            item_popularity=item_popularity,
            n_items=n_items,
            device=device,
            top_k=cfg.top_k,
        )

        # Recall@10 de relevantes por corte (Retrieval Solo)
        slice_rec_retrieval = compute_slice_recalls(
            recommendations_by_user={u: retrieval_topk[u, :cfg.top_k] for u in eval_users},
            test_relevant_by_user=test_relevant_by_user,
            eval_users=eval_users,
            item_buckets=item_buckets,
            cutoffs=(cfg.top_k,),
        )

        # Sesgo de popularidad (Retrieval Solo)
        bias_retrieval = compute_popularity_bias(
            recommendations_by_user={u: retrieval_topk[u] for u in eval_users},
            eval_users=eval_users,
            item_buckets=item_buckets,
            top_k=cfg.top_k,
        )

        # c) Evaluación de Two-Stage con N candidatos
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
                name=f"Two-Stage N={n_cand} (Var {var})",
                rec_fn=rec_fn_ts,
                eval_users=eval_users,
                test_relevant_by_user=test_relevant_by_user,
                item_popularity=item_popularity,
                n_items=n_items,
                device=device,
                top_k=cfg.top_k,
            )
            two_stage_metrics[n_cand] = eval_ts

        # GUARD AUTOMÁTICO: Verificar que Variante A en escenario estándar reproduce Etapa 3
        if seed == 42 and scenario == "standard" and var == "A" and not cfg.quick:
            ndcg_ret = eval_retrieval["ndcg_at_10"]
            ndcg_ts50 = two_stage_metrics[50]["ndcg_at_10"]
            assert abs(ndcg_ret - 0.190736) < 1e-4, f"BLOQUEANTE: Variante A retrieval solo ({ndcg_ret:.6f}) no reprodujo Etapa 3 (0.190736)"
            assert abs(ndcg_ts50 - 0.217769) < 1e-4, f"BLOQUEANTE: Variante A Two-Stage N=50 ({ndcg_ts50:.6f}) no reprodujo Etapa 3 (0.217769)"
            print(f"[Guardia de Regresión] Variante A superó la verificación exacta (Retrieval: {ndcg_ret:.4f}, TS N=50: {ndcg_ts50:.4f})")

        row = {
            "scenario": scenario,
            "seed": seed,
            "variant": var,
            "trainable_params": params_info["trainable_params"],
            "ranker_loss": float(final_ranker_loss),
            "retrieval_loss_final": float(retrieval_losses_all[var][-1]),
            # PRIMARIOS
            "ndcg_at_10_retrieval_solo": float(eval_retrieval["ndcg_at_10"]),
            "ndcg_at_10_two_stage_N50": float(two_stage_metrics[50]["ndcg_at_10"]) if 50 in two_stage_metrics else None,
            "cand_recall_cut0_at_200": float(slice_cand_recalls["0"]["recall_at_200"]),
            "cand_recall_cut1_5_at_200": float(slice_cand_recalls["1-5"]["recall_at_200"]),
            "cand_recall_cold_at_200": cand_recall_cold_200,
            # NORMAS DE COMPONENTES (Para análisis de dominancia en C y C2)
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
            # BASELINE ALEATORIO ANALÍTICO (K=10, 50, 200)
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
            # EXPLORATORIOS - Two-Stage N=200
            "ndcg_at_10_two_stage_N200": float(two_stage_metrics[200]["ndcg_at_10"]) if 200 in two_stage_metrics else None,
            "p_at_10_two_stage_N200": float(two_stage_metrics[200]["precision_at_10"]) if 200 in two_stage_metrics else None,
            "r_at_10_two_stage_N200": float(two_stage_metrics[200]["recall_at_10"]) if 200 in two_stage_metrics else None,
            # EXPLORATORIOS - Candidatos otros cortes
            "cand_recall_cut6_20_at_200": float(slice_cand_recalls["6-20"]["recall_at_200"]),
            "cand_recall_cut_gt20_at_200": float(slice_cand_recalls[">20"]["recall_at_200"]),
            "cand_recall_cut0_at_50": float(slice_cand_recalls["0"]["recall_at_50"]),
            "cand_recall_cut1_5_at_50": float(slice_cand_recalls["1-5"]["recall_at_50"]),
            # EXPLORATORIOS - Sesgo de catálogo
            "bias_tail_0_5_retrieval_solo": float(bias_retrieval["bias_fraction_tail_0_5"]),
            "bias_cut0_retrieval_solo": float(bias_retrieval["bias_fraction_0"]),
            "bias_cut1_5_retrieval_solo": float(bias_retrieval["bias_fraction_1-5"]),
        }
        rows.append(row)

        print(
            f"Resultados Primarios Var {var}: NDCG@10 (Retr)={row['ndcg_at_10_retrieval_solo']:.4f} | "
            f"NDCG@10 (TS N=50)={row['ndcg_at_10_two_stage_N50']:.4f} | "
            f"CandRec@200 [0]={row['cand_recall_cut0_at_200']:.4f} | "
            f"CandRec@200 [1-5]={row['cand_recall_cut1_5_at_200']:.4f}"
            + (f" | CandRec@200 [Cold]={cand_recall_cold_200:.4f}" if cand_recall_cold_200 is not None else "")
        )

    return rows, {
        "scenario": scenario,
        "seed": seed,
        "ranker_final_loss": float(final_ranker_loss),
        "ranker_losses": [float(x) for x in ranker_losses],
        "random_baseline": random_baseline_metrics,
    }


def compute_summary_and_contrasts(df_results: pd.DataFrame, out_dir: Path, suffix: str):
    """Calcula el resumen agregado por variante/escenario y los contrastes pareados primarios."""
    metric_cols = [
        "ndcg_at_10_retrieval_solo",
        "ndcg_at_10_two_stage_N50",
        "cand_recall_cut0_at_200",
        "cand_recall_cut1_5_at_200",
        "cand_recall_cold_at_200",
        "cand_recall_cut0_at_10",
        "cand_recall_cut0_at_50",
        "cand_recall_cut1_5_at_10",
        "cand_recall_cut1_5_at_50",
        "cand_recall_cold_at_10",
        "cand_recall_cold_at_50",
        "p_at_10_retrieval_solo",
        "r_at_10_retrieval_solo",
        "catalog_coverage_retrieval_solo",
        "mean_pop_retrieval_solo",
        "p_at_10_two_stage_N50",
        "r_at_10_two_stage_N50",
        "catalog_coverage_two_stage_N50",
        "ndcg_at_10_two_stage_N200",
        "p_at_10_two_stage_N200",
        "r_at_10_two_stage_N200",
        "cand_recall_cut6_20_at_200",
        "cand_recall_cut_gt20_at_200",
        "bias_tail_0_5_retrieval_solo",
        "bias_cut0_retrieval_solo",
        "bias_cut1_5_retrieval_solo",
        "mean_norm_id",
        "mean_norm_content",
        "ratio_id_to_content",
    ]
    valid_cols = [c for c in metric_cols if c in df_results.columns]

    # 1. Resumen: mean, std, min, max
    summary_rows = []
    for (scenario, var), grp in df_results.groupby(["scenario", "variant"], sort=False):
        row = {
            "scenario": scenario,
            "variant": var,
            "n_seeds": len(grp),
        }
        for col in valid_cols:
            vals = grp[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{col}_min"] = float(vals.min())
                row[f"{col}_max"] = float(vals.max())
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
                row[f"{col}_min"] = None
                row[f"{col}_max"] = None
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"stage4_summary{suffix}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nResumen por variante guardado en: {summary_path}")

    # 2. Contrastes pareados PRIMARIOS
    # - Escenario estándar:
    #   - NDCG@10 retrieval solo: B - A, C2 - A, C - D, B - R
    #   - NDCG@10 Two-Stage N=50: B - A, C2 - A
    # - Escenario en frío:
    #   - CandRecall@200 sobre ítems fríos: B - R, B - E, C2 - C-cero, C2 - B
    contrast_specs = [
        ("standard", "ndcg_at_10_retrieval_solo", "B", "A"),
        ("standard", "ndcg_at_10_retrieval_solo", "C2", "A"),
        ("standard", "ndcg_at_10_retrieval_solo", "C", "D"),
        ("standard", "ndcg_at_10_retrieval_solo", "B", "R"),
        ("standard", "ndcg_at_10_two_stage_N50", "B", "A"),
        ("standard", "ndcg_at_10_two_stage_N50", "C2", "A"),
        ("cold_start", "cand_recall_cold_at_200", "B", "R"),
        ("cold_start", "cand_recall_cold_at_200", "B", "E"),
        ("cold_start", "cand_recall_cold_at_200", "C2", "C-cero"),
        ("cold_start", "cand_recall_cold_at_200", "C2", "B"),
    ]

    contrast_rows = []
    print("\n" + "=" * 90)
    print("CONTRASTES PAREADOS PRIMARIOS")
    print("=" * 90)
    print(f"{'Escenario':<12} | {'Métrica':<27} | {'Contraste':<14} | {'Media':>8} | {'Desvío':>8} | {'Mín':>8} | {'Máx':>8} | {'Positivos':>9}")
    print("-" * 90)

    for scenario, metric, v1, v2 in contrast_specs:
        sub = df_results[df_results["scenario"] == scenario]
        piv = sub.pivot(index="seed", columns="variant", values=metric)
        if v1 in piv.columns and v2 in piv.columns:
            diffs = piv[v1] - piv[v2]
            mean_d = float(diffs.mean())
            std_d = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
            min_d = float(diffs.min())
            max_d = float(diffs.max())
            n_pos = int((diffs > 0).sum())
            n_tot = len(diffs)

            contrast_label = f"{v1} - {v2}"
            contrast_rows.append({
                "scenario": scenario,
                "metric": metric,
                "contrast": contrast_label,
                "v1": v1,
                "v2": v2,
                "mean_diff": mean_d,
                "std_diff": std_d,
                "min_diff": min_d,
                "max_diff": max_d,
                "n_positive": n_pos,
                "n_total": n_tot,
                "pct_positive": (n_pos / n_tot) * 100.0 if n_tot > 0 else 0.0,
            })
            print(f"{scenario:<12} | {metric:<27} | {contrast_label:<14} | {mean_d:>+8.4f} | {std_d:>8.4f} | {min_d:>+8.4f} | {max_d:>+8.4f} | {n_pos:>3}/{n_tot}")

    df_contrasts = pd.DataFrame(contrast_rows)
    contrasts_path = out_dir / f"stage4_paired_contrasts{suffix}.csv"
    df_contrasts.to_csv(contrasts_path, index=False)
    print(f"\nContrastes pareados guardados en: {contrasts_path}")


def compute_e_diagnostics(
    base_data,
    eval_users: List[int],
    seen_mask_np: np.ndarray,
    vectors_dir: Path,
    seed: int = 42,
) -> Dict:
    """Diagnóstico sobre la Variante E:
    - Cuántos vectores de género distintos hay entre los ítems
    - Tamaño medio de los bloques de ítems con vector idéntico
    - Fracción de los 200 candidatos que queda decidida por desempates exactos
    """
    item_genres = base_data.item_genre_np
    unique_rows, inverse_indices, counts = np.unique(
        item_genres, axis=0, return_inverse=True, return_counts=True
    )
    n_unique_genres = len(unique_rows)
    mean_block_size = float(np.mean(counts))
    max_block_size = int(np.max(counts))

    # Cargar vectores guardados de E si están disponibles para medir empates en la frontera del top-200
    vec_path = vectors_dir / f"standard_seed{seed}_E_vectors.npz"
    tie_fractions = []
    if vec_path.exists():
        vecs = np.load(vec_path)
        user_vecs = vecs["user_vectors"]
        item_vecs = vecs["item_vectors"]

        for u in eval_users:
            u_v = user_vecs[u]
            scores = item_vecs @ u_v
            unseen = np.where(~seen_mask_np[u])[0]
            unseen_scores = scores[unseen]

            sorted_idx = np.argsort(-unseen_scores)
            if len(sorted_idx) >= 200:
                score_200 = unseen_scores[sorted_idx[199]]
                tied_mask = np.abs(unseen_scores - score_200) < 1e-5
                n_tied = np.sum(tied_mask)
                n_tied_in_top200 = np.sum(np.abs(unseen_scores[sorted_idx[:200]] - score_200) < 1e-5)
                if n_tied > 1:
                    tie_fractions.append(n_tied_in_top200 / 200.0)
                else:
                    tie_fractions.append(0.0)

    mean_tie_fraction_in_top200 = float(np.mean(tie_fractions)) if tie_fractions else 0.0

    diag = {
        "n_items_total": int(len(item_genres)),
        "n_unique_genre_vectors": int(n_unique_genres),
        "mean_block_size": mean_block_size,
        "max_block_size": max_block_size,
        "mean_tie_fraction_in_top200": mean_tie_fraction_in_top200,
    }
    return diag


def main():
    total_start = time.perf_counter()
    args = parse_args()

    # Resolver dispositivo
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolver directorio de salida
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
    print("ETAPA 4 — RETRIEVAL SEMÁNTICO MULTIMODAL")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Dispositivo activo: {device} ({dev_name})")
    print(f"Directorio de salida: {target_results_dir}")
    print(f"Semillas: {args.seeds} | Quick: {args.quick} | Escenarios: {args.scenarios}")
    print("=" * 80)

    # Cargar datos base
    base_data = load_and_prepare_data(cfg)

    # Cargar embeddings de texto (Fase 1)
    text_embs, metadata_json = generate_or_load_item_embeddings(
        movies_df=base_data.movies,
        item2idx=base_data.item2idx,
        cache_dir=args.data_dir / "cache",
        device="cuda" if device.type == "cuda" else "cpu",
    )

    all_rows = []
    run_diagnostics = []

    for seed in args.seeds:
        seed_suffix = f"_seed{seed}"
        if args.quick:
            seed_suffix += "_quick"
        seed_csv_path = target_results_dir / f"stage4_results{seed_suffix}.csv"

        # 8 variantes evaluadas ("A", "B", "C", "C-cero", "C2", "D", "E", "R") x escenarios
        expected_rows = len(args.scenarios) * 8

        if args.resume and seed_csv_path.exists():
            try:
                df_prev = pd.read_csv(seed_csv_path)
                if len(df_prev) >= expected_rows:
                    print(f"\n[Resume] Semilla {seed} ya completada previamente ({len(df_prev)} filas en {seed_csv_path}). Omitiendo.")
                    all_rows.extend(df_prev.to_dict(orient="records"))
                    continue
            except Exception as e:
                print(f"[Resume] Advertencia al leer {seed_csv_path}: {e}. Se reejecutará la semilla {seed}.")

        seed_rows = []
        for scenario in args.scenarios:
            rows, diag = run_experiment_seed(
                seed=seed,
                scenario=scenario,
                cfg=cfg,
                base_data=base_data,
                text_embs=text_embs,
                metadata_json=metadata_json,
                device=device,
                candidate_ns=args.candidate_ns,
                out_dir=target_results_dir,
            )
            seed_rows.extend(rows)
            run_diagnostics.append(diag)

        # Checkpoint: guardar CSV de la semilla inmediatamente al terminar
        df_seed = pd.DataFrame(seed_rows)
        df_seed.to_csv(seed_csv_path, index=False)
        print(f"\n[Checkpoint] Resultados de semilla {seed} guardados en: {seed_csv_path}")
        all_rows.extend(seed_rows)

    df_results = pd.DataFrame(all_rows)

    # Guardar CSV de corridas
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{len(args.seeds)}seeds" if len(args.seeds) > 1 else f"_seed{args.seeds[0]}"
    if args.quick:
        suffix += "_quick"

    csv_path = target_results_dir / f"stage4_results{suffix}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")

    # 1. Calcular y guardar Resumen Agregado y Contrastes Pareados Primarios
    compute_summary_and_contrasts(df_results, target_results_dir, suffix)

    # 2. Diagnóstico de la Variante E (empates y colapso de géneros)
    diag_e = compute_e_diagnostics(
        base_data=base_data,
        eval_users=base_data.eval_users,
        seen_mask_np=base_data.seen_mask_np,
        vectors_dir=target_results_dir / "vectors",
        seed=args.seeds[0],
    )
    diag_e_path = target_results_dir / "diagnostic_variant_e.json"
    with open(diag_e_path, "w", encoding="utf-8") as f:
        json.dump(diag_e, f, indent=2, ensure_ascii=False)
    print(f"\nDiagnóstico Variante E guardado en: {diag_e_path}")
    print(f"[Diagnóstico E] Vectores de género únicos: {diag_e['n_unique_genre_vectors']} / {diag_e['n_items_total']}")
    print(f"[Diagnóstico E] Tamaño medio de bloque de ítems idénticos: {diag_e['mean_block_size']:.2f} (máx: {diag_e['max_block_size']})")
    print(f"[Diagnóstico E] Fracción media de candidatos top-200 decidida por desempates: {diag_e['mean_tie_fraction_in_top200']:.4f}")

    # Guardar metadatos JSON
    metadata_out = {
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "quick": args.quick,
        "device": {
            "type": device.type,
            "name": dev_name,
            "torch": torch.__version__,
        },
        "text_model": metadata_json,
        "elapsed_total_seconds": time.perf_counter() - total_start,
        "diagnostics": run_diagnostics,
        "diagnostic_variant_e": diag_e,
    }
    json_path = target_results_dir / f"run_{timestamp_str}{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2, ensure_ascii=False)
    print(f"Metadatos guardados en: {json_path}")

    print(f"\nEjecución finalizada con éxito en {time.perf_counter() - total_start:.2f} segundos.")


if __name__ == "__main__":
    main()
