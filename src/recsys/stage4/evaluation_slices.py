"""Evaluación segmentada por cortes de popularidad y cálculo de sesgo de recomendación."""

from typing import Dict, List, Set, Tuple, Union
import numpy as np


def assign_popularity_buckets(
    item_train_counts: Union[np.ndarray, Dict[int, int]],
    n_items: int,
) -> Dict[int, str]:
    """Asigna a cada ítem un corte de popularidad según sus interacciones en train:
    - '0': 0 interacciones
    - '1-5': 1 a 5 interacciones
    - '6-20': 6 a 20 interacciones
    - '>20': más de 20 interacciones
    """
    buckets: Dict[int, str] = {}
    for item_idx in range(n_items):
        cnt = item_train_counts[item_idx] if isinstance(item_train_counts, (np.ndarray, list)) else item_train_counts.get(item_idx, 0)
        if cnt == 0:
            buckets[item_idx] = "0"
        elif 1 <= cnt <= 5:
            buckets[item_idx] = "1-5"
        elif 6 <= cnt <= 20:
            buckets[item_idx] = "6-20"
        else:
            buckets[item_idx] = ">20"
    return buckets


def compute_slice_recalls(
    recommendations_by_user: Dict[int, Union[List[int], np.ndarray]],
    test_relevant_by_user: Dict[int, Set[int]],
    eval_users: List[int],
    item_buckets: Dict[int, str],
    cutoffs: Tuple[int, ...] = (10, 50, 200),
) -> Dict[str, Dict[str, float]]:
    """Calcula recall@k por corte de popularidad.
    REGLA CLAVE: Promedia SOLO sobre los usuarios que tienen al menos un ítem relevante en ese corte.
    Si un usuario no tiene relevantes en el corte, no se incluye en el denominador del promedio.
    """
    slice_names = ["0", "1-5", "6-20", ">20"]
    results: Dict[str, Dict[str, float]] = {s: {} for s in slice_names}
    
    # Pre-identificar relevantes por corte para cada usuario
    user_rel_by_slice: Dict[str, Dict[int, Set[int]]] = {s: {} for s in slice_names}
    for u in eval_users:
        user_rel = test_relevant_by_user.get(u, set())
        for item in user_rel:
            bucket = item_buckets.get(item)
            if bucket in user_rel_by_slice:
                if u not in user_rel_by_slice[bucket]:
                    user_rel_by_slice[bucket][u] = set()
                user_rel_by_slice[bucket][u].add(item)

    for s in slice_names:
        users_with_rel = user_rel_by_slice[s]
        n_eligible_users = len(users_with_rel)
        results[s]["n_eval_users"] = float(n_eligible_users)

        for k in cutoffs:
            metric_key = f"recall_at_{k}"
            if n_eligible_users == 0:
                results[s][metric_key] = 0.0
                continue

            user_recalls = []
            for u, rel_set in users_with_rel.items():
                recs = recommendations_by_user.get(u, [])
                top_k_recs = set(recs[:k])
                hits = len(top_k_recs & rel_set)
                user_recalls.append(hits / len(rel_set))

            results[s][metric_key] = float(np.mean(user_recalls))

    return results


def compute_popularity_bias(
    recommendations_by_user: Dict[int, Union[List[int], np.ndarray]],
    eval_users: List[int],
    item_buckets: Dict[int, str],
    top_k: int = 10,
) -> Dict[str, float]:
    """Calcula la fracción del total de recomendaciones en top-K que pertenecen a cada corte."""
    total_recs = 0
    slice_counts = {"0": 0, "1-5": 0, "6-20": 0, ">20": 0}

    for u in eval_users:
        recs = recommendations_by_user.get(u, [])[:top_k]
        for item in recs:
            total_recs += 1
            bucket = item_buckets.get(item)
            if bucket in slice_counts:
                slice_counts[bucket] += 1

    if total_recs == 0:
        return {f"bias_fraction_{k}": 0.0 for k in slice_counts}

    bias = {f"bias_fraction_{k}": slice_counts[k] / total_recs for k in slice_counts}
    bias["bias_fraction_tail_0_5"] = (slice_counts["0"] + slice_counts["1-5"]) / total_recs
    return bias
