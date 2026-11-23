"""Módulo de particionado temporal por usuario para la Etapa 5.
Implementa el split determinista 80/10/20 con desempate aleatorio reproducible y
regla de bordes Opción C (borde de evento más cercano) para garantizar separación estricta
por timestamp sin fugas entre train, val y test.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from recsys.stage4.evaluation_slices import assign_popularity_buckets


@dataclass
class TemporalDatasetBundle:
    """Contenedor de datos procesados bajo el protocolo temporal por usuario."""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    train_full_df: pd.DataFrame
    train_core_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    idx2item: Dict[int, int]
    n_users: int
    n_items: int

    user_train_seq: Dict[int, List[int]]
    user_val_seq: Dict[int, List[int]]
    user_test_seq: Dict[int, List[int]]

    user_seen_train: Dict[int, Set[int]]
    seen_mask_np: np.ndarray

    test_relevant_by_user: Dict[int, Set[int]]
    val_relevant_by_user: Dict[int, Set[int]]
    eval_users_test: List[int]
    eval_users_val: List[int]

    item_popularity: np.ndarray
    item_buckets: Dict[int, str]
    train_item_indices: Set[int]
    cold_item_indices: Set[int]

    users_cut_inside_tie: List[int]
    users_marked_strict_test: List[int]
    users_marked_strict_val: List[int]


def _apply_boundary_rule_c(
    target_k: int,
    n_total: int,
    timestamps: np.ndarray,
) -> Tuple[int, bool]:
    """Aplica la regla Opción C: ajusta el corte al borde de evento más cercano.
    
    Returns:
        (k_adjusted, was_marked_strict)
    """
    if target_k <= 0 or target_k >= n_total:
        return target_k, False

    # Verificar si el corte cae dentro de un bloque con timestamp idéntico
    if timestamps[target_k - 1] == timestamps[target_k]:
        target_ts = timestamps[target_k]
        tie_positions = np.where(timestamps == target_ts)[0]
        start_idx = int(tie_positions[0])
        end_idx = int(tie_positions[-1]) # inclusivo

        dist_start = target_k - start_idx
        dist_end = (end_idx + 1) - target_k

        candidate = start_idx if dist_start <= dist_end else (end_idx + 1)

        # Si el borde más cercano deja al usuario sin test/val, usar el otro borde
        if candidate == n_total:
            candidate = start_idx
        elif candidate == 0:
            candidate = end_idx + 1

        # Si aun así no hay partición posterior o anterior (todos los timestamps son idénticos)
        if candidate == n_total or candidate == 0:
            return target_k, True
        else:
            return candidate, False

    return target_k, False


def create_temporal_split(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio_of_train: float = 0.10,
    rating_threshold: float = 3.5,
    seed: int = 42,
    round_mode: str = "round",
    use_event_boundary_rule: bool = True,
) -> TemporalDatasetBundle:
    """Genera la partición temporal por usuario.
    
    Regla de ordenamiento y desempate:
    - Ordena por (userId ASC, timestamp ASC, rand_tie ASC).
    - rand_tie es una permutación determinista generada con la semilla especificada,
      evitando el sesgo de movieId ascendente presente en ratings.csv.
    - Si use_event_boundary_rule=True, aplica la Opción C para desplazar el corte
      al límite del evento más cercano garantizando separación estricta por timestamp.
    """
    ratings = ratings.copy()
    user_ids = ratings["userId"].unique()
    item_ids = ratings["movieId"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}
    idx2item = {i: m for m, i in item2idx.items()}

    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["item_idx"] = ratings["movieId"].map(item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)

    # Desempate determinista con permutación aleatoria de semilla fija
    rng = np.random.RandomState(seed)
    ratings["rand_tie"] = rng.permutation(len(ratings))

    sorted_ratings = ratings.sort_values(
        by=["user_idx", "timestamp", "rand_tie"]
    ).reset_index(drop=True)

    round_fn = np.round if round_mode == "round" else np.floor

    train_core_indices = []
    val_indices = []
    train_full_indices = []
    test_indices = []

    user_train_seq: Dict[int, List[int]] = {}
    user_val_seq: Dict[int, List[int]] = {}
    user_test_seq: Dict[int, List[int]] = {}
    users_cut_inside_tie: List[int] = []
    users_marked_strict_test: List[int] = []
    users_marked_strict_val: List[int] = []

    for u_idx, u_df in sorted_ratings.groupby("user_idx", sort=False):
        n_u = len(u_df)
        u_timestamps = u_df["timestamp"].values
        u_row_indices = u_df.index.values
        u_item_indices = u_df["item_idx"].values.tolist()

        k_target = int(round_fn(train_ratio * n_u))
        if k_target == n_u and n_u >= 2:
            k_target = n_u - 1

        # Verificar si el corte caería en un empate
        if 0 < k_target < n_u and u_timestamps[k_target - 1] == u_timestamps[k_target]:
            users_cut_inside_tie.append(int(u_idx))

        if use_event_boundary_rule:
            k_full, marked_strict_t = _apply_boundary_rule_c(k_target, n_u, u_timestamps)
            if marked_strict_t:
                users_marked_strict_test.append(int(u_idx))
        else:
            k_full = k_target

        # 2. Corte Train Core / Val dentro de k_full
        k_val_target = int(round_fn(val_ratio_of_train * k_full))
        k_core_target = k_full - k_val_target
        if k_core_target == k_full and k_full >= 2:
            k_core_target = k_full - 1

        if use_event_boundary_rule:
            k_core, marked_strict_v = _apply_boundary_rule_c(
                k_core_target, k_full, u_timestamps[:k_full]
            )
            if marked_strict_v:
                users_marked_strict_val.append(int(u_idx))
        else:
            k_core = k_core_target

        train_core_indices.extend(u_row_indices[:k_core])
        val_indices.extend(u_row_indices[k_core:k_full])
        train_full_indices.extend(u_row_indices[:k_full])
        test_indices.extend(u_row_indices[k_full:])

        user_train_seq[u_idx] = u_item_indices[:k_full]
        user_val_seq[u_idx] = u_item_indices[k_core:k_full]
        user_test_seq[u_idx] = u_item_indices[k_full:]

    train_full_df = sorted_ratings.iloc[train_full_indices].copy()
    train_core_df = sorted_ratings.iloc[train_core_indices].copy()
    val_df = sorted_ratings.iloc[val_indices].copy()
    test_df = sorted_ratings.iloc[test_indices].copy()

    user_seen_train = train_full_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    seen_mask_np = np.zeros((n_users, n_items), dtype=bool)
    seen_mask_np[train_full_df["user_idx"].values, train_full_df["item_idx"].values] = True

    # Relevantes (rating >= threshold)
    test_relevant_by_user = (
        test_df[test_df["rating"] >= rating_threshold]
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )
    val_relevant_by_user = (
        val_df[val_df["rating"] >= rating_threshold]
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    eval_users_test = sorted(list(test_relevant_by_user.keys()))
    eval_users_val = sorted(list(val_relevant_by_user.keys()))

    # Popularidad en Train Full
    item_popularity = np.zeros(n_items, dtype=np.float64)
    train_counts = train_full_df["item_idx"].value_counts()
    item_popularity[train_counts.index.values] = train_counts.values

    item_buckets = assign_popularity_buckets(item_popularity, n_items)

    train_item_indices = set(train_counts.index.values)
    cold_item_indices = set(range(n_items)) - train_item_indices

    return TemporalDatasetBundle(
        ratings=sorted_ratings,
        movies=movies,
        train_full_df=train_full_df,
        train_core_df=train_core_df,
        val_df=val_df,
        test_df=test_df,
        user2idx=user2idx,
        item2idx=item2idx,
        idx2item=idx2item,
        n_users=n_users,
        n_items=n_items,
        user_train_seq=user_train_seq,
        user_val_seq=user_val_seq,
        user_test_seq=user_test_seq,
        user_seen_train=user_seen_train,
        seen_mask_np=seen_mask_np,
        test_relevant_by_user=test_relevant_by_user,
        val_relevant_by_user=val_relevant_by_user,
        eval_users_test=eval_users_test,
        eval_users_val=eval_users_val,
        item_popularity=item_popularity,
        item_buckets=item_buckets,
        train_item_indices=train_item_indices,
        cold_item_indices=cold_item_indices,
        users_cut_inside_tie=users_cut_inside_tie,
        users_marked_strict_test=users_marked_strict_test,
        users_marked_strict_val=users_marked_strict_val,
    )


def generate_sliding_windows(
    sequence: List[int],
    max_len: int = 50,
    max_windows_per_user: Optional[int] = None,
    stride: int = 1,
) -> List[Tuple[List[int], int]]:
    """Genera pares (contexto, target) para entrenamiento secuencial autorregresivo.
    
    Args:
        sequence: Lista cronológica de índices de ítems.
        max_len: Longitud máxima de la ventana de contexto.
        max_windows_per_user: Límite superior de ventanas para controlar el sesgo
                             hacia usuarios hiperactivos. Si se especifica, se toman
                             las últimas `max_windows_per_user` ventanas.
        stride: Salto entre ventanas.
    
    Returns:
        Lista de tuplas (context_tokens, target_token).
    """
    n = len(sequence)
    if n < 2:
        return []

    windows = []
    for t in range(1, n, stride):
        context = sequence[max(0, t - max_len) : t]
        target = sequence[t]
        windows.append((context, target))

    if max_windows_per_user is not None and len(windows) > max_windows_per_user:
        windows = windows[-max_windows_per_user:]

    return windows
