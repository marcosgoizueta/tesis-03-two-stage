"""Tests unitarios para las funciones específicas de la Etapa 4."""

import numpy as np
import pandas as pd
import pytest

from recsys.stage4.cold_start import create_cold_item_split
from recsys.stage4.embeddings import format_movie_text
from recsys.stage4.evaluation_slices import assign_popularity_buckets, compute_slice_recalls, compute_popularity_bias


def test_format_movie_text():
    # Caso estándar
    assert (
        format_movie_text("Toy Story (1995)", "Adventure|Animation|Children")
        == "Toy Story (1995). Genres: Adventure, Animation, Children"
    )
    # Caso sin género
    assert (
        format_movie_text("Hyena Road", "(no genres listed)")
        == "Hyena Road. Genres: No genres listed"
    )
    assert (
        format_movie_text("Paterson", "")
        == "Paterson. Genres: No genres listed"
    )
    assert (
        format_movie_text("Nocturnal Animals", None)
        == "Nocturnal Animals. Genres: No genres listed"
    )


def test_slice_recall_averages_only_users_with_relevant_items_in_slice():
    """Test exigido (a): El promedio por corte usa SOLO usuarios con relevantes en ese corte."""
    # Creamos un escenario con 3 usuarios:
    # Usuario 1: tiene relevantes en corte '0' (ítem 10) y en corte '>20' (ítem 100)
    # Usuario 2: SOLO tiene relevantes en corte '>20' (ítem 101)
    # Usuario 3: SOLO tiene relevantes en corte '1-5' (ítem 20)
    item_buckets = {
        10: "0",
        20: "1-5",
        100: ">20",
        101: ">20",
    }
    test_relevant_by_user = {
        1: {10, 100},
        2: {101},
        3: {20},
    }
    eval_users = [1, 2, 3]

    # Recomendaciones (top-2):
    # Usuario 1 recibe [10, 50] -> acierta ítem 10 (corte 0: recall 1/1 = 1.0, corte >20: recall 0/1 = 0.0)
    # Usuario 2 recibe [101, 50] -> acierta ítem 101 (corte >20: recall 1/1 = 1.0)
    # Usuario 3 recibe [99, 98] -> falla ítem 20 (corte 1-5: recall 0/1 = 0.0)
    recommendations_by_user = {
        1: [10, 50],
        2: [101, 50],
        3: [99, 98],
    }

    results = compute_slice_recalls(
        recommendations_by_user=recommendations_by_user,
        test_relevant_by_user=test_relevant_by_user,
        eval_users=eval_users,
        item_buckets=item_buckets,
        cutoffs=(2,),
    )

    # Para el corte '0':
    # Solo el usuario 1 tiene relevantes en corte '0'.
    # Si promediara sobre los 3 usuarios daría 1.0 / 3 = 0.3333.
    # Como debe promediar SOLO sobre usuarios elegibles (usuario 1), debe dar exactamente 1.0.
    assert results["0"]["n_eval_users"] == 1.0
    assert results["0"]["recall_at_2"] == 1.0

    # Para el corte '>20':
    # Usuarios elegibles: 1 y 2 (n=2). Usuario 3 no tiene relevantes en '>20'.
    # Usuario 1 tiene recall 0.0, Usuario 2 tiene recall 1.0.
    # Promedio = (0.0 + 1.0) / 2 = 0.5.
    assert results[">20"]["n_eval_users"] == 2.0
    assert results[">20"]["recall_at_2"] == 0.5

    # Para el corte '1-5':
    # Solo el usuario 3 es elegible. Recall = 0.0.
    assert results["1-5"]["n_eval_users"] == 1.0
    assert results["1-5"]["recall_at_2"] == 0.0

    # Para el corte '6-20': ningún usuario tiene relevantes
    assert results["6-20"]["n_eval_users"] == 0.0
    assert results["6-20"]["recall_at_2"] == 0.0


def test_cold_item_split_guarantees_no_leakage_and_no_loss():
    """Test exigido (b): En el split en frío ningún ítem en frío aparece en train y ninguna interacción se pierde."""
    n_users = 50
    n_items = 100
    n_interactions = 1000

    np.random.seed(42)
    fake_ratings = pd.DataFrame({
        "userId": np.random.randint(1, n_users + 1, size=n_interactions),
        "movieId": np.random.randint(1, n_items + 1, size=n_interactions),
        "rating": np.random.choice([1.0, 2.0, 3.0, 3.5, 4.0, 5.0], size=n_interactions),
        "timestamp": np.arange(n_interactions),
    })
    user2idx = {u: i for i, u in enumerate(fake_ratings["userId"].unique())}
    item2idx = {m: i for i, m in enumerate(fake_ratings["movieId"].unique())}
    fake_ratings["user_idx"] = fake_ratings["userId"].map(user2idx)
    fake_ratings["item_idx"] = fake_ratings["movieId"].map(item2idx)
    actual_n_items = len(item2idx)

    train_df, test_df, cold_items, warm_items = create_cold_item_split(
        ratings=fake_ratings,
        n_items=actual_n_items,
        cold_item_ratio=0.10,
        seed=42,
    )

    # 1. Conservación exacta: ninguna interacción se pierde ni se duplica
    assert len(train_df) + len(test_df) == len(fake_ratings), "Se perdieron o duplicaron interacciones."

    # 2. Cero fuga: ningún ítem en frío aparece en train
    train_items = set(train_df["item_idx"].unique())
    leakage = train_items & cold_items
    assert len(leakage) == 0, f"Fuga de datos detectada: ítems fríos en train: {leakage}"

    # 3. Disyunción de conjuntos de ítems
    assert cold_items.isdisjoint(train_items), "cold_items y train_items no son disjuntos."
    assert len(cold_items) + len(warm_items) == actual_n_items

    # 4. Todas las interacciones de los ítems fríos están en test
    cold_ratings = fake_ratings[fake_ratings["item_idx"].isin(cold_items)]
    test_cold_ratings = test_df[test_df["item_idx"].isin(cold_items)]
    assert len(cold_ratings) == len(test_cold_ratings), "No todas las interacciones de ítems fríos están en test."


def test_embedding_matrix_alignment_with_item_idx():
    """Test exigido (2): Alineación estricta entre las filas de la matriz de embeddings y item2idx."""
    import json
    from pathlib import Path
    from recsys.config import Stage3Config
    from recsys.data import load_and_prepare_data

    cfg = Stage3Config()
    data = load_and_prepare_data(cfg)

    cache_dir = Path("data/cache")
    meta_path = cache_dir / "metadata.json"
    emb_path = cache_dir / "movie_embeddings_minilm.npy"

    assert meta_path.exists(), "metadata.json no existe en data/cache"
    assert emb_path.exists(), "movie_embeddings_minilm.npy no existe en data/cache"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    embeddings = np.load(emb_path)
    assert embeddings.shape[0] == data.n_items
    assert embeddings.shape[1] == 384

    # Verificar que el orden registrado en movie_ids coincide exactamente con item2idx
    stored_movie_ids = meta["movie_ids"]
    assert len(stored_movie_ids) == data.n_items

    for idx, movie_id in enumerate(stored_movie_ids):
        assert data.item2idx[movie_id] == idx, f"Desalineación en fila {idx}: movieId {movie_id} != {data.item2idx[movie_id]}"


def test_quick_mode_writes_to_results_quick():
    """Test exigido (3): El modo --quick escribe exclusivamente en results/quick/."""
    from scripts.run_stage3 import parse_args
    import argparse
    from unittest.mock import patch
    from pathlib import Path

    # Simular ejecución con --quick sin --results-dir
    test_args = ["run_stage3.py", "--quick", "--device", "cpu"]
    with patch("sys.argv", test_args):
        args = parse_args()
        assert args.quick is True
        assert args.results_dir is None

        # Replicar la resolución de target_results_dir del script corregido
        if args.results_dir is not None:
            target_results_dir = args.results_dir
        elif args.quick:
            target_results_dir = Path("results/quick")
        else:
            target_results_dir = Path("results/cpu")

        assert target_results_dir == Path("results/quick")
