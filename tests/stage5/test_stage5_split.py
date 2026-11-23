"""Tests unitarios para el módulo de split temporal y desempate determinista de la Etapa 5."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from recsys.stage5.data_split import create_temporal_split, generate_sliding_windows


@pytest.fixture
def mock_data():
    """Genera datos sintéticos con empates intencionales en timestamp."""
    users = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    movies = [500, 10, 200, 30, 40, 1, 2, 3, 4, 5]
    timestamps = [100, 100, 100, 200, 300, 10, 20, 30, 40, 50]
    ratings_vals = [4.0, 3.0, 5.0, 2.0, 4.5, 3.5, 4.0, 1.0, 5.0, 4.0]

    ratings_df = pd.DataFrame({
        "userId": users,
        "movieId": movies,
        "rating": ratings_vals,
        "timestamp": timestamps,
    })

    all_movies = sorted(list(set(movies) | {999, 1000}))
    movies_df = pd.DataFrame({
        "movieId": all_movies,
        "title": [f"Movie {m}" for m in all_movies],
        "genres": ["Action|Comedy" for _ in all_movies],
    })

    return ratings_df, movies_df


def test_split_reproducibility_and_proportions(mock_data):
    ratings_df, movies_df = mock_data
    bundle1 = create_temporal_split(ratings_df, movies_df, seed=42)
    bundle2 = create_temporal_split(ratings_df, movies_df, seed=42)

    assert bundle1.train_full_df.equals(bundle2.train_full_df)
    assert bundle1.test_df.equals(bundle2.test_df)
    assert bundle1.val_df.equals(bundle2.val_df)


def test_tie_breaking_avoids_movie_id_sort():
    n_ties = 30
    movie_ids = list(range(100, 100 + n_ties))
    ratings_df = pd.DataFrame({
        "userId": [1] * n_ties,
        "movieId": movie_ids,
        "rating": [4.0] * n_ties,
        "timestamp": [1000] * n_ties,
    })
    movies_df = pd.DataFrame({
        "movieId": movie_ids,
        "title": [f"Movie {m}" for m in movie_ids],
        "genres": ["Drama" for _ in movie_ids],
    })

    bundle = create_temporal_split(ratings_df, movies_df, seed=42, use_event_boundary_rule=False)
    seq = bundle.user_train_seq[0]

    is_strictly_sorted = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    assert not is_strictly_sorted, "El desempate NO debe ser por movieId ascendente."


def test_strict_timestamp_separation_option_c():
    """Verifica separación estricta: max(train_ts) < min(test_ts) y max(core_ts) < min(val_ts)."""
    ratings_path = Path("data/ml-latest-small/ratings.csv")
    movies_path = Path("data/ml-latest-small/movies.csv")

    if not ratings_path.exists():
        pytest.skip("Dataset real no disponible")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    bundle = create_temporal_split(ratings, movies, seed=42, use_event_boundary_rule=True)

    # 1. Separación estricta Train Full vs Test
    for u_idx in range(bundle.n_users):
        u_train = bundle.train_full_df[bundle.train_full_df["user_idx"] == u_idx]
        u_test = bundle.test_df[bundle.test_df["user_idx"] == u_idx]

        if len(u_train) > 0 and len(u_test) > 0 and u_idx not in bundle.users_marked_strict_test:
            max_train_ts = u_train["timestamp"].max()
            min_test_ts = u_test["timestamp"].min()
            assert max_train_ts < min_test_ts, f"Violación de separación estricta Train/Test en user {u_idx}: {max_train_ts} >= {min_test_ts}"

    # 2. Separación estricta Train Core vs Val
    for u_idx in range(bundle.n_users):
        u_core = bundle.train_core_df[bundle.train_core_df["user_idx"] == u_idx]
        u_val = bundle.val_df[bundle.val_df["user_idx"] == u_idx]

        if len(u_core) > 0 and len(u_val) > 0 and u_idx not in bundle.users_marked_strict_val:
            max_core_ts = u_core["timestamp"].max()
            min_val_ts = u_val["timestamp"].min()
            assert max_core_ts < min_val_ts, f"Violación de separación estricta Core/Val en user {u_idx}: {max_core_ts} >= {min_val_ts}"


def test_sliding_windows_generator():
    seq = list(range(100))
    windows = generate_sliding_windows(seq, max_len=10)
    assert len(windows) == 99
    for ctx, tgt in windows:
        assert len(ctx) <= 10
        assert len(ctx) >= 1

    windows_capped = generate_sliding_windows(seq, max_len=10, max_windows_per_user=15)
    assert len(windows_capped) == 15
    assert windows_capped[-1] == windows[-1]
