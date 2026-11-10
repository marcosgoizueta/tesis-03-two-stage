import io
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from recsys.config import Stage3Config


def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


@dataclass
class DatasetBundle:
    """Contenedor de todos los datos procesados, mapeos y matrices de características."""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_positive: pd.DataFrame

    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    n_users: int
    n_items: int

    user_seen_train: Dict[int, Set[int]]
    seen_mask_np: np.ndarray

    item_genre_np: np.ndarray
    user_genre_np: np.ndarray
    item_popularity: np.ndarray
    item_logpop_np: np.ndarray

    test_relevant_by_user: Dict[int, Set[int]]
    eval_users: List[int]
    all_genres: List[str]


def ensure_dataset(data_dir: Path, dataset_url: str) -> Path:
    """Descarga y extrae el dataset MovieLens ml-latest-small si no existe localmente."""
    ml_dir = data_dir / "ml-latest-small"
    ratings_csv = ml_dir / "ratings.csv"
    movies_csv = ml_dir / "movies.csv"

    if ratings_csv.exists() and movies_csv.exists():
        return ml_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando dataset desde {dataset_url}...")
    ctx = _get_ssl_context()
    with urllib.request.urlopen(dataset_url, context=ctx) as response:
        zip_bytes = response.read()

    print("Extrayendo archivos...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(data_dir)

    print(f"Dataset listo en {ml_dir}")
    return ml_dir


def compute_feature_matrices(
    movies_df: pd.DataFrame,
    item2idx: Dict[int, int],
    train_positive: pd.DataFrame,
    n_users: int,
    n_items: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Calcula matrices de características de contenido respetando la lógica exacta del notebook."""
    movie_genres = dict(zip(movies_df["movieId"], movies_df["genres"].str.split("|")))
    all_genres = sorted(
        {g for gl in movie_genres.values() for g in gl if g != "(no genres listed)"}
    )
    genre2col = {g: c for c, g in enumerate(all_genres)}
    n_genres = len(all_genres)

    item_genre_np = np.zeros((n_items, n_genres), dtype=np.float32)
    for movie_id, idx in item2idx.items():
        for g in movie_genres.get(movie_id, []):
            if g in genre2col:
                item_genre_np[idx, genre2col[g]] = 1.0

    # Perfil de géneros del usuario: distribución de géneros entre sus positivos de train
    user_pos_np = np.zeros((n_users, n_items), dtype=np.float32)
    user_pos_np[
        train_positive["user_idx"].values, train_positive["item_idx"].values
    ] = 1.0
    user_genre_np = user_pos_np @ item_genre_np
    user_genre_np = user_genre_np / np.maximum(
        user_genre_np.sum(axis=1, keepdims=True), 1.0
    )

    # Popularidad del ítem: conteo de positivos en train
    item_popularity = np.zeros(n_items, dtype=np.float64)
    pop_counts = train_positive["item_idx"].value_counts()
    item_popularity[pop_counts.index.values] = pop_counts.values
    item_logpop_np = np.log1p(item_popularity)
    item_logpop_np = (item_logpop_np / item_logpop_np.max()).astype(np.float32)

    return item_genre_np, user_genre_np, item_popularity, item_logpop_np, all_genres


def load_and_prepare_data(cfg: Stage3Config) -> DatasetBundle:
    """Carga los archivos, genera los splits y construye todas las estructuras necesarias."""
    ml_dir = ensure_dataset(cfg.data_dir, cfg.dataset_url)

    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")

    user_ids = ratings["userId"].unique()
    item_ids = ratings["movieId"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}

    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["item_idx"] = ratings["movieId"].map(item2idx)

    n_users = len(user2idx)
    n_items = len(item2idx)

    train_df, test_df = train_test_split(
        ratings, test_size=cfg.test_size, random_state=cfg.random_state_split
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
        all_genres,
    ) = compute_feature_matrices(
        movies, item2idx, train_positive, n_users, n_items
    )

    test_relevant_by_user = (
        test_df[test_df["rating"] >= cfg.rating_threshold]
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )
    eval_users = list(test_relevant_by_user.keys())

    return DatasetBundle(
        ratings=ratings,
        movies=movies,
        train_df=train_df,
        test_df=test_df,
        train_positive=train_positive,
        user2idx=user2idx,
        item2idx=item2idx,
        n_users=n_users,
        n_items=n_items,
        user_seen_train=user_seen_train,
        seen_mask_np=seen_mask_np,
        item_genre_np=item_genre_np,
        user_genre_np=user_genre_np,
        item_popularity=item_popularity,
        item_logpop_np=item_logpop_np,
        test_relevant_by_user=test_relevant_by_user,
        eval_users=eval_users,
        all_genres=all_genres,
    )
