"""Generación y almacenamiento en caché de embeddings de texto para ítems."""

import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def format_movie_text(title: str, genres: str) -> str:
    """Aplica la plantilla en inglés: '<title>. Genres: <comma-separated genres>'."""
    title_clean = str(title).strip()
    if pd.isna(genres) or not genres or genres == "(no genres listed)":
        genres_clean = "No genres listed"
    else:
        genres_clean = ", ".join(genres.split("|"))
    return f"{title_clean}. Genres: {genres_clean}"


def get_model_revision(model_name: str) -> str:
    """Obtiene el hash del commit del snapshot en la caché de Hugging Face si existe."""
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    if not model_dir_name.startswith("models--sentence-transformers--") and "/" not in model_name:
        model_dir_name = f"models--sentence-transformers--{model_name}"
    
    pattern = os.path.expanduser(f"~/.cache/huggingface/hub/{model_dir_name}/snapshots/*")
    snapshots = glob.glob(pattern)
    if snapshots:
        return os.path.basename(snapshots[0])
    return "unknown"


def generate_or_load_item_embeddings(
    movies_df: pd.DataFrame,
    item2idx: Dict[int, int],
    cache_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "auto",
    batch_size: int = 64,
) -> Tuple[np.ndarray, Dict]:
    """Genera una sola vez los embeddings de texto para el catálogo y los persiste en caché."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    emb_path = cache_dir / "movie_embeddings_minilm.npy"
    meta_path = cache_dir / "metadata.json"

    # Si la caché ya existe y coincide en dimensiones, cargar directamente
    n_items = len(item2idx)
    if emb_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        embeddings = np.load(emb_path)
        if embeddings.shape[0] == n_items:
            print(f"[Caché] Embeddings cargados desde {emb_path} (forma: {embeddings.shape})")
            return embeddings, metadata

    print(f"[FASE 1] Generando embeddings de texto con modelo '{model_name}'...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    import transformers

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        # Fallback por si no está en caché offline
        model = SentenceTransformer(model_name)

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device
    model = model.to(target_device)

    # Ordenar películas según item_idx
    movie_dict = dict(zip(movies_df["movieId"], zip(movies_df["title"], movies_df["genres"])))
    
    texts: List[str] = []
    movie_ids_ordered: List[int] = []
    missing_in_movies = 0
    for movie_id, idx in sorted(item2idx.items(), key=lambda x: x[1]):
        movie_ids_ordered.append(int(movie_id))
        if movie_id in movie_dict:
            title, genres = movie_dict[movie_id]
            texts.append(format_movie_text(title, genres))
        else:
            missing_in_movies += 1
            texts.append(f"Movie {movie_id}. Genres: No genres listed")

    if missing_in_movies > 0:
        print(f"[Aviso] {missing_in_movies} ítems no estaban en movies.csv y se usó texto genérico.")

    # Codificar textos
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=target_device,
        normalize_embeddings=True,
    ).astype(np.float32)

    revision = get_model_revision(model_name)
    metadata = {
        "model_name": model_name,
        "revision_commit": revision,
        "embedding_dim": int(embeddings.shape[1]),
        "n_items": int(embeddings.shape[0]),
        "normalized": True,
        "template": "<title>. Genres: <genres>",
        "created_at": datetime.now().isoformat(),
        "device_used": target_device,
        "movie_ids": movie_ids_ordered,
        "libraries": {
            "sentence_transformers": getattr(SentenceTransformer, "__version__", "3.0.1"),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    }

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[FASE 1] Embeddings guardados en {emb_path}")
    print(f"[FASE 1] Metadatos guardados en {meta_path}")

    return embeddings, metadata
