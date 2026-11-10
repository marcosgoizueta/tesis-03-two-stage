import json
import os
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

from recsys.config import Stage3Config


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """Obtiene metadata detallada del dispositivo de cómputo y versiones de CUDA."""
    is_cuda = device.type == "cuda"
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    cuda_version = torch.version.cuda if is_cuda else None

    return {
        "device_type": device.type,
        "device_name": device_name,
        "cuda_version": cuda_version,
        "torch_variant": torch.__version__,
        "cuda_device_count": torch.cuda.device_count() if is_cuda else 0,
        "cuda_capability": (
            list(torch.cuda.get_device_capability(0)) if is_cuda else None
        ),
    }


def save_comparativa_csv(
    results_df: pd.DataFrame,
    cfg: Stage3Config,
    device: torch.device,
) -> Path:
    """Actualiza comparativa_modelos.csv preservando etapas previas y registrando la columna dispositivo."""
    if cfg.quick:
        print("Modo --quick activo: se omite la escritura en comparativa_modelos.csv.")
        return cfg.comparativa_csv

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.comparativa_csv

    new_df = results_df.copy()
    dev_info = get_device_info(device)
    new_df["dispositivo"] = dev_info["device_name"]
    new_df["protocolo"] = cfg.protocolo_full
    new_df["fecha"] = datetime.now().isoformat(timespec="seconds")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if os.path.exists(results_path):
            previous = pd.read_csv(results_path)
            if "protocolo" not in previous.columns:
                previous["protocolo"] = None
            if "dispositivo" not in previous.columns:
                previous["dispositivo"] = "CPU (histórico)"

            # Etapas 1 y 2: se etiquetan como pointwise si todavía no tenían protocolo
            old_stages = (
                previous["modelo"].isin(["SVD (baseline)", "NCF (NeuMF)"])
                & previous["protocolo"].isna()
            )
            previous.loc[old_stages, "protocolo"] = cfg.protocolo_pointwise

            # Se descartan corridas anteriores de esta etapa para no duplicar filas
            drop = previous["modelo"].str.startswith(
                ("Two-Stage", "Retrieval", "Ranker", "Popularidad")
            )
            results = pd.concat([previous[~drop], new_df], ignore_index=True)
        else:
            results = new_df

    results.to_csv(results_path, index=False)
    print(f"Resultados actualizados guardados en {results_path}")
    return results_path


def save_run_metadata(
    results_list: List[Dict[str, Any]],
    candidate_recalls: Dict[int, float],
    retrieval_losses: List[float],
    ranker_losses: Dict[str, List[float]],
    cfg: Stage3Config,
    device: torch.device,
    elapsed_total_seconds: float,
) -> Path:
    """Guarda un registro detallado en JSON dentro del directorio de runs correspondiente."""
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dev_suffix = "gpu" if device.type == "cuda" else "cpu"
    json_path = cfg.runs_dir / f"run_{now_str}_seed{cfg.seed}_{dev_suffix}.json"

    dev_info = get_device_info(device)

    run_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "quick": cfg.quick,
        "seed": cfg.seed,
        "device": dev_info,
        "elapsed_total_seconds": elapsed_total_seconds,
        "fast_negative_sampling": cfg.fast_negative_sampling,
        "system": {
            "os": platform.platform(),
            "os_name": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version,
        },
        "libraries": {
            "torch": torch.__version__,
            "cuda_torch": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "hyperparameters": {
            "test_size": cfg.test_size,
            "rating_threshold": cfg.rating_threshold,
            "embed_dim_retrieval": cfg.embed_dim_retrieval,
            "n_neg_per_pos_retrieval": cfg.n_neg_per_pos_retrieval,
            "batch_size_retrieval": cfg.batch_size_retrieval,
            "lr_retrieval": cfg.lr_retrieval,
            "weight_decay_retrieval": cfg.weight_decay_retrieval,
            "n_epochs_retrieval": 1 if cfg.quick else cfg.n_epochs_retrieval,
            "max_candidates": cfg.max_candidates,
            "embed_dim_ranker": cfg.embed_dim_ranker,
            "hidden_ranker": list(cfg.hidden_ranker),
            "dropout_ranker": cfg.dropout_ranker,
            "lr_ranker": cfg.lr_ranker,
            "weight_decay_ranker": cfg.weight_decay_ranker,
            "batch_size_ranker": cfg.batch_size_ranker,
            "n_epochs_ranker": 1 if cfg.quick else cfg.n_epochs_ranker,
            "hard_pool": cfg.hard_pool,
            "top_k": cfg.top_k,
        },
        "candidate_recalls": {str(k): v for k, v in candidate_recalls.items()},
        "losses": {
            "retrieval": retrieval_losses,
            "rankers": ranker_losses,
        },
        "metrics": results_list,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2, ensure_ascii=False)

    print(f"Metadata de la corrida guardada en {json_path}")
    return json_path
