from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Stage3Config:
    """Configuración e hiperparámetros para el pipeline de dos etapas."""

    # Rutas base
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    dataset_url: str = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    # Semilla y dispositivo
    seed: int = 42
    device: str = "auto"  # 'auto', 'cpu', 'cuda'

    # Datos y split
    test_size: float = 0.2
    random_state_split: int = 42
    rating_threshold: float = 3.5

    # Retrieval (Two-Tower)
    embed_dim_retrieval: int = 32
    n_neg_per_pos_retrieval: int = 4
    batch_size_retrieval: int = 256
    lr_retrieval: float = 5e-3
    weight_decay_retrieval: float = 1e-6
    n_epochs_retrieval: int = 30
    max_candidates: int = 500

    # Ranker
    embed_dim_ranker: int = 32
    hidden_ranker: Tuple[int, ...] = (128, 64)
    dropout_ranker: float = 0.2
    lr_ranker: float = 2e-3
    weight_decay_ranker: float = 1e-5
    batch_size_ranker: int = 1024
    n_epochs_ranker: int = 8
    hard_pool: int = 100

    # Evaluación y candidatos
    top_k: int = 10
    candidate_recall_cutoffs: Tuple[int, ...] = (10, 50, 100, 200, 500)
    candidate_ns_to_eval: Tuple[int, ...] = (50, 200)

    # Flags operacionales
    quick: bool = False
    fast_negative_sampling: bool = False

    # Protocolos
    protocolo_full: str = "full-ranking (top-10 sobre catálogo no visto en train)"
    protocolo_pointwise: str = "pointwise (rating predicho vs. umbral 3.5)"

    @property
    def ml_dir(self) -> Path:
        return self.data_dir / "ml-latest-small"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "stage3_runs"

    @property
    def comparativa_csv(self) -> Path:
        return self.results_dir / "comparativa_modelos.csv"
