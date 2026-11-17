"""Generador del split de evaluación para el escenario de ítems en frío (Fase 4)."""

from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_cold_item_split(
    ratings: pd.DataFrame,
    n_items: int,
    cold_item_ratio: float = 0.10,
    seed: int = 42,
    rating_threshold: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int]]:
    """Separa al azar una fracción de ítems (default 10%) como ítems en frío:
    - Todas las interacciones de los ítems en frío van exclusivamente a test.
    - Las interacciones de los ítems restantes (warm) se dividen 80/20 con split aleatorio.
    - Ninguna interacción se descarta: len(train) + len(test) == len(ratings).
    - Ningún ítem en frío aparece en train.
    """
    rng = np.random.RandomState(seed)
    all_items = np.arange(n_items)
    n_cold = int(np.round(n_items * cold_item_ratio))
    
    cold_items = set(rng.choice(all_items, size=n_cold, replace=False))
    warm_items = set(all_items) - cold_items

    is_cold_interaction = ratings["item_idx"].isin(cold_items)
    cold_interactions = ratings[is_cold_interaction]
    warm_interactions = ratings[~is_cold_interaction]

    # Split 80/20 sobre los ítems warm
    train_warm, test_warm = train_test_split(
        warm_interactions, test_size=0.2, random_state=seed
    )

    # El test final combina el test warm con TODAS las interacciones de los ítems fríos
    train_df = train_warm.copy().reset_index(drop=True)
    test_df = pd.concat([test_warm, cold_interactions], ignore_index=True)

    return train_df, test_df, cold_items, warm_items
