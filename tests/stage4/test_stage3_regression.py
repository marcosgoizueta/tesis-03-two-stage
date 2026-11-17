"""Test de regresión para verificar que la Etapa 3 mantiene sus resultados de referencia."""

import random
from pathlib import Path
import numpy as np
import pytest
import torch

from recsys.config import Stage3Config
from recsys.data import load_and_prepare_data
from recsys.evaluation import evaluate_model, make_rec_retrieval_only
from recsys.retrieval import (
    RetrievalDataset,
    TwoTower,
    compute_retrieval_candidates,
    train_retrieval,
)


def set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@pytest.mark.slow
@pytest.mark.regression
def test_stage3_retrieval_exact_reproducibility():
    """Verifica que el retrieval de la Etapa 3 con semilla 42 reproduce exactamente las métricas de referencia:
    precision@10: 0.1478
    recall@10: 0.1275
    ndcg@10: 0.1907
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    set_seed(seed, device)
    cfg = Stage3Config(seed=seed, device="cuda" if device.type == "cuda" else "cpu")
    data = load_and_prepare_data(cfg)

    retrieval_dataset = RetrievalDataset(
        positive_df=data.train_positive,
        user_seen_train=data.user_seen_train,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        n_neg_per_pos=cfg.n_neg_per_pos_retrieval,
        fast_sampling=False,
    )

    retrieval_model = TwoTower(
        n_users=data.n_users,
        n_items=data.n_items,
        embed_dim=cfg.embed_dim_retrieval,
    ).to(device)

    train_retrieval(
        model=retrieval_model,
        dataset=retrieval_dataset,
        cfg=cfg,
        device=device,
    )

    retrieval_topk = compute_retrieval_candidates(
        model=retrieval_model,
        n_users=data.n_users,
        n_items=data.n_items,
        seen_mask_np=data.seen_mask_np,
        max_candidates=cfg.max_candidates,
        device=device,
    )

    metrics = evaluate_model(
        name="Retrieval (two-tower) solo",
        rec_fn=make_rec_retrieval_only(retrieval_topk, cfg.top_k),
        eval_users=data.eval_users,
        test_relevant_by_user=data.test_relevant_by_user,
        item_popularity=data.item_popularity,
        n_items=data.n_items,
        device=device,
        top_k=cfg.top_k,
    )

    p10 = metrics["precision_at_10"]
    r10 = metrics["recall_at_10"]
    n10 = metrics["ndcg_at_10"]

    print(f"\n[Test Regresión Etapa 3] P@10: {p10:.6f}, R@10: {r10:.6f}, NDCG@10: {n10:.6f}")

    assert abs(p10 - 0.147754) < 1e-4, f"P@10 esperado 0.1478, obtenido {p10:.4f}"
    assert abs(r10 - 0.127539) < 1e-4, f"R@10 esperado 0.1275, obtenido {r10:.4f}"
    assert abs(n10 - 0.190736) < 1e-4, f"NDCG@10 esperado 0.1907, obtenido {n10:.4f}"
