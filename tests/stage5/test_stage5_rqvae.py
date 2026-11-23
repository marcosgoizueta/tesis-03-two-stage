"""Tests unitarios para el módulo RQ-VAE y gestión de Semantic IDs."""

import numpy as np
import pytest
import torch

from recsys.stage5.rq_vae import RQVAE, SemanticIdManager, torch_kmeans


def test_rqvae_forward_shapes_and_losses():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, D = 32, 384
    model = RQVAE(in_dim=D, hidden_dim=64, latent_dim=64, num_levels=3, codebook_size=64).to(device)
    x = torch.randn(B, D, device=device)

    recon, codes, total_loss, rec_loss, com_loss = model(x)

    assert recon.shape == (B, D)
    assert codes.shape == (B, 3)
    assert total_loss.dim() == 0
    assert total_loss.item() > 0.0
    assert (codes >= 0).all() and (codes < 64).all()


def test_torch_kmeans_centroids_and_coverage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D, K = 500, 32, 16
    X = torch.randn(N, D, device=device)
    centroids, labels = torch_kmeans(X, n_clusters=K, max_iter=20, seed=42)

    assert centroids.shape == (K, D)
    assert labels.shape == (N,)
    unique_labels = torch.unique(labels)
    # Casi todos o todos los clusters deben usarse
    assert len(unique_labels) >= K * 0.90


def test_semantic_id_manager_cold_item_policy():
    manager = SemanticIdManager(max_disambiguation=10)
    train_items = [0, 1, 2]
    # Supongamos que ítems 0 y 1 tienen el mismo código (10, 20, 30), e ítem 2 tiene (5, 5, 5)
    codes_np = np.array([
        [10, 20, 30],
        [10, 20, 30],
        [5, 5, 5],
        [10, 20, 30], # Ítem 3 es frío y colisiona con 0 y 1
    ])
    train_counts = {0: 100, 1: 50, 2: 10}

    # 1. Ajuste con ítems de train
    manager.fit_train_items(train_items, codes_np, train_counts)
    assert manager.item_to_semantic_id[0] == (10, 20, 30, 0) # Más popular
    assert manager.item_to_semantic_id[1] == (10, 20, 30, 1) # Segundo más popular
    assert manager.item_to_semantic_id[2] == (5, 5, 5, 0)

    # 2. Asignación a ítem frío
    manager.assign_cold_items([3], codes_np)
    # Debe recibir d = 2
    assert manager.item_to_semantic_id[3] == (10, 20, 30, 2)
    assert manager.semantic_id_to_item[(10, 20, 30, 2)] == 3
