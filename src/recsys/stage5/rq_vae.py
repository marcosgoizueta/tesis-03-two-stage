"""Módulo RQ-VAE (Residual Quantized Variational AutoEncoder / Quantizer) para la Etapa 5.
Implementa cuantización vectorial residual en 3 niveles con codebooks jerárquicos,
estandarización fit sobre train, desambiguación y proyección de ítems fríos.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def torch_kmeans(
    X: torch.Tensor,
    n_clusters: int = 256,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KMeans acelerado en PyTorch para GPU o CPU con re-inicialización de clusters vacíos."""
    device = X.device
    generator = torch.Generator(device=device).manual_seed(seed)
    N, D = X.shape
    perm = torch.randperm(N, generator=generator, device=device)
    centroids = X[perm[:n_clusters]].clone()

    for _ in range(max_iter):
        x_sq = (X ** 2).sum(dim=1, keepdim=True)
        c_sq = (centroids ** 2).sum(dim=1, keepdim=True).t()
        dists = x_sq - 2 * torch.mm(X, centroids.t()) + c_sq
        labels = torch.argmin(dists, dim=1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=n_clusters).float().unsqueeze(1)
        new_centroids.index_add_(0, labels, X)

        empty_mask = (counts.squeeze() == 0)
        if empty_mask.any():
            n_empty = empty_mask.sum().item()
            rand_idx = torch.randperm(N, generator=generator, device=device)[:n_empty]
            new_centroids[empty_mask] = X[rand_idx]
            counts[empty_mask] = 1.0

        new_centroids = new_centroids / counts
        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < 1e-4:
            break

    return centroids, labels


class ResidualVectorQuantizer(nn.Module):
    """Cuantizador vectorial residual jerárquico."""

    def __init__(
        self,
        num_levels: int = 3,
        codebook_size: int = 256,
        dim: int = 384,
        beta: float = 0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim)) for _ in range(num_levels)
        ])

    def init_from_kmeans(self, X_train: torch.Tensor, seed: int = 42) -> List[int]:
        """Inicializa los codebooks secuencialmente por niveles usando K-Means."""
        curr_res = X_train.clone()
        used_counts = []
        for l in range(self.num_levels):
            centers, labels = torch_kmeans(
                curr_res, n_clusters=self.codebook_size, max_iter=100, seed=seed + l
            )
            self.codebooks[l].data.copy_(centers)
            curr_res = curr_res - centers[labels]
            used = len(torch.unique(labels))
            used_counts.append(used)
        return used_counts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        quantized_out = torch.zeros_like(x)
        all_codes = []
        commit_loss = torch.tensor(0.0, device=x.device)

        for l in range(self.num_levels):
            cb = self.codebooks[l]
            dist = (
                torch.sum(residual ** 2, dim=-1, keepdim=True)
                - 2 * torch.matmul(residual, cb.t())
                + torch.sum(cb ** 2, dim=-1, keepdim=True).t()
            )
            code = torch.argmin(dist, dim=-1)
            all_codes.append(code)

            quant = cb[code]

            # Commitment loss con straight-through estimator
            commit_loss = commit_loss + torch.mean((residual.detach() - quant) ** 2) + self.beta * torch.mean((residual - quant.detach()) ** 2)
            quant_st = residual + (quant - residual).detach()

            quantized_out = quantized_out + quant_st
            residual = residual - quant_st

        codes_tensor = torch.stack(all_codes, dim=-1)
        return quantized_out, codes_tensor, commit_loss


class RQVAE(nn.Module):
    """Autoencoder Cuantizado Residual con conexiones residuales."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 384,
        latent_dim: int = 384,
        num_levels: int = 3,
        codebook_size: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.enc_res = nn.Identity() if in_dim == latent_dim else nn.Linear(in_dim, latent_dim)

        self.rvq = ResidualVectorQuantizer(
            num_levels=num_levels, codebook_size=codebook_size, dim=latent_dim, beta=beta
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.dec_res = nn.Identity() if latent_dim == in_dim else nn.Linear(latent_dim, in_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x) + self.enc_res(x)
        z_q, codes, commit_loss = self.rvq(h)
        x_recon = self.decoder(z_q) + self.dec_res(z_q)
        recon_loss = torch.mean((x - x_recon) ** 2)
        total_loss = recon_loss + commit_loss
        return x_recon, codes, total_loss, recon_loss, commit_loss

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x) + self.enc_res(x)
        _, codes, _ = self.rvq(h)
        return codes


class SemanticIdManager:
    """Gestiona la asignación de Semantic IDs (c1, c2, c3, d) y resuelve colisiones."""

    def __init__(self, max_disambiguation: int = 16):
        self.max_disambiguation = max_disambiguation
        self.tuple_to_items: Dict[Tuple[int, ...], List[int]] = {}
        self.item_to_semantic_id: Dict[int, Tuple[int, int, int, int]] = {}
        self.semantic_id_to_item: Dict[Tuple[int, int, int, int], int] = {}
        self.next_disambiguation: Dict[Tuple[int, ...], int] = {}

    def fit_train_items(
        self,
        train_item_indices: List[int],
        codes_np: np.ndarray,
        item_train_counts: Dict[int, int],
    ) -> None:
        """Asigna Semantic IDs a los ítems de train ordenados por popularidad descendente."""
        from collections import defaultdict
        groups = defaultdict(list)
        for idx in train_item_indices:
            c_tuple = tuple(int(c) for c in codes_np[idx])
            groups[c_tuple].append(idx)

        for c_tuple, items in groups.items():
            # Los ítems más populares obtienen el token d más bajo (0, 1, ...)
            items.sort(key=lambda i: item_train_counts.get(i, 0), reverse=True)
            for d, item_idx in enumerate(items):
                d_clamped = min(d, self.max_disambiguation - 1)
                full_id = (c_tuple[0], c_tuple[1], c_tuple[2], d_clamped)
                self.item_to_semantic_id[item_idx] = full_id
                self.semantic_id_to_item[full_id] = item_idx
            self.next_disambiguation[c_tuple] = len(items)

    def assign_cold_items(
        self,
        cold_item_indices: List[int],
        codes_np: np.ndarray,
    ) -> None:
        """Asigna Semantic IDs a los ítems fríos respetando la política explícita de colisiones."""
        for idx in cold_item_indices:
            c_tuple = tuple(int(c) for c in codes_np[idx])
            d = self.next_disambiguation.get(c_tuple, 0)
            d_clamped = min(d, self.max_disambiguation - 1)
            full_id = (c_tuple[0], c_tuple[1], c_tuple[2], d_clamped)

            self.item_to_semantic_id[idx] = full_id
            # Si hay colisión con d_clamped saturado, mapear preservando el primer registro
            if full_id not in self.semantic_id_to_item:
                self.semantic_id_to_item[full_id] = idx
            self.next_disambiguation[c_tuple] = d + 1
