"""Baselines de las Etapas 1 y 2 (regresión de rating) reimplementados para evaluarlos con el protocolo
temporal full-ranking de la Etapa 5.

- FunkSVD: replica el algoritmo de `surprise.SVD` (SVD sesgado entrenado por SGD, sin mezclar el orden
  de las muestras entre épocas, init N(0, 0.1) de los factores, sesgos en 0, lr 0.005, reg 0.02).
  Se reimplementa en numpy porque scikit-surprise no se instala en este entorno sin riesgo para NumPy.
- NeuMF: la arquitectura exacta de notebooks/02_ncf.ipynb (GMF + MLP, salida sigmoide escalada a [0.5, 5]),
  con la inicialización por defecto de nn.Embedding (N(0, 1)), como en el notebook.
"""
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class FunkSVD:
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr: float = 0.005, reg: float = 0.02,
                 init_std: float = 0.1, seed: int = 42):
        self.n_factors, self.n_epochs, self.lr, self.reg = n_factors, n_epochs, lr, reg
        self.init_std, self.seed = init_std, seed

    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray, n_users: int, n_items: int):
        rng = np.random.RandomState(self.seed)
        self.mu = float(np.mean(ratings))
        self.bu = np.zeros(n_users); self.bi = np.zeros(n_items)
        self.pu = rng.normal(0, self.init_std, (n_users, self.n_factors))
        self.qi = rng.normal(0, self.init_std, (n_items, self.n_factors))
        self.item_known = np.zeros(n_items, dtype=bool); self.item_known[np.unique(items)] = True
        self.user_known = np.zeros(n_users, dtype=bool); self.user_known[np.unique(users)] = True
        lr, reg = self.lr, self.reg
        bu, bi, pu, qi, mu = self.bu, self.bi, self.pu, self.qi, self.mu
        for _ in range(self.n_epochs):          # Surprise no baraja: mismo orden en cada época
            for u, i, r in zip(users, items, ratings):
                p, q = pu[u], qi[i]
                err = r - (mu + bu[u] + bi[i] + q @ p)
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                p_old = p.copy()
                pu[u] += lr * (err * q - reg * p)
                qi[i] += lr * (err * p_old - reg * q)
        return self

    def predict_matrix(self, user_idx: Sequence[int]) -> np.ndarray:
        """Rating predicho (len(user_idx), n_items). Ítems no vistos en train: mu + bu (como Surprise)."""
        u = np.asarray(user_idx)
        s = self.mu + self.bu[u][:, None] + self.bi[None, :] + self.pu[u] @ self.qi.T
        base = self.mu + self.bu[u][:, None]
        return np.where(self.item_known[None, :], s, base)

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        s = self.mu + self.bu[users] + self.bi[items] + np.einsum("ij,ij->i", self.pu[users], self.qi[items])
        return np.where(self.item_known[items], s, self.mu + self.bu[users])


class NeuMF(nn.Module):
    """Idéntico a notebooks/02_ncf.ipynb (no cambiar sin documentarlo)."""

    def __init__(self, n_users, n_items, gmf_dim=16, mlp_dim=32, mlp_layers=(64, 32, 16), rating_range=(0.5, 5.0)):
        super().__init__()
        self.rating_min, self.rating_max = rating_range
        self.user_emb_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_emb_gmf = nn.Embedding(n_items, gmf_dim)
        self.user_emb_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_dim)
        mods, inp = [], mlp_dim * 2
        for size in mlp_layers:
            mods += [nn.Linear(inp, size), nn.ReLU()]
            inp = size
        self.mlp = nn.Sequential(*mods)
        self.output_layer = nn.Linear(gmf_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, item_idx):
        gmf = self.user_emb_gmf(user_idx) * self.item_emb_gmf(item_idx)
        mlp = self.mlp(torch.cat([self.user_emb_mlp(user_idx), self.item_emb_mlp(item_idx)], dim=-1))
        out = self.sigmoid(self.output_layer(torch.cat([gmf, mlp], dim=-1))).squeeze(-1)
        return out * (self.rating_max - self.rating_min) + self.rating_min


def train_neumf(users: np.ndarray, items: np.ndarray, ratings: np.ndarray, n_users: int, n_items: int,
                device: torch.device, seed: int, epochs: int = 15, batch_size: int = 256, lr: float = 1e-3,
                weight_decay: float = 1e-5, val: Optional[tuple] = None):
    """Entrena NeuMF como en el notebook (MSE, Adam). Si `val=(u, i, r)` registra el RMSE de validación por época."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    model = NeuMF(n_users, n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    U = torch.tensor(users, dtype=torch.long); I = torch.tensor(items, dtype=torch.long)
    R = torch.tensor(ratings, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(R), generator=gen)
        tot = 0.0
        for s in range(0, len(R), batch_size):
            idx = perm[s:s + batch_size]
            u, i, r = U[idx].to(device), I[idx].to(device), R[idx].to(device)
            opt.zero_grad()
            loss = crit(model(u, i), r)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        rec = {"epoch": ep, "train_mse": tot / len(R)}
        if val is not None:
            rec["val_rmse"] = rmse_neumf(model, *val, device)
        history.append(rec)
    return model, history


@torch.no_grad()
def rmse_neumf(model, users, items, ratings, device) -> float:
    model.eval()
    p = model(torch.tensor(users, dtype=torch.long, device=device), torch.tensor(items, dtype=torch.long, device=device))
    return float(torch.sqrt(torch.mean((p.cpu() - torch.tensor(ratings, dtype=torch.float32)) ** 2)))


@torch.no_grad()
def score_all_items_neumf(model, user_idx: Sequence[int], n_items: int, device) -> np.ndarray:
    model.eval()
    items = torch.arange(n_items, device=device)
    out = np.zeros((len(user_idx), n_items), dtype=np.float32)
    for k, u in enumerate(user_idx):
        out[k] = model(torch.full((n_items,), int(u), dtype=torch.long, device=device), items).cpu().numpy()
    return out
