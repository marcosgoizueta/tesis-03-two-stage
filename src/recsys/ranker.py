from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from recsys.config import Stage3Config


class Ranker(nn.Module):
    """Modelo Ranker (MLP) con características de contenido y embeddings."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_genre: np.ndarray,
        item_genre: np.ndarray,
        item_logpop: np.ndarray,
        embed_dim: int = 32,
        hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.1)

        self.register_buffer("user_genre", torch.tensor(user_genre, dtype=torch.float32))
        self.register_buffer("item_genre", torch.tensor(item_genre, dtype=torch.float32))
        self.register_buffer("item_logpop", torch.tensor(item_logpop, dtype=torch.float32))

        g = item_genre.shape[1]
        in_dim = 3 * embed_dim + 3 * g + 1
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        ue, ie = self.user_emb(user_idx), self.item_emb(item_idx)
        ug, ig = self.user_genre[user_idx], self.item_genre[item_idx]
        lp = self.item_logpop[item_idx].unsqueeze(-1)
        x = torch.cat([ue, ie, ue * ie, ug, ig, ug * ig, lp], dim=-1)
        return self.mlp(x).squeeze(-1)


def sample_ranker_negatives(
    pos_users: np.ndarray,
    n_uniform: int,
    n_hard: int,
    n_items: int,
    seen_mask_np: np.ndarray,
    retrieval_topk: np.ndarray,
    hard_pool: int = 100,
) -> np.ndarray:
    """Muestreo de negativos para el ranker (uniformes y/o difíciles del retrieval)."""
    P = len(pos_users)
    parts = []
    if n_uniform > 0:
        neg = np.random.randint(0, n_items, size=(P, n_uniform))
        bad = seen_mask_np[pos_users[:, None], neg]
        while bad.any():
            neg[bad] = np.random.randint(0, n_items, size=int(bad.sum()))
            bad = seen_mask_np[pos_users[:, None], neg]
        parts.append(neg)
    if n_hard > 0:
        cols = np.random.randint(0, hard_pool, size=(P, n_hard))
        parts.append(retrieval_topk[pos_users[:, None], cols])
    return np.concatenate(parts, axis=1)


def train_ranker(
    pos_users: np.ndarray,
    pos_items: np.ndarray,
    n_users: int,
    n_items: int,
    user_genre_np: np.ndarray,
    item_genre_np: np.ndarray,
    item_logpop_np: np.ndarray,
    seen_mask_np: np.ndarray,
    retrieval_topk: np.ndarray,
    n_uniform: int,
    n_hard: int,
    cfg: Stage3Config,
    device: torch.device,
    seed: int,
) -> Tuple[Ranker, List[float]]:
    """Entrena el modelo Ranker reproduciendo el reinicio de semilla local."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = Ranker(
        n_users=n_users,
        n_items=n_items,
        user_genre=user_genre_np,
        item_genre=item_genre_np,
        item_logpop=item_logpop_np,
        embed_dim=cfg.embed_dim_ranker,
        hidden=cfg.hidden_ranker,
        dropout=cfg.dropout_ranker,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr_ranker, weight_decay=cfg.weight_decay_ranker
    )
    criterion = nn.BCEWithLogitsLoss()

    k = n_uniform + n_hard
    losses = []
    epochs = 1 if cfg.quick else cfg.n_epochs_ranker

    for epoch in range(1, epochs + 1):
        neg = sample_ranker_negatives(
            pos_users=pos_users,
            n_uniform=n_uniform,
            n_hard=n_hard,
            n_items=n_items,
            seen_mask_np=seen_mask_np,
            retrieval_topk=retrieval_topk,
            hard_pool=cfg.hard_pool,
        )

        u_all = np.concatenate([pos_users, np.repeat(pos_users, k)])
        i_all = np.concatenate([pos_items, neg.reshape(-1)])
        y_all = np.concatenate(
            [np.ones(len(pos_users)), np.zeros(len(pos_users) * k)]
        ).astype(np.float32)

        u_t = torch.tensor(u_all, dtype=torch.long, device=device)
        i_t = torch.tensor(i_all, dtype=torch.long, device=device)
        y_t = torch.tensor(y_all, dtype=torch.float32, device=device)
        perm = torch.randperm(len(y_t), device=device)

        model.train()
        total = 0.0
        for s in range(0, len(perm), cfg.batch_size_ranker):
            idx = perm[s : s + cfg.batch_size_ranker]
            optimizer.zero_grad()
            loss = criterion(model(u_t[idx], i_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        losses.append(total / len(perm))
        print(f"Epoch {epoch}/{epochs} - Ranker BCE: {losses[-1]:.4f}")

    model.eval()
    return model, losses

