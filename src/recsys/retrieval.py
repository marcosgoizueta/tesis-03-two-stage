from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from recsys.config import Stage3Config


def sample_negatives(
    user_idx_array: np.ndarray,
    user_seen_train: Dict[int, Set[int]],
    n_items: int,
    n_neg_per_pos: int = 4,
) -> np.ndarray:
    """Implementación de referencia idéntica al notebook: bucle en Python."""
    neg_items = np.empty((len(user_idx_array), n_neg_per_pos), dtype=np.int64)
    for row, u in enumerate(user_idx_array):
        seen = user_seen_train.get(int(u), set())
        count = 0
        while count < n_neg_per_pos:
            candidate = np.random.randint(0, n_items)
            if candidate not in seen:
                neg_items[row, count] = candidate
                count += 1
    return neg_items


def sample_negatives_fast(
    user_idx_array: np.ndarray,
    seen_mask_np: np.ndarray,
    n_items: int,
    n_neg_per_pos: int = 4,
) -> np.ndarray:
    """Implementación vectorizada opcional (altera el consumo del generador aleatorio)."""
    P = len(user_idx_array)
    neg = np.random.randint(0, n_items, size=(P, n_neg_per_pos), dtype=np.int64)
    bad = seen_mask_np[user_idx_array[:, None], neg]
    while bad.any():
        neg[bad] = np.random.randint(0, n_items, size=int(bad.sum()), dtype=np.int64)
        bad = seen_mask_np[user_idx_array[:, None], neg]
    return neg


class RetrievalDataset(Dataset):
    """Dataset para la torre de retrieval: (usuario, ítem_positivo, [ítems_negativos])."""

    def __init__(
        self,
        positive_df,
        user_seen_train: Dict[int, Set[int]],
        n_items: int,
        seen_mask_np: np.ndarray,
        n_neg_per_pos: int = 4,
        fast_sampling: bool = False,
    ):
        self.users = positive_df["user_idx"].values
        self.pos_items = positive_df["item_idx"].values
        self.user_seen_train = user_seen_train
        self.n_items = n_items
        self.seen_mask_np = seen_mask_np
        self.n_neg_per_pos = n_neg_per_pos
        self.fast_sampling = fast_sampling
        self.neg_items = self._sample()

    def _sample(self) -> np.ndarray:
        if self.fast_sampling:
            return sample_negatives_fast(
                self.users, self.seen_mask_np, self.n_items, self.n_neg_per_pos
            )
        return sample_negatives(
            self.users, self.user_seen_train, self.n_items, self.n_neg_per_pos
        )

    def resample_negatives(self) -> None:
        self.neg_items = self._sample()

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.pos_items[idx], dtype=torch.long),
            torch.tensor(self.neg_items[idx], dtype=torch.long),
        )


class TwoTower(nn.Module):
    """Modelo Two-Tower para Candidate Generation."""

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

    def forward_user(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_idx)

    def forward_item(self, item_idx: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_idx)

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.forward_user(user_idx)
        i = self.forward_item(item_idx)
        return (u * i).sum(dim=-1)


def train_retrieval(
    model: TwoTower,
    dataset: RetrievalDataset,
    cfg: Stage3Config,
    device: torch.device,
) -> List[float]:
    """Entrena el modelo TwoTower reproduciendo paso a paso la lógica original."""
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size_retrieval, shuffle=True
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr_retrieval, weight_decay=cfg.weight_decay_retrieval
    )

    losses = []
    epochs = 1 if cfg.quick else cfg.n_epochs_retrieval

    for epoch in range(1, epochs + 1):
        dataset.resample_negatives()
        model.train()
        epoch_loss = 0.0

        for users_batch, pos_items_batch, neg_items_batch in loader:
            users_batch = users_batch.to(device)
            pos_items_batch = pos_items_batch.to(device)
            neg_items_batch = neg_items_batch.to(device)

            optimizer.zero_grad()

            pos_scores = model.score(users_batch, pos_items_batch)
            pos_labels = torch.ones_like(pos_scores)

            users_expanded = (
                users_batch.unsqueeze(1)
                .expand(-1, neg_items_batch.size(1))
                .reshape(-1)
            )
            neg_items_flat = neg_items_batch.reshape(-1)
            neg_scores = model.score(users_expanded, neg_items_flat)
            neg_labels = torch.zeros_like(neg_scores)

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([pos_labels, neg_labels])

            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(users_batch)

        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs} - Retrieval BCE: {epoch_loss:.4f}")

    return losses


def compute_retrieval_candidates(
    model: TwoTower,
    n_users: int,
    n_items: int,
    seen_mask_np: np.ndarray,
    max_candidates: int,
    device: torch.device,
) -> np.ndarray:
    """Calcula la matriz de candidatos top-K ordenados para todos los usuarios."""
    model.eval()
    with torch.no_grad():
        all_users_t = torch.arange(n_users, dtype=torch.long, device=device)
        all_items_t = torch.arange(n_items, dtype=torch.long, device=device)
        user_embeds = model.forward_user(all_users_t)
        item_embeds = model.forward_item(all_items_t)
        retr_scores = user_embeds @ item_embeds.T
        retr_scores = retr_scores.masked_fill(
            torch.from_numpy(seen_mask_np).to(device), -float("inf")
        )
        retrieval_topk = (
            torch.topk(retr_scores, max_candidates, dim=1).indices.cpu().numpy()
        )
    return retrieval_topk


def compute_candidate_recalls(
    retrieval_topk: np.ndarray,
    test_relevant_by_user: Dict[int, Set[int]],
    eval_users: List[int],
    cutoffs: Tuple[int, ...] = (10, 50, 100, 200, 500),
) -> Dict[int, float]:
    """Calcula el recall de candidatos en los diferentes umbrales."""
    recalls = {}
    for n in cutoffs:
        cand_recall = float(
            np.mean(
                [
                    len(set(retrieval_topk[u, :n]) & test_relevant_by_user[u])
                    / len(test_relevant_by_user[u])
                    for u in eval_users
                ]
            )
        )
        recalls[n] = cand_recall
    return recalls

