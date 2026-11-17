"""Arquitecturas de la torre de ítems para la Etapa 4 (Variantes A, B, C, C2, D y E)."""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from recsys.config import Stage3Config
from recsys.retrieval import RetrievalDataset


class ContentMLP(nn.Module):
    """Red densa para proyectar características de contenido a 32 dimensiones.
    Arquitectura: Linear(in_features, 64) -> ReLU -> Linear(64, 32).
    Se inicializa usando un torch.Generator propio para no alterar el RNG global de PyTorch.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, out_dim: int = 32, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        if generator is not None:
            self._init_weights_with_generator(generator)

    def _init_weights_with_generator(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            std1 = math.sqrt(2.0 / self.fc1.weight.shape[1])
            self.fc1.weight.normal_(mean=0.0, std=std1, generator=generator)
            if self.fc1.bias is not None:
                self.fc1.bias.zero_()

            std2 = math.sqrt(2.0 / self.fc2.weight.shape[1])
            self.fc2.weight.normal_(mean=0.0, std=std2, generator=generator)
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TwoTowerStage4(nn.Module):
    """Modelo Two-Tower generalizado para la Etapa 4 con 6 variantes de torre de ítems:
    - 'A': Solo ID (reproducción exacta de Etapa 3).
    - 'B': Solo contenido de texto (texto fijo -> ContentMLP).
    - 'C': Híbrida (ID + salida de ContentMLP sobre texto).
    - 'C2': Híbrida con ID Dropout (p=0.5 en train; en eval anula ID solo para ítems con 0 interacciones en train).
    - 'D': Control sin LLM con ID (ID + salida de ContentMLP sobre géneros multi-hot).
    - 'E': Control de B sin ID (Solo usuario + salida de ContentMLP sobre géneros multi-hot).
    """

    def __init__(
        self,
        variant: str,
        n_users: int,
        n_items: int,
        embed_dim: int = 32,
        text_embeddings_np: Optional[np.ndarray] = None,
        item_genres_np: Optional[np.ndarray] = None,
        is_cold_mask_np: Optional[np.ndarray] = None,
        random_text_embeddings_np: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.variant = variant.upper()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.is_cold_mask_np = is_cold_mask_np

        valid_variants = {"A", "B", "C", "C2", "D", "D2", "E", "R"}
        if self.variant not in valid_variants:
            raise ValueError(f"Variante desconocida: {variant}. Opciones válidas: {valid_variants}")

        # 1. Inicialización de embeddings por ID
        # REGLA CLAVE: Para garantizar reproducibilidad idéntica a la Etapa 3,
        # los embeddings por ID se inicializan con el RNG global de PyTorch en el orden exacto:
        # primero instanciación de user e item, luego normal_ de user e item.
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        if self.variant in {"A", "C", "C2", "D", "D2"}:
            self.item_id_embedding = nn.Embedding(n_items, embed_dim)
            dummy_item_embedding = None
        else:
            # En variantes sin ID de ítem (B, E y R), consumimos el RNG global de la misma forma
            # para que el estado del RNG global después del init sea 100% idéntico al de A, C, C2, D y D2.
            self.item_id_embedding = None
            dummy_item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        if self.variant in {"A", "C", "C2", "D", "D2"}:
            nn.init.normal_(self.item_id_embedding.weight, mean=0.0, std=0.1)
        else:
            nn.init.normal_(dummy_item_embedding.weight, mean=0.0, std=0.1)

        # 2. Generador aleatorio propio para redes de contenido y dropout (desacoplado del RNG global)
        content_generator = torch.Generator().manual_seed(seed)
        self.id_dropout_generator = torch.Generator().manual_seed(seed + 9999)

        # Para asegurar que la instanciación interna de nn.Linear no consuma números del RNG global,
        # guardamos y restauramos el estado del RNG global de PyTorch.
        global_rng_state = torch.get_rng_state()
        global_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        # 3. Componentes específicos de cada variante
        if self.variant in {"B", "C", "C2"}:
            if text_embeddings_np is None:
                raise ValueError(f"La variante {self.variant} requiere text_embeddings_np.")
            text_dim = text_embeddings_np.shape[1]
            self.register_buffer(
                "fixed_text_embeddings",
                torch.tensor(text_embeddings_np, dtype=torch.float32),
            )
            self.content_net = ContentMLP(
                in_features=text_dim,
                hidden_dim=64,
                out_dim=embed_dim,
                generator=content_generator,
            )
        elif self.variant == "R":
            # Variante R: Control con embeddings aleatorios normalizados (fijos por ítem)
            if random_text_embeddings_np is not None:
                rand_embs = torch.tensor(random_text_embeddings_np, dtype=torch.float32)
            else:
                r_gen = torch.Generator().manual_seed(42)
                raw_r = torch.randn(n_items, 384, generator=r_gen)
                rand_embs = nn.functional.normalize(raw_r, p=2, dim=-1)
            self.register_buffer("fixed_text_embeddings", rand_embs)
            self.content_net = ContentMLP(
                in_features=384,
                hidden_dim=64,
                out_dim=embed_dim,
                generator=content_generator,
            )
        else:
            self.content_net = None

        if self.variant in {"D", "D2", "E"}:
            if item_genres_np is None:
                raise ValueError(f"La variante {self.variant} requiere item_genres_np.")
            genre_dim = item_genres_np.shape[1]
            self.register_buffer(
                "fixed_genre_features",
                torch.tensor(item_genres_np, dtype=torch.float32),
            )
            self.genre_net = ContentMLP(
                in_features=genre_dim,
                hidden_dim=64,
                out_dim=embed_dim,
                generator=content_generator,
            )
        else:
            self.genre_net = None

        # Máscara de ítems fríos (0 interacciones en train) para C2 y D2 en evaluación
        if self.variant in {"C2", "D2"}:
            if is_cold_mask_np is not None:
                self.register_buffer(
                    "is_cold_mask",
                    torch.tensor(is_cold_mask_np, dtype=torch.bool),
                )
            else:
                self.register_buffer(
                    "is_cold_mask",
                    torch.zeros(n_items, dtype=torch.bool),
                )
        else:
            self.is_cold_mask = None

        # Restauramos el estado del RNG global para dejarlo idéntico a la variante A
        torch.set_rng_state(global_rng_state)
        if global_cuda_state is not None:
            torch.cuda.set_rng_state_all(global_cuda_state)

    def forward_user(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_idx)

    def forward_item(self, item_idx: torch.Tensor) -> torch.Tensor:
        if self.variant == "A":
            return self.item_id_embedding(item_idx)

        elif self.variant in {"B", "R"}:
            raw_text = self.fixed_text_embeddings[item_idx]
            return self.content_net(raw_text)

        elif self.variant == "C":
            id_emb = self.item_id_embedding(item_idx)
            raw_text = self.fixed_text_embeddings[item_idx]
            text_proj = self.content_net(raw_text)
            return id_emb + text_proj

        elif self.variant == "C2":
            id_emb = self.item_id_embedding(item_idx)
            raw_text = self.fixed_text_embeddings[item_idx]
            text_proj = self.content_net(raw_text)

            if self.training:
                # ID Dropout: para cada ítem de cada muestra, con probabilidad 0.5 se anula el embedding por ID
                # Generamos en CPU con el generador local determinista y transferimos al dispositivo
                mask_prob = torch.rand(item_idx.shape, generator=self.id_dropout_generator).to(item_idx.device)
                drop_mask = (mask_prob >= 0.5).float().unsqueeze(-1)
                return id_emb * drop_mask + text_proj
            else:
                # En evaluación: se usa completo, salvo para ítems sin interacciones en train (is_cold_mask=True),
                # cuyo embedding por ID se anula
                cold_flags = self.is_cold_mask[item_idx].unsqueeze(-1)
                eval_mask = (~cold_flags).float()
                return id_emb * eval_mask + text_proj

        elif self.variant == "D":
            id_emb = self.item_id_embedding(item_idx)
            raw_genre = self.fixed_genre_features[item_idx]
            genre_proj = self.genre_net(raw_genre)
            return id_emb + genre_proj

        elif self.variant == "D2":
            id_emb = self.item_id_embedding(item_idx)
            raw_genre = self.fixed_genre_features[item_idx]
            genre_proj = self.genre_net(raw_genre)

            if self.training:
                # ID Dropout: para cada ítem de cada muestra, con probabilidad 0.5 se anula el embedding por ID
                mask_prob = torch.rand(item_idx.shape, generator=self.id_dropout_generator).to(item_idx.device)
                drop_mask = (mask_prob >= 0.5).float().unsqueeze(-1)
                return id_emb * drop_mask + genre_proj
            else:
                # En evaluación: se usa completo, salvo para ítems sin interacciones en train (is_cold_mask=True),
                # cuyo embedding por ID se anula
                cold_flags = self.is_cold_mask[item_idx].unsqueeze(-1)
                eval_mask = (~cold_flags).float()
                return id_emb * eval_mask + genre_proj

        elif self.variant == "E":
            raw_genre = self.fixed_genre_features[item_idx]
            return self.genre_net(raw_genre)

        raise RuntimeError(f"Variante no soportada: {self.variant}")

    def forward_all_items(
        self,
        zero_cold_id: bool = False,
        is_cold_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calcula el vector de representación de TODOS los ítems del catálogo para retrieval.
        Si zero_cold_id=True (ej. C-cero), anula el embedding por ID de los ítems con 0 interacciones en train.
        """
        all_indices = torch.arange(self.n_items, device=self.user_embedding.weight.device)
        if not zero_cold_id or self.variant not in {"C", "D"}:
            return self.forward_item(all_indices)

        # Modo C-cero: anulación de ID para ítems fríos
        mask = is_cold_mask
        if mask is None:
            if hasattr(self, "is_cold_mask") and self.is_cold_mask is not None:
                mask = self.is_cold_mask
            elif hasattr(self, "is_cold_mask_np") and self.is_cold_mask_np is not None:
                mask = torch.tensor(self.is_cold_mask_np, dtype=torch.bool, device=all_indices.device)

        if mask is not None:
            cold_flags = mask.unsqueeze(-1)
            eval_mask = (~cold_flags).float()
        else:
            eval_mask = 1.0

        id_emb = self.item_id_embedding(all_indices) * eval_mask
        if self.variant == "C":
            raw_text = self.fixed_text_embeddings[all_indices]
            text_proj = self.content_net(raw_text)
            return id_emb + text_proj
        elif self.variant == "D":
            raw_genre = self.fixed_genre_features[all_indices]
            genre_proj = self.genre_net(raw_genre)
            return id_emb + genre_proj

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.forward_user(user_idx)
        i = self.forward_item(item_idx)
        return (u * i).sum(dim=-1)

    def count_parameters(self) -> Dict[str, int]:
        """Calcula el número de parámetros entrenables, no entrenables (buffers) y totales."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for b in self.buffers())
        total = trainable + buffers
        return {
            "trainable_params": trainable,
            "buffer_params": buffers,
            "total_params": total,
        }

    def compute_component_norms(self) -> Dict[str, float]:
        """Mide la norma L2 media de la contribución por ID y por contenido sobre todos los ítems."""
        if self.variant not in {"C", "C2", "D", "D2"}:
            return {}
        with torch.no_grad():
            all_idx = torch.arange(self.n_items, device=self.user_embedding.weight.device)
            id_embs = self.item_id_embedding(all_idx)
            norm_id = float(torch.norm(id_embs, p=2, dim=-1).mean().item())

            if self.variant in {"C", "C2"}:
                raw_text = self.fixed_text_embeddings[all_idx]
                content_embs = self.content_net(raw_text)
                norm_content = float(torch.norm(content_embs, p=2, dim=-1).mean().item())
            elif self.variant in {"D", "D2"}:
                raw_genre = self.fixed_genre_features[all_idx]
                content_embs = self.genre_net(raw_genre)
                norm_content = float(torch.norm(content_embs, p=2, dim=-1).mean().item())

            return {
                "mean_norm_id": norm_id,
                "mean_norm_content": norm_content,
                "ratio_id_to_content": norm_id / max(norm_content, 1e-8),
            }


def train_retrieval_stage4(
    model: TwoTowerStage4,
    dataset: RetrievalDataset,
    cfg: Stage3Config,
    device: torch.device,
) -> List[float]:
    """Entrenamiento de la torre de retrieval con hiperparámetros idénticos a la Etapa 3:
    30 épocas, batch size 256, Adam lr 5e-3, weight decay 1e-6.
    """
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

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

    return losses


def compute_retrieval_candidates_stage4(
    model: TwoTowerStage4,
    n_users: int,
    n_items: int,
    seen_mask_np: np.ndarray,
    max_candidates: int,
    device: torch.device,
    override_item_embs: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Genera candidatos para todos los usuarios en GPU/CPU en bloques.
    Si override_item_embs no es None, utiliza esos vectores de ítem (ej. para evaluación C-cero).
    """
    model.eval()
    with torch.no_grad():
        all_user_idx = torch.arange(n_users, device=device)
        user_embs = model.forward_user(all_user_idx)  # (n_users, 32)
        if override_item_embs is not None:
            item_embs = override_item_embs
        else:
            item_embs = model.forward_all_items()        # (n_items, 32)

        batch_size = 128
        retrieval_candidates = []

        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            u_batch = user_embs[start:end]           # (B, 32)
            scores = torch.matmul(u_batch, item_embs.T) # (B, n_items)

            # Enmascarar ítems ya vistos en train
            seen_chunk = torch.tensor(
                seen_mask_np[start:end], dtype=torch.bool, device=device
            )
            scores[seen_chunk] = -1e9

            _, top_items = torch.topk(scores, k=max_candidates, dim=1)
            retrieval_candidates.append(top_items.cpu().numpy())

        return np.concatenate(retrieval_candidates, axis=0)
