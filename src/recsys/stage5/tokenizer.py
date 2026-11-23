"""Tokenizadores para modelos generativos de la Etapa 5 (G1, G2 y G3)."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class GenerativeTokenizer:
    """Tokenizador que gestiona el vocabulario y la codificación de secuencias para G1, G2 y G3.
    
    Tokens especiales:
    - 0: <pad>
    - 1: <s> (decoder_start_token)
    - 2: </s> (eos)
    - 3: <unk>
    """

    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3
    N_SPECIAL = 4

    def __init__(
        self,
        variant: str,
        n_items: int = 9724,
        semantic_id_map: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        random_id_map: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
    ):
        self.variant = variant.upper()
        self.n_items = n_items

        if self.variant == "G1":
            if semantic_id_map is None:
                raise ValueError("G1 requiere semantic_id_map (item_idx -> (c1, c2, c3, d)).")
            self.id_map = semantic_id_map
            # Rangos de tokens discretizados:
            # c1: 4 .. 259 (256 códigos)
            # c2: 260 .. 515 (256 códigos)
            # c3: 516 .. 771 (256 códigos)
            # d:  772 .. 787 (16 tokens)
            self.vocab_size = self.N_SPECIAL + 256 + 256 + 256 + 16 # 788
            self.c1_offset = self.N_SPECIAL
            self.c2_offset = self.c1_offset + 256
            self.c3_offset = self.c2_offset + 256
            self.d_offset = self.c3_offset + 256

        elif self.variant == "G2":
            if random_id_map is None:
                raise ValueError("G2 requiere random_id_map (item_idx -> (r1, r2, r3, d)).")
            self.id_map = random_id_map
            self.vocab_size = self.N_SPECIAL + 256 + 256 + 256 + 16
            self.c1_offset = self.N_SPECIAL
            self.c2_offset = self.c1_offset + 256
            self.c3_offset = self.c2_offset + 256
            self.d_offset = self.c3_offset + 256

        elif self.variant == "G3":
            # Token atómico por película: 4 .. 4 + n_items - 1
            self.vocab_size = self.N_SPECIAL + n_items
            self.item_offset = self.N_SPECIAL
            self.id_map = {}

        else:
            raise ValueError(f"Variante desconocida: {variant}. Opciones válidas: G1, G2, G3.")

        # Mapeo inverso: secuencia de tokens target -> item_idx
        self.target_tokens_to_item: Dict[Tuple[int, ...], int] = {}
        for item_idx in range(n_items):
            tgt = tuple(self.encode_target_item(item_idx))
            self.target_tokens_to_item[tgt] = item_idx

    def encode_target_item(self, item_idx: int) -> List[int]:
        """Convierte un ítem en su secuencia target para el decoder (sin tokens especiales)."""
        if self.variant in {"G1", "G2"}:
            c1, c2, c3, d = self.id_map[item_idx]
            return [
                self.c1_offset + c1,
                self.c2_offset + c2,
                self.c3_offset + c3,
                self.d_offset + d,
            ]
        else: # G3
            return [self.item_offset + item_idx]

    def encode_input_sequence(self, item_sequence: List[int], max_tokens: int = 200) -> List[int]:
        """Convierte una lista cronológica de ítems en una secuencia continua de tokens para el encoder."""
        tokens = []
        for item_idx in item_sequence:
            tokens.extend(self.encode_target_item(item_idx))
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
        return tokens

    def decode_target_tokens(self, tokens: Union[List[int], Tuple[int, ...]]) -> Optional[int]:
        """Decodifica una secuencia de tokens emitida por el decoder al item_idx original."""
        # Limpiar tokens especiales como BOS o EOS
        clean = [t for t in tokens if t not in {self.PAD_TOKEN_ID, self.BOS_TOKEN_ID, self.EOS_TOKEN_ID}]
        return self.target_tokens_to_item.get(tuple(clean), None)


def build_random_id_map(
    semantic_id_map: Dict[int, Tuple[int, int, int, int]],
    seed: int = 42,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Construye un mapeo para G2 permutando aleatoriamente las asignaciones de G1.
    
    Preserva exactamente la misma distribución marginal y correlacional de colisiones y frecuencias,
    aislando rigurosamente el impacto del contenido semántico textual del sesgo estructural de árbol.
    """
    rng = np.random.RandomState(seed)
    items = list(semantic_id_map.keys())
    tuples = list(semantic_id_map.values())

    perm = rng.permutation(len(tuples))
    return {items[i]: tuples[perm[i]] for i in range(len(items))}
