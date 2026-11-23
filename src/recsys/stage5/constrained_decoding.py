"""Decodificación restringida mediante Trie (Prefix Tree) para la Etapa 5.
Garantiza matemáticamente 0% de alucinaciones y 0% de ítems vistos en las recomendaciones.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from recsys.stage5.tokenizer import GenerativeTokenizer


class TrieNode:
    """Nodo del árbol de prefijos para secuencias discretas de recomendación."""

    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.item_idx: Optional[int] = None
        self.is_leaf: bool = False


class ItemTrie:
    """Árbol de prefijos que almacena las secuencias válidas de tokens de todos los ítems del catálogo."""

    def __init__(self, tokenizer: GenerativeTokenizer):
        self.tokenizer = tokenizer
        self.root = TrieNode()
        self._build_catalog_trie()

    def _build_catalog_trie(self) -> None:
        """Inserta en el Trie todas las secuencias válidas de ítems del catálogo."""
        for item_idx in range(self.tokenizer.n_items):
            tokens = self.tokenizer.encode_target_item(item_idx)
            curr = self.root
            for tok in tokens:
                if tok not in curr.children:
                    curr.children[tok] = TrieNode()
                curr = curr.children[tok]
            curr.is_leaf = True
            curr.item_idx = item_idx

    def build_user_allowed_fn(self, seen_items: Set[int]):
        """Construye una función `prefix_allowed_tokens_fn` para HuggingFace generate().
        
        Args:
            seen_items: Conjunto de item_idx que el usuario ya consumió y no deben recomendarse.
        """
        # Prefiltrar los caminos válidos que no terminan en un ítem visto
        # Para eficiencia en memoria y CPU, navegamos el Trie principal comprobando si
        # la rama conduce a al menos un ítem no visto.
        valid_items = set(range(self.tokenizer.n_items)) - seen_items
        eos_id = self.tokenizer.EOS_TOKEN_ID
        bos_id = self.tokenizer.BOS_TOKEN_ID

        def prefix_allowed_tokens_fn(batch_id: int, input_ids_tensor) -> List[int]:
            # input_ids_tensor contiene la secuencia generada en el decoder hasta ahora
            # (generalmente inicia con decoder_start_token_id / BOS)
            tokens = input_ids_tensor.tolist()
            # Descartar token de inicio si está presente
            clean_prefix = [t for t in tokens if t not in {self.tokenizer.PAD_TOKEN_ID, bos_id}]

            curr = self.root
            for tok in clean_prefix:
                if tok in curr.children:
                    curr = curr.children[tok]
                else:
                    # Prefijo inválido fuera de catálogo
                    return [eos_id]

            if curr.is_leaf:
                # Ya completó la tupla de ítem (ej. 4 tokens en G1 o 1 en G3) -> forzar EOS
                return [eos_id]

            # Recolectar hijos que conduzcan a al menos un ítem no visto
            allowed = []
            for child_tok, child_node in curr.children.items():
                if self._node_has_unseen_leaf(child_node, seen_items):
                    allowed.append(child_tok)

            if not allowed:
                return [eos_id]
            return allowed

        return prefix_allowed_tokens_fn

    def _node_has_unseen_leaf(self, node: TrieNode, seen_items: Set[int]) -> bool:
        """Verifica recursivamente si el sub-árbol contiene al menos un ítem no visto."""
        if node.is_leaf:
            return node.item_idx not in seen_items
        for child in node.children.values():
            if self._node_has_unseen_leaf(child, seen_items):
                return True
        return False
