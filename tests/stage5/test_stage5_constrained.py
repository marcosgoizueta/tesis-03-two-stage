"""Tests para la decodificación restringida mediante Trie."""

import torch
from recsys.stage5.tokenizer import GenerativeTokenizer
from recsys.stage5.constrained_decoding import ItemTrie


def test_trie_blocks_seen_items_and_invalid_tokens():
    n_items = 10
    semantic_map = {i: (0, 0, i, 0) for i in range(n_items)}
    tok = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_map)
    trie = ItemTrie(tok)

    # Usuario ya vio el ítem 3
    seen = {3}
    allowed_fn = trie.build_user_allowed_fn(seen)

    # Nivel 1 (c1)
    allowed_l1 = allowed_fn(0, torch.tensor([1])) # después de BOS (1)
    assert tok.c1_offset in allowed_l1

    # Nivel 2 (c2)
    allowed_l2 = allowed_fn(0, torch.tensor([1, tok.c1_offset]))
    assert tok.c2_offset in allowed_l2

    # Nivel 3 (c3): deben estar todos menos el correspondiente a ítem 3 (tok.c3_offset + 3)
    allowed_l3 = allowed_fn(0, torch.tensor([1, tok.c1_offset, tok.c2_offset]))
    assert (tok.c3_offset + 3) not in allowed_l3
    assert (tok.c3_offset + 0) in allowed_l3


def test_trie_forces_eos_after_leaf():
    n_items = 5
    tok = GenerativeTokenizer("G3", n_items=n_items)
    trie = ItemTrie(tok)

    seen = {0}
    allowed_fn = trie.build_user_allowed_fn(seen)

    # Tras generar el token atómico del ítem 1, debe permitir SOLO EOS (2)
    allowed_eos = allowed_fn(0, torch.tensor([1, tok.item_offset + 1]))
    assert allowed_eos == [tok.EOS_TOKEN_ID]
