"""Tests para los tokenizadores de G1, G2 y G3."""

import pytest
from recsys.stage5.tokenizer import GenerativeTokenizer, build_random_id_map


def test_tokenizer_g1_and_g2_roundtrip():
    n_items = 100
    semantic_map = {i: (i % 256, (i * 2) % 256, (i * 3) % 256, i % 16) for i in range(n_items)}
    tok_g1 = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_map)

    # Encode and decode roundtrip
    for item in range(n_items):
        target = tok_g1.encode_target_item(item)
        assert len(target) == 4
        decoded = tok_g1.decode_target_tokens(target)
        assert decoded == item

    # G2
    random_map = build_random_id_map(semantic_map, seed=42)
    tok_g2 = GenerativeTokenizer("G2", n_items=n_items, random_id_map=random_map)
    for item in range(n_items):
        target = tok_g2.encode_target_item(item)
        assert len(target) == 4
        decoded = tok_g2.decode_target_tokens(target)
        assert decoded == item


def test_tokenizer_g3_atomic():
    n_items = 500
    tok_g3 = GenerativeTokenizer("G3", n_items=n_items)
    assert tok_g3.vocab_size == 4 + n_items

    for item in range(n_items):
        target = tok_g3.encode_target_item(item)
        assert len(target) == 1
        decoded = tok_g3.decode_target_tokens(target)
        assert decoded == item
