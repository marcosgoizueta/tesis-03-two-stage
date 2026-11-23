"""Tests para el modelo T5Recommender."""

import torch
from recsys.stage5.tokenizer import GenerativeTokenizer
from recsys.stage5.constrained_decoding import ItemTrie
from recsys.stage5.models import T5Recommender


def test_t5_recommender_forward_and_recommend():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_items = 20
    semantic_map = {i: (0, 0, i, 0) for i in range(n_items)}
    tok = GenerativeTokenizer("G1", n_items=n_items, semantic_id_map=semantic_map)
    trie = ItemTrie(tok)

    model = T5Recommender(
        tokenizer=tok,
        d_model=64,
        d_kv=32,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
    ).to(device)

    # Paso forward con labels
    input_ids = torch.tensor([[10, 20, 30]], device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[tok.c1_offset, tok.c2_offset, tok.c3_offset, tok.d_offset]], device=device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert out.loss.dim() == 0

    # Inferencia con decodificación restringida
    seen = {1, 2}
    history = [0, 1, 2]
    recs = model.recommend_for_user(
        history_item_indices=history,
        item_trie=trie,
        seen_items=seen,
        device=device,
        num_beams=5,
        top_k=3,
    )

    # Las recomendaciones deben ser ítems válidos y NO estar en seen
    for r in recs:
        assert 0 <= r < n_items
        assert r not in seen
