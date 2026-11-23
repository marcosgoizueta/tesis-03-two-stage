"""Modelos generativos basados en T5 entrenados desde cero para la Etapa 5 (G1, G2 y G3)."""

from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

from recsys.stage5.tokenizer import GenerativeTokenizer
from recsys.stage5.constrained_decoding import ItemTrie


class T5Recommender(nn.Module):
    """Recomendador generativo encoder-decoder tipo T5 entrenado desde cero."""

    def __init__(
        self,
        tokenizer: GenerativeTokenizer,
        d_model: int = 128,
        d_kv: int = 64,
        d_ff: int = 1024,
        num_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 6,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=32,
            dropout_rate=dropout_rate,
            feed_forward_proj="relu",
            decoder_start_token_id=tokenizer.BOS_TOKEN_ID,
            pad_token_id=tokenizer.PAD_TOKEN_ID,
            eos_token_id=tokenizer.EOS_TOKEN_ID,
        )
        self.model = T5ForConditionalGeneration(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def recommend_for_user(
        self,
        history_item_indices: List[int],
        item_trie: ItemTrie,
        seen_items: Set[int],
        device: torch.device,
        num_beams: int = 50,
        top_k: int = 50,
        max_context_tokens: int = 200,
    ) -> List[int]:
        """Genera recomendaciones top-K para un usuario con decodificación restringida y beam search."""
        self.eval()
        input_tokens = self.tokenizer.encode_input_sequence(
            history_item_indices, max_tokens=max_context_tokens
        )
        if not input_tokens:
            return []

        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        allowed_fn = item_trie.build_user_allowed_fn(seen_items)

        max_new_tokens = 6 if self.tokenizer.variant in {"G1", "G2"} else 3
        num_return = min(num_beams, top_k)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            prefix_allowed_tokens_fn=allowed_fn,
            early_stopping=True,
        )

        recs: List[int] = []
        for beam_seq in outputs:
            tokens_list = beam_seq.tolist()
            item = self.tokenizer.decode_target_tokens(tokens_list)
            if item is not None and item not in seen_items and item not in recs:
                recs.append(item)
            if len(recs) >= top_k:
                break

        return recs
