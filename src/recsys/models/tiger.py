"""TIGER: T5-style encoder-decoder that generates the next item's Semantic ID (Rajput et al. 2023).

Decoding is trie-constrained beam search: at step d only codes that extend some existing item's Semantic ID
prefix are allowed, so every finished beam is a valid item. The decoder is re-run on the full (<= 4 token)
prefix at each step; with 4 steps this is cheaper to get right than a KV cache. use_cache=False matters: HF T5
otherwise returns cross-attention keys/values for every (row, beam), which on MPS peaked at ~20 GB per batch of
256 users x 30 beams (vs ~5 GB without; identical outputs).
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from recsys.data.tiger_data import EOS, FIRST_CODE_TOKEN, PAD


def make_tiger(vocab_size: int, num_layers: int = 4, num_decoder_layers: int = 4, num_heads: int = 6,
               d_kv: int = 64, d_model: int = 128, d_ff: int = 1024, dropout: float = 0.1) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=vocab_size, d_model=d_model, d_kv=d_kv, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers,
        num_decoder_layers=num_decoder_layers, dropout_rate=dropout, feed_forward_proj="relu",
        pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD,
    )
    return T5ForConditionalGeneration(config)


class SemanticIDTrie:
    """Valid Semantic ID prefixes. Prefixes of length d are int64 keys in base codebook_size.

    keys[d]: sorted unique keys of the length-d prefixes (keys[0] = [0], the empty prefix)
    allowed[d]: (len(keys[d]), codebook_size) bool, which code may follow each prefix
    item_of: item ID of each full Semantic ID, aligned with keys[num_positions]
    items: optional item IDs to include (default all 1..N), e.g. only the seen items in the cold-start setting
    """

    def __init__(self, item_sids: np.ndarray, codebook_size: int = 256, device: str | torch.device = "cpu",
                 items: np.ndarray | None = None):
        ids = np.arange(1, len(item_sids)) if items is None else np.unique(np.asarray(items, dtype=np.int64))
        sids = np.asarray(item_sids, dtype=np.int64)[ids]  # row 0 = padding, never included
        self.codebook_size, self.num_positions = codebook_size, sids.shape[1]
        self.keys, self.allowed = [], []
        prefix_keys = np.zeros(len(sids), dtype=np.int64)
        for d in range(self.num_positions):
            keys, inverse = np.unique(prefix_keys, return_inverse=True)
            allowed = np.zeros((len(keys), codebook_size), dtype=bool)
            allowed[inverse, sids[:, d]] = True
            self.keys.append(torch.as_tensor(keys, device=device))
            self.allowed.append(torch.as_tensor(allowed, device=device))
            prefix_keys = prefix_keys * codebook_size + sids[:, d]
        order = np.argsort(prefix_keys)
        if len(np.unique(prefix_keys)) != len(prefix_keys):
            raise ValueError("Semantic IDs are not unique")
        self.full_keys = torch.as_tensor(prefix_keys[order], device=device)
        self.item_of = torch.as_tensor(ids[order], device=device)

    @staticmethod
    def _lookup(keys: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Index of each query in sorted keys, and whether it is present."""
        idx = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
        return idx, keys[idx] == query

    def allowed_next(self, depth: int, prefixes: torch.Tensor) -> torch.Tensor:
        idx, found = self._lookup(self.keys[depth], prefixes)
        return self.allowed[depth][idx] & found[..., None]

    def items(self, prefixes: torch.Tensor) -> torch.Tensor:
        """Item ID of each full prefix, -1 if it is not a valid Semantic ID."""
        idx, found = self._lookup(self.full_keys, prefixes)
        return torch.where(found, self.item_of[idx], torch.full_like(prefixes, -1))


@torch.no_grad()
def constrained_beam_search(model: T5ForConditionalGeneration, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            trie: SemanticIDTrie, beam_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`beam_size` valid items per row by sequence log-probability: (items (B, K), scores (B, K)).

    Items are -1 where fewer than K valid Semantic IDs could be reached (score -inf).
    """
    B, k_codes = input_ids.shape[0], trie.codebook_size
    enc = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    prefixes = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
    scores = torch.zeros(B, 1, device=input_ids.device)
    dec = torch.full((B, 1, 1), PAD, dtype=torch.long, device=input_ids.device)  # decoder start token
    for d in range(trie.num_positions):
        nb = prefixes.shape[1]
        out = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc.repeat_interleave(nb, 0)),
                    attention_mask=attention_mask.repeat_interleave(nb, 0), decoder_input_ids=dec.view(B * nb, d + 1),
                    use_cache=False)
        first = FIRST_CODE_TOKEN + d * k_codes
        logp = F.log_softmax(out.logits[:, -1].float(), dim=-1)[:, first:first + k_codes].view(B, nb, k_codes)
        logp = logp.masked_fill(~trie.allowed_next(d, prefixes), -torch.inf)
        top, idx = (scores[..., None] + logp).view(B, -1).topk(min(beam_size, nb * k_codes), dim=1)
        beam, code = idx // k_codes, idx % k_codes
        prefixes = prefixes.gather(1, beam) * k_codes + code
        dec = torch.cat([dec.gather(1, beam[..., None].expand(-1, -1, d + 1)), (first + code)[..., None]], dim=2)
        scores = top
    items = trie.items(prefixes)
    return torch.where(torch.isfinite(scores), items, torch.full_like(items, -1)), scores
