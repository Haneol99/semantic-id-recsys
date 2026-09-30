"""Token sequences for TIGER.

Vocabulary: 0 = pad (also the decoder start token), 1 = eos, then one token per (position, code) for the 4
Semantic ID positions (separate per position), then `num_user_tokens` hashed user tokens:
    code token  = 2 + position * codebook_size + code
    user token  = 2 + 4 * codebook_size + md5(reviewerID) % num_user_tokens

Encoder input = [user token] + Semantic ID tokens of the last `max_history` items (oldest first), right-padded.
Decoder target = the next item's 4 Semantic ID tokens.
"""

import hashlib
from collections.abc import Sequence

import numpy as np

PAD, EOS = 0, 1
FIRST_CODE_TOKEN = 2


class TigerTokenizer:
    def __init__(self, item_sids: np.ndarray, codebook_size: int = 256, num_user_tokens: int = 2000,
                 max_history: int = 20):
        """item_sids: (num_items + 1, 4) Semantic IDs indexed by item ID (row 0 unused)."""
        self.item_sids = np.asarray(item_sids, dtype=np.int64)
        self.codebook_size = codebook_size
        self.num_positions = self.item_sids.shape[1]
        self.num_user_tokens = num_user_tokens
        self.max_history = max_history
        self.first_user_token = FIRST_CODE_TOKEN + self.num_positions * codebook_size
        self.vocab_size = self.first_user_token + num_user_tokens
        self.max_input_len = 1 + max_history * self.num_positions
        offsets = FIRST_CODE_TOKEN + np.arange(self.num_positions) * codebook_size
        self.item_tokens = self.item_sids + offsets[None]  # (num_items + 1, 4) token IDs per item

    def user_token(self, reviewer_id: str) -> int:
        h = int(hashlib.md5(reviewer_id.encode()).hexdigest(), 16)
        return self.first_user_token + h % self.num_user_tokens

    def encode_inputs(self, histories: Sequence[Sequence[int]], user_tokens: Sequence[int]) -> np.ndarray:
        """(len(histories), max_input_len) right-padded encoder inputs."""
        out = np.full((len(histories), self.max_input_len), PAD, dtype=np.int64)
        for i, (hist, ut) in enumerate(zip(histories, user_tokens, strict=True)):
            hist = list(hist)[-self.max_history:]
            tokens = self.item_tokens[hist].reshape(-1) if hist else np.empty(0, dtype=np.int64)
            out[i, 0] = ut
            out[i, 1:1 + len(tokens)] = tokens
        return out

    def encode_targets(self, items: Sequence[int]) -> np.ndarray:
        return self.item_tokens[np.asarray(items, dtype=np.int64)]


def training_examples(train_seqs: Sequence[Sequence[int]]) -> tuple[list[int], list[list[int]], list[int]]:
    """Every next-item position inside the train part: (user index, history train[:t], target train[t]), t >= 1."""
    users, histories, targets = [], [], []
    for u, seq in enumerate(train_seqs):
        for t in range(1, len(seq)):
            users.append(u)
            histories.append(seq[:t])
            targets.append(seq[t])
    return users, histories, targets
