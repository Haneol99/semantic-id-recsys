"""Popularity baseline: every user gets the same scores, the item's train interaction count."""

from collections.abc import Sequence

import numpy as np
import torch


class Popularity:
    def __init__(self, num_items: int):
        self.num_items = num_items  # item IDs 1..num_items; 0 is padding
        self.counts = torch.zeros(num_items + 1)

    def fit(self, train_seqs: Sequence[Sequence[int]]) -> "Popularity":
        counts = np.bincount(np.concatenate([np.asarray(s, dtype=np.int64) for s in train_seqs]),
                             minlength=self.num_items + 1)
        self.counts = torch.as_tensor(counts, dtype=torch.float32)
        return self

    def score(self, user_ids: np.ndarray) -> torch.Tensor:
        return self.counts.expand(len(user_ids), -1)
