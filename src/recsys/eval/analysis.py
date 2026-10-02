"""Helpers for the Phase 4 analysis."""

import numpy as np


def prefix_depth(top_items: np.ndarray, targets: np.ndarray, item_sids: np.ndarray) -> np.ndarray:
    """(users,) max number of leading Semantic-ID codes shared by the target and any listed item (0..num_codes).

    top_items: (users, k) item IDs, 0 = empty slot (ignored); item_sids: (num_items + 1, num_codes), row 0 unused.
    num_codes means the target itself is in the list (Semantic IDs are unique).
    """
    t = item_sids[targets][:, None, :]
    shared = np.cumprod(item_sids[top_items] == t, axis=2).sum(2)
    shared[top_items == 0] = 0
    return shared.max(1)
