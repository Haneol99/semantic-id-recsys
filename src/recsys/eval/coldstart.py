"""Cold-start retrieval lists (TIGER paper Sec. 4.3).

epsilon = maximum share of unseen items in a top-K list, as a slot count `unseen_slots(eps, K)`:
  "ceil"  (primary): ceil(eps * K), so any eps > 0 allows at least one unseen item, also for K < 1 / eps;
  "floor" (strict share): floor(eps * K), so e.g. eps = 0.1 allows no unseen item for K < 10.

TIGER: walk the beam's generated Semantic IDs in score order. A generated ID that is a seen item's exact 4-code ID
adds that item; every unseen item whose first 3 codes match the generated ID is added right after it (item-ID
order). Unseen items are added only while fewer than unseen_slots are in the list; history items are skipped.

Hybrid: a seen-item ranking (SASRec) fills the first K - unseen_slots places and an unseen-item ranking
(Semantic-KNN) the last unseen_slots places.
"""

import math
from collections.abc import Sequence

import numpy as np


def unseen_slots(eps: float, k: int, rule: str = "ceil") -> int:
    x = eps * k
    if rule == "ceil":
        return int(math.ceil(x - 1e-9))
    if rule == "floor":
        return int(math.floor(x + 1e-9))
    raise ValueError(f"unknown rule {rule!r}")


def prefix_keys(item_sids: np.ndarray, codebook_size: int, n_codes: int = 3) -> np.ndarray:
    """(num_items + 1,) int64 key of each item's first n_codes codes."""
    keys = np.zeros(len(item_sids), dtype=np.int64)
    for d in range(n_codes):
        keys = keys * codebook_size + item_sids[:, d]
    return keys


def tiger_candidates(generated: Sequence[int], history: set[int], unseen: set[int], key3: np.ndarray,
                     unseen_by_key3: dict[int, list[int]]) -> list[tuple[int, bool]]:
    """Ordered (item, is_unseen) candidates from one user's generated item IDs (best first; -1 = none)."""
    out, added = [], set()
    for g in generated:
        if g < 0:
            continue
        if g not in unseen and g not in history and g not in added:
            out.append((g, False))
            added.add(g)
        for u in unseen_by_key3.get(int(key3[g]), ()):
            if u not in history and u not in added:
                out.append((u, True))
                added.add(u)
    return out


def capped_list(candidates: Sequence[tuple[int, bool]], k: int, max_unseen: int) -> list[int]:
    """First k candidates, skipping unseen ones once max_unseen are in the list."""
    out, n_unseen = [], 0
    for item, is_unseen in candidates:
        if len(out) == k:
            break
        if is_unseen:
            if n_unseen >= max_unseen:
                continue
            n_unseen += 1
        out.append(item)
    return out


def hybrid_list(seen_ranked: Sequence[int], unseen_ranked: Sequence[int], k: int, n_unseen: int) -> list[int]:
    """Top k - n_unseen seen items, then the top n_unseen unseen items."""
    n_unseen = min(n_unseen, k)
    return list(seen_ranked[:k - n_unseen]) + list(unseen_ranked[:n_unseen])


def recall_ndcg(lists: Sequence[Sequence[int]], targets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-user hit (recall@len) and NDCG of each list (target position, 0-based, at 1 / log2(pos + 2))."""
    hit = np.zeros(len(lists))
    ndcg = np.zeros(len(lists))
    for i, (lst, t) in enumerate(zip(lists, targets)):
        if t in lst:
            hit[i] = 1.0
            ndcg[i] = 1.0 / math.log2(list(lst).index(t) + 2)
    return hit, ndcg
