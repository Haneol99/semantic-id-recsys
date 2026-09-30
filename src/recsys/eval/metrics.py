"""Ranking metrics for one relevant item per user, computed from the target's 0-based rank."""

import numpy as np

KS = (5, 10)


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-user Recall@K (= HitRate@K with one relevant item): 1 if rank < k else 0."""
    return (np.asarray(ranks) < k).astype(np.float64)


def ndcg_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per-user NDCG@K with one relevant item: 1 / log2(rank + 2) if rank < k else 0 (ideal DCG = 1)."""
    ranks = np.asarray(ranks)
    return np.where(ranks < k, 1.0 / np.log2(ranks + 2.0), 0.0)


def per_user_metrics(ranks: np.ndarray, ks=KS) -> dict[str, np.ndarray]:
    out = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranks, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranks, k)
    return out
