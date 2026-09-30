"""Full-ranking evaluator shared by all models.

Scores cover item IDs 0..N (column 0 = padding). For each user the padding item and every item in the
user's input history get score -inf. The target's rank is its 0-based position when items are sorted by
score descending, with ties broken by lower item ID first:
    rank = #{items with score > s_t} + #{items with score == s_t and id < target}
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch

from recsys.eval.metrics import KS, per_user_metrics

ScoreFn = Callable[[np.ndarray], torch.Tensor]  # user_ids (B,) -> scores (B, N+1)


def target_ranks(scores: torch.Tensor, histories: Sequence[Sequence[int]], targets: torch.Tensor) -> torch.Tensor:
    """Mask padding + history in place and return the 0-based rank of each target.

    Raises ValueError if a target is in its own input history: masking would hide it.
    """
    in_history = [i for i, (h, t) in enumerate(zip(histories, targets.tolist())) if t in h]
    if in_history:
        raise ValueError(f"target is in the masked input history for {len(in_history)} row(s), e.g. row {in_history[0]}")
    scores = scores.float()
    scores[:, 0] = -torch.inf
    rows = torch.cat([torch.full((len(h),), i, dtype=torch.long) for i, h in enumerate(histories)])
    cols = torch.cat([torch.as_tensor(h, dtype=torch.long) for h in histories])
    scores[rows.to(scores.device), cols.to(scores.device)] = -torch.inf

    targets = targets.to(scores.device)
    target_scores = scores.gather(1, targets[:, None])
    item_ids = torch.arange(scores.shape[1], device=scores.device)
    higher = (scores > target_scores).sum(1)
    tied_before = ((scores == target_scores) & (item_ids[None, :] < targets[:, None])).sum(1)
    return (higher + tied_before).cpu()


def evaluate(
    score_fn: ScoreFn,
    histories: Sequence[Sequence[int]],
    targets: Sequence[int],
    ks=KS,
    batch_size: int = 1024,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Rank every item for every user; return (mean metrics, per-user arrays incl. "rank")."""
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    ranks = []
    for start in range(0, len(targets), batch_size):
        user_ids = np.arange(start, min(start + batch_size, len(targets)))
        scores = score_fn(user_ids).clone()
        ranks.append(target_ranks(scores, [histories[u] for u in user_ids], targets_t[user_ids]))
    ranks = torch.cat(ranks).numpy()

    per_user = {"rank": ranks, **per_user_metrics(ranks, ks)}
    means = {name: float(arr.mean()) for name, arr in per_user.items() if name != "rank"}
    return means, per_user


def evaluate_score_matrix(scores, histories, targets, ks=KS, batch_size: int = 1024):
    """Same as `evaluate`, for a precomputed (num_users x N+1) score matrix (numpy or torch)."""
    scores = torch.as_tensor(scores)
    return evaluate(lambda u: scores[torch.as_tensor(u)], histories, targets, ks, batch_size)


MISS_RANK = 1_000_000_000  # rank stored for a target that is not in a model's ranked list


def filter_ranked(items: np.ndarray, histories: Sequence[Sequence[int]], top_k: int) -> list[list[int]]:
    """Per row: drop invalid (<= 0) and history items from a ranked candidate list, keep the first top_k."""
    out = []
    for row, hist in zip(items, histories, strict=True):
        seen = set(hist)
        out.append([int(i) for i in row if i > 0 and i not in seen][:top_k])
    return out


def evaluate_ranked_lists(
    ranked: Sequence[Sequence[int]], histories: Sequence[Sequence[int]], targets: Sequence[int], ks=KS
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Metrics for models that return a ranked list instead of full scores (e.g. beam search).

    rank = 0-based position of the target in the list, MISS_RANK if absent, so a target outside the list is a
    miss even when the list is shorter than K. Lists must not contain history items (same masking as `evaluate`).
    """
    ranks = np.full(len(targets), MISS_RANK, dtype=np.int64)
    for u, (lst, hist, t) in enumerate(zip(ranked, histories, targets, strict=True)):
        if t in hist:
            raise ValueError(f"target is in the input history for row {u}")
        if not set(lst).isdisjoint(hist):
            raise ValueError(f"ranked list contains history items for row {u}")
        if t in lst:
            ranks[u] = lst.index(t)
    per_user = {"rank": ranks, **per_user_metrics(ranks, ks)}
    means = {name: float(arr.mean()) for name, arr in per_user.items() if name != "rank"}
    return means, per_user
