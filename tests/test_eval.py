import math

import numpy as np
import pytest
import torch

from recsys.eval.bootstrap import bootstrap_ci, paired_bootstrap_ci
from recsys.eval.evaluator import evaluate_score_matrix, target_ranks
from recsys.eval.metrics import ndcg_at_k, per_user_metrics, recall_at_k
from recsys.models.popularity import Popularity

# ---------- metrics (hand-computed) ----------

RANKS = np.array([0, 1, 4, 5, 9, 10])


def test_recall_hand_computed():
    assert recall_at_k(RANKS, 5).tolist() == [1, 1, 1, 0, 0, 0]
    assert recall_at_k(RANKS, 10).tolist() == [1, 1, 1, 1, 1, 0]


def test_ndcg_hand_computed():
    # rank r (0-based) -> 1 / log2(r + 2): r=0 -> 1, r=1 -> 1/log2(3), r=4 -> 1/log2(6), ...
    expected5 = [1.0, 0.6309298, 0.3868528, 0, 0, 0]
    expected10 = [1.0, 0.6309298, 0.3868528, 0.3562072, 0.2890648, 0]
    np.testing.assert_allclose(ndcg_at_k(RANKS, 5), expected5, atol=1e-6)
    np.testing.assert_allclose(ndcg_at_k(RANKS, 10), expected10, atol=1e-6)


def test_per_user_metrics_keys_and_means():
    m = per_user_metrics(RANKS)
    assert set(m) == {"recall@5", "ndcg@5", "recall@10", "ndcg@10"}
    assert m["recall@10"].mean() == pytest.approx(5 / 6)
    assert m["ndcg@5"].mean() == pytest.approx((1 + 1 / math.log2(3) + 1 / math.log2(6)) / 6)


# ---------- evaluator: ranking and masking ----------

def test_history_and_padding_are_masked():
    # item 0 (padding) and item 1 (in history) outrank the target; both must be removed
    scores = torch.tensor([[9.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    assert target_ranks(scores.clone(), [[]], torch.tensor([3])).item() == 2  # items 1, 2 above; pad masked
    assert target_ranks(scores.clone(), [[1]], torch.tensor([3])).item() == 1  # item 2 remains above
    assert target_ranks(scores.clone(), [[1, 2]], torch.tensor([3])).item() == 0


def test_target_in_history_raises():
    scores = torch.ones(2, 6)
    with pytest.raises(ValueError, match="row 1"):
        target_ranks(scores, [[1], [2, 3]], torch.tensor([3, 3]))


def test_ties_broken_by_lower_item_id():
    scores = torch.ones(1, 6)
    assert target_ranks(scores.clone(), [[]], torch.tensor([3])).item() == 2  # items 1, 2 come first
    assert target_ranks(scores.clone(), [[1]], torch.tensor([3])).item() == 1


def test_evaluate_score_matrix_end_to_end():
    scores = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0],  # user 0: order 4,3,2,1; history {4}; target 3 -> rank 0
        [0.0, 4.0, 3.0, 2.0, 1.0],  # user 1: order 1,2,3,4; history {1}; target 4 -> rank 2
    ])
    means, per_user = evaluate_score_matrix(scores, [[4], [1]], [3, 4], ks=[1, 5], batch_size=1)
    assert per_user["rank"].tolist() == [0, 2]
    assert per_user["recall@1"].tolist() == [1, 0]
    assert means["recall@5"] == 1.0
    assert means["ndcg@5"] == pytest.approx((1 + 1 / math.log2(4)) / 2)


# ---------- bootstrap ----------

def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = np.random.default_rng(1).binomial(1, 0.05, size=5000).astype(float)
    ci = bootstrap_ci(values)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["ci_low"] < ci["ci_high"]
    assert ci == bootstrap_ci(values)


def test_paired_bootstrap():
    rng = np.random.default_rng(2)
    a = rng.binomial(1, 0.10, size=5000).astype(float)
    same = paired_bootstrap_ci(a, a)
    assert same["frac_resamples_diff_le_0"] == 1.0
    assert same["mean"] == same["ci_low"] == same["ci_high"] == 0.0
    diff = paired_bootstrap_ci(a, np.zeros_like(a))
    assert diff["ci_low"] <= diff["mean"] <= diff["ci_high"] and diff["ci_low"] > 0
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, a[:10])


# ---------- popularity ----------

def test_popularity_counts_train_interactions():
    model = Popularity(num_items=4).fit([[1, 2, 2], [2, 3]])
    assert model.counts.tolist() == [0, 1, 3, 1, 0]
    assert model.score(np.arange(2)).shape == (2, 5)
