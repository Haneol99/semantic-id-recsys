import numpy as np

from recsys.eval.analysis import prefix_depth

# items 1..4 (row 0 = padding)
SIDS = np.array([[0, 0, 0, 0], [5, 1, 2, 0], [5, 1, 3, 0], [5, 2, 2, 0], [7, 1, 2, 0]])


def test_prefix_depth_is_longest_shared_leading_prefix():
    lists = np.array([[2, 3, 0], [4, 0, 0], [1, 4, 0], [3, 4, 0]])
    targets = np.array([1, 1, 1, 2])
    # target 1 = (5,1,2,0): item 2 shares (5,1); item 4 shares nothing (first code differs); item 1 itself = 4
    # target 2 = (5,1,3,0): item 3 = (5,2,..) shares only the first code
    assert prefix_depth(lists, targets, SIDS).tolist() == [2, 0, 4, 1]


def test_prefix_depth_ignores_empty_slots():
    # slot 0 maps to the all-zero padding row; it must not count as a match for a target whose codes start with 0
    sids = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [3, 3, 3, 0]])
    assert prefix_depth(np.array([[2, 0]]), np.array([1]), sids).tolist() == [0]
