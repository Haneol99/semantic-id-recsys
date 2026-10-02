import numpy as np

from recsys.eval.coldstart import (capped_list, hybrid_list, prefix_keys, recall_ndcg, tiger_candidates,
                                   unseen_slots)


def test_unseen_slots_rules():
    assert [unseen_slots(0.1, k) for k in (1, 5, 10, 11)] == [1, 1, 1, 2]
    assert [unseen_slots(0.1, k, "floor") for k in (1, 9, 10, 20)] == [0, 0, 1, 2]
    assert unseen_slots(0.0, 10) == 0 and unseen_slots(0.3, 10) == 3 and unseen_slots(0.2, 10, "floor") == 2


def test_tiger_candidates_match_unseen_items_by_first_three_codes():
    # items 1..5; items 1 (seen) and 4 (unseen) share prefix (0,1,2); item 5 is unseen with its own prefix
    sids = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [3, 3, 3, 0], [1, 1, 1, 0], [0, 1, 2, 1], [2, 2, 2, 0]])
    key3 = prefix_keys(sids, 4)
    unseen = {4, 5}
    by_key = {int(key3[4]): [4], int(key3[5]): [5]}
    cands = tiger_candidates([1, 5, 3, 2, -1], history={3}, unseen=unseen, key3=key3, unseen_by_key3=by_key)
    # 1 exact (seen) -> its unseen sibling 4; generated 5 (unseen ID) -> 5 via its prefix; 3 is history; then 2
    assert cands == [(1, False), (4, True), (5, True), (2, False)]
    assert capped_list(cands, 3, max_unseen=1) == [1, 4, 2]
    assert capped_list(cands, 3, max_unseen=0) == [1, 2]  # short list: only two seen candidates
    assert capped_list(cands, 2, max_unseen=5) == [1, 4]


def test_hybrid_list_puts_unseen_items_in_the_last_slots():
    assert hybrid_list([7, 8, 9, 10], [1, 2], k=3, n_unseen=1) == [7, 8, 1]
    assert hybrid_list([7, 8, 9, 10], [1, 2], k=3, n_unseen=0) == [7, 8, 9]


def test_recall_ndcg_from_lists():
    hit, ndcg = recall_ndcg([[5, 6, 7], [1, 2], []], [6, 3, 4])
    assert hit.tolist() == [1, 0, 0] and np.isclose(ndcg[0], 1 / np.log2(3)) and ndcg[1:].tolist() == [0, 0]
