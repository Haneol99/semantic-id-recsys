import json
from pathlib import Path

import pytest

from recsys.data.preprocess import MIN_INTERACTIONS, build_item_map, build_sequences, leave_one_out

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_TIESHUFFLE = PROCESSED.parent / "processed_tieshuffle"


# ---------- unit tests on synthetic data ----------

def test_sequences_sorted_by_time_and_filtered():
    reviews = [("u1", f"a{i}", 10 - i) for i in range(5)] + [("u2", "a0", 1)] * 4
    seqs = build_sequences(reviews)
    assert set(seqs) == {"u1"}  # u2 has only 4 interactions
    assert [ts for _, ts in seqs["u1"]] == [6, 7, 8, 9, 10]


def test_item_map_is_asin_sorted_not_first_appearance():
    seqs = {"u": [("zzz", 1), ("aaa", 2), ("mmm", 3)]}
    assert build_item_map(seqs) == {"aaa": 1, "mmm": 2, "zzz": 3}


def test_leave_one_out_synthetic():
    assert leave_one_out([1, 2, 3, 4, 5]) == ([1, 2, 3], 4, 5)


# ---------- checks on the real processed data ----------

@pytest.fixture(scope="module")
def processed():
    if not (PROCESSED / "splits.json").exists():
        pytest.skip("data/processed not found; run `python -m recsys.data.preprocess`")
    load = lambda name: json.loads((PROCESSED / name).read_text())
    return load("sequences.json"), load("splits.json"), load("id_maps.json"), load("item_meta.json")


def test_no_user_below_min_interactions(processed):
    seqs = processed[0]["items"]
    assert min(len(s) for s in seqs) >= MIN_INTERACTIONS


def test_test_item_is_last_item(processed):
    seqs, splits = processed[0]["items"], processed[1]
    assert all(t == s[-1] for s, t in zip(seqs, splits["test"], strict=True))


def assert_splits_do_not_overlap(seqs, splits):
    for seq, train, valid, test in zip(seqs, splits["train"], splits["valid"], splits["test"], strict=True):
        # train = positions [0, n-2), valid = n-2, test = n-1; together they are exactly the sequence
        assert len(train) == len(seq) - 2
        assert train + [valid, test] == seq
        assert len({*train, valid, test}) == len(seq)  # no item in two parts


def test_split_positions_do_not_overlap(processed):
    assert_splits_do_not_overlap(processed[0]["items"], processed[1])


def test_timestamps_non_decreasing(processed):
    for ts in processed[0]["timestamps"]:
        assert all(a <= b for a, b in zip(ts, ts[1:]))


def test_item_ids_contiguous_and_asin_sorted(processed):
    sequences, _, id_maps, item_meta = processed
    seqs = sequences["items"]
    item2asin = id_maps["item2asin"]
    used = {i for s in seqs for i in s}
    assert used == set(range(1, len(item2asin)))  # 1..N, 0 reserved for padding
    assert item2asin[0] is None
    assert item2asin[1:] == sorted(item2asin[1:])
    assert set(item_meta) == {str(i) for i in used}


def test_tie_seed_shuffles_only_same_timestamp_reviews():
    reviews = [("u", "a", 1), ("u", "b", 2), ("u", "c", 2), ("u", "d", 2), ("u", "e", 2), ("u", "f", 3)]
    raw = [a for a, _ in build_sequences(reviews)["u"]]
    assert raw == ["a", "b", "c", "d", "e", "f"]  # ties keep input order
    orders = {tuple(a for a, _ in build_sequences(reviews, tie_seed=s)["u"]) for s in range(20)}
    assert all(o[0] == "a" and o[-1] == "f" and sorted(o[1:5]) == ["b", "c", "d", "e"] for o in orders)
    assert len(orders) > 1  # different seeds give different tie orders
    assert build_sequences(reviews, tie_seed=3) == build_sequences(reviews, tie_seed=3)  # reproducible


# ---------- checks on the shuffled-ties dataset (main dataset) ----------

@pytest.fixture(scope="module")
def tieshuffle():
    if not (PROCESSED_TIESHUFFLE / "splits.json").exists() or not (PROCESSED / "sequences.json").exists():
        pytest.skip("data/processed_tieshuffle or data/processed not found; see scripts/reproduce.sh")
    load = lambda d, name: json.loads((d / name).read_text())
    return load(PROCESSED, "sequences.json"), load(PROCESSED_TIESHUFFLE, "sequences.json"), \
        load(PROCESSED_TIESHUFFLE, "splits.json"), load(PROCESSED_TIESHUFFLE, "stats.json")


def test_tieshuffle_same_items_and_timestamps_per_user_as_original(tieshuffle):
    orig, shuf, _, stats = tieshuffle
    assert stats["tie_order"] == "shuffled(seed=0)"
    for a, b in zip(orig["items"], shuf["items"], strict=True):
        assert sorted(a) == sorted(b)
    assert orig["timestamps"] == shuf["timestamps"]  # only same-timestamp items may move


def test_tieshuffle_splits_do_not_overlap(tieshuffle):
    _, shuf, splits, _ = tieshuffle
    assert_splits_do_not_overlap(shuf["items"], splits)
