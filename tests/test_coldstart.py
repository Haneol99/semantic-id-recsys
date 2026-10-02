import json
from pathlib import Path

import numpy as np
import pytest

from recsys.data.coldstart import make_coldstart_split, select_held_out
from recsys.data.dataset import Split, load_split
from recsys.data.tiger_data import training_examples
from recsys.train.sasrec_trainer import build_training_pairs

COLD_DIR = Path("data/processed_coldstart")


def toy_split():
    return Split(train=[[1, 2, 3], [4, 5, 6, 7], [2, 8, 9], [3, 4, 1]], valid=[4, 8, 5, 9], test=[5, 2, 7, 6],
                 num_items=9)


def assert_no_held_out_in_training(split: Split, held: set[int]) -> None:
    assert not any(i in held for seq in split.train for i in seq)
    inputs, targets = build_training_pairs(split.train, max_len=50)  # SASRec
    assert not np.isin(inputs, list(held)).any() and not np.isin(targets, list(held)).any()
    _, histories, tgts = training_examples(split.train)  # TIGER
    assert not any(i in held for h in histories for i in h) and not any(t in held for t in tgts)


def test_select_held_out_is_a_seeded_share_of_distinct_test_targets():
    targets = list(range(1, 101)) * 3
    a = select_held_out(targets, 0.05, seed=0)
    assert a == select_held_out(targets, 0.05, seed=0) and len(a) == 5 and set(a) <= set(targets)
    assert a != select_held_out(targets, 0.05, seed=1)


def test_coldstart_removes_held_out_items_from_training_and_validation():
    split = toy_split()
    cold, info = make_coldstart_split(split, [2, 4])
    assert cold.train == [[1, 3], [5, 6, 7], [8, 9], [3, 1]]
    assert_no_held_out_in_training(cold, {2, 4})
    assert cold.valid_users == [1, 2, 3]  # user 0's valid target (4) is held out
    assert cold.eval_users("valid").tolist() == [1, 2, 3] and cold.eval_users("test").tolist() == [0, 1, 2, 3]
    assert (cold.valid, cold.test) == (split.valid, split.test)
    assert info["users_with_unseen_test_target"] == 1 and info["train_interactions_removed"] == 4


@pytest.mark.skipif(not COLD_DIR.exists(), reason="run scripts/make_coldstart.py first")
def test_saved_coldstart_split_has_no_held_out_item_in_training():
    meta = json.loads((COLD_DIR / "coldstart.json").read_text())
    held = set(meta["held_out_items"])
    cold, source = load_split(COLD_DIR), load_split(meta["source_dir"])
    assert held <= set(source.test) and len(held) == round(0.05 * len(set(source.test)))
    assert_no_held_out_in_training(cold, held)
    assert cold.valid_users == [u for u, v in enumerate(cold.valid) if v not in held]
    assert cold.test == source.test and cold.valid == source.valid
    # nothing else removed: filtering the source train reproduces the cold-start train
    assert cold.train == [[i for i in s if i not in held] for s in source.train]
