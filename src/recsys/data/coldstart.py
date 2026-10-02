"""Cold-start split (TIGER paper Sec. 4.3): hold out a share of the distinct test-target items from training.

Held-out ("unseen") items: `fraction` of the distinct test-target items, drawn with numpy default_rng(seed) from
the sorted list. Every interaction with them is removed from every user's train sequence; valid and test targets
are unchanged. Users whose valid target is held out are left out of validation (model selection); their test
target is still evaluated. Inputs at evaluation time are the filtered train (+ the valid item for test), so a held-out
valid item can appear in a test input, but no held-out item is ever a training input or target.
"""

import json
from pathlib import Path

import numpy as np

from recsys.data.dataset import Split


def select_held_out(test_targets: list[int], fraction: float = 0.05, seed: int = 0) -> list[int]:
    distinct = np.array(sorted(set(test_targets)))
    n = int(round(fraction * len(distinct)))
    return sorted(int(i) for i in np.random.default_rng(seed).choice(distinct, size=n, replace=False))


def make_coldstart_split(split: Split, held_out: list[int]) -> tuple[Split, dict]:
    """Split with held-out items removed from train and valid_users set; plus counts for the record."""
    held = set(held_out)
    train = [[i for i in seq if i not in held] for seq in split.train]
    valid_users = [u for u, v in enumerate(split.valid) if v not in held]
    cold = Split(train, list(split.valid), list(split.test), split.num_items, valid_users=valid_users)
    removed = sum(len(a) - len(b) for a, b in zip(split.train, train))
    info = {
        "num_held_out_items": len(held),
        "num_distinct_test_targets": len(set(split.test)),
        "train_interactions_removed": removed,
        "users_with_train_interactions_removed": sum(len(a) != len(b) for a, b in zip(split.train, train)),
        "users_with_unseen_test_target": sum(t in held for t in split.test),
        "users_with_unseen_valid_target_excluded_from_valid": split.num_users - len(valid_users),
        "users_with_any_held_out_interaction": sum(
            any(i in held for i in [*a, v, t]) for a, v, t in zip(split.train, split.valid, split.test)),
        "users_with_empty_train_after_removal": sum(len(s) == 0 for s in train),
        "users_with_unseen_valid_item_in_test_input": sum(v in held for v in split.valid),
    }
    return cold, info


def save_coldstart(out_dir: Path, source_dir: Path, cold: Split, held_out: list[int], info: dict,
                   fraction: float, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(
        {"train": cold.train, "valid": cold.valid, "test": cold.test, "valid_users": cold.valid_users}))
    stats = json.loads((source_dir / "stats.json").read_text())
    stats["coldstart_source"] = str(source_dir)
    (out_dir / "stats.json").write_text(json.dumps(stats))
    for name in ("id_maps.json", "item_meta.json"):
        (out_dir / name).write_text((source_dir / name).read_text())
    (out_dir / "coldstart.json").write_text(json.dumps(
        {"source_dir": str(source_dir), "fraction": fraction, "seed": seed, **info, "held_out_items": held_out},
        indent=2))
