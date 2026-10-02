"""Load the processed leave-one-out split."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Split:
    train: list[list[int]]  # per user: items at positions [0, n-2)
    valid: list[int]  # per user: item at position n-2
    test: list[int]  # per user: item at position n-1
    num_items: int  # item IDs are 1..num_items; 0 is padding
    valid_users: list[int] | None = None  # users used for validation; None = all (cold-start split excludes some)

    @property
    def num_users(self) -> int:
        return len(self.train)

    def inputs(self, split: str) -> list[list[int]]:
        """Input history for predicting the `split` target: train for valid, train + valid for test."""
        if split == "valid":
            return self.train
        if split == "test":
            return [h + [v] for h, v in zip(self.train, self.valid)]
        raise ValueError(f"unknown split {split!r}")

    def targets(self, split: str) -> list[int]:
        return {"valid": self.valid, "test": self.test}[split]

    def eval_users(self, split: str) -> np.ndarray:
        """Users evaluated on `split`: all for test; valid_users (default all) for valid."""
        if split == "valid" and self.valid_users is not None:
            return np.asarray(self.valid_users, dtype=np.int64)
        if split not in ("valid", "test"):
            raise ValueError(f"unknown split {split!r}")
        return np.arange(self.num_users)


def load_split(processed_dir: str | Path = "data/processed") -> Split:
    processed_dir = Path(processed_dir)
    splits = json.loads((processed_dir / "splits.json").read_text())
    stats = json.loads((processed_dir / "stats.json").read_text())
    return Split(splits["train"], splits["valid"], splits["test"], stats["num_items"], splits.get("valid_users"))
