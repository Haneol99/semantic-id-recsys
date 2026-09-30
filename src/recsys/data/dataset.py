"""Load the processed leave-one-out split."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Split:
    train: list[list[int]]  # per user: items at positions [0, n-2)
    valid: list[int]  # per user: item at position n-2
    test: list[int]  # per user: item at position n-1
    num_items: int  # item IDs are 1..num_items; 0 is padding

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


def load_split(processed_dir: str | Path = "data/processed") -> Split:
    processed_dir = Path(processed_dir)
    splits = json.loads((processed_dir / "splits.json").read_text())
    stats = json.loads((processed_dir / "stats.json").read_text())
    return Split(splits["train"], splits["valid"], splits["test"], stats["num_items"])
