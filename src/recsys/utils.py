"""Helpers shared by run scripts: seeding, config loading, and writing results/<run_name>/."""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from recsys.eval.bootstrap import bootstrap_ci


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def git_hash() -> str:
    """Current commit, with a "-dirty" suffix if code or configs have uncommitted (incl. untracked) changes."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs"], capture_output=True, text=True
    ).stdout
    dirty = bool(status.strip())
    return head + ("-dirty" if head and dirty else "")


def save_run(out_dir: Path, config: dict, results: dict[str, tuple[dict, dict]], bootstrap: dict) -> dict:
    """Write config.yaml, metrics.json and <split>_<metric>.npy for each evaluated split.

    results: {split: (mean_metrics, per_user_arrays)} as returned by recsys.eval.evaluator.evaluate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    metrics = {"run_name": config["run_name"], "seed": config["seed"], "git_hash": git_hash(), "splits": {}}
    for split, (means, per_user) in results.items():
        for name, arr in per_user.items():
            np.save(out_dir / f"{split}_{name.replace('@', '')}.npy", arr)
        metrics["splits"][split] = {
            "num_users": len(per_user["rank"]),
            **{name: bootstrap_ci(per_user[name], **bootstrap) for name in means},
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
