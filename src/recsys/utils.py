"""Helpers shared by run scripts: seeding, config loading, data checksums, and writing results/<run_name>/."""

import hashlib
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


def get_device() -> torch.device:
    """MPS if available, else CPU (spec §7)."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_checksums(data_dir: str | Path) -> dict[str, str]:
    """SHA-256 of every processed data file (*.json) in data_dir, keyed by file name."""
    return {p.name: file_sha256(p) for p in sorted(Path(data_dir).glob("*.json"))}


def seed_run_config(config: dict, seed: int) -> dict:
    """Config for one training seed: the config's own seed keeps run_name, other seeds get run_name_seed<N>."""
    run_name = config["run_name"] if seed == config["seed"] else f"{config['run_name']}_seed{seed}"
    return {**config, "run_name": run_name, "seed": seed}


def finished_run_matches(out_dir: Path, config: dict) -> bool:
    """True if out_dir holds a finished run (metrics.json) whose saved config equals `config`."""
    if not (out_dir / "metrics.json").exists() or not (out_dir / "config.yaml").exists():
        return False
    return yaml.safe_load((out_dir / "config.yaml").read_text()) == config


def save_run(
    out_dir: Path, config: dict, results: dict[str, tuple[dict, dict]], bootstrap: dict, extra: dict | None = None
) -> dict:
    """Write config.yaml, metrics.json and <split>_<metric>.npy for each evaluated split.

    results: {split: (mean_metrics, per_user_arrays)} as returned by recsys.eval.evaluator.evaluate.
    extra: optional run info (e.g. device, best epoch, timings) stored under metrics.json["run_info"].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    metrics = {
        "run_name": config["run_name"],
        "seed": config["seed"],
        "git_hash": git_hash(),
        "data": {"dir": config["data_dir"], "sha256": data_checksums(config["data_dir"])},
        "splits": {},
    }
    if extra:
        metrics["run_info"] = extra
    for split, (means, per_user) in results.items():
        for name, arr in per_user.items():
            np.save(out_dir / f"{split}_{name.replace('@', '')}.npy", arr)
        metrics["splits"][split] = {
            "num_users": len(per_user["rank"]),
            **{name: bootstrap_ci(per_user[name], **bootstrap) for name in means},
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
