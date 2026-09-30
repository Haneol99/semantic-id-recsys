"""Train SASRec with early stopping on valid NDCG@10, then evaluate the best checkpoint on valid and test (once).

Usage: python scripts/train_sasrec.py [--config configs/sasrec.yaml]
"""

import argparse
import json
from pathlib import Path

from recsys.data.dataset import load_split
from recsys.models.sasrec import SASRec
from recsys.train.sasrec_trainer import evaluate_split, train_sasrec
from recsys.utils import get_device, load_config, save_run, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sasrec.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()
    split = load_split(config["data_dir"])
    out_dir = Path(config["results_dir"]) / config["run_name"]

    model = SASRec(split.num_items, **config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"device {device}  params {num_params:,}  users {split.num_users:,}  items {split.num_items:,}")

    info = train_sasrec(model, split, config["train"], device, out_dir / "best.pt", config["seed"])
    log = info.pop("log")

    # Best checkpoint is loaded; test is evaluated exactly once here.
    results = {s: evaluate_split(model, split, s) for s in ("valid", "test")}
    metrics = save_run(out_dir, config, results, config["bootstrap"],
                       extra={"device": str(device), "num_params": num_params, **info})
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    print(f"\nbest epoch {info['best_epoch']} / {info['epochs_run']}  "
          f"train {info['train_sec_per_epoch_mean']:.1f}s/epoch  total {info['total_sec']:.0f}s  "
          f"git {metrics['git_hash']}")
    for s, m in metrics["splits"].items():
        print(f"[{s}] " + "  ".join(f"{k}={v['mean']:.4f}" for k, v in m.items() if k != "num_users"))


if __name__ == "__main__":
    main()
