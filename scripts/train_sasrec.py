"""Train SASRec with early stopping on valid NDCG@10, then evaluate the best checkpoint on valid and test (once).

With --seeds, trains one run per seed: the config's own seed keeps run_name, other seeds write to
results/<run_name>_seed<N>/. A seed whose finished run already exists with an identical config is skipped
(use --force to retrain).

SASRec training on MPS is not bit-for-bit deterministic, so a rerun matches a recorded run only up to
seed-level noise; report mean ± std across seeds.

Usage: python scripts/train_sasrec.py [--config configs/sasrec.yaml] [--seeds 42 43 44] [--force]
"""

import argparse
import json
from pathlib import Path

from recsys.data.dataset import load_split
from recsys.models.sasrec import SASRec
from recsys.train.sasrec_trainer import evaluate_split, train_sasrec
from recsys.utils import finished_run_matches, get_device, load_config, save_run, seed_run_config, set_seed


def run(config: dict, out_dir: Path) -> None:
    set_seed(config["seed"])
    device = get_device()
    split = load_split(config["data_dir"])

    model = SASRec(split.num_items, **config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{config['run_name']}] device {device}  params {num_params:,}  users {split.num_users:,}  "
          f"items {split.num_items:,}  seed {config['seed']}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sasrec.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="one run per seed (default: config seed)")
    parser.add_argument("--force", action="store_true", help="retrain even if an identical finished run exists")
    args = parser.parse_args()

    base = load_config(args.config)
    for seed in args.seeds or [base["seed"]]:
        config = seed_run_config(base, seed)
        out_dir = Path(config["results_dir"]) / config["run_name"]
        if not args.force and finished_run_matches(out_dir, config):
            print(f"skip {out_dir}: finished run with the same config exists")
            continue
        run(config, out_dir)


if __name__ == "__main__":
    main()
