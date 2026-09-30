"""Evaluate the Popularity baseline on valid and test. Usage: python scripts/run_popularity.py [--config ...]"""

import argparse
from pathlib import Path

from recsys.data.dataset import load_split
from recsys.eval.evaluator import evaluate
from recsys.models.popularity import Popularity
from recsys.utils import load_config, save_run, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/popularity.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    split = load_split(config["data_dir"])
    model = Popularity(split.num_items).fit(split.train)  # counts from train positions only

    results = {
        s: evaluate(model.score, split.inputs(s), split.targets(s), ks=config["ks"])
        for s in config["splits"]
    }
    out_dir = Path(config["results_dir"]) / config["run_name"]
    metrics = save_run(out_dir, config, results, config["bootstrap"])

    print(f"git {metrics['git_hash']}  ->  {out_dir}")
    for s, m in metrics["splits"].items():
        print(f"[{s}] " + "  ".join(
            f"{k}={v['mean']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]" for k, v in m.items() if k != "num_users"
        ))


if __name__ == "__main__":
    main()
