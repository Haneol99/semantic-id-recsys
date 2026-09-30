"""Compare runs on one split: means with 95% bootstrap CIs, paired-difference CIs vs a baseline run,
and relative gap to the paper's numbers. Writes results/summary_<split>.md.

Usage: python scripts/compare_runs.py --runs popularity sasrec --baseline popularity [--split test]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from recsys.eval.bootstrap import paired_bootstrap_ci

METRICS = ["recall@5", "ndcg@5", "recall@10", "ndcg@10"]
# TIGER paper (Rajput et al., 2023), Amazon Beauty, full ranking.
PAPER = {
    "sasrec": {"recall@5": 0.0387, "ndcg@5": 0.0249, "recall@10": 0.0605, "ndcg@10": 0.0318},
    "tiger": {"recall@5": 0.0454, "ndcg@5": 0.0321, "recall@10": 0.0648, "ndcg@10": 0.0384},
}


def fmt_ci(d: dict, digits: int = 4) -> str:
    return f"{d['mean']:.{digits}f} [{d['ci_low']:.{digits}f}, {d['ci_high']:.{digits}f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    root = Path(args.results_dir)
    load = lambda run, m: np.load(root / run / f"{args.split}_{m.replace('@', '')}.npy")
    header = "| Model | " + " | ".join(m.replace("recall", "Recall").replace("ndcg", "NDCG") for m in METRICS) + " |"
    lines = [f"### {args.split} — mean [95% bootstrap CI], {'{:,}'} users", "", header, "|---" * (len(METRICS) + 1) + "|"]

    for run in args.runs:
        metrics = json.loads((root / run / "metrics.json").read_text())["splits"][args.split]
        lines.append(f"| {run} (ours) | " + " | ".join(fmt_ci(metrics[m]) for m in METRICS) + " |")
        n_users = metrics["num_users"]
        if run in PAPER:
            lines.append(f"| {run} (paper) | " + " | ".join(f"{PAPER[run][m]:.4f}" for m in METRICS) + " |")
            lines.append(f"| {run} ours vs paper | " + " | ".join(
                f"{100 * (metrics[m]['mean'] / PAPER[run][m] - 1):+.1f}%" for m in METRICS) + " |")
    lines[0] = lines[0].format(n_users)

    if args.baseline:
        lines += ["", f"Paired difference vs {args.baseline} (mean [95% CI], share of resamples with diff <= 0):", "",
                  header, "|---" * (len(METRICS) + 1) + "|"]
        for run in args.runs:
            if run == args.baseline:
                continue
            cells = []
            for m in METRICS:
                d = paired_bootstrap_ci(load(run, m), load(args.baseline, m))
                cells.append(f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] (p≤0: {d['p_diff_le_0']:.3f})")
            lines.append(f"| {run} − {args.baseline} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    (root / f"summary_{args.split}.md").write_text(text)


if __name__ == "__main__":
    main()
