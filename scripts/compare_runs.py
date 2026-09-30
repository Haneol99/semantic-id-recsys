"""Compare runs on one split: means with 95% bootstrap CIs, relative gap to a paper model, and paired-difference
CIs between runs. Paired differences are only allowed between runs that use the same data_dir.

Usage:
  python scripts/compare_runs.py --runs popularity sasrec sasrec_bce --paper sasrec \
      --pairs sasrec:popularity sasrec:sasrec_bce [--split test] [--out summary_test.md]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from recsys.eval.bootstrap import paired_bootstrap_ci

METRICS = ["recall@5", "ndcg@5", "recall@10", "ndcg@10"]
# TIGER paper (Rajput et al., 2023), Amazon Beauty, full ranking.
PAPER = {
    "sasrec": {"recall@5": 0.0387, "ndcg@5": 0.0249, "recall@10": 0.0605, "ndcg@10": 0.0318},
    "tiger": {"recall@5": 0.0454, "ndcg@5": 0.0321, "recall@10": 0.0648, "ndcg@10": 0.0384},
}


def fmt_ci(d: dict) -> str:
    return f"{d['mean']:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"


def table_header() -> list[str]:
    names = [m.replace("recall", "Recall").replace("ndcg", "NDCG") for m in METRICS]
    return ["| Run | " + " | ".join(names) + " |", "|---" * (len(METRICS) + 1) + "|"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--paper", choices=sorted(PAPER), default=None, help="paper model to compare against")
    parser.add_argument("--pairs", nargs="*", default=[], help="run_a:run_b -> paired CI of (a - b)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=None, help="file name under results-dir (default summary_<split>.md)")
    args = parser.parse_args()

    root = Path(args.results_dir)
    metrics = {r: json.loads((root / r / "metrics.json").read_text()) for r in args.runs}
    data_dir = {r: yaml.safe_load((root / r / "config.yaml").read_text())["data_dir"] for r in args.runs}
    load = lambda run, m: np.load(root / run / f"{args.split}_{m.replace('@', '')}.npy")

    n_users = {metrics[r]["splits"][args.split]["num_users"] for r in args.runs}
    lines = [f"### {args.split} — mean [95% bootstrap CI over {', '.join(f'{n:,}' for n in n_users)} users]", ""]
    lines += table_header()
    for run in args.runs:
        m = metrics[run]["splits"][args.split]
        lines.append(f"| {run} | " + " | ".join(fmt_ci(m[k]) for k in METRICS) + " |")
    if args.paper:
        ref = PAPER[args.paper]
        lines.append(f"| *paper {args.paper}* | " + " | ".join(f"{ref[k]:.4f}" for k in METRICS) + " |")
        lines += ["", f"Relative to paper {args.paper}:", ""] + table_header()
        for run in args.runs:
            m = metrics[run]["splits"][args.split]
            lines.append(f"| {run} | " + " | ".join(f"{100 * (m[k]['mean'] / ref[k] - 1):+.1f}%" for k in METRICS) + " |")

    if args.pairs:
        lines += ["", "Paired difference a − b: mean [95% CI] (share of resamples with diff ≤ 0):", ""] + table_header()
        for pair in args.pairs:
            a, b = pair.split(":")
            if data_dir[a] != data_dir[b]:
                raise ValueError(f"cannot pair {a} ({data_dir[a]}) with {b} ({data_dir[b]}): different data")
            cells = []
            for k in METRICS:
                d = paired_bootstrap_ci(load(a, k), load(b, k))
                cells.append(f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] ({d['p_diff_le_0']:.3f})")
            lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")

    lines += ["", "Data: " + ", ".join(f"{r} → `{data_dir[r]}`" for r in args.runs)]
    text = "\n".join(lines) + "\n"
    print(text)
    (root / (args.out or f"summary_{args.split}.md")).write_text(text)


if __name__ == "__main__":
    main()
