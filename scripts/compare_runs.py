"""Compare runs on one split: means with 95% bootstrap CIs, relative gap to a paper model, mean ± std across
training seeds, and paired-difference CIs between runs.

Paired differences need the same data: same resolved data_dir and same splits.json SHA-256 (recorded in each
run's metrics.json). With --cross-data, runs on different data are paired by user ID instead; this requires the
same users in the same order (user2reviewer in both data dirs' id_maps.json). Targets may differ per user.

--seeds N ...: for each run that has results/<run>_seed<N>/ for every N, report mean ± std (ddof=1) across the
run itself and those seeds. The CI table stays the per-user bootstrap CI of the base run.

Usage:
  python scripts/compare_runs.py --runs popularity sasrec sasrec_bce --paper sasrec [tiger] \
      --pairs sasrec:popularity sasrec:sasrec_bce [--seeds 43 44] [--cross-data] [--split test] \
      [--out summary_test.md] [--note TEXT]
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


class Runs:
    def __init__(self, root: Path):
        self.root = root
        self._metrics, self._config = {}, {}

    def metrics(self, run: str) -> dict:
        if run not in self._metrics:
            self._metrics[run] = json.loads((self.root / run / "metrics.json").read_text())
        return self._metrics[run]

    def config(self, run: str) -> dict:
        if run not in self._config:
            self._config[run] = yaml.safe_load((self.root / run / "config.yaml").read_text())
        return self._config[run]

    def data_dir(self, run: str) -> Path:
        return Path(self.config(run)["data_dir"]).resolve()

    def splits_sha(self, run: str) -> str:
        data = self.metrics(run).get("data")
        if not data:
            raise ValueError(f"run {run} has no data checksum in metrics.json; rerun it")
        return data["sha256"]["splits.json"]

    def per_user(self, run: str, split: str, metric: str) -> np.ndarray:
        return np.load(self.root / run / f"{split}_{metric.replace('@', '')}.npy")


def same_data(runs: Runs, a: str, b: str) -> bool:
    return runs.data_dir(a) == runs.data_dir(b) and runs.splits_sha(a) == runs.splits_sha(b)


def check_same_users(runs: Runs, a: str, b: str) -> None:
    users = [json.loads((runs.data_dir(r) / "id_maps.json").read_text())["user2reviewer"] for r in (a, b)]
    if users[0] != users[1]:
        raise ValueError(f"cannot pair {a} with {b} by user ID: different users or user order")


def seed_rows(runs: Runs, run_names: list[str], seeds: list[int], split: str) -> list[str]:
    rows = []
    for run in run_names:
        variants = [f"{run}_seed{s}" for s in seeds]
        present = [(runs.root / v / "metrics.json").exists() for v in variants]
        if not any(present):
            continue
        if not all(present):
            missing = [v for v, p in zip(variants, present) if not p]
            raise FileNotFoundError(f"missing seed runs for {run}: {missing}")
        group = [run, *variants]
        for v in variants:
            if not same_data(runs, run, v):
                raise ValueError(f"{v} uses different data than {run}")
        means = np.array([[runs.metrics(r)["splits"][split][k]["mean"] for k in METRICS] for r in group])
        seed_list = ", ".join(str(runs.metrics(r)["seed"]) for r in group)
        cells = [f"{m:.4f} ± {s:.4f}" for m, s in zip(means.mean(0), means.std(0, ddof=1))]
        rows.append(f"| {run} (seeds {seed_list}) | " + " | ".join(cells) + " |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--paper", choices=sorted(PAPER), nargs="*", default=[], help="paper models to compare against")
    parser.add_argument("--pairs", nargs="*", default=[], help="run_a:run_b -> paired CI of (a - b)")
    parser.add_argument("--cross-data", action="store_true", help="allow pairs across data dirs, paired by user ID")
    parser.add_argument("--seeds", type=int, nargs="*", default=[], help="extra seeds: runs <run>_seed<N>")
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=None, help="file name under results-dir (default summary_<split>.md)")
    parser.add_argument("--note", default=None, help="text placed under the title")
    args = parser.parse_args()

    runs = Runs(Path(args.results_dir))
    n_users = sorted({runs.metrics(r)["splits"][args.split]["num_users"] for r in args.runs})
    lines = [f"### {args.split} — mean [95% bootstrap CI over {', '.join(f'{n:,}' for n in n_users)} users]", ""]
    if args.note:
        lines += [f"> {args.note}", ""]
    lines += table_header()
    for run in args.runs:
        m = runs.metrics(run)["splits"][args.split]
        lines.append(f"| {run} | " + " | ".join(fmt_ci(m[k]) for k in METRICS) + " |")
    for paper in args.paper:
        lines.append(f"| *paper {paper}* | " + " | ".join(f"{PAPER[paper][k]:.4f}" for k in METRICS) + " |")

    if args.seeds:
        rows = seed_rows(runs, args.runs, args.seeds, args.split)
        if rows:
            lines += ["", "Across training seeds: mean ± std (sample std, ddof=1). SASRec training on MPS is not "
                      "bit-for-bit deterministic; seed variation covers that noise.", ""] + table_header() + rows

    for paper in args.paper:
        ref = PAPER[paper]
        lines += ["", f"Relative to paper {paper} (single runs above):", ""] + table_header()
        for run in args.runs:
            m = runs.metrics(run)["splits"][args.split]
            lines.append(f"| {run} | " + " | ".join(f"{100 * (m[k]['mean'] / ref[k] - 1):+.1f}%" for k in METRICS) + " |")

    if args.pairs:
        how = "paired by user ID across data" if args.cross_data else "paired by user"
        lines += ["", f"Paired difference a − b ({how}): mean [95% CI] (fraction of resamples with diff ≤ 0):", ""]
        lines += table_header()
        for pair in args.pairs:
            a, b = pair.split(":")
            if not same_data(runs, a, b):
                if not args.cross_data:
                    raise ValueError(f"cannot pair {a} ({runs.data_dir(a)}) with {b} ({runs.data_dir(b)}): "
                                     "different data (pass --cross-data to pair by user ID)")
                check_same_users(runs, a, b)
            cells = []
            for k in METRICS:
                d = paired_bootstrap_ci(runs.per_user(a, args.split, k), runs.per_user(b, args.split, k))
                cells.append(f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
                             f"({d['frac_resamples_diff_le_0']:.3f})")
            lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")

    lines += ["", "Data: " + ", ".join(f"{r} → `{runs.config(r)['data_dir']}`" for r in args.runs)]
    text = "\n".join(lines) + "\n"
    print(text)
    (runs.root / (args.out or f"summary_{args.split}.md")).write_text(text)


if __name__ == "__main__":
    main()
