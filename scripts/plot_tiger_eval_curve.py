"""TIGER eval curve: subset valid NDCG@10 (2,000 fixed valid users, beam 30) per eval step, with training loss.

Reads <run_dir>/train_log.json and writes results/tiger/eval_curve.png.

Usage: python scripts/plot_tiger_eval_curve.py [--run-dir results/tiger_tieshuffle] [--out results/tiger/eval_curve.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("results/tiger_tieshuffle"))
    parser.add_argument("--out", type=Path, default=Path("results/tiger/eval_curve.png"))
    args = parser.parse_args()

    log = json.loads((args.run_dir / "train_log.json").read_text())
    evals = [e for e in log if "subset_valid_ndcg@10" in e]
    losses = [e for e in log if "loss" in e]
    steps = [e["step"] for e in evals]
    ndcg = [e["subset_valid_ndcg@10"] for e in evals]
    best = max(range(len(evals)), key=lambda i: ndcg[i])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, ndcg, "o-", color="tab:blue", ms=3, label="subset valid NDCG@10 (2,000 users)")
    ax.plot(steps[best], ndcg[best], "*", color="tab:red", ms=14,
            label=f"best.pt: {ndcg[best]:.4f} @ step {steps[best]:,}")
    ax.set_xlabel("training step")
    ax.set_ylabel("subset valid NDCG@10", color="tab:blue")
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot([e["step"] for e in losses], [e["loss"] for e in losses], color="0.6", lw=0.8, label="train loss")
    ax2.set_ylabel("train loss (per token)", color="0.4")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}: {len(evals)} evals, best {ndcg[best]:.4f} @ {steps[best]}, last {ndcg[-1]:.4f} @ {steps[-1]}")


if __name__ == "__main__":
    main()
