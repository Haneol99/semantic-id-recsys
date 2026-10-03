"""TIGER eval curve: subset valid NDCG@10 (2,000 fixed valid users, beam 30) per eval step, and training loss.

Two stacked panels on a shared step axis (no second y-scale); a dotted line marks the best.pt step.
Reads <run_dir>/train_log.json and writes results/tiger/eval_curve.png.

Usage: python scripts/plot_tiger_eval_curve.py [--run-dir results/tiger_tieshuffle] [--out results/tiger/eval_curve.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recsys.plotting import INK, INK_MUTED, MODEL_COLORS, style_axes  # noqa: E402


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
    color = MODEL_COLORS["TIGER"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax.plot(steps, ndcg, "o-", color=color, ms=3, lw=2)
    ax.plot(steps[best], ndcg[best], "o", ms=10, mfc="none", mec=INK, mew=1.5)
    ax.annotate(f"best.pt: {ndcg[best]:.4f} @ step {steps[best]:,}", (steps[best], ndcg[best]),
                xytext=(14, 2), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.annotate(f"{ndcg[-1]:.4f} @ step {steps[-1]:,}", (steps[-1], ndcg[-1]), xytext=(-4, -14),
                textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_ylabel("subset valid NDCG@10\n(2,000 users, beam 30)", color=INK)
    ax.set_ylim(0, max(ndcg) * 1.18)
    ax.set_title("TIGER on Amazon Beauty: validation peaks at step 22k while train loss keeps falling",
                 fontsize=10, color=INK)
    ax2.plot([e["step"] for e in losses], [e["loss"] for e in losses], color=color, lw=0.8)
    ax2.set_ylabel("train loss\n(per token)", color=INK)
    ax2.set_xlabel("training step", color=INK)
    for a in (ax, ax2):
        a.axvline(steps[best], color=INK_MUTED, ls=":", lw=1)
        style_axes(a)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}: {len(evals)} evals, best {ndcg[best]:.4f} @ {steps[best]}, last {ndcg[-1]:.4f} @ {steps[-1]}")


if __name__ == "__main__":
    main()
