"""Main test results (shuffled ties): Recall@10 and NDCG@10 per model with 95% bootstrap CIs, against the paper.

Filled dot + bar: the seed-42 run, 95% bootstrap CI over the 22,363 test users (from each run's metrics.json).
Hollow dots: the other SASRec training seeds (43, 44). Vertical lines: the TIGER paper's SASRec and TIGER numbers.

Writes results/analysis/main_results_test.png.

Usage: python scripts/plot_main_results.py [--results-dir results]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_runs import PAPER  # noqa: E402  (scripts/ is on sys.path when run as a script)
from recsys.plotting import INK, INK_MUTED, MODEL_COLORS, style_axes  # noqa: E402

MODELS = {  # top to bottom; first run = seed 42 (CI shown), the rest = other training seeds
    "SASRec-CE": ["sasrec_tieshuffle", "sasrec_tieshuffle_seed43", "sasrec_tieshuffle_seed44"],
    "TIGER": ["tiger_tieshuffle"],
    "SASRec-BCE": ["sasrec_bce_tieshuffle", "sasrec_bce_tieshuffle_seed43", "sasrec_bce_tieshuffle_seed44"],
    "Popularity": ["popularity_tieshuffle"],
}
METRICS = ["recall@10", "ndcg@10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    test = {r: json.loads((args.results_dir / r / "metrics.json").read_text())["splits"]["test"]
            for runs in MODELS.values() for r in runs}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    ys = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}
    for ax, metric in zip(axes, METRICS):
        for model, runs in MODELS.items():
            y, color, main = ys[model], MODEL_COLORS[model], test[runs[0]][metric]
            ax.errorbar(main["mean"], y, xerr=[[main["mean"] - main["ci_low"]], [main["ci_high"] - main["mean"]]],
                        fmt="o", color=color, ms=8, elinewidth=2, capsize=4, zorder=3)
            ax.plot([test[r][metric]["mean"] for r in runs[1:]], [y] * (len(runs) - 1), "o", ms=8, mfc="white",
                    mec=color, mew=1.5, zorder=3)
            ax.annotate(f"{main['mean']:.4f}", (main["ci_high"], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK, zorder=4,
                        bbox={"fc": "white", "ec": "none", "pad": 1})  # keeps reference lines off the number
        for paper, ls, ha, dx in (("sasrec", ":", "right", -3), ("tiger", "--", "left", 3)):
            x = PAPER[paper][metric]
            ax.axvline(x, color=INK_MUTED if paper == "sasrec" else INK, ls=ls, lw=1.2, zorder=1)
            ax.annotate(f"paper {'SASRec' if paper == 'sasrec' else 'TIGER'}\n{x:.4f}", (x, len(MODELS) - 0.45),
                        xytext=(dx, 0), textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=INK)
        ax.set_xlim(0, max(test[r][metric]["ci_high"] for runs in MODELS.values() for r in runs) * 1.15)
        ax.set_ylim(-0.6, len(MODELS) + 0.2)
        ax.set_xlabel(metric.replace("recall", "Recall").replace("ndcg", "NDCG"), color=INK)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e4e3dc", linewidth=0.6)
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([m + (" ×3 seeds" if len(r) > 1 else "") for m, r in MODELS.items()])
    fig.text(0.5, 0.01, "Test split, 22,363 users, shuffled same-day ties. Filled dot + bar: seed-42 run, 95% bootstrap "
             "CI over users. Hollow dots: SASRec seeds 43 and 44. TIGER: one seed.", ha="center", fontsize=8,
             color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = args.results_dir / "analysis" / "main_results_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
