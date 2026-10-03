"""README cold-start figure: Recall@10 vs eps for the four main methods, from the saved cold-start results.

Left: users whose test target is an unseen (held-out) item. Right: all test users (the cost on seen items).
Methods: TIGER (paper method), TIGER with exactly scored unseen items, Semantic-KNN, and the SASRec-CE + Semantic-KNN
hybrid; bands are 95% bootstrap CIs over users. The 2-/1-code matching sensitivities and Recall@K curves are in
results/coldstart_test.md and the full figure results/coldstart/fig5_coldstart.png (scripts/eval_coldstart.py).

Reads results/coldstart/coldstart_test.json (no model is run) and writes results/coldstart/coldstart_readme.png.

Usage: python scripts/plot_coldstart.py [--results-dir results]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recsys.plotting import INK, INK_MUTED, MODEL_COLORS, style_axes  # noqa: E402

METHODS = {  # key in coldstart_test.json -> label
    "TIGER": "TIGER (paper method)",
    "TIGER (exact-scored unseen)": "TIGER, unseen items scored exactly",
    "Semantic-KNN": "Semantic-KNN",
    "Hybrid": "Hybrid: SASRec-CE + Semantic-KNN",
}


def end_labels(ax, ends: dict[str, float], x: float, min_gap: float) -> None:
    """Value labels at the right end of each line, nudged apart vertically so they do not overlap."""
    placed = []
    for m, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_text = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_text)
        ax.annotate(f"{y:.4f}", (x, y), xytext=(x, y_text), textcoords="data", xycoords="data",
                    ha="left", va="center", fontsize=11, color=INK,
                    bbox={"fc": "white", "ec": "none", "pad": 0.5})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    saved = json.loads((args.results_dir / "coldstart" / "coldstart_test.json").read_text())
    res, info, rule = saved["results"], saved["info"], saved["rule"]
    eps_keys = list(res["unseen_targets_recall10_by_eps"])
    eps = [float(e) for e in eps_keys]
    slot = math.ceil if rule == "ceil" else math.floor

    panels = [
        (res["unseen_targets_recall10_by_eps"], lambda row, m: row[m],
         f"Users with an unseen test target (n = {info['n_cold']:,})"),
        (res["all_users_by_eps"], lambda row, m: row["recall@10"][m],
         f"All test users (n = {info['n_users']:,}): cost on seen items"),
    ]
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (table, get, title) in zip(axes, panels):
        ends = {}
        for m, label in METHODS.items():
            cells = [get(table[e], m) for e in eps_keys]
            color = MODEL_COLORS[m]
            ax.fill_between(eps, [c["ci_low"] for c in cells], [c["ci_high"] for c in cells], color=color, alpha=0.15,
                            lw=0)
            ax.plot(eps, [c["mean"] for c in cells], "o-", color=color, lw=2.5, ms=7, label=label, zorder=3,
                    clip_on=False)
            ends[m] = cells[-1]["mean"]
        if ax is axes[0]:  # keep TIGER's line at 0 off the x-axis spine
            ax.set_ylim(bottom=-0.006)
        lo, hi = ax.get_ylim()
        end_labels(ax, ends, eps[-1] + 0.008, min_gap=0.06 * (hi - lo))
        ax.set_xlim(-0.01, eps[-1] + 0.045)
        ax.set_xticks(eps)
        ax.set_xticklabels([f"{e:g}\n({slot(round(e * 10, 6))} slot{'s' if slot(round(e * 10, 6)) != 1 else ''})"
                            for e in eps], fontsize=11)
        ax.set_xlabel("eps: max share of the top 10 that may be unseen items", color=INK)
        ax.set_ylabel("Recall@10", color=INK)
        ax.set_title(title, fontsize=13, color=INK)
        style_axes(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, f"Cold start (TIGER paper Sec. 4.3): 5% of test-target items held out of training. Test split; "
             f"bands = 95% bootstrap CI over users; unseen slots = {rule}(eps·10).", ha="center", fontsize=10,
             color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = args.results_dir / "coldstart" / "coldstart_readme.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
