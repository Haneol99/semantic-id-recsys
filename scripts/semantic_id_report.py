"""Semantic ID report: codebook usage, collisions, and first-code category purity vs random assignment.

Category = second level of the item's first category path (the first level is "Beauty" for every item).
Purity of a first code = share of its items in the code's most common category; the overall number is the
item-weighted mean (= sum of per-code majority counts / N). Random baseline: the same code sizes with category
labels shuffled across items (mean ± std over `--permutations` shuffles).

Writes results/rqvae/quality.json and results/rqvae/first_code_categories.png (cf. TIGER Fig. 4a).

Usage: python scripts/semantic_id_report.py [--sids data/processed/semantic_ids.json]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recsys.data.item_text import top_category  # noqa: E402
from recsys.data.semantic_ids import codebook_usage, collision_stats, load_semantic_ids  # noqa: E402

# Categorical slots 1-6 (validated order, light mode), then a neutral gray for "Other / none".
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
OTHER_COLOR = "#a3a29b"


def purity(codes: np.ndarray, labels: np.ndarray) -> tuple[float, dict[int, float]]:
    per_code, majority_total = {}, 0
    for c in np.unique(codes):
        counts = Counter(labels[codes == c])
        top = counts.most_common(1)[0][1]
        per_code[int(c)] = top / (codes == c).sum()
        majority_total += top
    return majority_total / len(codes), per_code


def plot(codes: np.ndarray, labels: np.ndarray, cats: list[str], path: Path, observed: float, rand: float) -> None:
    code_ids = np.unique(codes)
    counts = np.array([[np.sum((codes == c) & (labels == cat)) for cat in cats] for c in code_ids])
    dominant = counts.argmax(1)
    order = np.lexsort((-counts.sum(1), dominant))  # group codes by dominant category, larger first
    counts, code_ids = counts[order], code_ids[order]

    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=150)
    x, bottom = np.arange(len(code_ids)), np.zeros(len(code_ids))
    for j, cat in enumerate(cats):
        color = COLORS[j] if j < len(COLORS) else OTHER_COLOR
        ax.bar(x, counts[:, j], bottom=bottom, width=1.0, color=color, edgecolor="white", linewidth=0.3, label=cat)
        bottom += counts[:, j]
    ax.set_xlim(-0.5, len(code_ids) - 0.5)
    ax.set_xticks([])
    ax.set_xlabel(f"First Semantic ID code ({len(code_ids)} used), grouped by dominant category")
    ax.set_ylabel("Items")
    ax.set_title(f"Category mix per first code: item-weighted purity {observed:.3f} vs random {rand:.3f}",
                 loc="left", fontsize=11, pad=22)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3dc", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(cats), frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sids", default="data/processed/semantic_ids.json")
    parser.add_argument("--meta", default="data/processed/item_meta.json")
    parser.add_argument("--out-dir", type=Path, default=Path("results/rqvae"))
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sids = load_semantic_ids(args.sids)
    k = sids["codebook_size"]
    n = len(sids["item_to_sid"])
    all_codes = np.array([sids["item_to_sid"][str(i)] for i in range(1, n + 1)])
    prefix = all_codes[:, :3]

    meta = json.loads(Path(args.meta).read_text())
    labels = np.array([top_category(meta[str(i)]) or "(none)" for i in range(1, n + 1)])
    by_size = [c for c, _ in Counter(labels).most_common()]
    main_cats = [c for c in by_size if c != "(none)"][: len(COLORS)]
    labels_plot = np.array([lab if lab in main_cats else "Other / none" for lab in labels])
    cats = main_cats + (["Other / none"] if (labels_plot == "Other / none").any() else [])

    first = all_codes[:, 0]
    observed, per_code = purity(first, labels)
    rng = np.random.default_rng(args.seed)
    rand = np.array([purity(first, rng.permutation(labels))[0] for _ in range(args.permutations)])

    report = {
        "num_items": n,
        "codebook_usage_per_level": codebook_usage(prefix, k),
        **collision_stats(prefix),
        "max_4th_token": int(all_codes[:, 3].max()),
        "category_counts": dict(Counter(labels).most_common()),
        "first_code_purity": {
            "item_weighted": observed,
            "unweighted_mean_over_codes": float(np.mean(list(per_code.values()))),
            "random_baseline_mean": float(rand.mean()),
            "random_baseline_std": float(rand.std(ddof=1)),
            "random_permutations": args.permutations,
            "largest_category_share": Counter(labels).most_common(1)[0][1] / n,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "quality.json").write_text(json.dumps(report, indent=2))
    plot(first, labels_plot, cats, args.out_dir / "first_code_categories.png", observed, float(rand.mean()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
