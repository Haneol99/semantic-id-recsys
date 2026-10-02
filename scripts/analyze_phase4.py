"""Phase 4 analysis on existing runs (no training), test split, main dataset (shuffled ties).

1. Buckets of test users, per model: Recall@10 / NDCG@10 with 95% bootstrap CIs, and paired bootstrap CIs for
   TIGER − SASRec-CE and TIGER − SASRec-BCE.
     - item frequency: train interactions of the test target (0–5 incl. never-in-train, 6–20, >20)
     - history length: items in the test input (train + valid; TIGER reads the last 20, SASRec the last 50)
     - same-day vs different-day valid/test pair (day-granular review timestamps)
   SASRec rows use all three seeds: bucket mean ± std over seeds; CIs and paired CIs use the per-user mean over
   seeds (paired against TIGER's single seed). Per-user metrics come from each run's saved test_*.npy files.
2. Semantic-ID prefix analysis: for test users a model misses (target not in its top 10), how often the top-10
   list holds an item sharing the target's first 1 / 2 / 3 codes. Chance level: the same lists scored against
   another user's target (random permutation). TIGER vs SASRec is also compared paired, on the users both miss
   (TIGER and every seed of that SASRec), with a bootstrap CI of the difference in prefix-hit shares. Top-10 lists are recomputed by inference with each run's best.pt
   (cached in results/analysis/top10_<run>.npy) and checked against the saved per-user ranks.

Writes results/analysis_phase4_test.md (tables), results/analysis/phase4_test.json, and
results/analysis/{buckets_test.png, prefix_test.png}.

Usage: python scripts/analyze_phase4.py [--results-dir results]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recsys.data.dataset import load_split  # noqa: E402
from recsys.data.semantic_ids import load_semantic_ids  # noqa: E402
from recsys.eval.analysis import prefix_depth  # noqa: E402
from recsys.eval.bootstrap import bootstrap_ci, paired_bootstrap_ci  # noqa: E402
from recsys.eval.evaluator import MISS_RANK, filter_ranked  # noqa: E402
from recsys.models.sasrec import SASRec  # noqa: E402
from recsys.models.tiger import SemanticIDTrie, constrained_beam_search, make_tiger  # noqa: E402
from recsys.train.tiger_trainer import TigerData  # noqa: E402
from recsys.utils import get_device  # noqa: E402

DATA_DIR = "data/processed_tieshuffle"
SEEDS = ["", "_seed43", "_seed44"]
MODELS = {  # display name -> run names (several = training seeds)
    "Popularity": ["popularity_tieshuffle"],
    "SASRec-BCE": [f"sasrec_bce_tieshuffle{s}" for s in SEEDS],
    "SASRec-CE": [f"sasrec_tieshuffle{s}" for s in SEEDS],
    "TIGER": ["tiger_tieshuffle"],
}
METRICS = ["recall@10", "ndcg@10"]
COLORS = {"Popularity": "0.6", "SASRec-BCE": "tab:orange", "SASRec-CE": "tab:green", "TIGER": "tab:blue"}


# ---------------- buckets ----------------

def make_groupings(split, timestamps) -> dict[str, dict[str, np.ndarray]]:
    """{grouping: {bucket label: boolean mask over users}}; the never-in-train row is a subset of 0–5."""
    counts = np.bincount(np.concatenate([np.asarray(s) for s in split.train]), minlength=split.num_items + 1)
    freq = counts[np.asarray(split.test)]
    hist_len = np.array([len(h) for h in split.inputs("test")])
    same_day = np.array([t[-1] == t[-2] for t in timestamps])
    return {
        "Test-target train interactions": {
            "0–5": freq <= 5, "  of which 0 (never in train)": freq == 0, "6–20": (freq > 5) & (freq <= 20),
            ">20": freq > 20},
        "Test history length (items)": {
            "4–5": hist_len <= 5, "6–10": (hist_len > 5) & (hist_len <= 10), "11–20": (hist_len > 10) & (hist_len <= 20),
            ">20": hist_len > 20},
        "Valid/test pair": {"same day": same_day, "different day": ~same_day},
    }


def load_per_user(root: Path, run: str) -> dict[str, np.ndarray]:
    return {m: np.load(root / run / f"test_{m.replace('@', '')}.npy") for m in [*METRICS, "rank"]}


def bucket_tables(groupings, per_run, bootstrap) -> dict:
    out = {}
    for gname, buckets in groupings.items():
        out[gname] = {}
        for label, mask in buckets.items():
            row = {"n": int(mask.sum())}
            avg = {}
            for model, runs in MODELS.items():
                for m in METRICS:
                    vals = np.stack([per_run[r][m][mask] for r in runs])  # (seeds, users)
                    avg[(model, m)] = vals.mean(0)
                    seed_means = vals.mean(1)
                    row[f"{model}|{m}"] = {
                        **bootstrap_ci(avg[(model, m)], **bootstrap),
                        "seed_std": float(seed_means.std(ddof=1)) if len(runs) > 1 else None}
            for base in ("SASRec-CE", "SASRec-BCE"):
                for m in METRICS:
                    d = paired_bootstrap_ci(avg[("TIGER", m)], avg[(base, m)], **bootstrap)
                    per_seed = [paired_bootstrap_ci(per_run["tiger_tieshuffle"][m][mask], per_run[r][m][mask], **bootstrap)
                                for r in MODELS[base]]
                    d["per_seed_mean"] = [round(p["mean"], 5) for p in per_seed]
                    d["per_seed_ci_excludes_0_same_sign"] = sum(
                        (p["ci_low"] > 0 and d["mean"] > 0) or (p["ci_high"] < 0 and d["mean"] < 0) for p in per_seed)
                    row[f"TIGER−{base}|{m}"] = d
            out[gname][label] = row
    return out


def fmt_model(cell: dict) -> str:
    if cell["seed_std"] is not None:
        return f"{cell['mean']:.4f} ± {cell['seed_std']:.4f}"
    return f"{cell['mean']:.4f} [{cell['ci_low']:.4f}, {cell['ci_high']:.4f}]"


def fmt_diff(d: dict) -> str:
    sig = d["ci_low"] > 0 or d["ci_high"] < 0
    s = f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
    return f"**{s}**" if sig else s


# ---------------- top-10 lists ----------------

@torch.no_grad()
def sasrec_top10(root: Path, run: str, split, device) -> np.ndarray:
    cfg = yaml.safe_load((root / run / "config.yaml").read_text())
    model = SASRec(split.num_items, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(root / run / "best.pt", map_location=device))
    model.eval()
    hist = split.inputs("test")
    out = np.zeros((split.num_users, 10), dtype=np.int64)
    for i in range(0, split.num_users, 1024):
        users = range(i, min(i + 1024, split.num_users))
        scores = model.score([hist[u] for u in users]).float()
        scores[:, 0] = -torch.inf
        for j, u in enumerate(users):
            scores[j, torch.as_tensor(hist[u], device=device)] = -torch.inf
        # ties broken by lower item ID, as in the evaluator: stable sort on -score
        order = torch.sort(-scores, dim=1, stable=True).indices[:, :10]
        out[i:i + len(users)] = order.cpu().numpy()
    return out


@torch.no_grad()
def tiger_top10(root: Path, run: str, split, device) -> np.ndarray:
    cfg = yaml.safe_load((root / run / "config.yaml").read_text())
    reviewers = json.loads((Path(cfg["data_dir"]) / "id_maps.json").read_text())["user2reviewer"]
    sids = load_semantic_ids(cfg["semantic_ids_path"])
    item_sids = np.zeros((split.num_items + 1, sids["num_tokens"]), dtype=np.int64)
    for item, sid in sids["item_to_sid"].items():
        item_sids[int(item)] = sid
    data = TigerData.build(split, reviewers, item_sids, **cfg["tokens"])
    trie = SemanticIDTrie(item_sids, data.tok.codebook_size, device=device)
    model = make_tiger(data.tok.vocab_size, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(root / run / "best.pt", map_location=device))
    model.eval()
    hist = split.inputs("test")
    out = np.zeros((split.num_users, 10), dtype=np.int64)  # 0 = empty slot (short list)
    for i in range(0, split.num_users, 256):
        users = np.arange(i, min(i + 256, split.num_users))
        x = data.eval_inputs("test", users).to(device)
        items, _ = constrained_beam_search(model, x, (x != 0).long(), trie, cfg["train"]["beam_size"])
        for j, lst in enumerate(filter_ranked(items.cpu().numpy(), [hist[u] for u in users], 10)):
            out[i + j, :len(lst)] = lst
        if device.type == "mps":
            torch.mps.empty_cache()
    return out


def rank_agreement(top10: np.ndarray, targets: np.ndarray, saved_rank: np.ndarray) -> float:
    """Share of users whose top-10 position of the target (or miss) equals the saved rank (or rank >= 10)."""
    hit = top10 == targets[:, None]
    pos = np.where(hit.any(1), hit.argmax(1), MISS_RANK)
    saved = np.where(saved_rank < 10, saved_rank, MISS_RANK)
    return float((pos == saved).mean())


def popularity_top10(split) -> np.ndarray:
    counts = np.bincount(np.concatenate([np.asarray(s) for s in split.train]), minlength=split.num_items + 1)
    counts[0] = -1
    order = np.argsort(-counts, kind="stable")[:200]  # ties -> lower item ID first
    lists = filter_ranked(np.tile(order, (split.num_users, 1)), split.inputs("test"), 10)
    return np.array(lists, dtype=np.int64)


# ---------------- prefix analysis ----------------

def prefix_stats(depth: np.ndarray, missed: np.ndarray) -> dict:
    d = depth[missed]
    return {"n_missed": int(missed.sum()), **{f"ge{k}": float((d >= k).mean()) for k in (1, 2, 3)}}


# ---------------- plots ----------------

def plot_buckets(tables: dict, out: Path) -> None:
    fig, axes = plt.subplots(len(tables), 2, figsize=(12, 3.6 * len(tables)))
    models = list(MODELS)
    width = 0.8 / len(models)
    for row, (gname, buckets) in enumerate(tables.items()):
        labels = [b for b in buckets if not b.startswith("  ")]
        for col, m in enumerate(METRICS):
            ax = axes[row, col]
            for k, model in enumerate(models):
                cells = [buckets[b][f"{model}|{m}"] for b in labels]
                x = np.arange(len(labels)) + (k - (len(models) - 1) / 2) * width
                means = [c["mean"] for c in cells]
                err = [[c["mean"] - c["ci_low"] for c in cells], [c["ci_high"] - c["mean"] for c in cells]]
                ax.bar(x, means, width, yerr=err, capsize=2, color=COLORS[model],
                       label=model + (" (3 seeds)" if len(MODELS[model]) > 1 else ""))
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels([f"{b}\n(n={buckets[b]['n']:,})" for b in labels], fontsize=9)
            ax.set_title(f"{gname}: {m.replace('recall', 'Recall').replace('ndcg', 'NDCG')}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Test split, shuffled ties. Error bars: 95% bootstrap CI over users "
             "(SASRec: per-user mean over 3 seeds).", ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def share(v: float) -> str:
    return f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"


def plot_prefix(prefix: dict, out: Path) -> None:
    """One panel per prefix depth (own y-scale): observed share of missed users vs chance."""
    models = list(prefix)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, d, name in zip(axes, (1, 2, 3), ("first code", "first 2 codes", "first 3 codes")):
        obs = [prefix[m]["missed"][f"ge{d}"] for m in models]
        chance = [prefix[m]["chance"][f"ge{d}"] for m in models]
        x = np.arange(len(models))
        ax.bar(x, obs, 0.7, color=[COLORS[m] for m in models])
        ax.scatter(x, chance, marker="_", s=600, color="k", zorder=3, label="chance (other user's target)")
        for xi, v in zip(x, obs):
            ax.text(xi, v, share(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_title(f"shares the target's {name}", fontsize=10)
        ax.set_ylim(0, max(obs + chance) * 1.2)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("share of missed test users")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Missed targets: does the top-10 list hold an item with the target's Semantic-ID prefix? "
                 "(SASRec: mean over 3 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------- report ----------------

def markdown(tables: dict, prefix: dict, prefix_by_freq: dict, prefix_paired: dict, checks: dict) -> str:
    L = ["# Phase 4 analysis — test split, main dataset (shuffled ties)", "",
         "Per bucket: Popularity and TIGER (one seed) as mean [95% bootstrap CI]; SASRec as mean ± std over "
         "seeds 42/43/44. TIGER − SASRec: paired bootstrap over the bucket's users against the per-user mean over "
         "the three SASRec seeds, mean [95% CI]; **bold** = CI excludes 0. `seeds` = how many of the three single-seed "
         "pairings have a CI excluding 0 with the same sign. 1,000 resamples, seed 0.", ""]
    for m in METRICS:
        name = m.replace("recall", "Recall").replace("ndcg", "NDCG")
        L += [f"## {name}", ""]
        for gname, buckets in tables.items():
            L += [f"### {gname}", "",
                  f"| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | "
                  f"TIGER − SASRec-BCE | seeds |", "|---|---:|---|---|---|---|---|---|---|---|"]
            for b, row in buckets.items():
                ce, bce = row[f"TIGER−SASRec-CE|{m}"], row[f"TIGER−SASRec-BCE|{m}"]
                L.append(f"| {b.strip() if not b.startswith('  ') else '↳ ' + b.strip()} | {row['n']:,} | "
                         + " | ".join(fmt_model(row[f"{mod}|{m}"]) for mod in MODELS)
                         + f" | {fmt_diff(ce)} | {ce['per_seed_ci_excludes_0_same_sign']}/3 | {fmt_diff(bce)} | "
                           f"{bce['per_seed_ci_excludes_0_same_sign']}/3 |")
            L.append("")
    L += ["## Semantic-ID prefix analysis (missed test users)", "",
          "Share of users whose target is not in the model's top 10 but whose top-10 list holds an item with the "
          "same first 1 / 2 / 3 Semantic-ID codes as the target. Chance: the same lists against a random other "
          "user's target. SASRec: mean over 3 seeds. Codes: 256 per level; first code ≈ product category "
          "(purity 0.915).", "",
          "| Model | missed users | ≥ first code | ≥ first 2 | ≥ first 3 | chance ≥1 | chance ≥2 | chance ≥3 |",
          "|---|---:|---|---|---|---|---|---|"]
    for model, p in prefix.items():
        L.append(f"| {model} | {p['missed']['n_missed']:,.0f} | " + " | ".join(share(p['missed'][f'ge{k}']) for k in (1, 2, 3))
                 + " | " + " | ".join(share(p['chance'][f'ge{k}']) for k in (1, 2, 3)) + " |")
    L += ["", "Paired, on users missed by TIGER and by all three seeds of the SASRec variant (SASRec share = mean "
          "over seeds), difference TIGER − SASRec [95% CI]:", "",
          "| vs | users both missed | ≥ first code: TIGER / SASRec | diff | ≥ first 2: TIGER / SASRec | diff |",
          "|---|---:|---|---|---|---|"]
    for base, p in prefix_paired.items():
        L.append(f"| {base} | {p['n_both_missed']:,} | " + " | ".join(
            f"{share(p[f'ge{k}']['tiger'])} / {share(p[f'ge{k}']['sasrec'])} | {fmt_diff(p[f'ge{k}']['diff'])}" for k in (1, 2)) + " |")
    L += ["", "By test-target train interactions (missed users; share with ≥ first code / ≥ first 2 codes):", "",
          "| Bucket | " + " | ".join(prefix_by_freq) + " |", "|---" * (len(prefix_by_freq) + 1) + "|"]
    for b in next(iter(prefix_by_freq.values())):
        L.append(f"| {b.strip()} | " + " | ".join(
            f"{share(prefix_by_freq[mod][b]['ge1'])} / {share(prefix_by_freq[mod][b]['ge2'])}" for mod in prefix_by_freq) + " |")
    L += ["", "Checks: top-10 lists recomputed from best.pt agree with the saved per-user ranks for "
          + ", ".join(f"{r} {v:.4f}" for r, v in checks.items()) + " of users.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = args.results_dir
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(DATA_DIR)
    timestamps = json.loads((Path(DATA_DIR) / "sequences.json").read_text())["timestamps"]
    runs = [r for rs in MODELS.values() for r in rs]
    shas = {r: json.loads((root / r / "metrics.json").read_text())["data"]["sha256"]["splits.json"] for r in runs}
    if len(set(shas.values())) != 1:
        raise ValueError(f"runs use different splits: {shas}")
    per_run = {r: load_per_user(root, r) for r in runs}
    bootstrap = yaml.safe_load((root / "tiger_tieshuffle" / "config.yaml").read_text())["bootstrap"]

    groupings = make_groupings(split, timestamps)
    tables = bucket_tables(groupings, per_run, bootstrap)

    # top-10 lists (cached) and their agreement with the saved ranks
    device = get_device()
    targets = np.asarray(split.test)
    top10, checks = {}, {}
    for r in runs:
        path = out_dir / f"top10_{r}.npy"
        if not path.exists():
            if r.startswith("popularity"):
                lists = popularity_top10(split)
            elif r.startswith("tiger"):
                lists = tiger_top10(root, r, split, device)
            else:
                lists = sasrec_top10(root, r, split, device)
            np.save(path, lists)
        top10[r] = np.load(path)
        checks[r] = rank_agreement(top10[r], targets, per_run[r]["rank"])
        print(f"top-10 {r}: agreement with saved ranks {checks[r]:.4f}", flush=True)

    sids = load_semantic_ids(yaml.safe_load((root / "tiger_tieshuffle" / "config.yaml").read_text())["semantic_ids_path"])
    item_sids = np.zeros((split.num_items + 1, sids["num_tokens"]), dtype=np.int64)
    for item, sid in sids["item_to_sid"].items():
        item_sids[int(item)] = sid
    perm_targets = targets[np.random.default_rng(0).permutation(len(targets))]
    prefix, prefix_by_freq = {}, {}
    freq_buckets = {b: msk for b, msk in groupings["Test-target train interactions"].items()}
    for model, rs in MODELS.items():
        missed_stats, chance_stats, by_freq = [], [], {b: [] for b in freq_buckets}
        for r in rs:
            missed = ~(top10[r] == targets[:, None]).any(1)
            depth = prefix_depth(top10[r], targets, item_sids)
            missed_stats.append(prefix_stats(depth, missed))
            chance = perm_targets != targets
            chance_stats.append(prefix_stats(prefix_depth(top10[r], perm_targets, item_sids), chance & ~(top10[r] == perm_targets[:, None]).any(1)))
            for b, msk in freq_buckets.items():
                by_freq[b].append(prefix_stats(depth, missed & msk))
        avg = lambda lst: {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}  # noqa: E731
        prefix[model] = {"missed": avg(missed_stats), "chance": avg(chance_stats)}
        prefix_by_freq[model] = {b: avg(v) for b, v in by_freq.items()}

    # paired: users missed by TIGER and by every seed of the SASRec variant; per-user indicator of a prefix hit
    depth = {r: prefix_depth(top10[r], targets, item_sids) for r in runs}
    hit = {r: (top10[r] == targets[:, None]).any(1) for r in runs}
    prefix_paired = {}
    for base in ("SASRec-CE", "SASRec-BCE"):
        both = ~hit["tiger_tieshuffle"] & np.all([~hit[r] for r in MODELS[base]], axis=0)
        prefix_paired[base] = {"n_both_missed": int(both.sum())}
        for k in (1, 2):
            a = (depth["tiger_tieshuffle"][both] >= k).astype(float)
            b = np.mean([(depth[r][both] >= k) for r in MODELS[base]], axis=0)
            prefix_paired[base][f"ge{k}"] = {"tiger": float(a.mean()), "sasrec": float(b.mean()),
                                             "diff": paired_bootstrap_ci(a, b, **bootstrap)}

    (out_dir / "phase4_test.json").write_text(json.dumps(
        {"buckets": tables, "prefix": prefix, "prefix_by_freq": prefix_by_freq, "prefix_paired": prefix_paired,
         "rank_agreement": checks}, indent=2))
    (root / "analysis_phase4_test.md").write_text(markdown(tables, prefix, prefix_by_freq, prefix_paired, checks))
    plot_buckets(tables, out_dir / "buckets_test.png")
    plot_prefix(prefix, out_dir / "prefix_test.png")
    print((root / "analysis_phase4_test.md").read_text())


if __name__ == "__main__":
    main()
