"""Cold-start evaluation (TIGER paper Sec. 4.3, Fig. 5) on the test split of data/processed_coldstart.

Methods (all trained on the cold-start train split; see recsys.eval.coldstart for the list rules):
  TIGER        best.pt of results/tiger_coldstart; beam search over the Semantic IDs of all items (seen + unseen),
               seen items by exact 4-code match, unseen items by first-3-code match, at most unseen_slots(eps, K).
  Hybrid       SASRec-CE (results/sasrec_coldstart, seed 42) top seen items + Semantic-KNN's top unseen items in
               the last unseen_slots(eps, K) places. At eps = 0 this is plain SASRec-CE.
  Semantic-KNN cosine between Sentence-T5 item embeddings and the mean embedding of the user's last n history
               items (n picked on valid users by Recall@10), all items ranked, same eps cap as TIGER.
History items are never recommended. Evaluated users: (a) users whose test target is unseen, (b) all users.
Paired bootstrap (1,000 resamples, seed 0) for TIGER − Hybrid.

Writes results/coldstart_test.md, results/coldstart/coldstart_test.json, results/coldstart/fig5_coldstart.png
(cached per-user model outputs in results/coldstart/*.npy).

Usage: python scripts/eval_coldstart.py [--rule ceil]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recsys.data.dataset import load_split  # noqa: E402
from recsys.data.semantic_ids import load_semantic_ids  # noqa: E402
from recsys.eval.bootstrap import bootstrap_ci, paired_bootstrap_ci  # noqa: E402
from recsys.eval.coldstart import (capped_list, hybrid_list, prefix_keys, recall_ndcg, tiger_candidates,  # noqa: E402
                                   unseen_slots)
from recsys.models.sasrec import SASRec  # noqa: E402
from recsys.models.tiger import SemanticIDTrie, constrained_beam_search, make_tiger  # noqa: E402
from recsys.train.tiger_trainer import TigerData  # noqa: E402
from recsys.utils import get_device  # noqa: E402

EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
KS = list(range(1, 11))
KNN_NS = [1, 2, 3, 5, 10]
METHODS = ["TIGER", "Hybrid", "Semantic-KNN"]
COLORS = {"TIGER": "tab:blue", "Hybrid": "tab:green", "Semantic-KNN": "tab:purple"}


@torch.no_grad()
def tiger_generated(cfg: dict, run_dir: Path, split, item_sids: np.ndarray, device) -> np.ndarray:
    """(users, beam) generated item IDs over the trie of all items, best first (-1 = none)."""
    reviewers = json.loads((Path(cfg["data_dir"]) / "id_maps.json").read_text())["user2reviewer"]
    data = TigerData.build(split, reviewers, item_sids, **cfg["tokens"])
    trie = SemanticIDTrie(item_sids, data.tok.codebook_size, device=device)  # all items: unseen IDs reachable
    model = make_tiger(data.tok.vocab_size, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    out = []
    for i in range(0, split.num_users, 256):
        users = np.arange(i, min(i + 256, split.num_users))
        x = data.eval_inputs("test", users).to(device)
        items, _ = constrained_beam_search(model, x, (x != 0).long(), trie, cfg["train"]["beam_size"])
        out.append(items.cpu().numpy())
        if device.type == "mps":
            torch.mps.empty_cache()
    return np.concatenate(out)


@torch.no_grad()
def sasrec_seen_top(run_dir: Path, split, unseen: np.ndarray, device, k: int = 10) -> np.ndarray:
    """(users, k) SASRec top seen items (history and unseen items masked; ties -> lower item ID)."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = SASRec(split.num_items, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    hist = split.inputs("test")
    out = np.zeros((split.num_users, k), dtype=np.int64)
    for i in range(0, split.num_users, 1024):
        users = range(i, min(i + 1024, split.num_users))
        s = model.score([hist[u] for u in users]).float()
        s[:, 0] = -torch.inf
        s[:, torch.as_tensor(unseen, device=device)] = -torch.inf
        for j, u in enumerate(users):
            s[j, torch.as_tensor(hist[u], device=device)] = -torch.inf
        out[i:i + len(users)] = torch.sort(-s, dim=1, stable=True).indices[:, :k].cpu().numpy()
    return out


def knn_scores(emb: torch.Tensor, histories, n: int, rows: range) -> torch.Tensor:
    profile = torch.stack([emb[torch.as_tensor(histories[u][-n:])].mean(0) for u in rows])
    s = F.normalize(profile, dim=1) @ emb.T
    s[:, 0] = -torch.inf
    for j, u in enumerate(rows):
        s[j, torch.as_tensor(histories[u])] = -torch.inf
    return s


def knn_pick_n(emb: torch.Tensor, split) -> tuple[int, dict]:
    """n (last history items averaged) with the best valid Recall@10 over valid users, all items ranked."""
    users = split.eval_users("valid")
    hist, targets = split.inputs("valid"), np.asarray(split.targets("valid"))
    res = {}
    for n in KNN_NS:
        hits = 0
        for i in range(0, len(users), 2048):
            rows = users[i:i + 2048]
            top = torch.topk(knn_scores(emb, hist, n, rows), 10, dim=1).indices.numpy()
            hits += (top == targets[rows][:, None]).any(1).sum()
        res[n] = float(hits / len(users))
    return max(res, key=res.get), res


def knn_rankings(emb: torch.Tensor, split, n: int, unseen_mask: np.ndarray, depth: int = 200):
    """Per test user: top `depth` items overall (for standalone KNN) and top 10 unseen items (for the hybrid)."""
    hist = split.inputs("test")
    overall = np.zeros((split.num_users, depth), dtype=np.int64)
    unseen_top = np.zeros((split.num_users, 10), dtype=np.int64)
    um = torch.as_tensor(unseen_mask)
    for i in range(0, split.num_users, 2048):
        rows = range(i, min(i + 2048, split.num_users))
        s = knn_scores(emb, hist, n, rows)
        overall[i:i + len(rows)] = torch.topk(s, depth, dim=1).indices.numpy()
        unseen_top[i:i + len(rows)] = torch.topk(s.masked_fill(~um[None], -torch.inf), 10, dim=1).indices.numpy()
    return overall, unseen_top


def build_lists(method: str, eps: float, k: int, rule: str, ctx: dict) -> list[list[int]]:
    slots = unseen_slots(eps, k, rule)
    if method == "TIGER":
        return [capped_list(c, k, slots) for c in ctx["tiger_cands"]]
    if method == "Semantic-KNN":
        return [capped_list(c, k, slots) for c in ctx["knn_cands"]]
    return [hybrid_list(s, u, k, slots) for s, u in zip(ctx["sasrec_top"], ctx["knn_unseen_top"])]


def evaluate_all(rule: str, ctx: dict, targets: np.ndarray, cold_users: np.ndarray, bootstrap: dict) -> dict:
    res = {"unseen_targets_recall_at_k_eps0.1": {}, "unseen_targets_recall10_by_eps": {}, "all_users_by_eps": {}}
    per = {}
    for method in METHODS:
        for eps in EPS:
            for k in (KS if eps == 0.1 else [10]):
                hit, ndcg = recall_ndcg(build_lists(method, eps, k, rule, ctx), targets)
                per[(method, eps, k)] = (hit, ndcg)
    for k in KS:
        row = {m: bootstrap_ci(per[(m, 0.1, k)][0][cold_users], **bootstrap) for m in METHODS}
        row["TIGER−Hybrid"] = paired_bootstrap_ci(per[("TIGER", 0.1, k)][0][cold_users],
                                                  per[("Hybrid", 0.1, k)][0][cold_users], **bootstrap)
        res["unseen_targets_recall_at_k_eps0.1"][k] = row
    for eps in EPS:
        row = {m: bootstrap_ci(per[(m, eps, 10)][0][cold_users], **bootstrap) for m in METHODS}
        row["TIGER−Hybrid"] = paired_bootstrap_ci(per[("TIGER", eps, 10)][0][cold_users],
                                                  per[("Hybrid", eps, 10)][0][cold_users], **bootstrap)
        res["unseen_targets_recall10_by_eps"][eps] = row
        allrow = {}
        for name, idx in (("recall@10", 0), ("ndcg@10", 1)):
            allrow[name] = {m: bootstrap_ci(per[(m, eps, 10)][idx], **bootstrap) for m in METHODS}
            allrow[name]["TIGER−Hybrid"] = paired_bootstrap_ci(per[("TIGER", eps, 10)][idx],
                                                               per[("Hybrid", eps, 10)][idx], **bootstrap)
        res["all_users_by_eps"][eps] = allrow
    return res


def ci(d: dict, signed: bool = False) -> str:
    f = "+.4f" if signed else ".4f"
    s = f"{d['mean']:{f}} [{d['ci_low']:{f}}, {d['ci_high']:{f}}]"
    return f"**{s}**" if signed and (d["ci_low"] > 0 or d["ci_high"] < 0) else s


def markdown(res: dict, sens: dict, info: dict, rule: str) -> str:
    L = ["# Cold-start evaluation — test split, `data/processed_coldstart` (paper Sec. 4.3)", "",
         f"{info['num_unseen']} held-out (unseen) items = 5% of distinct test targets; {info['n_cold']:,} of "
         f"{info['n_users']:,} test users have an unseen target. eps = maximum share of unseen items in the top K, "
         f"as unseen slots = {rule}(eps·K). Semantic-KNN: mean Sentence-T5 embedding of the last {info['knn_n']} "
         f"history items (picked on valid: {info['knn_valid']}). Mean [95% bootstrap CI]; TIGER − Hybrid paired, "
         "**bold** = CI excludes 0.", "",
         "## (a) Users with an unseen test target: Recall@K at eps = 0.1", "",
         "| K | unseen slots | TIGER | Hybrid (SASRec-CE + KNN) | Semantic-KNN | TIGER − Hybrid |", "|---:|---:|---|---|---|---|"]
    for k, row in res["unseen_targets_recall_at_k_eps0.1"].items():
        L.append(f"| {k} | {unseen_slots(0.1, int(k), rule)} | " + " | ".join(ci(row[m]) for m in METHODS)
                 + f" | {ci(row['TIGER−Hybrid'], True)} |")
    L += ["", "## (a) Users with an unseen test target: Recall@10 by eps", "",
          "| eps | unseen slots | TIGER | Hybrid | Semantic-KNN | TIGER − Hybrid |", "|---:|---:|---|---|---|---|"]
    for eps, row in res["unseen_targets_recall10_by_eps"].items():
        L.append(f"| {eps} | {unseen_slots(float(eps), 10, rule)} | " + " | ".join(ci(row[m]) for m in METHODS)
                 + f" | {ci(row['TIGER−Hybrid'], True)} |")
    for name in ("recall@10", "ndcg@10"):
        L += ["", f"## (b) All {info['n_users']:,} test users: {name.replace('recall', 'Recall').replace('ndcg', 'NDCG')} by eps", "",
              "| eps | TIGER | Hybrid | Semantic-KNN | TIGER − Hybrid |", "|---:|---|---|---|---|"]
        for eps, row in res["all_users_by_eps"].items():
            r = row[name]
            L.append(f"| {eps} | " + " | ".join(ci(r[m]) for m in METHODS) + f" | {ci(r['TIGER−Hybrid'], True)} |")
    other = "floor" if rule == "ceil" else "ceil"
    L += ["", f"## Sensitivity: unseen slots = {other}(eps·K)", "",
          "| setting | TIGER | Hybrid | Semantic-KNN | TIGER − Hybrid |", "|---|---|---|---|---|"]
    for k in (1, 5, 10):
        row = sens["unseen_targets_recall_at_k_eps0.1"][k]
        L.append(f"| unseen targets, Recall@{k}, eps 0.1 | " + " | ".join(ci(row[m]) for m in METHODS)
                 + f" | {ci(row['TIGER−Hybrid'], True)} |")
    for eps, row in sens["unseen_targets_recall10_by_eps"].items():
        L.append(f"| unseen targets, Recall@10, eps {eps} | " + " | ".join(ci(row[m]) for m in METHODS)
                 + f" | {ci(row['TIGER−Hybrid'], True)} |")
    L += ["", "Reference (decoding over seen items only, from each run's metrics.json, all test users): "
          + "; ".join(f"{k} {v}" for k, v in info["reference"].items()), ""]
    return "\n".join(L)


def plot(res: dict, out: Path, rule: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    a = res["unseen_targets_recall_at_k_eps0.1"]
    for m in METHODS:
        y = [a[k][m]["mean"] for k in KS]
        lo = [a[k][m]["ci_low"] for k in KS]
        hi = [a[k][m]["ci_high"] for k in KS]
        axes[0].plot(KS, y, "o-", color=COLORS[m], label=m, ms=4)
        axes[0].fill_between(KS, lo, hi, color=COLORS[m], alpha=0.15)
    axes[0].set(xlabel="K", ylabel="Recall@K", title="(a) unseen test targets, eps = 0.1")
    b = res["unseen_targets_recall10_by_eps"]
    for m in METHODS:
        axes[1].plot(EPS, [b[e][m]["mean"] for e in EPS], "o-", color=COLORS[m], label=m, ms=4)
        axes[1].fill_between(EPS, [b[e][m]["ci_low"] for e in EPS], [b[e][m]["ci_high"] for e in EPS],
                             color=COLORS[m], alpha=0.15)
    axes[1].set(xlabel="eps (max share of unseen items)", ylabel="Recall@10", title="(b) unseen test targets")
    c = res["all_users_by_eps"]
    for m in METHODS:
        axes[2].plot(EPS, [c[e]["ndcg@10"][m]["mean"] for e in EPS], "o-", color=COLORS[m], label=f"{m} NDCG@10", ms=4)
        axes[2].plot(EPS, [c[e]["recall@10"][m]["mean"] for e in EPS], "s--", color=COLORS[m], label=f"{m} Recall@10", ms=4)
    axes[2].set(xlabel="eps", ylabel="metric@10", title="(c) all test users: cost of unseen slots")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"Cold-start retrieval (cf. TIGER Fig. 5); unseen slots = {rule}(eps·K); shaded: 95% bootstrap CI",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiger-run", type=Path, default=Path("results/tiger_coldstart"))
    parser.add_argument("--sasrec-run", type=Path, default=Path("results/sasrec_coldstart"))
    parser.add_argument("--emb", type=Path, default=Path("data/processed/item_emb.npy"))
    parser.add_argument("--rule", choices=["ceil", "floor"], default="ceil")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = yaml.safe_load((args.tiger_run / "config.yaml").read_text())
    split = load_split(cfg["data_dir"])
    sids = load_semantic_ids(cfg["semantic_ids_path"])
    item_sids = np.zeros((split.num_items + 1, sids["num_tokens"]), dtype=np.int64)
    for item, sid in sids["item_to_sid"].items():
        item_sids[int(item)] = sid
    unseen = np.asarray(sids["unseen_items"], dtype=np.int64)
    held = json.loads((Path(cfg["data_dir"]) / "coldstart.json").read_text())["held_out_items"]
    if sorted(held) != unseen.tolist():
        raise ValueError("semantic_ids.json unseen_items differ from coldstart.json held_out_items")
    unseen_set = set(unseen.tolist())
    unseen_mask = np.zeros(split.num_items + 1, dtype=bool)
    unseen_mask[unseen] = True
    targets = np.asarray(split.test)
    cold_users = np.flatnonzero(unseen_mask[targets])
    hist = [set(h) for h in split.inputs("test")]
    bootstrap = cfg["bootstrap"]
    device = get_device()
    out_dir = args.results_dir / "coldstart"
    out_dir.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        path = out_dir / f"{name}.npy"
        if not path.exists():
            np.save(path, fn())
        return np.load(path)

    generated = cached("tiger_generated_test", lambda: tiger_generated(cfg, args.tiger_run, split, item_sids, device))
    sasrec_top = cached("sasrec_seen_top10_test", lambda: sasrec_seen_top(args.sasrec_run, split, unseen, device))
    emb = F.normalize(torch.as_tensor(np.load(args.emb), dtype=torch.float32), dim=1)
    knn_n, knn_valid = knn_pick_n(emb, split)
    print(f"Semantic-KNN: n = {knn_n} (valid Recall@10 {knn_valid})", flush=True)
    knn_overall, knn_unseen_top = knn_rankings(emb, split, knn_n, unseen_mask)

    key3 = prefix_keys(item_sids, sids["codebook_size"])
    unseen_by_key3: dict[int, list[int]] = {}
    for u in unseen.tolist():
        unseen_by_key3.setdefault(int(key3[u]), []).append(u)
    ctx = {
        "tiger_cands": [tiger_candidates(g.tolist(), hist[i], unseen_set, key3, unseen_by_key3)
                        for i, g in enumerate(generated)],
        "knn_cands": [[(int(i), bool(unseen_mask[i])) for i in row] for row in knn_overall],
        "sasrec_top": sasrec_top.tolist(), "knn_unseen_top": knn_unseen_top.tolist(),
    }
    res = evaluate_all(args.rule, ctx, targets, cold_users, bootstrap)
    sens = evaluate_all("floor" if args.rule == "ceil" else "ceil", ctx, targets, cold_users, bootstrap)

    reference = {}
    for name, run in (("TIGER", args.tiger_run), ("SASRec-CE", args.sasrec_run)):
        t = json.loads((run / "metrics.json").read_text())["splits"]["test"]
        reference[name] = f"R@10 {t['recall@10']['mean']:.4f}, N@10 {t['ndcg@10']['mean']:.4f}"
    gen_unseen = float(np.isin(generated, unseen).mean())
    info ={"num_unseen": len(unseen), "n_cold": len(cold_users), "n_users": split.num_users, "knn_n": knn_n,
            "knn_valid": {n: round(v, 4) for n, v in knn_valid.items()}, "reference": reference,
            "tiger_beam_share_unseen_ids": gen_unseen,
            "tiger_short_lists_eps0": int(sum(len(capped_list(c, 10, 0)) < 10 for c in ctx["tiger_cands"]))}
    (out_dir / "coldstart_test.json").write_text(json.dumps(
        {"rule": args.rule, "info": info, "results": res, f"sensitivity_{'floor' if args.rule == 'ceil' else 'ceil'}": sens},
        indent=2, default=str))
    (args.results_dir / "coldstart_test.md").write_text(markdown(res, sens, info, args.rule))
    plot(res, out_dir / "fig5_coldstart.png", args.rule)
    print((args.results_dir / "coldstart_test.md").read_text())
    print(json.dumps({k: v for k, v in info.items() if k != "reference"}, indent=2))


if __name__ == "__main__":
    main()
