"""Cold-start evaluation (TIGER paper Sec. 4.3, Fig. 5) on the test split of data/processed_coldstart.

Methods (all trained on the cold-start train split; see recsys.eval.coldstart for the list rules):
  TIGER        (paper method, primary) best.pt of results/tiger_coldstart; beam search over the Semantic IDs of all
               items (seen + unseen), seen items by exact 4-code match, unseen items by first-3-code match, at most
               unseen_slots(eps, K).
  TIGER (2-code / 1-code match)  evaluation-only sensitivity: unseen items matched by the first 2 / 1 codes of a
               generated ID (within a matched group in item-ID order).
  TIGER (exact-scored unseen)  labeled variant: TIGER's seen items from the beam fill the first K - unseen_slots
               places; the unseen items with the highest full TIGER log-prob (all 4 codes, scored exactly, not by
               beam search) fill the last unseen_slots places -- the same layout as the hybrid. Scored only for
               users with an unseen target (for the others an unseen slot is never a hit).
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
from transformers.modeling_outputs import BaseModelOutput  # noqa: E402

EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
KS = list(range(1, 11))
KNN_NS = [1, 2, 3, 5, 10]
T_EXACT, T2, T1 = "TIGER (exact-scored unseen)", "TIGER (2-code match)", "TIGER (1-code match)"
METHODS = ["TIGER", T_EXACT, "Hybrid", "Semantic-KNN", T2, T1]
MAIN = ["TIGER", T_EXACT, "Hybrid", "Semantic-KNN"]
PAIRS = [("TIGER", "Hybrid"), (T_EXACT, "Hybrid"), ("TIGER", "Semantic-KNN")]
COLORS = {"TIGER": "tab:blue", T_EXACT: "tab:cyan", "Hybrid": "tab:green", "Semantic-KNN": "tab:purple",
          T2: "tab:gray", T1: "0.7"}


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
def tiger_unseen_exact_top(cfg: dict, run_dir: Path, split, item_sids: np.ndarray, unseen: np.ndarray, device,
                           users: np.ndarray, k: int = 10, users_per_batch: int = 32) -> np.ndarray:
    """(all users, k) unseen items with the highest full TIGER log-prob (sum over the 4 code positions, full-vocab
    log-softmax as in beam search), history items excluded; computed for `users` only, 0 for the others.

    Only users with an unseen target need it: for the others an unseen slot is never a hit, so which unseen item
    fills it does not change any metric (0 = padding, never a target)."""
    reviewers = json.loads((Path(cfg["data_dir"]) / "id_maps.json").read_text())["user2reviewer"]
    data = TigerData.build(split, reviewers, item_sids, **cfg["tokens"])
    model = make_tiger(data.tok.vocab_size, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    tok = torch.as_tensor(data.tok.item_tokens[unseen], device=device)  # (n_unseen, 4)
    n = len(unseen)
    dec_in = torch.cat([torch.zeros(n, 1, dtype=torch.long, device=device), tok[:, :3]], 1)
    hist = split.inputs("test")
    pos = {int(u): j for j, u in enumerate(unseen)}
    out = np.zeros((split.num_users, k), dtype=np.int64)
    all_users = np.asarray(users)
    for i in range(0, len(all_users), users_per_batch):
        users = all_users[i:i + users_per_batch]
        x = data.eval_inputs("test", users).to(device)
        mask = (x != 0).long()
        enc = model.get_encoder()(input_ids=x, attention_mask=mask).last_hidden_state
        b = len(users)
        logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc.repeat_interleave(n, 0)),
                       attention_mask=mask.repeat_interleave(n, 0), decoder_input_ids=dec_in.repeat(b, 1),
                       use_cache=False).logits
        lp = F.log_softmax(logits.float(), -1).gather(2, tok.repeat(b, 1)[..., None]).squeeze(-1).sum(1).view(b, n)
        for j, u in enumerate(users):
            for h in hist[u]:
                if h in pos:
                    lp[j, pos[h]] = -torch.inf
        out[users] = unseen[torch.topk(lp, k, dim=1).indices.cpu().numpy()]
        if device.type == "mps":
            torch.mps.empty_cache()
    return out


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
    match = {"TIGER": 3, T2: 2, T1: 1}
    if method in match:
        return [capped_list(c, k, slots) for c in ctx["tiger_cands"][match[method]]]
    if method == "Semantic-KNN":
        return [capped_list(c, k, slots) for c in ctx["knn_cands"]]
    if method == T_EXACT:
        return [hybrid_list(s, u, k, slots) for s, u in zip(ctx["tiger_seen"], ctx["tiger_exact_unseen_top"])]
    return [hybrid_list(s, u, k, slots) for s, u in zip(ctx["sasrec_top"], ctx["knn_unseen_top"])]


def add_pairs(row: dict, get, bootstrap: dict) -> dict:
    for a, b in PAIRS:
        row[f"{a} − {b}"] = paired_bootstrap_ci(get(a), get(b), **bootstrap)
    return row


def evaluate_all(rule: str, ctx: dict, targets: np.ndarray, cold_users: np.ndarray, bootstrap: dict) -> dict:
    res = {"unseen_targets_recall_at_k_eps0.1": {}, "unseen_targets_recall10_by_eps": {}, "all_users_by_eps": {}}
    per = {}
    for method in METHODS:
        for eps in EPS:
            for k in (KS if eps == 0.1 else [10]):
                hit, ndcg = recall_ndcg(build_lists(method, eps, k, rule, ctx), targets)
                per[(method, eps, k)] = (hit, ndcg)
    for k in KS:
        get = lambda m, k=k: per[(m, 0.1, k)][0][cold_users]  # noqa: E731
        res["unseen_targets_recall_at_k_eps0.1"][k] = add_pairs(
            {m: bootstrap_ci(get(m), **bootstrap) for m in METHODS}, get, bootstrap)
    for eps in EPS:
        get = lambda m, eps=eps: per[(m, eps, 10)][0][cold_users]  # noqa: E731
        res["unseen_targets_recall10_by_eps"][eps] = add_pairs(
            {m: bootstrap_ci(get(m), **bootstrap) for m in METHODS}, get, bootstrap)
        allrow = {}
        for name, idx in (("recall@10", 0), ("ndcg@10", 1)):
            get = lambda m, eps=eps, idx=idx: per[(m, eps, 10)][idx]  # noqa: E731
            allrow[name] = add_pairs({m: bootstrap_ci(get(m), **bootstrap) for m in METHODS}, get, bootstrap)
        res["all_users_by_eps"][eps] = allrow
    return res


def ci(d: dict, signed: bool = False) -> str:
    f = "+.4f" if signed else ".4f"
    s = f"{d['mean']:{f}} [{d['ci_low']:{f}}, {d['ci_high']:{f}}]"
    return f"**{s}**" if signed and (d["ci_low"] > 0 or d["ci_high"] < 0) else s


def markdown(res: dict, sens: dict, info: dict, rule: str) -> str:
    pair_names = [f"{a} − {b}" for a, b in PAIRS]
    head_main = "| " + " | ".join(MAIN) + " | " + " | ".join(pair_names) + " |"
    sep_main = "|---" * (len(MAIN) + len(pair_names)) + "|"
    cells = lambda row: " | ".join(ci(row[m]) for m in MAIN) + " | " + " | ".join(ci(row[p], True) for p in pair_names)  # noqa: E731
    L = ["# Cold-start evaluation — test split, `data/processed_coldstart` (paper Sec. 4.3)", "",
         f"{info['num_unseen']} held-out (unseen) items = 5% of distinct test targets; {info['n_cold']:,} of "
         f"{info['n_users']:,} test users have an unseen target. eps = maximum share of unseen items in the top K, "
         f"as unseen slots = {rule}(eps·K) (primary). Mean [95% bootstrap CI over users]; differences paired, "
         "**bold** = CI excludes 0.", "",
         "- **TIGER** (paper method, primary): beam search (30) over all items' Semantic IDs; seen items by exact "
         "4-code match, unseen items by first-3-code match, at most `unseen slots` of them.",
         f"- **{T_EXACT}** (labeled variant, not the paper's method): TIGER's beam seen items in the first K − slots "
         "places, the unseen items with the highest exact full TIGER log-prob in the last slots.",
         "- **Hybrid**: SASRec-CE (seed 42) top seen items + Semantic-KNN's top unseen items in the last slots "
         "(eps 0 = plain SASRec-CE).",
         f"- **Semantic-KNN**: cosine to the mean Sentence-T5 embedding of the last {info['knn_n']} history items "
         f"(n picked on valid Recall@10: {info['knn_valid']}); all items ranked, same cap as TIGER.",
         f"- TIGER's beam generated an unseen item's ID in {info['tiger_beam_share_unseen_ids']:.4%} of beam slots; "
         f"short lists at eps 0: {info['tiger_short_lists_eps0']}.", "",
         "## (a) Users with an unseen test target: Recall@K at eps = 0.1", "",
         "| K | unseen slots " + head_main, "|---:|---:" + sep_main]
    for k, row in res["unseen_targets_recall_at_k_eps0.1"].items():
        L.append(f"| {k} | {unseen_slots(0.1, int(k), rule)} | " + cells(row) + " |")
    L += ["", "## (a) Users with an unseen test target: Recall@10 by eps", "",
          "| eps | unseen slots " + head_main, "|---:|---:" + sep_main]
    for eps, row in res["unseen_targets_recall10_by_eps"].items():
        L.append(f"| {eps} | {unseen_slots(float(eps), 10, rule)} | " + cells(row) + " |")
    for name in ("recall@10", "ndcg@10"):
        L += ["", f"## (b) All {info['n_users']:,} test users: "
              f"{name.replace('recall', 'Recall').replace('ndcg', 'NDCG')} by eps (cost on seen items)", "",
              "| eps " + head_main, "|---:" + sep_main]
        for eps, row in res["all_users_by_eps"].items():
            L.append(f"| {eps} | " + cells(row[name]) + " |")
    L += ["", "## Sensitivity (evaluation only): TIGER matching unseen items by the first 2 / 1 codes", "",
          f"Recall@10 of users with an unseen target, and all-users Recall@10 / NDCG@10, by eps ({rule} rule).", "",
          "| eps | TIGER 3-code (paper) | 2-code | 1-code | all users R@10: 3 / 2 / 1-code | all users N@10: 3 / 2 / 1-code |",
          "|---:|---|---|---|---|---|"]
    for eps, row in res["unseen_targets_recall10_by_eps"].items():
        a = res["all_users_by_eps"][eps]
        L.append(f"| {eps} | {ci(row['TIGER'])} | {ci(row[T2])} | {ci(row[T1])} | "
                 + " / ".join(f"{a['recall@10'][m]['mean']:.4f}" for m in ("TIGER", T2, T1)) + " | "
                 + " / ".join(f"{a['ndcg@10'][m]['mean']:.4f}" for m in ("TIGER", T2, T1)) + " |")
    other = "floor" if rule == "ceil" else "ceil"
    L += ["", f"## Sensitivity: unseen slots = {other}(eps·K)", "",
          "| setting | unseen slots " + head_main, "|---|---:" + sep_main]
    for k in (1, 5, 10):
        L.append(f"| unseen targets, Recall@{k}, eps 0.1 | {unseen_slots(0.1, k, other)} | "
                 + cells(sens["unseen_targets_recall_at_k_eps0.1"][k]) + " |")
    for eps, row in sens["unseen_targets_recall10_by_eps"].items():
        L.append(f"| unseen targets, Recall@10, eps {eps} | {unseen_slots(float(eps), 10, other)} | " + cells(row) + " |")
    L += ["", "Reference (decoding over seen items only, from each run's metrics.json, all test users): "
          + "; ".join(f"{k} {v}" for k, v in info["reference"].items()), ""]
    return "\n".join(L)


def plot(res: dict, out: Path, rule: str) -> None:
    """(a) Recall@K vs K on unseen targets (eps 0.1), (b) Recall@10 vs eps on unseen targets, (c, d) all-user
    Recall@10 / NDCG@10 vs eps. Main methods solid with 95% CI bands; 2-/1-code matching dotted."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    style = lambda m: dict(ls="-", marker="o", ms=4) if m in MAIN else dict(ls=":", marker=".", ms=3, lw=1)  # noqa: E731

    def curve(ax, xs, cells, m, band=True):
        ax.plot(xs, [c["mean"] for c in cells], color=COLORS[m], label=m, **style(m))
        if band and m in MAIN:
            ax.fill_between(xs, [c["ci_low"] for c in cells], [c["ci_high"] for c in cells], color=COLORS[m], alpha=0.15)

    a, b, c = (res["unseen_targets_recall_at_k_eps0.1"], res["unseen_targets_recall10_by_eps"], res["all_users_by_eps"])
    for m in METHODS:
        curve(axes[0], KS, [a[k][m] for k in KS], m)
        curve(axes[1], EPS, [b[e][m] for e in EPS], m)
        curve(axes[2], EPS, [c[e]["recall@10"][m] for e in EPS], m)
        curve(axes[3], EPS, [c[e]["ndcg@10"][m] for e in EPS], m)
    titles = [("K", "Recall@K", f"(a) unseen test targets (n={len(a)}), eps = 0.1"),
              ("eps (max share of unseen items)", "Recall@10", "(b) unseen test targets"),
              ("eps", "Recall@10", "(c) all test users: Recall@10"),
              ("eps", "NDCG@10", "(d) all test users: NDCG@10")]
    for ax, (xl, yl, t) in zip(axes, titles):
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_title("(a) unseen test targets, eps = 0.1", fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(METHODS), frameon=False, fontsize=9)
    fig.suptitle(f"Cold-start retrieval (cf. TIGER Fig. 5): unseen slots = {rule}(eps·K); bands = 95% bootstrap CI; "
                 "dotted = TIGER with 2-/1-code matching (sensitivity)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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

    exact_unseen = cached("tiger_exact_unseen_top10_test_coldusers",
                          lambda: tiger_unseen_exact_top(cfg, args.tiger_run, split, item_sids, unseen, device,
                                                         users=cold_users))
    tiger_cands = {}
    for n_codes in (3, 2, 1):
        key = prefix_keys(item_sids, sids["codebook_size"], n_codes)
        unseen_by_key: dict[int, list[int]] = {}
        for u in unseen.tolist():
            unseen_by_key.setdefault(int(key[u]), []).append(u)
        tiger_cands[n_codes] = [tiger_candidates(g.tolist(), hist[i], unseen_set, key, unseen_by_key)
                                for i, g in enumerate(generated)]
    ctx = {
        "tiger_cands": tiger_cands,
        "tiger_seen": [[i for i, is_unseen in c if not is_unseen][:10] for c in tiger_cands[3]],
        "tiger_exact_unseen_top": exact_unseen.tolist(),
        "knn_cands": [[(int(i), bool(unseen_mask[i])) for i in row] for row in knn_overall],
        "sasrec_top": sasrec_top.tolist(), "knn_unseen_top": knn_unseen_top.tolist(),
    }
    res = evaluate_all(args.rule, ctx, targets, cold_users, bootstrap)
    sens = evaluate_all("floor" if args.rule == "ceil" else "ceil", ctx, targets, cold_users, bootstrap)

    reference = {}
    for name, run in (("TIGER", args.tiger_run), ("SASRec-CE", args.sasrec_run)):
        if not (run / "metrics.json").exists():
            reference[name] = "n/a (no metrics.json)"
            continue
        t = json.loads((run / "metrics.json").read_text())["splits"]["test"]
        reference[name] = f"R@10 {t['recall@10']['mean']:.4f}, N@10 {t['ndcg@10']['mean']:.4f}"
    gen_unseen = float(np.isin(generated, unseen).mean())
    info ={"num_unseen": len(unseen), "n_cold": len(cold_users), "n_users": split.num_users, "knn_n": knn_n,
            "knn_valid": {n: round(v, 4) for n, v in knn_valid.items()}, "reference": reference,
            "tiger_beam_share_unseen_ids": gen_unseen,
            "tiger_short_lists_eps0": int(sum(len(capped_list(c, 10, 0)) < 10 for c in ctx["tiger_cands"][3]))}
    (out_dir / "coldstart_test.json").write_text(json.dumps(
        {"rule": args.rule, "info": info, "results": res, f"sensitivity_{'floor' if args.rule == 'ceil' else 'ceil'}": sens},
        indent=2, default=str))
    (args.results_dir / "coldstart_test.md").write_text(markdown(res, sens, info, args.rule))
    plot(res, out_dir / "fig5_coldstart.png", args.rule)
    print((args.results_dir / "coldstart_test.md").read_text())
    print(json.dumps({k: v for k, v in info.items() if k != "reference"}, indent=2))


if __name__ == "__main__":
    main()
