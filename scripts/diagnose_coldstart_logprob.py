"""Cold-start diagnosis: teacher-forced per-position log-prob of the test target's Semantic-ID codes under TIGER.

For every test user of data/processed_coldstart, the decoder is fed the target's own codes (decoder start 0, then
codes 1-3) and the log-prob of the target code at each of the 4 positions is read from a full-vocab log-softmax, as
in beam search (recsys.models.tiger.constrained_beam_search). Users are grouped by whether their test target is an
unseen (held-out) item or a seen one; each group gets the mean per position with a 95% bootstrap CI over users.

Checkpoints of results/tiger_coldstart: best.pt (step 28k, used by scripts/eval_coldstart.py) and last.pt (step 30k,
end of training). Reference: a uniform distribution over one position's 256 codes gives log(1/256) = -5.55.

Writes results/coldstart_logprob_test.md and results/coldstart/logprob_test.json, plus per-user arrays
results/coldstart/logprob_test_<ckpt>.npy (users x 4).

Usage: python scripts/diagnose_coldstart_logprob.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from recsys.data.dataset import load_split
from recsys.data.semantic_ids import load_semantic_ids
from recsys.eval.bootstrap import bootstrap_ci
from recsys.models.tiger import make_tiger
from recsys.train.tiger_trainer import TigerData
from recsys.utils import get_device

CHECKPOINTS = ["best", "last"]


def load_weights(run_dir: Path, ckpt: str, device) -> tuple[dict, int]:
    """(model state_dict, training step) of best.pt or last.pt."""
    last = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
    if ckpt == "last":
        return last["model"], int(last["state"]["step"])
    return torch.load(run_dir / "best.pt", map_location=device), int(last["state"]["best_step"])


@torch.no_grad()
def target_logprobs(model, data: TigerData, targets: np.ndarray, device, batch: int = 512) -> np.ndarray:
    """(users, num_tokens) log-prob of each target code given the user's test input and the previous target codes."""
    tok = torch.as_tensor(data.tok.item_tokens[targets], device=device)  # (users, num_tokens)
    out = []
    for i in range(0, len(targets), batch):
        users = np.arange(i, min(i + batch, len(targets)))
        x = data.eval_inputs("test", users).to(device)
        t = tok[users]
        dec_in = torch.cat([torch.zeros(len(users), 1, dtype=torch.long, device=device), t[:, :-1]], 1)
        logits = model(input_ids=x, attention_mask=(x != 0).long(), decoder_input_ids=dec_in, use_cache=False).logits
        out.append(F.log_softmax(logits.float(), -1).gather(2, t[..., None]).squeeze(-1).cpu().numpy())
        if device.type == "mps":
            torch.mps.empty_cache()
    return np.concatenate(out)


def markdown(res: dict, info: dict) -> str:
    n_pos = info["num_tokens"]
    L = ["# Cold-start diagnosis — teacher-forced log-prob of the test target's codes (TIGER, `tiger_coldstart`)", "",
         f"Test split of `data/processed_coldstart`: {info['n_unseen_users']:,} users with an unseen (held-out) target, "
         f"{info['n_seen_users']:,} with a seen target. Decoder fed the target's own previous codes; full-vocab "
         "log-softmax as in beam search. Mean log-prob [95% bootstrap CI over users]. Uniform over one position's "
         f"{info['codebook_size']} codes: {math.log(1 / info['codebook_size']):.2f}. Code 4 is the collision token "
         "(0 unless items share codes 1–3).", "",
         "| checkpoint | step | targets | " + " | ".join(f"code {p + 1}" for p in range(n_pos)) + " | sum (codes 1–4) |",
         "|---|---:|---|" + "---|" * (n_pos + 1)]
    for ckpt, r in res.items():
        for group in ("unseen", "seen"):
            g = r[group]
            L.append(f"| {ckpt}.pt | {r['step']:,} | {group} | "
                     + " | ".join(f"{c['mean']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}]" for c in g["per_position"])
                     + f" | {g['sum']['mean']:.2f} [{g['sum']['ci_low']:.2f}, {g['sum']['ci_high']:.2f}] |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiger-run", type=Path, default=Path("results/tiger_coldstart"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = yaml.safe_load((args.tiger_run / "config.yaml").read_text())
    split = load_split(cfg["data_dir"])
    sids = load_semantic_ids(cfg["semantic_ids_path"])
    item_sids = np.zeros((split.num_items + 1, sids["num_tokens"]), dtype=np.int64)
    for item, sid in sids["item_to_sid"].items():
        item_sids[int(item)] = sid
    unseen = np.asarray(sids["unseen_items"], dtype=np.int64)
    unseen_mask = np.zeros(split.num_items + 1, dtype=bool)
    unseen_mask[unseen] = True
    targets = np.asarray(split.test)
    is_unseen = unseen_mask[targets]
    reviewers = json.loads((Path(cfg["data_dir"]) / "id_maps.json").read_text())["user2reviewer"]
    data = TigerData.build(split, reviewers, item_sids, **cfg["tokens"])
    device = get_device()
    out_dir = args.results_dir / "coldstart"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = {}
    for ckpt in CHECKPOINTS:
        weights, step = load_weights(args.tiger_run, ckpt, device)
        model = make_tiger(data.tok.vocab_size, **cfg["model"]).to(device)
        model.load_state_dict(weights)
        model.eval()
        lp = target_logprobs(model, data, targets, device)
        np.save(out_dir / f"logprob_test_{ckpt}.npy", lp)
        res[ckpt] = {"step": step}
        for group, mask in (("unseen", is_unseen), ("seen", ~is_unseen)):
            res[ckpt][group] = {"per_position": [bootstrap_ci(lp[mask, p], **cfg["bootstrap"]) for p in range(lp.shape[1])],
                                "sum": bootstrap_ci(lp[mask].sum(1), **cfg["bootstrap"])}
        print(ckpt, step, {g: [round(c["mean"], 2) for c in res[ckpt][g]["per_position"]] for g in ("unseen", "seen")},
              flush=True)

    info = {"device": str(device), "n_unseen_users": int(is_unseen.sum()), "n_seen_users": int((~is_unseen).sum()),
            "num_tokens": int(sids["num_tokens"]), "codebook_size": int(sids["codebook_size"])}
    (out_dir / "logprob_test.json").write_text(json.dumps({"info": info, "results": res}, indent=2))
    (args.results_dir / "coldstart_logprob_test.md").write_text(markdown(res, info))


if __name__ == "__main__":
    main()
