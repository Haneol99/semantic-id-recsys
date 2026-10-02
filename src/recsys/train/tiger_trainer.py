"""TIGER training: teacher-forced cross-entropy on the next item's Semantic ID. Model selection (best.pt) by NDCG@10
of trie-constrained beam search over a fixed random subset of valid users; optional early stopping (patience).
Checkpoints (last.pt) allow resuming.

Learning rate (paper): `lr` for the first `constant_steps` steps, then lr * sqrt(constant_steps / step); optional
linear warmup over the first `warmup_steps` steps (0 for the paper schedule).

Collapse check: every eval logs the number of distinct items across the subset's top-k lists and warns when it is
below `min_distinct_items`. A model that ignores its input gives (nearly) the same list to every user; the first
Adafactor run without parameter scaling gave 15 distinct items for 2,000 users.
"""

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers.optimization import Adafactor

from recsys.data.dataset import Split
from recsys.data.tiger_data import TigerTokenizer, training_examples
from recsys.eval.evaluator import evaluate_ranked_lists, filter_ranked
from recsys.models.tiger import SemanticIDTrie, constrained_beam_search


@dataclass
class TigerData:
    split: Split
    tok: TigerTokenizer
    user_tokens: np.ndarray  # (num_users,)
    train_inputs: torch.Tensor  # (num_examples, max_input_len)
    train_targets: torch.Tensor  # (num_examples, 4)

    @classmethod
    def build(cls, split: Split, reviewers: list[str], item_sids: np.ndarray, **tok_kwargs) -> "TigerData":
        tok = TigerTokenizer(item_sids, **tok_kwargs)
        user_tokens = np.array([tok.user_token(r) for r in reviewers], dtype=np.int64)
        users, histories, targets = training_examples(split.train)
        inputs = tok.encode_inputs(histories, user_tokens[users])
        return cls(split, tok, user_tokens, torch.as_tensor(inputs), torch.as_tensor(tok.encode_targets(targets)))

    def eval_inputs(self, name: str, users: np.ndarray) -> torch.Tensor:
        histories = self.split.inputs(name)
        return torch.as_tensor(self.tok.encode_inputs([histories[u] for u in users], self.user_tokens[users]))


def lr_factor(step: int, constant_steps: int, warmup_steps: int = 0) -> float:
    """LambdaLR factor; `step` = optimizer steps already taken, so the first update uses step 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0 if step <= constant_steps else math.sqrt(constant_steps / step)


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    if cfg["optimizer"] == "adafactor":
        # scale_parameter=True (T5 / Mesh-TF default): each update is lr x the parameter's RMS, i.e. relative.
        # With False the update size is ~lr per weight regardless of scale; at lr 0.01 that is the whole init scale
        # of T5's attention weights per step, and the encoder collapsed (STATUS, Phase 3).
        return Adafactor(model.parameters(), lr=cfg["lr"], scale_parameter=cfg["scale_parameter"], relative_step=False,
                         warmup_init=False, weight_decay=cfg.get("weight_decay", 0.0))
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")


def release_mps_cache(device) -> None:
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def evaluate_users(model, data: TigerData, trie: SemanticIDTrie, name: str, users: np.ndarray, device,
                   beam_size: int, top_k: int = 10, batch_size: int = 256):
    """Beam search -> drop history items -> top_k list -> shared ranked-list evaluator.

    Returns (means, per_user, list_stats); list_stats = short lists (< top_k items) and distinct items over all lists.
    """
    model.eval()
    histories = data.split.inputs(name)
    ranked = []
    for i in range(0, len(users), batch_size):
        ub = users[i:i + batch_size]
        x = data.eval_inputs(name, ub).to(device)
        items, _ = constrained_beam_search(model, x, (x != 0).long(), trie, beam_size)
        ranked += filter_ranked(items.cpu().numpy(), [histories[u] for u in ub], top_k)
        release_mps_cache(device)  # cached beam-search buffers otherwise push training into swap
    hist_u = [histories[u] for u in users]
    targets = [data.split.targets(name)[u] for u in users]
    means, per_user = evaluate_ranked_lists(ranked, hist_u, targets)
    list_stats = {"short_lists": int(sum(len(r) < top_k for r in ranked)),
                  "distinct_items": len({i for r in ranked for i in r})}
    model.train()
    return means, per_user, list_stats


def train_tiger(model, data: TigerData, trie: SemanticIDTrie, cfg: dict, device, out_dir: Path, seed: int,
                resume: bool = False, max_minutes: float | None = None) -> dict:
    """Train until max_steps, early stop (patience evals without a better subset NDCG@10; patience None = off),
    or max_minutes.

    Saves best.pt (best subset NDCG@10) and last.pt (full state for resuming) at every eval.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, cfg["constant_steps"], cfg.get("warmup_steps", 0)))
    n, bs = len(data.train_inputs), cfg["batch_size"]
    steps_per_epoch = math.ceil(n / bs)
    subset = np.sort(np.random.default_rng(cfg["valid_subset_seed"]).choice(
        data.split.num_users, size=min(cfg["valid_subset_size"], data.split.num_users), replace=False))

    state = {"step": 0, "best": -1.0, "best_step": 0, "evals_since_best": 0, "log": [], "train_sec": 0.0}
    if resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        print(f"resumed at step {state['step']}", flush=True)

    def save_last():
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "state": state}, out_dir / "last.pt")

    model.train()
    start, window_loss, window_steps, window_t0 = time.perf_counter(), 0.0, 0, time.perf_counter()
    stop_reason = "max_steps"
    while state["step"] < cfg["max_steps"]:
        epoch, pos = divmod(state["step"], steps_per_epoch)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch))
        for b in range(pos, steps_per_epoch):
            idx = perm[b * bs:(b + 1) * bs]
            x, y = data.train_inputs[idx].to(device), data.train_targets[idx].to(device)
            loss = model(input_ids=x, attention_mask=(x != 0).long(), labels=y).loss
            opt.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            state["step"] += 1
            window_loss += loss.item()
            window_steps += 1
            step = state["step"]

            if step % cfg["log_every"] == 0:
                dt = time.perf_counter() - window_t0
                state["train_sec"] += dt
                entry = {"step": step, "loss": window_loss / window_steps, "lr": sched.get_last_lr()[0],
                         "steps_per_sec": window_steps / dt, "elapsed_sec": round(time.perf_counter() - start, 1)}
                state["log"].append(entry)
                print(f"step {step:6d}  loss {entry['loss']:.4f}  lr {entry['lr']:.2e}  "
                      f"{entry['steps_per_sec']:.1f} steps/s", flush=True)
                if not math.isfinite(entry["loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                window_loss, window_steps, window_t0 = 0.0, 0, time.perf_counter()

            if step % cfg["eval_every"] == 0:
                t0 = time.perf_counter()
                means, _, lists = evaluate_users(model, data, trie, "valid", subset, device, cfg["beam_size"])
                metric = means["ndcg@10"]
                improved = metric > state["best"]
                if improved:
                    state.update(best=metric, best_step=step, evals_since_best=0)
                    torch.save(model.state_dict(), out_dir / "best.pt")
                else:
                    state["evals_since_best"] += 1
                state["log"].append({"step": step, **{f"subset_valid_{k}": v for k, v in means.items()},
                                     **lists, "eval_sec": round(time.perf_counter() - t0, 1)})
                print(f"  eval step {step}: subset valid ndcg@10 {metric:.4f} recall@10 {means['recall@10']:.4f}  "
                      f"best {state['best']:.4f}@{state['best_step']}  distinct items {lists['distinct_items']}  "
                      f"short lists {lists['short_lists']}  "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
                if lists["distinct_items"] < cfg.get("min_distinct_items", 0):
                    msg = (f"possible collapse at step {step}: only {lists['distinct_items']} distinct items in "
                           f"{len(subset)} users' top-10 lists (< {cfg['min_distinct_items']}); is the input ignored?")
                    print(f"  WARNING: {msg}", flush=True)
                    warnings.warn(msg, stacklevel=1)
                save_last()
                window_t0 = time.perf_counter()  # eval time is not training time
                if cfg["patience"] is not None and state["evals_since_best"] >= cfg["patience"]:
                    stop_reason = "early_stop"
                    break

            if max_minutes is not None and time.perf_counter() - start > max_minutes * 60:
                stop_reason = "max_minutes"
                break
            if step >= cfg["max_steps"]:
                break
        if stop_reason != "max_steps":
            break

    save_last()
    (out_dir / "train_log.json").write_text(json.dumps(state["log"], indent=2))
    return {"stop_reason": stop_reason, "steps": state["step"], "best_step": state["best_step"],
            "best_subset_valid_ndcg@10": state["best"], "valid_subset_size": len(subset),
            "steps_per_epoch": steps_per_epoch, "train_sec": round(state["train_sec"], 1)}
