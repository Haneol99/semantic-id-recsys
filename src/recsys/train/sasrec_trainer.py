"""SASRec training: next-item prediction at every position, full cross-entropy over all items,
early stopping on validation NDCG@10 (shared full-ranking evaluator)."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recsys.data.dataset import Split
from recsys.eval.evaluator import evaluate
from recsys.models.sasrec import SASRec, pad_left


def build_training_pairs(train_seqs: list[list[int]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Per user: inputs = train[:-1], targets = train[1:], both truncated to the last max_len and left-padded."""
    inputs = pad_left([s[:-1] for s in train_seqs], max_len)
    targets = pad_left([s[1:] for s in train_seqs], max_len)
    return inputs, targets


def sequence_ce_loss(model: SASRec, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over items 1..N at every non-padding position (padding item excluded from the softmax)."""
    hidden = model(inputs)
    valid = targets != 0
    logits = model.item_logits(hidden[valid])
    logits[:, 0] = -torch.inf
    return F.cross_entropy(logits, targets[valid])


def evaluate_split(model: SASRec, split: Split, name: str, batch_size: int = 1024):
    model.eval()
    histories = split.inputs(name)
    return evaluate(lambda users: model.score([histories[u] for u in users]), histories, split.targets(name),
                    batch_size=batch_size)


def train_sasrec(model: SASRec, split: Split, cfg: dict, device: torch.device, ckpt_path: Path, seed: int) -> dict:
    """Train with early stopping; the best model (by valid NDCG@10) is saved to ckpt_path and reloaded."""
    inputs, targets = build_training_pairs(split.train, model.max_len)
    inputs, targets = torch.as_tensor(inputs), torch.as_tensor(targets)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                           weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    metric, patience = cfg["early_stop_metric"], cfg["patience"]

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, evals_since_best = -1.0, 0, 0
    log, train_times = [], []
    start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(inputs), generator=gen)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            loss = sequence_ce_loss(model, inputs[idx].to(device), targets[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        train_times.append(time.perf_counter() - t0)
        entry = {"epoch": epoch, "loss": total_loss / n_batches, "train_sec": train_times[-1]}

        if epoch % cfg["eval_every"] == 0:
            t1 = time.perf_counter()
            means, _ = evaluate_split(model, split, "valid")
            entry.update({"eval_sec": time.perf_counter() - t1, **{f"valid_{k}": v for k, v in means.items()}})
            if means[metric] > best:
                best, best_epoch, evals_since_best = means[metric], epoch, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                evals_since_best += 1
            print(f"epoch {epoch:3d}  loss {entry['loss']:.4f}  valid {metric} {means[metric]:.4f}  "
                  f"best {best:.4f}@{best_epoch}  ({entry['train_sec']:.1f}s train, {entry['eval_sec']:.1f}s eval)",
                  flush=True)
        log.append(entry)
        if evals_since_best >= patience:
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return {
        "best_epoch": best_epoch,
        f"best_valid_{metric}": best,
        "epochs_run": epoch,
        "train_sec_per_epoch_mean": float(np.mean(train_times)),
        "total_sec": time.perf_counter() - start,
        "log": log,
    }
