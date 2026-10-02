"""Train the RQ-VAE on all item embeddings, then assign Semantic IDs (3 codes + collision token).

Trains on content embeddings of all 12,101 items (no interactions, so no split leakage). With `holdout_path`
(cold start, a coldstart.json with "held_out_items") it trains on the seen items only, then assigns codes to all
items; seen items get the collision tokens first (see recsys.data.semantic_ids).

Schedule: dead codes are reset every `reset_every` epochs up to epoch `reset_until` (0 = never). Final assignments
come from at least `stable_epochs` epochs without resets; after that, training stops at max_epochs or when
neither reconstruction loss nor min codebook usage has made progress for `patience` evals.

Writes results/rqvae/{config.yaml, model.pt, train_log.json, metrics.json} and the semantic_ids_path JSON.

Usage: python scripts/train_rqvae.py [--config configs/rqvae.yaml]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from recsys.data.semantic_ids import add_collision_token, build_lookup, codebook_usage, collision_stats, save_semantic_ids
from recsys.models.rqvae import RQVAE
from recsys.utils import file_sha256, get_device, git_hash, load_config, set_seed


@torch.no_grad()
def full_eval(model: RQVAE, x: torch.Tensor, codebook_size: int) -> dict:
    model.eval()
    out = model(x)
    codes = out["codes"].cpu().numpy()
    stats = collision_stats(codes)
    model.train()
    return {
        "recon_loss": out["recon_loss"].item(),
        "rq_loss": out["rq_loss"].item(),
        "usage": codebook_usage(codes, codebook_size),
        "collision_rate": stats["frac_nonzero_4th_token"],
        "unique_prefixes": stats["unique_prefixes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rqvae.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg, seed = config["train"], config["seed"]
    set_seed(seed)
    device = torch.device(config["device"]) if config.get("device") not in (None, "auto") else get_device()
    out_dir = Path(config["results_dir"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    x_all = torch.as_tensor(np.load(config["emb_path"])[1:], dtype=torch.float32, device=device)  # items 1..N
    seen_mask = np.ones(len(x_all), dtype=bool)
    held_out = None
    if config.get("holdout_path"):
        held_out = json.loads(Path(config["holdout_path"]).read_text())["held_out_items"]
        seen_mask[np.asarray(held_out) - 1] = False
    x = x_all[torch.as_tensor(seen_mask, device=device)]  # training items
    model = RQVAE(**config["model"]).to(device)
    k = config["model"]["codebook_size"]
    gen = torch.Generator().manual_seed(seed)
    if config.get("input_norm", "none") == "standardize":
        model.set_input_standardization(x)
    opt = {"adagrad": torch.optim.Adagrad, "adamw": torch.optim.AdamW}[cfg["optimizer"]](
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    reset_every, reset_until = cfg.get("reset_every", 0), cfg.get("reset_until", 0)
    stop_allowed_from = reset_until + cfg.get("stable_epochs", 0)
    resets_per_level = [0] * config["model"]["levels"]
    print(f"device {device}  items {len(x):,}  params {sum(p.numel() for p in model.parameters()):,}")

    log, best_recon, best_usage, since_progress = [], float("inf"), -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        perm = torch.randperm(len(x), generator=gen).to(device)
        if epoch == 1:
            model.kmeans_init(x[perm[:cfg["batch_size"]]], seed)  # k-means on the first batch, level by level
        total = 0.0
        for i in range(0, len(x), cfg["batch_size"]):
            loss = model(x[perm[i:i + cfg["batch_size"]]])["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        n_reset = None
        if reset_every and epoch <= reset_until and epoch % reset_every == 0:
            n_reset = model.reset_dead_codes(x, gen)
            resets_per_level = [a + b for a, b in zip(resets_per_level, n_reset)]

        if epoch == 1 or epoch % cfg["eval_every"] == 0 or epoch == cfg["max_epochs"]:
            ev = full_eval(model, x, k)
            entry = {"epoch": epoch, "sec": round(time.perf_counter() - start, 1), **ev,
                     "resets_so_far": list(resets_per_level)}
            log.append(entry)
            print(f"epoch {epoch:5d}  recon {ev['recon_loss']:.3e}  rq {ev['rq_loss']:.3e}  usage "
                  + "/".join(f"{u:.2f}" for u in ev["usage"])
                  + f"  collision {ev['collision_rate']:.4f}  resets {resets_per_level}  ({entry['sec']:.0f}s)",
                  flush=True)
            progress = ev["recon_loss"] < best_recon * (1 - cfg["min_rel_improvement"]) or min(ev["usage"]) > best_usage
            best_recon, best_usage = min(best_recon, ev["recon_loss"]), max(best_usage, min(ev["usage"]))
            since_progress = 0 if progress else since_progress + 1
            if epoch >= stop_allowed_from and since_progress >= cfg["patience"]:
                print(f"plateau: no progress for {cfg['patience']} evals")
                break

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    model.eval()
    codes_all = model.encode_codes(x_all).cpu().numpy()
    codes = codes_all[seen_mask]  # training items: usage and collision stats below
    sids = add_collision_token(codes_all, seen_mask if held_out is not None else None)
    if sids[:, -1].max() >= k:
        raise ValueError(f"collision group larger than codebook size {k}: 4th token {sids[:, -1].max()}")
    sid_path = Path(config["semantic_ids_path"])
    save_semantic_ids(sid_path, build_lookup(sids, k, held_out))

    final = log[-1]
    usage = codebook_usage(codes, k)
    metrics = {
        "run_name": config["run_name"], "seed": seed, "git_hash": git_hash(), "device": str(device),
        "data": {"emb_path": config["emb_path"], "sha256": file_sha256(config["emb_path"])},
        "epochs_run": final["epoch"], "optimizer": cfg["optimizer"], "input_norm": config.get("input_norm", "none"),
        "dead_code_resets_per_level": resets_per_level, "last_reset_epoch": min(reset_until, final["epoch"]), "total_sec": round(time.perf_counter() - start, 1),
        "recon_loss": final["recon_loss"], "rq_loss": final["rq_loss"],
        "codebook_usage": usage, "passes_usage_80": all(u >= 0.8 for u in usage),
        **collision_stats(codes),
        "semantic_ids": {"path": str(sid_path), "sha256": file_sha256(sid_path)},
    }
    if held_out is not None:
        unseen_codes = codes_all[~seen_mask]
        seen_prefixes = {tuple(c) for c in codes}
        metrics["coldstart"] = {
            "holdout_path": config["holdout_path"], "num_train_items": int(seen_mask.sum()),
            "num_unseen_items": len(held_out), "codebook_usage_all_items": codebook_usage(codes_all, k),
            "unseen_items_with_seen_3code_prefix": int(sum(tuple(c) in seen_prefixes for c in unseen_codes)),
            "unseen_items_with_seen_first_code": int(np.isin(unseen_codes[:, 0], codes[:, 0]).sum()),
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k_: v for k_, v in metrics.items() if k_ not in ("data", "semantic_ids")}, indent=2))


if __name__ == "__main__":
    main()
