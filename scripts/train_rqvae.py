"""Train the RQ-VAE on all item embeddings, then assign Semantic IDs (3 codes + collision token).

Trains on content embeddings of all 12,101 items (no interactions, so no split leakage). Stops at max_epochs or
when neither reconstruction loss nor min codebook usage has made progress for `patience` evals.

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

    x = torch.as_tensor(np.load(config["emb_path"])[1:], dtype=torch.float32, device=device)  # items 1..N
    model = RQVAE(**config["model"]).to(device)
    k = config["model"]["codebook_size"]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adagrad(model.parameters(), lr=cfg["lr"])
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

        if epoch == 1 or epoch % cfg["eval_every"] == 0 or epoch == cfg["max_epochs"]:
            ev = full_eval(model, x, k)
            entry = {"epoch": epoch, "sec": round(time.perf_counter() - start, 1), **ev}
            log.append(entry)
            print(f"epoch {epoch:5d}  recon {ev['recon_loss']:.3e}  rq {ev['rq_loss']:.3e}  usage "
                  + "/".join(f"{u:.2f}" for u in ev["usage"])
                  + f"  collision {ev['collision_rate']:.4f}  ({entry['sec']:.0f}s)", flush=True)
            progress = ev["recon_loss"] < best_recon * (1 - cfg["min_rel_improvement"]) or min(ev["usage"]) > best_usage
            best_recon, best_usage = min(best_recon, ev["recon_loss"]), max(best_usage, min(ev["usage"]))
            since_progress = 0 if progress else since_progress + 1
            if since_progress >= cfg["patience"]:
                print(f"plateau: no progress for {cfg['patience']} evals")
                break

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    model.eval()
    codes = model.encode_codes(x).cpu().numpy()
    sids = add_collision_token(codes)
    if sids[:, -1].max() >= k:
        raise ValueError(f"collision group larger than codebook size {k}: 4th token {sids[:, -1].max()}")
    sid_path = Path(config["semantic_ids_path"])
    save_semantic_ids(sid_path, build_lookup(sids, k))

    final = log[-1]
    usage = codebook_usage(codes, k)
    metrics = {
        "run_name": config["run_name"], "seed": seed, "git_hash": git_hash(), "device": str(device),
        "data": {"emb_path": config["emb_path"], "sha256": file_sha256(config["emb_path"])},
        "epochs_run": final["epoch"], "total_sec": round(time.perf_counter() - start, 1),
        "recon_loss": final["recon_loss"], "rq_loss": final["rq_loss"],
        "codebook_usage": usage, "passes_usage_80": all(u >= 0.8 for u in usage),
        **collision_stats(codes),
        "semantic_ids": {"path": str(sid_path), "sha256": file_sha256(sid_path)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k_: v for k_, v in metrics.items() if k_ not in ("data", "semantic_ids")}, indent=2))


if __name__ == "__main__":
    main()
