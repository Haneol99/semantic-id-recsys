"""Train TIGER, then evaluate the best checkpoint on full valid and test (once).

Usage:
  python scripts/train_tiger.py [--config configs/tiger_tieshuffle.yaml] [--resume]
  python scripts/train_tiger.py --pilot-minutes 15   # time-boxed pilot: results/<run_name>_pilot, no test eval

For the long run, keep the Mac awake:  caffeinate -i python scripts/train_tiger.py --resume
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from recsys.data.dataset import load_split
from recsys.data.semantic_ids import load_semantic_ids
from recsys.models.tiger import SemanticIDTrie, make_tiger
from recsys.train.tiger_trainer import TigerData, evaluate_users, train_tiger
from recsys.utils import file_sha256, get_device, load_config, save_run, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tiger_tieshuffle.yaml")
    parser.add_argument("--resume", action="store_true", help="continue from results/<run_name>/last.pt")
    parser.add_argument("--pilot-minutes", type=float, default=None, help="time-boxed pilot without test eval")
    parser.add_argument("--eval-every", type=int, default=None, help="override train.eval_every (e.g. pilot)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.pilot_minutes is not None:
        config = {**config, "run_name": config["run_name"] + "_pilot"}
    if args.eval_every is not None:
        config = {**config, "train": {**config["train"], "eval_every": args.eval_every}}
    set_seed(config["seed"])
    device = get_device()
    out_dir = Path(config["results_dir"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    split = load_split(config["data_dir"])
    reviewers = json.loads((Path(config["data_dir"]) / "id_maps.json").read_text())["user2reviewer"]
    sids = load_semantic_ids(config["semantic_ids_path"])
    item_sids = np.zeros((split.num_items + 1, sids["num_tokens"]), dtype=np.int64)
    for item, sid in sids["item_to_sid"].items():
        item_sids[int(item)] = sid
    data = TigerData.build(split, reviewers, item_sids, **config["tokens"])
    trie = SemanticIDTrie(item_sids, data.tok.codebook_size, device=device)

    model = make_tiger(data.tok.vocab_size, **config["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"device {device}  params {num_params:,}  vocab {data.tok.vocab_size}  "
          f"train examples {len(data.train_inputs):,}  users {split.num_users:,}", flush=True)

    info = train_tiger(model, data, trie, config["train"], device, out_dir, config["seed"], resume=args.resume,
                       max_minutes=args.pilot_minutes)
    print(json.dumps(info, indent=2))
    if args.pilot_minutes is not None:
        return  # pilot: no full valid / test evaluation

    import torch
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    users = np.arange(split.num_users)
    results, list_stats = {}, {}
    for s in ("valid", "test"):  # test is evaluated exactly once, with the best checkpoint
        means, per_user, list_stats[s] = evaluate_users(model, data, trie, s, users, device, config["train"]["beam_size"])
        results[s] = (means, per_user)
    metrics = save_run(out_dir, config, results, config["bootstrap"], extra={
        "device": str(device), "num_params": num_params, **info, "list_stats": list_stats,
        "semantic_ids_sha256": file_sha256(config["semantic_ids_path"]),
    })
    for s, m in metrics["splits"].items():
        print(f"[{s}] " + "  ".join(f"{k}={v['mean']:.4f}" for k, v in m.items() if k != "num_users"))


if __name__ == "__main__":
    main()
