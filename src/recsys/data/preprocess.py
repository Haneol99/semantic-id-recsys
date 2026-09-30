"""Preprocess Amazon 2014 Beauty (5-core) into per-user sequences, a leave-one-out split, and item metadata.

Outputs in data/processed/:
  sequences.json  {"items": [[item_id, ...], ...], "timestamps": [[unix_ts, ...], ...]}; list index = user_id
  splits.json     {"train": [[item_id, ...], ...], "valid": [item_id, ...], "test": [item_id, ...]}; index = user_id
  item_meta.json  {"<item_id>": {"asin", "title", "price", "brand", "categories"}}
  id_maps.json    {"item2asin": [...], "user2reviewer": [...]}; item2asin[0] is None (padding)
  stats.json      dataset statistics

Usage: python -m recsys.data.preprocess [--raw-dir data/raw] [--out-dir data/processed]
"""

import argparse
import ast
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REVIEWS_FILE = "reviews_Beauty_5.json.gz"
META_FILE = "meta_Beauty.json.gz"
MIN_INTERACTIONS = 5
META_FIELDS = ("title", "price", "brand", "categories")


def load_reviews(path: Path) -> list[tuple[str, str, int]]:
    """Return (reviewerID, asin, unixReviewTime) in file order."""
    with gzip.open(path, "rt") as f:
        return [(r["reviewerID"], r["asin"], int(r["unixReviewTime"])) for r in map(json.loads, f)]


def build_sequences(
    reviews: list[tuple[str, str, int]], min_interactions: int = MIN_INTERACTIONS
) -> dict[str, list[tuple[str, int]]]:
    """Group by user, sort each user's reviews by timestamp, and keep users with >= min_interactions.

    The sort is stable, so reviews with the same timestamp (the 2014 data has day resolution)
    keep their order in the raw file.
    """
    by_user: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for user, asin, ts in reviews:
        by_user[user].append((asin, ts))
    return {
        user: sorted(events, key=lambda e: e[1])
        for user, events in by_user.items()
        if len(events) >= min_interactions
    }


def build_item_map(sequences: dict[str, list[tuple[str, int]]]) -> dict[str, int]:
    """Map ASINs to 1..N in ASIN-sorted order (0 is padding), not in order of first appearance."""
    asins = sorted({asin for events in sequences.values() for asin, _ in events})
    return {asin: i for i, asin in enumerate(asins, start=1)}


def leave_one_out(seq: list[int]) -> tuple[list[int], int, int]:
    """Split one sequence into (train items, valid target, test target)."""
    return seq[:-2], seq[-2], seq[-1]


def load_meta(path: Path, asins: set[str]) -> dict[str, dict]:
    """Parse the 2014 meta file (Python dict literals, one per line), keeping only the given ASINs."""
    meta = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            record = ast.literal_eval(line)
            if record["asin"] in asins:
                meta[record["asin"]] = {k: record.get(k) for k in META_FIELDS}
    return meta


def sequence_stats(item_seqs: list[list[int]], num_items: int) -> dict:
    lengths = np.array([len(s) for s in item_seqs])
    return {
        "num_users": len(item_seqs),
        "num_items": num_items,
        "num_interactions": int(lengths.sum()),
        "mean_seq_len": float(lengths.mean()),
        "median_seq_len": float(np.median(lengths)),
        "min_seq_len": int(lengths.min()),
        "max_seq_len": int(lengths.max()),
    }


def preprocess(raw_dir: Path, out_dir: Path) -> dict:
    sequences = build_sequences(load_reviews(raw_dir / REVIEWS_FILE))
    item_map = build_item_map(sequences)

    users = sorted(sequences)  # user_id = index in reviewerID-sorted order
    item_seqs = [[item_map[asin] for asin, _ in sequences[u]] for u in users]
    ts_seqs = [[ts for _, ts in sequences[u]] for u in users]
    splits = [leave_one_out(s) for s in item_seqs]

    meta = load_meta(raw_dir / META_FILE, set(item_map))
    item_meta = {
        str(item_id): {"asin": asin, **meta.get(asin, dict.fromkeys(META_FIELDS))}
        for asin, item_id in item_map.items()
    }

    stats = sequence_stats(item_seqs, len(item_map))
    stats["items_with_meta"] = len(meta)
    stats["meta_field_coverage"] = {
        k: sum(m[k] is not None for m in item_meta.values()) for k in META_FIELDS
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sequences.json": {"items": item_seqs, "timestamps": ts_seqs},
        "splits.json": {
            "train": [s[0] for s in splits],
            "valid": [s[1] for s in splits],
            "test": [s[2] for s in splits],
        },
        "item_meta.json": item_meta,
        "id_maps.json": {"item2asin": [None, *sorted(item_map, key=item_map.get)], "user2reviewer": users},
        "stats.json": stats,
    }
    for name, obj in outputs.items():
        with open(out_dir / name, "w") as f:
            json.dump(obj, f)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Amazon 2014 Beauty 5-core.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    stats = preprocess(args.raw_dir, args.out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
