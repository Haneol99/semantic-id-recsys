"""Semantic IDs: RQ-VAE codes per item plus a 4th token that makes every ID unique.

Items sharing the same 3-code prefix get 4th tokens 0, 1, 2, ... in item-ID order; a unique prefix gets 0.

semantic_ids.json: {"codebook_size", "num_tokens", "item_to_sid": {"<item_id>": [c1, c2, c3, c4]},
                    "sid_to_item": {"c1-c2-c3-c4": item_id}}
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np


def add_collision_token(codes: np.ndarray) -> np.ndarray:
    """codes (N, L) for item IDs 1..N in order -> (N, L + 1) with the collision token appended."""
    seen: Counter = Counter()
    extra = np.zeros(len(codes), dtype=np.int64)
    for i, prefix in enumerate(map(tuple, codes)):
        extra[i] = seen[prefix]
        seen[prefix] += 1
    return np.hstack([codes.astype(np.int64), extra[:, None]])


def sid_key(sid) -> str:
    return "-".join(str(int(c)) for c in sid)


def build_lookup(sids: np.ndarray, codebook_size: int) -> dict:
    """sids (N, T) for item IDs 1..N in order -> the semantic_ids.json structure."""
    item_to_sid = {str(i): [int(c) for c in s] for i, s in enumerate(sids, start=1)}
    sid_to_item = {sid_key(s): i for i, s in enumerate(sids, start=1)}
    if len(sid_to_item) != len(item_to_sid):
        raise ValueError("semantic IDs are not unique")
    return {"codebook_size": codebook_size, "num_tokens": int(sids.shape[1]),
            "item_to_sid": item_to_sid, "sid_to_item": sid_to_item}


def save_semantic_ids(path: str | Path, lookup: dict) -> None:
    Path(path).write_text(json.dumps(lookup))


def load_semantic_ids(path: str | Path = "data/processed/semantic_ids.json") -> dict:
    return json.loads(Path(path).read_text())


def collision_stats(codes: np.ndarray) -> dict:
    """Stats of the L-code prefixes (before the collision token)."""
    groups = Counter(map(tuple, codes))
    n = len(codes)
    return {
        "unique_prefixes": len(groups),
        "max_collision_group": max(groups.values()),
        "frac_nonzero_4th_token": (n - len(groups)) / n,  # every item except the first of each group
    }


def codebook_usage(codes: np.ndarray, codebook_size: int) -> list[float]:
    """Per level: fraction of codes used by at least one item."""
    return [len(np.unique(codes[:, d])) / codebook_size for d in range(codes.shape[1])]
