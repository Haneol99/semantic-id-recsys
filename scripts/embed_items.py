"""Encode every item's metadata sentence with Sentence-T5 (768-d).

Writes data/processed/item_emb.npy (row = item ID, row 0 = zeros for padding) and item_emb.json
(model name, shape, SHA-256 of the .npy). Item IDs are identical in data/processed and data/processed_tieshuffle.

Usage: python scripts/embed_items.py [--processed-dir data/processed] [--model sentence-t5-base]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from recsys.data.item_text import load_item_sentences
from recsys.utils import file_sha256, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    sentences = load_item_sentences(args.processed_dir)
    for i in (1, len(sentences) // 2, len(sentences)):
        print(f"item {i}: {sentences[i - 1]}")

    device = get_device()
    print(f"device {device}  model {args.model}  items {len(sentences):,}")
    model = SentenceTransformer(args.model, device=str(device))
    emb = model.encode(sentences, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    emb = np.vstack([np.zeros((1, emb.shape[1]), dtype=np.float32), emb.astype(np.float32)])

    out = args.processed_dir / "item_emb.npy"
    np.save(out, emb)
    info = {"model": args.model, "device": str(device), "shape": list(emb.shape), "sha256": file_sha256(out),
            "row0": "zeros (padding)", "text_fields": ["title", "price", "brand", "categories"]}
    (args.processed_dir / "item_emb.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
