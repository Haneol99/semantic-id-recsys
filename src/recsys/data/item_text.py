"""One sentence per item from its metadata (title, price, brand, categories), for Sentence-T5 embeddings.

Missing or empty fields are left out. HTML entities in the raw text (e.g. "&amp;") are unescaped.
"""

import html
import json
from pathlib import Path


def _clean(text) -> str:
    return " ".join(html.unescape(str(text)).split()) if text is not None else ""


def item_sentence(meta: dict) -> str:
    """e.g. 'Title: X. Price: $5.04. Brand: Y. Categories: Beauty > Makeup > Face.'"""
    parts = []
    if title := _clean(meta.get("title")):
        parts.append(f"Title: {title}.")
    if meta.get("price") is not None:
        parts.append(f"Price: ${float(meta['price']):.2f}.")
    if brand := _clean(meta.get("brand")):
        parts.append(f"Brand: {brand}.")
    paths = [" > ".join(_clean(c) for c in path if _clean(c)) for path in meta.get("categories") or []]
    if paths := [p for p in paths if p]:
        parts.append(f"Categories: {'; '.join(paths)}.")
    return " ".join(parts)


def load_item_sentences(processed_dir: str | Path) -> list[str]:
    """Sentences for item IDs 1..N in order (index 0 = item 1)."""
    meta = json.loads((Path(processed_dir) / "item_meta.json").read_text())
    return [item_sentence(meta[str(i)]) for i in range(1, len(meta) + 1)]


def top_category(meta: dict) -> str | None:
    """Second level of the first category path (the first level is 'Beauty' for every item), e.g. 'Skin Care'."""
    cats = meta.get("categories") or []
    return cats[0][1] if cats and len(cats[0]) > 1 else None
