import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recsys.data.item_text import item_sentence, top_category
from recsys.data.semantic_ids import add_collision_token, build_lookup, codebook_usage, collision_stats, sid_key
from recsys.models.rqvae import RQVAE

SID_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "semantic_ids.json"
NUM_ITEMS = 12101


# ---------- item text ----------

def test_item_sentence_all_fields_and_html_unescape():
    meta = {"title": "Shea &amp; Cocoa", "price": 5.0, "brand": "B", "categories": [["Beauty", "Skin Care"]]}
    assert item_sentence(meta) == "Title: Shea & Cocoa. Price: $5.00. Brand: B. Categories: Beauty > Skin Care."


def test_item_sentence_skips_missing_fields():
    assert item_sentence({"title": "T", "price": None, "brand": "", "categories": None}) == "Title: T."
    assert item_sentence({"title": None, "price": None, "brand": None, "categories": [[]]}) == ""


def test_top_category_is_second_level():
    assert top_category({"categories": [["Beauty", "Makeup", "Eyes"]]}) == "Makeup"
    assert top_category({"categories": [["Beauty"]]}) is None


# ---------- collision token and lookup (synthetic) ----------

def test_collision_token_makes_ids_unique():
    codes = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3]])
    sids = add_collision_token(codes)
    assert sids[:, -1].tolist() == [0, 1, 0, 2]
    assert collision_stats(codes) == {"unique_prefixes": 2, "max_collision_group": 3, "frac_nonzero_4th_token": 0.5}
    lookup = build_lookup(sids, 256)
    assert lookup["sid_to_item"][sid_key([1, 2, 3, 2])] == 4
    assert codebook_usage(codes, 8) == [2 / 8, 2 / 8, 2 / 8]


# ---------- RQ-VAE model ----------

def test_rqvae_shapes_and_kmeans_init():
    torch.manual_seed(0)
    x = torch.randn(300, 16)
    model = RQVAE(input_dim=16, hidden_dims=(12, 8), latent_dim=4, levels=3, codebook_size=8)
    model.kmeans_init(x, seed=0)
    out = model(x)
    assert out["codes"].shape == (300, 3) and out["codes"].min() >= 0 and out["codes"].max() < 8
    assert codebook_usage(out["codes"].numpy(), 8)[0] == 1.0  # k-means init: every level-1 code is used
    out["loss"].backward()
    assert model.encoder[0].weight.grad is not None  # straight-through gradient reaches the encoder


# ---------- real Semantic IDs ----------

@pytest.fixture(scope="module")
def sids():
    if not SID_PATH.exists():
        pytest.skip("semantic_ids.json not found; run scripts/train_rqvae.py")
    return json.loads(SID_PATH.read_text())


def test_every_item_has_unique_4_token_id(sids):
    item_to_sid = sids["item_to_sid"]
    assert set(item_to_sid) == {str(i) for i in range(1, NUM_ITEMS + 1)}
    assert all(len(s) == 4 for s in item_to_sid.values())
    assert len({tuple(s) for s in item_to_sid.values()}) == NUM_ITEMS


def test_lookup_round_trips(sids):
    assert len(sids["sid_to_item"]) == NUM_ITEMS
    for item, sid in sids["item_to_sid"].items():
        assert sids["sid_to_item"][sid_key(sid)] == int(item)


def test_codes_within_codebook(sids):
    k = sids["codebook_size"]
    assert k == 256
    assert all(0 <= c < k for s in sids["item_to_sid"].values() for c in s)
