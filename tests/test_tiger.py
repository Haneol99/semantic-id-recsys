import numpy as np
import pytest
import torch
import torch.nn.functional as F

from recsys.data.dataset import Split
from recsys.data.tiger_data import FIRST_CODE_TOKEN, PAD, TigerTokenizer, training_examples
from recsys.eval.evaluator import MISS_RANK, evaluate_ranked_lists, filter_ranked
from recsys.models.tiger import SemanticIDTrie, constrained_beam_search, make_tiger
from recsys.train.tiger_trainer import TigerData

K = 4  # codebook size in the toy tests
# items 1..6 (row 0 = padding); items 1-3 share the prefix (0, 1, 2)
SIDS = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 2], [3, 0, 1, 0], [3, 3, 3, 0], [1, 2, 0, 0]])


def tok(max_history=2):
    return TigerTokenizer(SIDS, codebook_size=K, num_user_tokens=5, max_history=max_history)


# ---------- input / target alignment ----------

def test_training_examples_are_next_item_positions_inside_train():
    users, hist, targets = training_examples([[5, 6, 7], [1, 2]])
    assert users == [0, 0, 1]
    assert hist == [[5], [5, 6], [1]]
    assert targets == [6, 7, 2]


def test_encode_inputs_user_token_then_last_items_oldest_first():
    t = tok(max_history=2)
    x = t.encode_inputs([[1, 4, 6], [2]], [t.first_user_token + 3, t.first_user_token])
    code = lambda item: [FIRST_CODE_TOKEN + p * K + c for p, c in enumerate(SIDS[item])]
    assert x.shape == (2, 1 + 2 * 4)
    assert x[0].tolist() == [t.first_user_token + 3, *code(4), *code(6)]  # item 1 dropped (max_history 2)
    assert x[1].tolist() == [t.first_user_token, *code(2)] + [PAD] * 4
    assert t.encode_targets([5]).tolist() == [code(5)]
    assert t.vocab_size == 2 + 4 * K + 5


def test_code_tokens_are_separate_per_position():
    t = tok()
    assert t.encode_targets([1]).tolist() == [[2, 2 + 4 + 1, 2 + 8 + 2, 2 + 12 + 0]]


def test_eval_inputs_valid_uses_train_and_test_uses_train_plus_valid():
    split = Split(train=[[1, 2, 3]], valid=[4], test=[5], num_items=6)
    data = TigerData.build(split, ["r0"], SIDS, codebook_size=K, num_user_tokens=5, max_history=20)
    t = data.tok
    assert data.eval_inputs("valid", np.array([0])).tolist() == t.encode_inputs([[1, 2, 3]], data.user_tokens).tolist()
    assert data.eval_inputs("test", np.array([0])).tolist() == t.encode_inputs([[1, 2, 3, 4]], data.user_tokens).tolist()
    # training targets come from train only: 2 and 3, never the valid/test items
    assert data.train_targets.tolist() == t.encode_targets([2, 3]).tolist()


# ---------- trie-constrained beam search ----------

def test_trie_allows_only_existing_prefixes():
    trie = SemanticIDTrie(SIDS, codebook_size=K)
    assert trie.allowed_next(0, torch.tensor([0])).tolist() == [[True, True, False, True]]
    key = lambda *codes: sum(c * K ** (len(codes) - 1 - i) for i, c in enumerate(codes))
    assert trie.allowed_next(3, torch.tensor([key(0, 1, 2)])).tolist() == [[True, True, True, False]]
    assert trie.items(torch.tensor([key(3, 3, 3, 0), key(3, 3, 3, 1)])).tolist() == [5, -1]


def sequence_logprob(model, x, sids):
    """Exact log p(Semantic ID | x) for every item by teacher forcing."""
    n = len(sids)
    targets = torch.as_tensor(sids + FIRST_CODE_TOKEN + np.arange(4) * K)
    dec = torch.cat([torch.full((n, 1), PAD), targets[:, :-1]], 1)
    logits = model(input_ids=x.expand(n, -1), attention_mask=(x != 0).long().expand(n, -1),
                   decoder_input_ids=dec).logits
    return F.log_softmax(logits, -1).gather(2, targets[..., None]).squeeze(-1).sum(1)


def test_beam_search_returns_only_valid_items_and_matches_exact_ranking():
    torch.manual_seed(0)
    t = tok()
    model = make_tiger(t.vocab_size, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8, d_model=16,
                       d_ff=32).eval()
    trie = SemanticIDTrie(SIDS, codebook_size=K)
    x = torch.as_tensor(t.encode_inputs([[1, 4], [6]], [t.first_user_token, t.first_user_token + 1]))
    items, scores = constrained_beam_search(model, x, (x != 0).long(), trie, beam_size=6)
    assert set(items.flatten().tolist()) <= set(range(1, 7))  # only real items
    assert all(len(set(row)) == 6 for row in items.tolist())
    with torch.no_grad():
        for b in range(2):  # beam >= #items -> exact: same order as full sequence log-probs
            exact = sequence_logprob(model, x[b:b + 1], SIDS[1:])
            assert items[b].tolist() == (exact.argsort(descending=True) + 1).tolist()
            torch.testing.assert_close(scores[b], exact.sort(descending=True).values, atol=1e-4, rtol=1e-4)


def test_small_beam_still_returns_valid_items():
    torch.manual_seed(1)
    t = tok()
    model = make_tiger(t.vocab_size, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8, d_model=16,
                       d_ff=32).eval()
    x = torch.as_tensor(t.encode_inputs([[2]], [t.first_user_token]))
    items, _ = constrained_beam_search(model, x, (x != 0).long(), SemanticIDTrie(SIDS, K), beam_size=2)
    assert items.shape == (1, 2) and all(1 <= i <= 6 for i in items[0].tolist())


# ---------- history exclusion and ranked-list evaluation ----------

def test_filter_ranked_drops_history_and_invalid_then_truncates():
    items = np.array([[3, 1, -1, 5, 2, 6]])
    assert filter_ranked(items, [[1, 2]], top_k=3) == [[3, 5, 6]]


def test_ranked_list_eval_misses_and_rejects_history():
    means, per_user = evaluate_ranked_lists([[3, 5], [4, 6]], [[1], [2]], [5, 1], ks=[1, 10])
    assert per_user["rank"].tolist() == [1, MISS_RANK]  # short list: absent target is a miss, not rank 2
    assert means["recall@10"] == 0.5 and means["recall@1"] == 0.0
    with pytest.raises(ValueError, match="history"):
        evaluate_ranked_lists([[1, 3]], [[1]], [3])


# ---------- training: resume and early stopping ----------

def _toy_training(tmp_path, max_steps, patience, resume=False):
    from recsys.train.tiger_trainer import train_tiger
    split = Split(train=[[1, 2, 3], [4, 5, 6], [6, 1, 2], [3, 4, 5]], valid=[4, 1, 3, 6], test=[5, 2, 4, 1], num_items=6)
    data = TigerData.build(split, ["r0", "r1", "r2", "r3"], SIDS, codebook_size=K, num_user_tokens=5, max_history=20)
    trie = SemanticIDTrie(SIDS, codebook_size=K)
    torch.manual_seed(0)
    model = make_tiger(data.tok.vocab_size, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8, d_model=16,
                       d_ff=32, dropout=0.0)  # no dropout: the resumed run must match bit for bit on CPU
    cfg = {"optimizer": "adafactor", "lr": 0.01, "constant_steps": 3, "batch_size": 3, "max_steps": max_steps,
           "log_every": 1, "eval_every": 2, "patience": patience, "valid_subset_size": 4, "valid_subset_seed": 0,
           "beam_size": 3}
    info = train_tiger(model, data, trie, cfg, "cpu", tmp_path, seed=0, resume=resume)
    return model, info, torch.load(tmp_path / "last.pt", weights_only=False)


def test_resume_restores_step_optimizer_schedule_best_and_patience(tmp_path):
    full_model, full_info, full_ckpt = _toy_training(tmp_path / "full", max_steps=10, patience=None)
    _toy_training(tmp_path / "split", max_steps=4, patience=None)  # "pause" after the step-4 checkpoint
    model, info, ckpt = _toy_training(tmp_path / "split", max_steps=10, patience=None, resume=True)
    assert info["steps"] == full_info["steps"] == 10
    for k in ("step", "best", "best_step", "evals_since_best"):
        assert ckpt["state"][k] == full_ckpt["state"][k]
    assert [e.get("loss") for e in ckpt["state"]["log"]] == [e.get("loss") for e in full_ckpt["state"]["log"]]
    assert ckpt["scheduler"]["last_epoch"] == 10
    for a, b in zip(model.parameters(), full_model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_patience_none_trains_to_max_steps(tmp_path):
    _, info, _ = _toy_training(tmp_path, max_steps=12, patience=None)
    assert (info["stop_reason"], info["steps"]) == ("max_steps", 12)
