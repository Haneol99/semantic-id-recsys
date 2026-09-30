import numpy as np
import torch

from recsys.models.sasrec import SASRec, pad_left
from recsys.train.sasrec_trainer import build_training_pairs, sequence_ce_loss


def make_model(**kw):
    torch.manual_seed(0)
    return SASRec(num_items=20, max_len=6, hidden=16, **kw).eval()


def test_pad_left_truncates_and_pads():
    out = pad_left([[1, 2], [1, 2, 3, 4, 5], []], max_len=4)
    assert out.tolist() == [[0, 0, 1, 2], [2, 3, 4, 5], [0, 0, 0, 0]]


def test_training_pairs_are_shifted_by_one():
    inputs, targets = build_training_pairs([[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]], max_len=4)
    assert inputs.tolist() == [[0, 0, 1, 2], [7, 8, 9, 10]]
    assert targets.tolist() == [[0, 0, 2, 3], [8, 9, 10, 11]]


def test_causal_future_items_do_not_change_past_outputs():
    model = make_model()
    a = torch.tensor([[3, 4, 5, 6, 7, 8]])
    b = torch.tensor([[3, 4, 5, 9, 10, 11]])  # differs only from position 3 on
    ha, hb = model(a), model(b)
    assert torch.allclose(ha[:, :3], hb[:, :3], atol=1e-6)
    assert not torch.allclose(ha[:, 3:], hb[:, 3:])


def test_padding_does_not_change_real_positions():
    # same items, different amounts of left padding in a longer window -> identical last hidden state
    model = make_model()
    hist = [[5, 6, 7]]
    short = model(torch.tensor([[5, 6, 7]]))[:, -1]
    padded = model(torch.as_tensor(pad_left(hist, 6)))[:, -1]
    assert torch.allclose(short, padded, atol=1e-5)
    assert torch.isfinite(model(torch.zeros(1, 6, dtype=torch.long))).all()  # all-padding row: no NaN


def test_score_shape_and_multi_head():
    model = make_model(num_heads=2)
    scores = model.score([[1, 2, 3], [4]])
    assert scores.shape == (2, 21)


def test_overfits_tiny_dataset():
    torch.manual_seed(0)
    model = SASRec(num_items=10, max_len=5, hidden=32, dropout=0.0)
    inputs, targets = map(torch.as_tensor, build_training_pairs([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 5))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = sequence_ce_loss(model, inputs, targets).item()
    for _ in range(200):
        loss = sequence_ce_loss(model, inputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 < first
    model.eval()
    assert model.score([[1, 2, 3, 4]]).argmax().item() == 5


def test_sample_negatives_avoid_user_items_and_padding():
    from recsys.train.sasrec_trainer import sample_negatives
    targets = np.array([[0, 2, 3], [4, 5, 6]])
    user_items = [{1, 2, 3}, {4, 5, 6, 7}]
    negs = sample_negatives(targets, user_items, num_items=8, rng=np.random.default_rng(0))
    assert negs[0, 0] == 0  # padding position gets no negative
    for r, c in zip(*np.nonzero(targets)):
        assert 1 <= negs[r, c] <= 8 and negs[r, c] not in user_items[r]


def test_bce_loss_overfits_tiny_dataset():
    from recsys.train.sasrec_trainer import sample_negatives, sequence_bce_loss
    torch.manual_seed(0)
    seqs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    model = SASRec(num_items=10, max_len=5, hidden=32, dropout=0.0)
    inputs, targets = build_training_pairs(seqs, 5)
    negs = torch.as_tensor(sample_negatives(targets, [set(s) for s in seqs], 10, np.random.default_rng(0)))
    inputs, targets = torch.as_tensor(inputs), torch.as_tensor(targets)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = sequence_bce_loss(model, inputs, targets, negs).item()
    for _ in range(200):
        loss = sequence_bce_loss(model, inputs, targets, negs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 < first


def test_last_epoch_is_evaluated_when_eval_every_exceeds_max_epochs(tmp_path):
    from recsys.data.dataset import Split
    from recsys.train.sasrec_trainer import train_sasrec
    torch.manual_seed(0)
    split = Split(train=[[1, 2, 3], [4, 5, 6]], valid=[4, 7], test=[5, 8], num_items=8)
    model = SASRec(num_items=8, max_len=4, hidden=8)
    cfg = {"lr": 1e-3, "betas": [0.9, 0.98], "weight_decay": 0.0, "batch_size": 2, "max_epochs": 2,
           "eval_every": 5, "early_stop_metric": "ndcg@10", "patience": 10}
    info = train_sasrec(model, split, cfg, torch.device("cpu"), tmp_path / "best.pt", seed=0)
    assert info["best_epoch"] == 2 and (tmp_path / "best.pt").exists()


def test_seed_run_config_and_skip_check(tmp_path):
    import yaml
    from recsys.utils import finished_run_matches, seed_run_config
    base = {"run_name": "r", "seed": 42, "data_dir": "d"}
    assert seed_run_config(base, 42) == base
    assert seed_run_config(base, 43) == {"run_name": "r_seed43", "seed": 43, "data_dir": "d"}
    assert not finished_run_matches(tmp_path, base)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "metrics.json").write_text("{}")
    assert finished_run_matches(tmp_path, base)
    assert not finished_run_matches(tmp_path, seed_run_config(base, 43))
