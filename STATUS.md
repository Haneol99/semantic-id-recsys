# STATUS.md

> Claude Code: update this file at the end of every task — what was done, exact numbers, open issues, next step. Keep entries short.

## Current Phase
Phase 1 — Foundation (in progress)

## Done
- **1-A scaffold (2026-09-29):** directory layout per spec §8 (empty `__init__.py` in `src/recsys/{data,eval,models,train}`, `tests/`), `.gitignore`, `requirements.txt`, `scripts/check_env.py`.
  - Env (conda `recsys`): Python 3.11.16 arm64, PyTorch 2.14.0, MPS built + available, device = `mps`.
  - Installed: numpy 2.4.6, pandas 3.0.6, pyyaml 6.0.3, tqdm 4.70.1, pytest 9.1.1, scikit-learn 1.9.1, matplotlib 3.11.2.
  - check_env: 1024×1024 matmul on MPS ran; max |diff| vs CPU = 0.00e+00 (177 ms, first call incl. warm-up).
- **1-B data pipeline (2026-09-30):** `scripts/download_data.py` (SNAP URLs worked), `src/recsys/data/preprocess.py` (`python -m recsys.data.preprocess`), `tests/test_preprocess.py` (8 pass), `pyproject.toml` (package installed editable).
  - Stats **match spec**: 22,363 users, 12,101 items, 198,502 interactions, mean len 8.876, median 6 (min 5, max 204).
  - Item IDs 1..12,101 in ASIN-sorted order (0 = pad); user IDs 0..22,362 in reviewerID-sorted order.
  - Meta found for all 12,101 items; field coverage: title 12,094, price 11,516, brand 10,021, categories 12,101.
  - Outputs in `data/processed/`: sequences.json, splits.json, item_meta.json, id_maps.json, stats.json.
- **1-C evaluator + Popularity (2026-09-30):** `src/recsys/eval/{metrics,evaluator,bootstrap}.py`, `src/recsys/models/popularity.py`, `src/recsys/data/dataset.py`, `src/recsys/utils.py`, `scripts/run_popularity.py`, `configs/popularity.yaml`, `tests/test_eval.py` (17 tests pass in total).
  - Full ranking over items 1..12,101; padding + the user's input history get score −inf. Ties are broken by lower item ID (deterministic). No user has a repeated item, so masking never removes a target.
  - Inputs: valid → train items; test → train + valid item. Popularity counts use train positions only (both splits).
  - Bootstrap: 1,000 resamples, seed 0, percentile 95% CI; paired-difference CI helper ready.
  - Run output `results/popularity/`: config.yaml, metrics.json (means + CIs, seed 42, git hash), per-user `{split}_{rank,recall5,recall10,ndcg5,ndcg10}.npy`. Produced at commit 5d3b881 (clean).
- **1-D SASRec (2026-09-30):** `src/recsys/models/sasrec.py`, `src/recsys/train/sasrec_trainer.py`, `scripts/train_sasrec.py`, `configs/sasrec.yaml`, `scripts/compare_runs.py` (writes `results/summary_<split>.md`), `tests/test_sasrec.py` (23 tests pass in total).
  - 2 blocks, hidden 64, 1 head, dropout 0.2, max_len 50, learned positions; full-softmax CE at every position; Adam lr 1e-3, betas (0.9, 0.98), batch 256; early stop on valid NDCG@10, patience 10. 828,288 params.
  - MPS: 4.7 s/epoch train (mean), 0.6 s/valid eval; stopped at epoch 30, best epoch 20; total 158 s. Commit 8d37597 (clean). Test evaluated once.
  - **Outside ±15% of paper — on the high side** (test R@10 +48%, N@10 +70%). Not tuned; waiting on owner decision (see Open Issues).
- **1-D diagnostics (a)+(b) (2026-09-30, owner-approved):** `loss: bce` option (original SASRec: 1 negative per position, uniform over items not in the user's train seq, resampled each epoch); `preprocess --tie-seed` (per-user seeded shuffle of same-timestamp reviews) → `data/processed_tieshuffle/` (same counts; 6,554 test / 9,377 valid targets change; default output verified byte-identical). `compare_runs.py` now takes `--paper`, `--pairs a:b`, `--out` and refuses to pair runs on different data. Commit 6c167b5; 26 tests pass.
  - **Loss explains the gap:** SASRec-BCE is −7% to −9% vs paper (original ties) and −7% to −8% (shuffled ties) — within ±15%. CE − BCE (paired, original data): R@10 +0.0348 [+0.0315, +0.0382], N@10 +0.0249 [+0.0229, +0.0270].
  - **Tie order is a smaller effect:** CE SASRec test R@10 0.0897 → 0.0842, N@10 0.0539 → 0.0496 with shuffled ties (unpaired; CIs overlap). BCE unchanged within noise. Popularity rises (R@10 0.0114 → 0.0155).
  - Timing (MPS): BCE 2.7 s/epoch, best epoch 44/54 (188 s) orig, 58/68 (239 s) shuffled; CE shuffled 4.7 s/epoch, best 16/26 (138 s).
  - Tables: `results/summary_test.md` (original ties), `results/summary_test_tieshuffle.md` (shuffled ties).

## Results
| Run | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|---|---|
| popularity | valid | 0.0091 [0.0079, 0.0103] | 0.0056 [0.0048, 0.0065] | 0.0163 [0.0146, 0.0179] | 0.0079 [0.0071, 0.0088] | 22,363 users, 95% bootstrap CI |
| popularity | test | 0.0075 [0.0064, 0.0085] | 0.0041 [0.0035, 0.0048] | 0.0114 [0.0101, 0.0127] | 0.0054 [0.0047, 0.0061] | commit 5d3b881 |
| sasrec | valid | 0.0832 | 0.0603 | 0.1137 | 0.0701 | best epoch 20 (early-stop metric) |
| sasrec | test | 0.0635 [0.0606, 0.0666] | 0.0454 [0.0433, 0.0478] | 0.0897 [0.0863, 0.0935] | 0.0539 [0.0516, 0.0564] | commit 8d37597; +48–83% vs paper |
| sasrec − popularity | test | +0.0560 [+0.0529, +0.0591] | +0.0413 [+0.0391, +0.0436] | +0.0783 [+0.0745, +0.0824] | +0.0485 [+0.0460, +0.0510] | paired bootstrap |
| sasrec_bce | test | 0.0359 [0.0334, 0.0383] | 0.0228 [0.0212, 0.0245] | 0.0550 [0.0523, 0.0579] | 0.0290 [0.0273, 0.0306] | original loss; −7…−9% vs paper; commit 6c167b5 |
| popularity_tieshuffle | test | 0.0095 [0.0082, 0.0107] | 0.0057 [0.0049, 0.0065] | 0.0155 [0.0138, 0.0171] | 0.0076 [0.0067, 0.0085] | shuffled same-day ties |
| sasrec_tieshuffle | test | 0.0596 [0.0565, 0.0626] | 0.0417 [0.0394, 0.0441] | 0.0842 [0.0808, 0.0878] | 0.0496 [0.0472, 0.0521] | shuffled ties, CE |
| sasrec_bce_tieshuffle | test | 0.0355 [0.0332, 0.0378] | 0.0231 [0.0215, 0.0249] | 0.0559 [0.0529, 0.0591] | 0.0297 [0.0280, 0.0316] | shuffled ties, BCE; −7…−8% vs paper |
| *paper SASRec* | test | 0.0387 | 0.0249 | 0.0605 | 0.0318 | reference |
| *paper TIGER* | test | 0.0454 | 0.0321 | 0.0648 | 0.0384 | reference |

## Open Issues / Decisions
- **Timestamp ties — decided (owner, 2026-09-30):** keep raw-file order for same-day items; note in README. (9,719 / 22,363 users have tied valid/test timestamps.)
- **SASRec above paper — cause found, OPEN decision:** the gap is the loss (full CE vs original BCE). Owner to decide: (1) which SASRec is the main baseline (CE, BCE, or both); (2) which tie order is the main dataset (raw/ASIN vs shuffled). Note: CE SASRec (test R@10 0.0897) is already above the paper's TIGER (0.0648). Earlier read-only diagnostics:
  - Same-day valid/test users (43.5%): SASRec test R@10 0.1260 / N@10 0.0802 vs different-day 0.0618 / 0.0336 (different-day subgroup is close to the paper). The raw 5-core file is sorted by ASIN, so same-day items are in ASIN order.
  - History masking (valid): N@10 0.0701 masked vs 0.0519 unmasked; R@10 0.1137 vs 0.1037.
  - Loss: full-softmax CE vs original SASRec's BCE with one sampled negative (not yet measured).
  - Ruled out: exact score ties at target (0 in 2,000 valid users), repeated items (0), test used for selection (no).
- README must document: history masking, tie-breaking by item ID, Popularity counts from train positions only.
- `requirements.txt` is unpinned; versions above are what was installed. Pin before the final README if exact reproducibility is needed.

## Next Step
- Owner decides main SASRec baseline and main tie order; then Phase 2 (Semantic IDs).
