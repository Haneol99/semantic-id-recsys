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

## Results
| Run | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|---|---|
| popularity | valid | 0.0091 [0.0079, 0.0103] | 0.0056 [0.0048, 0.0065] | 0.0163 [0.0146, 0.0179] | 0.0079 [0.0071, 0.0088] | 22,363 users, 95% bootstrap CI |
| popularity | test | 0.0075 [0.0064, 0.0085] | 0.0041 [0.0035, 0.0048] | 0.0114 [0.0101, 0.0127] | 0.0054 [0.0047, 0.0061] | commit 5d3b881 |

## Open Issues / Decisions
- **Timestamp ties — decided (owner, 2026-09-30):** keep raw-file order for same-day items; note in README. (9,719 / 22,363 users have tied valid/test timestamps.)
- README must document: history masking, tie-breaking by item ID, Popularity counts from train positions only.
- `requirements.txt` is unpinned; versions above are what was installed. Pin before the final README if exact reproducibility is needed.

## Next Step
- Phase 1-D: SASRec (max_len 50, CE over all items) trained with valid-based model selection; target within ±15% of paper (R@10 0.0605, N@10 0.0318).
