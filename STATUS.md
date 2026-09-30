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

## Results
| Run | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Open Issues / Decisions
- **Timestamp ties:** timestamps are day-resolution; 9,719 / 22,363 users (43%) have the same timestamp on their valid and test items (19,586 have some tie). Ties keep raw-file order (stable sort). This decides which item is the test target for those users — document in README; revisit if SASRec lands outside ±15% of the paper.
- `requirements.txt` is unpinned; versions above are what was installed. Pin before the final README if exact reproducibility is needed.

## Next Step
- Phase 1-C: evaluation module (full ranking, history masking, Recall/NDCG@5/10, per-user .npy) + Popularity baseline.
