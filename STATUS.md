# STATUS.md

> Claude Code: update this file at the end of every task — what was done, exact numbers, open issues, next step. Keep entries short.

## Current Phase
Phase 1 — Foundation (in progress)

## Done
- **1-A scaffold (2026-09-29):** directory layout per spec §8 (empty `__init__.py` in `src/recsys/{data,eval,models,train}`, `tests/`), `.gitignore`, `requirements.txt`, `scripts/check_env.py`.
  - Env (conda `recsys`): Python 3.11.16 arm64, PyTorch 2.14.0, MPS built + available, device = `mps`.
  - Installed: numpy 2.4.6, pandas 3.0.6, pyyaml 6.0.3, tqdm 4.70.1, pytest 9.1.1, scikit-learn 1.9.1, matplotlib 3.11.2.
  - check_env: 1024×1024 matmul on MPS ran; max |diff| vs CPU = 0.00e+00 (177 ms, first call incl. warm-up).

## Results
| Run | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Open Issues / Decisions
- `requirements.txt` is unpinned; versions above are what was installed. Pin before the final README if exact reproducibility is needed.

## Next Step
- Phase 1-B: download Amazon Beauty 2014 (5-core reviews + meta), preprocess, and pass the sanity check (22,363 users / 12,101 items / mean len ~8.87 / median 6).
