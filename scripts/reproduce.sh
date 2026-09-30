#!/usr/bin/env bash
# Regenerate every Phase 1 run and summary table. Run from the repo root in the `recsys` env
# (pip install -r requirements.txt && pip install -e .).
#
# SASRec training on MPS is not bit-for-bit deterministic: a rerun matches the recorded numbers only up to
# seed-level noise (see the mean ± std table in results/summary_test_tieshuffle.md). Popularity and the data
# files are deterministic. train_sasrec.py skips a seed whose finished run exists with the same config
# (pass --force to retrain).
set -euo pipefail

# Data
python scripts/download_data.py
python -m recsys.data.preprocess                                                  # original (ASIN) tie order
python -m recsys.data.preprocess --tie-seed 0 --out-dir data/processed_tieshuffle  # main dataset: shuffled ties
python -m pytest -q

# Main dataset (shuffled same-day ties)
python scripts/run_popularity.py --config configs/popularity_tieshuffle.yaml
python scripts/train_sasrec.py --config configs/sasrec_tieshuffle.yaml --seeds 42 43 44      # CE: main baseline
python scripts/train_sasrec.py --config configs/sasrec_bce_tieshuffle.yaml --seeds 42 43 44  # BCE: paper check

# Sensitivity check (original ASIN tie order; carries the ASIN-order artifact)
python scripts/run_popularity.py --config configs/popularity.yaml
python scripts/train_sasrec.py --config configs/sasrec.yaml
python scripts/train_sasrec.py --config configs/sasrec_bce.yaml

# Summary tables
python scripts/compare_runs.py --runs popularity_tieshuffle sasrec_tieshuffle sasrec_bce_tieshuffle \
  --paper sasrec --seeds 43 44 \
  --pairs sasrec_tieshuffle:popularity_tieshuffle sasrec_bce_tieshuffle:popularity_tieshuffle \
          sasrec_tieshuffle:sasrec_bce_tieshuffle \
  --note "Main dataset: same-day ties shuffled (tie-seed 0). CIs and pairs are from the seed-42 runs." \
  --out summary_test_tieshuffle.md
python scripts/compare_runs.py --runs popularity sasrec sasrec_bce --paper sasrec \
  --pairs sasrec:popularity sasrec_bce:popularity sasrec:sasrec_bce \
  --note "Sensitivity check only. Original raw-file order: same-day items are in ASIN (= item ID) order, so on all 9,719 same-day valid/test pairs the test item has the higher ID. These numbers carry that ASIN-order artifact." \
  --out summary_test.md
python scripts/compare_runs.py --runs popularity_tieshuffle popularity sasrec_tieshuffle sasrec sasrec_bce_tieshuffle sasrec_bce \
  --cross-data \
  --pairs popularity_tieshuffle:popularity sasrec_tieshuffle:sasrec sasrec_bce_tieshuffle:sasrec_bce \
  --note "Tie order effect: shuffled − original, same model and seed (42), paired by user ID. Test targets differ for 6,554 of 22,363 users. Single training seed per side." \
  --out summary_test_tieorder.md
