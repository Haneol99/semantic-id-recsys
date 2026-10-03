#!/usr/bin/env bash
# Regenerate every run, summary table and README figure (Phases 1-5). Run from the repo root in the `recsys` env
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

# Phase 2: Semantic IDs (item IDs are identical in both data dirs; RQ-VAE trains on CPU, deterministic)
python scripts/embed_items.py
python scripts/train_rqvae.py --config configs/rqvae.yaml
python scripts/semantic_id_report.py

# Phase 3: TIGER (MPS, ~14 h). The recorded run was stopped by hand at step 100k after subset valid NDCG@10 had
# declined since step 22k (overfitting); --max-steps 100000 reproduces that stop. Final valid + test eval uses best.pt.
python scripts/train_tiger.py --config configs/tiger_tieshuffle.yaml --max-steps 100000
python scripts/plot_tiger_eval_curve.py

# Phase 4: bucket and Semantic-ID prefix analysis of the test split (inference only, no training)
python scripts/analyze_phase4.py

# Phase 4: cold start (paper Sec. 4.3). Hold out 5% of the distinct test-target items (seed 0) from all training
# data, retrain the RQ-VAE (seen items only), SASRec-CE (seed 42) and TIGER (30k steps, ~4 h on MPS), then evaluate
# retrieval of the held-out items and the per-code-position log-prob diagnosis.
python scripts/make_coldstart.py                                      # -> data/processed_coldstart
python scripts/train_rqvae.py --config configs/rqvae_coldstart.yaml   # unseen items get IDs from the trained model
python scripts/train_sasrec.py --config configs/sasrec_coldstart.yaml
python scripts/train_tiger.py --config configs/tiger_coldstart.yaml
python scripts/eval_coldstart.py                                      # results/coldstart_test.md, fig5_coldstart.png
python scripts/diagnose_coldstart_logprob.py                          # results/coldstart_logprob_test.md

# Sensitivity check (original ASIN tie order; carries the ASIN-order artifact)
python scripts/run_popularity.py --config configs/popularity.yaml
python scripts/train_sasrec.py --config configs/sasrec.yaml
python scripts/train_sasrec.py --config configs/sasrec_bce.yaml

# Summary tables
python scripts/compare_runs.py --runs popularity_tieshuffle sasrec_tieshuffle sasrec_bce_tieshuffle tiger_tieshuffle \
  --paper tiger sasrec --seeds 43 44 \
  --pairs tiger_tieshuffle:popularity_tieshuffle \
          tiger_tieshuffle:sasrec_tieshuffle tiger_tieshuffle:sasrec_tieshuffle_seed43 tiger_tieshuffle:sasrec_tieshuffle_seed44 \
          tiger_tieshuffle:sasrec_bce_tieshuffle tiger_tieshuffle:sasrec_bce_tieshuffle_seed43 tiger_tieshuffle:sasrec_bce_tieshuffle_seed44 \
          sasrec_tieshuffle:popularity_tieshuffle sasrec_bce_tieshuffle:popularity_tieshuffle \
          sasrec_tieshuffle:sasrec_bce_tieshuffle \
  --note "Main dataset: same-day ties shuffled (tie-seed 0). SASRec CIs and SASRec pairs are from the seed-42 runs; TIGER (one seed, 42; best.pt at step 22k of a run stopped at 100k) is paired with every SASRec seed." \
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

# README figures (docs/assets/ is committed; results/ run outputs are not)
python scripts/plot_main_results.py
python scripts/plot_coldstart.py
mkdir -p docs/assets
cp results/analysis/main_results_test.png results/analysis/buckets_test.png results/analysis/prefix_test.png \
   results/tiger/eval_curve.png results/rqvae/first_code_categories.png docs/assets/
cp results/coldstart/coldstart_readme.png docs/assets/coldstart_test.png
