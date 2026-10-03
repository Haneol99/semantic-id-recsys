# STATUS.md

> Claude Code: update this file at the end of every task — what was done, exact numbers, open issues, next step. Keep entries short.

## Current Phase
Phase 1 — Foundation (done); Phase 2 — Semantic IDs (done, usage pass); Phase 3 — Generative model (TIGER done: stopped by hand at step 100k, best.pt @ 22k evaluated once on valid + test); Phase 4 — Analysis (buckets, Semantic-ID prefix analysis and cold-start experiment done)

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
  - **Loss explains the gap:** SASRec-BCE (seed 42) is −7.3% to −9.2% vs paper (original ties) and −6.6% to −8.4% (shuffled ties); 3-seed mean on shuffled ties −10.5% to −12.4% — within ±15%. CE − BCE (paired, original data): R@10 +0.0348 [+0.0315, +0.0382], N@10 +0.0249 [+0.0229, +0.0270].
  - **Tie order (paired by user ID, seed 42, `results/summary_test_tieorder.md`):** shuffled − original for CE SASRec: R@10 −0.0056 [−0.0093, −0.0021], N@10 −0.0043 [−0.0066, −0.0021], so shuffling significantly lowers CE. (An earlier note here read overlapping unpaired CIs as "no difference". That reading was wrong.) BCE: R@10 +0.0009 [−0.0025, +0.0043], N@10 +0.0008 [−0.0014, +0.0027]: no detectable effect. Popularity rises: R@10 +0.0041 [+0.0026, +0.0054]. Single training seed per side: the CE drop (0.0056) is ~4× the CE seed std (0.0015).
  - Timing (MPS): BCE 2.7 s/epoch, best epoch 44/54 (188 s) orig, 58/68 (239 s) shuffled; CE shuffled 4.7 s/epoch, best 16/26 (138 s).
  - Tables: `results/summary_test.md` (original ties), `results/summary_test_tieshuffle.md` (shuffled ties).
- **Review fixes (2026-09-30):** code at commit 6abbed4; 31 tests pass.
  - `train_sasrec.py --seeds` trains one run per seed. The config's seed keeps `run_name`, other seeds write `<run>_seed<N>`, and finished runs with an identical config are skipped. The seed-42 shuffled runs were kept (config unchanged).
  - **MPS training is not bit-for-bit deterministic:** the same seed and commit gave a different loss from epoch 1 (6th decimal). Retrained `sasrec_tieshuffle` went best epoch 16 → 18, test R@10 0.0842 → 0.0855; `sasrec` R@10 0.0897 → 0.0906. So report mean ± std across seeds 42/43/44; that seed variation covers the nondeterminism. A rerun reproduces numbers only up to this noise. Popularity and the data files are exactly reproducible (verified byte-identical).
  - New seed runs (MPS): CE 131 s / 117 s (best epoch 16 of 26 both); BCE 168 s / 173 s (best 45/55, 42/52).
  - `metrics.json` now records `data.sha256` for every processed data file. The 6 earlier runs were backfilled (`data.backfilled`), after checking that every data file was last modified before each run finished. The backfill happened before `data/processed/stats.json` was regenerated with current code (adds `"tie_order": "raw_file"`; the other 4 files are byte-identical). Its checksum therefore differs from the backfilled one, while `splits.json` (used for pairing) is unchanged.
  - `compare_runs.py`: same data = same resolved `data_dir` and same `splits.json` SHA-256. `--cross-data` pairs by user ID (checks `user2reviewer` match). `--seeds` adds a mean ± std table, and `--note` adds a note. The paired column is renamed `frac_resamples_diff_le_0` (was `p_diff_le_0`; it is not a p-value).
  - The evaluator raises if a target is in its masked history. The last epoch is always evaluated, so a checkpoint exists even when `eval_every > max_epochs`. New tests cover the shuffled dataset (same items and timestamps per user as original, no split overlap).
  - `scripts/reproduce.sh` has the exact commands for every run and all 3 summary tables. `requirements.txt` is pinned to the installed versions.

- **Phase 2 Semantic IDs (2026-09-30):**
  - Code: `src/recsys/data/item_text.py`, `src/recsys/data/semantic_ids.py`, `src/recsys/models/rqvae.py`, `scripts/embed_items.py`, `scripts/train_rqvae.py`, `scripts/semantic_id_report.py`, `configs/rqvae.yaml`, `configs/rqvae_paper.yaml`, `tests/test_semantic_ids.py`. 40 tests pass. Run at commit 5feaee3 (clean).
  - **Item text:** one sentence per item, e.g. `Title: … Price: $3.50. Brand: Maybelline. Categories: Beauty > Makeup > Lips > Lip Stains.` Missing fields are skipped and HTML entities unescaped.
  - **Embeddings:** `sentence-transformers/sentence-t5-base` on MPS gives `data/processed/item_emb.npy` (12,102 × 768; row 0 = zeros), unit-norm rows. Metadata and SHA-256 are in `item_emb.json`.
  - **The paper's RQ-VAE settings collapse on these embeddings.** With Adagrad lr 0.4 all items map to one code after epoch 1 (usage 0.00/0.00/0.00, recon 8e6). In short diagnostics (150–300 epochs), raw input collapsed under every optimizer tried (Adagrad 0.4/0.05/0.01, AdamW 1e-3), each settling at recon 2.07e-4, the MSE of predicting the mean embedding. Standardized input + AdamW 1e-3 reached usage 0.33/1.00/1.00; adding a dead-code reset fixed level 1. Owner chose this fix (2026-09-30).
  - **RQ-VAE run** (`configs/rqvae.yaml`, seed 42, CPU): 3,000 epochs in 625 s. Dead-code resets happened every 10 epochs up to epoch 2,000: 240 / 41 / 21 codes reset in total at levels 1 / 2 / 3, all within the first 100 epochs (none from epoch 100 to 2,000, although resets were still enabled). The final 1,000 epochs had no resets. Plateau stop at epoch 3,000. Final recon MSE 0.360 (standardized space), RQ loss 0.407.
  - **Semantic IDs** (`data/processed/semantic_ids.json`; item_to_sid plus reverse sid_to_item; item IDs are identical in `data/processed_tieshuffle`):
    - codebook usage 1.00 / 1.00 / 1.00, so the ≥80% pass criterion is met;
    - 11,816 unique 3-code prefixes; max collision group 9 (4th token ≤ 8);
    - 2.36% of items (285) need a non-zero 4th token.
  - **Quality** (`results/rqvae/quality.json`, `results/rqvae/first_code_categories.png`, cf. paper Fig. 4a): category = second level of the first category path, because the first level is "Beauty" for every item. First-code purity is 0.915 item-weighted (0.908 mean over codes) vs 0.344 ± 0.002 for random assignment (100 shuffles, same code sizes); the largest category is 31.5% of items.

- **Phase 3 TIGER (2026-10-01 → 10-02):**
  - Code (commit c13de7b): `src/recsys/data/tiger_data.py`, `src/recsys/models/tiger.py` (trie-constrained beam search), `src/recsys/train/tiger_trainer.py`, `scripts/train_tiger.py`, `configs/tiger_tieshuffle.yaml`, `tests/test_tiger.py`. Model 4+4 layers, 6×64 heads, d_model 128, FFN 1024, dropout 0.1, 4.85M params; vocab 3,026 (4×256 code tokens + 2,000 hashed user tokens + pad/eos); last 20 items; Adafactor lr 0.01, constant 10k steps then inverse sqrt; batch 256; beam 30 → drop history items → top 10 → shared evaluator.
  - Pilot (15 min, MPS): loss 5.39 → 3.75 in 1,000 steps; subset valid N@10 0.0039. MPS 2.11 steps/s vs CPU 0.44.
  - Long run paused at step 4,000 (best subset valid N@10 0.0084 @ step 2,000), then resumed from `last.pt`.
  - **Slow-eval fix (2026-10-01):** subset evals took 30 s → 461 s (step 2k → 4k; pilot 43 s → 415 s). Cause: one beam-search batch (256 users × 30 beams) peaked at ~20.6 GB MPS driver memory, kept cached after the eval; on a 32 GB Mac this pushed training and the next eval into swap (in a diagnostic, 300 training steps after two evals did not finish in ~8 min vs ~140 s normally). Fixes: (1) `use_cache=False` in the beam-search decoder calls — HF T5 otherwise returns cross-attention K/V for every beam; peak 20.6 → 5.0 GB, batch 5.4 → 2.1 s, identical items and scores; (2) `torch.mps.empty_cache()` after every eval batch. With (2) alone, evals after training took 25–32 s and training stayed at 2.1 steps/s. Eval interval stays 2,000 steps.
  - **No early stopping (owner, 2026-10-01):** `patience: null`, train the full 200k steps as in the paper. Model selection unchanged: best.pt = best NDCG@10 on the fixed 2,000-user valid subset (beam 30), used for the one-time full valid + test evaluation. The resumed run continues the step-4,000 state (step, optimizer, LR schedule, best score, data order); the patience counter is still recorded but no longer stops training. Tests check that a resumed run matches an uninterrupted one bit for bit (CPU, no dropout) and that `patience: null` trains to `max_steps`. 51 tests pass.

  - **First long run collapsed (paused by owner at step ~20k, 2026-10-01).** Subset valid N@10 stayed at 0.005–0.008 (Popularity level) from step 2k to 20k while train loss fell. Read-only diagnostics on last.pt (20k) and best.pt (2k):
    - Top-10 lists of 2,000 valid users contain 15 distinct items (mean pairwise overlap 0.98); swapping in another user's input leaves top-10 unchanged (overlap 1.00; exact scores over all items correlate 1.0000). In-sample Recall@10 on 2,000 train examples 0.022 ≈ valid 0.018.
    - Teacher-forced NLL per position (step 20k, valid): 5.44 / 3.78 / 0.69 / 0.06. The first code is at the train-marginal level (5.42; uniform 5.55), worse than a first-code bigram table (5.00, acc 9.2% vs the model's 1.2%). Train loss fell only through positions 2–4, which follow from the prefix.
    - Encoder output collapsed (pooled cosine between users 0.958 at 2k → 0.999 at 20k); zeroing the encoder output changes position-0 predictions by KL 0.0003. Attention/FFN weights grew 3.7–6.4× their init norm by step 2k and 7–17× by 20k (embeddings 1.9×).
    - Ruled out: train/eval input layout, target Semantic IDs (all 131,413 match `semantic_ids.json`), label shift (decoder start 0), item-ID maps, trie, HF T5 cross-attention wiring, beam search (scores match exact full scoring within 2e-5; exact full-ranking Recall@10 was also 0.010).
    - **Cause: Adafactor with `scale_parameter=False` at lr 0.01** (my choice in c13de7b). Updates are then ~lr per weight regardless of scale, i.e. about the whole init std of T5 attention weights (q 0.011) per step.
  - **Fix (2026-10-01):** `scale_parameter: true` (T5 / Mesh-TF default: update = lr × parameter RMS), lr 0.01, paper schedule unchanged. Not a paper deviation: the paper gives only lr and schedule. Also added: optional `warmup_steps` (0 here; for an AdamW fallback), `distinct_items` in every eval log entry, and a warning when it is below `min_distinct_items: 100` (collapse check). 54 tests pass.
  - **2,000-step probe with the fix** (real `train_tiger`, MPS, seed 42; outputs in a scratch dir, not `results/`), pass criteria fixed before running, all passed:
    - position-0 valid NLL 4.81 < bigram baseline 5.00 (per position 4.81 / 3.84 / 1.49 / 0.05);
    - distinct items in subset top-10 lists 519 ≥ 100 (323 at step 1k);
    - subset valid N@10 0.0298 (R@10 0.057) ≥ 1.5 × Popularity on the same users (0.0087); 0.0258 at step 1k;
    - top-10 overlap with another user's input 0.118 < 0.5;
    - zero-encoder KL at position 0 0.837 > 0.01; encoder pooled cosine between users 0.32.
    - Speed 2.0 steps/s; subset eval 13.5 s. The AdamW fallback (lr 1e-3, 1k warmup) was not needed.
  - Collapsed run moved to `results/tiger_tieshuffle_collapsed_adafactor_noscale` (best.pt, last.pt at step 20k); it cannot be resumed with the new optimizer setting.
  - **Final run** (`results/tiger_tieshuffle`, log `results/tiger/train.log`; started 2026-10-01 18:55 from clean commit db14412, seed 42, MPS):
    - **Stopped by hand at step ~100k (owner, 2026-10-02 08:58; SIGTERM to the process group), not by the normal path. The paper's 200k-step schedule was not completed.** Reason: overfitting. Subset valid NDCG@10 peaked at **0.0477 at step 22k** (best.pt), then declined to ~0.027 by step 100k (0.0266 at the step-100k eval; 50 evals) while train loss kept falling (2.57 @ 2k → 1.82 @ 22k → 1.39 @ 50k → 1.08 @ 100k). No collapse: ≥ 550 distinct items in the subset top-10 lists at every eval (2,236 at 20k); 2.0 steps/s throughout, 13.5 s per subset eval; 13.8 h of training.
    - Eval curve (step vs subset valid NDCG@10, with train loss): `results/tiger/eval_curve.png` (`scripts/plot_tiger_eval_curve.py`; gitignored like other run outputs, copy into README assets in Phase 5).
    - Final valid + test evaluation **once**, with best.pt (step 22k), via the new `train_tiger.py --eval-only --stop-reason manual_stop_overfitting` (reads the run state from last.pt; refuses to run if metrics.json exists). metrics.json `git_hash` is `db14412…-dirty`: the uncommitted change was only this eval-only path (and `compare_runs.py --paper` taking several models); evaluation code unchanged. `--max-steps 100000` now reproduces the stop (`scripts/reproduce.sh`).
    - Full valid (22,363 users): R@5 0.0516, N@5 0.0338, R@10 0.0787, N@10 0.0426 (the 2,000-user subset used for selection gave 0.0477, optimistic). Test: R@5 0.0414, N@5 0.0270, **R@10 0.0651, N@10 0.0346**. Short lists (< 10 items after dropping history): 19 valid / 24 test users; 4,375 / 4,275 distinct items in the top-10 lists.
    - **vs the paper's TIGER** (test): R@10 +0.5% (paper 0.0648 inside our CI [0.0614, 0.0686]); R@5 −8.9%, N@5 −15.8%, N@10 −9.8% (paper values above our CIs). One training seed.
    - **vs baselines** (paired bootstrap over 22,363 test users, `results/summary_test_tieshuffle.md`): TIGER − Popularity N@10 +0.0270 [+0.0249, +0.0293]. TIGER − SASRec-CE N@10 −0.0150 [−0.0170, −0.0129] (seed 42; −0.0160 / −0.0159 vs seeds 43 / 44; all metrics, all seeds CI < 0). TIGER − SASRec-BCE N@10 +0.0049 [+0.0029, +0.0070] (seed 42; +0.0069 / +0.0067 vs seeds 43 / 44; all metrics, all seeds CI > 0). So TIGER beats the paper-style SASRec (BCE), as in the paper, but not the full-softmax SASRec-CE baseline.
  - **Unexplained slowdown in the collapsed run:** around step 20k training ran at 0.7 steps/s (vs 2.1 before) and the step-20k subset eval took 41.5 s (vs ~15 s). Cause not investigated (other load on the Mac is possible). Every eval now also logs `train_steps_per_sec`: training-only speed since the previous eval, so a slowdown shows within one eval interval.

- **Phase 4 analysis — buckets and Semantic-ID prefixes (2026-10-02; existing runs, inference only, test split, shuffled ties):**
  - Code: `scripts/analyze_phase4.py`, `src/recsys/eval/analysis.py` (`prefix_depth`), `tests/test_analysis.py` (56 tests pass). Tables: `results/analysis_phase4_test.md`. Plots: `results/analysis/buckets_test.png` (grouped bars per bucket, Recall@10 / NDCG@10, 95% CIs) and `results/analysis/prefix_test.png`; numbers in `results/analysis/phase4_test.json`. The plots and JSON are gitignored like other run outputs.
  - Method: per-user test metrics from each run's saved `test_*.npy`. SASRec-CE and SASRec-BCE use all three seeds: bucket mean ± std over seeds, and CIs on the per-user mean over seeds. TIGER (one seed) is paired against that per-user mean, 1,000 resamples; how many single-seed pairings agree is also reported. Top-10 lists were recomputed from each run's best.pt and match the saved ranks for 100% of users in all 8 runs.
  - Buckets:
    - test-target train interactions: 0–5 (5,114 users, incl. the 64 never-in-train targets), 6–20 (8,620), >20 (8,629);
    - test history length: 4–5 (11,383), 6–10 (7,433), 11–20 (2,528), >20 (1,019); TIGER reads the last 20 items, SASRec the last 50;
    - valid/test pair on the same day (9,719) vs different days (12,644).
  - **TIGER vs SASRec-CE: lower in every bucket** (all paired CIs < 0, 3/3 seeds). Largest gaps: history >20 items (N@10 −0.0425 [−0.0548, −0.0305]) and 6–20-interaction targets, where TIGER is at about half of CE (N@10 0.0140 vs 0.0300).
  - **TIGER vs SASRec-BCE:**
    - better for popular targets (>20 interactions: N@10 +0.0188 [+0.0147, +0.0226], R@10 +0.0321), at every history length, and on same-day pairs (N@10 +0.0106) and different-day pairs (+0.0028);
    - no difference for 6–20 targets (N@10 −0.0006 [−0.0025, +0.0010]);
    - **worse for rare targets** (0–5: N@10 −0.0036 [−0.0050, −0.0022]; R@10 0.0047 vs BCE 0.0096 vs CE 0.0167).
  - Rare items are TIGER's weakest bucket here, the opposite of the paper's motivation that shared Semantic-ID codes help rare items. No model hits any of the 64 never-in-train targets.
  - All models do about 2× better on same-day valid/test pairs than different-day ones.
  - **Semantic-ID prefixes of missed targets:** the share of missed users whose top 10 holds an item with the target's first code is TIGER 0.163 vs SASRec-CE / BCE 0.153 / 0.153 vs Popularity 0.067 (chance, the same lists against another user's target: 0.031–0.072).
    - Paired on users both models miss: TIGER +0.0064 [+0.0018, +0.0110] vs CE and +0.0152 [+0.0104, +0.0198] vs BCE.
    - First 2 codes: TIGER 0.013 vs CE 0.010 (paired +0.0021 [+0.0008, +0.0034]). First 3 codes: ~0.001 for every model.
    - So TIGER's misses land in the target's coarse code group slightly more often, but near-misses deeper than the first code are rare for every model.
  - Caveats: one TIGER seed (best.pt at step 22k of a run that later overfit); buckets were chosen after the test split had been evaluated, and the many bucket comparisons have no multiple-comparison correction.

- **Phase 4 cold-start experiment (paper Sec. 4.3 / Fig. 5; 2026-10-02):**
  - Code: commits 1e93828 (split, seen-only RQ-VAE, seen-only trie), b6e02ed + 15b5b59 (`scripts/eval_coldstart.py`, `src/recsys/eval/coldstart.py`), and this commit (eval variants below). 65 tests pass. Table: `results/coldstart_test.md`; plot `results/coldstart/fig5_coldstart.png`, numbers `results/coldstart/coldstart_test.json` (gitignored).
  - **Split** (`data/processed_coldstart`, from the shuffled-ties data, seed 0): 415 items held out = 5% of 8,302 distinct test targets; their 5,649 train interactions (4,720 users) removed. 1,098 / 22,363 test users have an unseen target; 913 users with an unseen valid target are left out of validation (21,450 valid users).
  - **RQ-VAE on the 11,686 seen items only** (`results/rqvae_coldstart`, same settings as the main run): usage 1.00 / 1.00 / 1.00, recon 0.364, 11,444 unique 3-code prefixes, 2.07% non-zero 4th token. Unseen items get IDs from the trained model: all 415 share their first code with some seen item, but **only 3 / 415 share a full 3-code prefix with a seen item**.
  - **Runs** (seed 42, MPS, one seed each): `sasrec_coldstart` (CE, best epoch 15; test R@10 0.0807, N@10 0.0477); `tiger_coldstart` (30k steps max, best.pt @ 28k, subset valid N@10 0.0458; full valid R@10 0.0750 / N@10 0.0410; test, seen items only, R@10 0.0576 / N@10 0.0316; 4.1 h).
  - **Methods** (unseen slots = ceil(eps·K), floor as sensitivity): TIGER = paper method (beam 30 over all items' IDs, unseen items via first-3-code match of generated IDs); TIGER (exact-scored unseen) = labeled variant, the unseen items with the highest exact 4-code TIGER log-prob in the unseen slots; Hybrid = SASRec-CE seen items + Semantic-KNN's top unseen items in the slots; Semantic-KNN = cosine to the last history item's Sentence-T5 embedding (n = 1 picked on valid R@10). Sensitivity: TIGER with 2-/1-code matching.
  - **Unseen targets (1,098 users), Recall@10, eps 0.1** (one unseen slot): TIGER **0.0000** [0, 0]; TIGER exact-scored 0.0128 [0.0064, 0.0200]; Semantic-KNN 0.0565 [0.0428, 0.0692]; **Hybrid 0.0811** [0.0647, 0.0965]. At eps 0.3: 0.0000 / 0.0319 / 0.0619 / 0.1357. TIGER is 0 at every K and eps, under both slot rules: its beam never generated an unseen item's ID (0.0000% of beam slots), and 3-code matching can only reach the 3 unseen items with a seen prefix.
  - **Relaxed matching (evaluation only):** 2-code 0.0091, 1-code 0.0191 at eps 0.1; 1-code reaches 0.0847 at eps 0.3, but all-user N@10 drops 0.0316 → 0.0250.
  - **All test users (cost on seen items):** Hybrid R@10 0.0807 → 0.0804 at eps 0.1 (0.0738 at eps 0.3); TIGER − Hybrid N@10 −0.0160 [−0.0180, −0.0138] at eps 0.1 (CI < 0 at every eps).
  - **Why TIGER retrieves no unseen item** (teacher-forced per-position log-prob of the target code, final checkpoint, step 30k; diagnosis run in the session, not saved to a results file): unseen targets −5.14 / −7.93 / −12.52 vs seen targets −5.19 / −4.00 / −0.53 for code positions 1–3. The first code is predicted equally well; from the second code on, the model gives unseen items' code combinations very low probability, because it has learned which (code 1, code 2, code 3) combinations exist in training. Generalization through shared codes reaches only the first code.
  - **The paper's cold-start result (Fig. 5: TIGER retrieves unseen items) is not reproduced.** Content-based retrieval (Semantic-KNN, Hybrid) beats every TIGER variant on unseen targets.
  - Caveats: one seed per model; TIGER cold-start trained for 30k steps (paper 200k), best.pt at 28k so it may still have been improving; the exact-scored variant and 2-/1-code matching are our additions, not the paper's method; the log-prob diagnosis used the final checkpoint, the evaluation used best.pt (28k).

## Results
| Run | Split | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|---|---|
| popularity | valid | 0.0091 [0.0079, 0.0103] | 0.0056 [0.0048, 0.0065] | 0.0163 [0.0146, 0.0179] | 0.0079 [0.0071, 0.0088] | 22,363 users, 95% bootstrap CI |
| popularity | test | 0.0075 [0.0064, 0.0085] | 0.0041 [0.0035, 0.0048] | 0.0114 [0.0101, 0.0127] | 0.0054 [0.0047, 0.0061] | commit 5d3b881 |
| sasrec | valid | 0.0832 | 0.0603 | 0.1137 | 0.0701 | best epoch 20 (early-stop metric) |
| sasrec | test | 0.0635 [0.0606, 0.0666] | 0.0454 [0.0433, 0.0478] | 0.0897 [0.0863, 0.0935] | 0.0539 [0.0516, 0.0564] | commit 8d37597; +48–83% vs paper |
| sasrec − popularity | test | +0.0560 [+0.0529, +0.0591] | +0.0413 [+0.0391, +0.0436] | +0.0783 [+0.0745, +0.0824] | +0.0485 [+0.0460, +0.0510] | paired bootstrap |
| sasrec_bce | test | 0.0359 [0.0334, 0.0383] | 0.0228 [0.0212, 0.0245] | 0.0550 [0.0523, 0.0579] | 0.0290 [0.0273, 0.0306] | original loss; −7.3…−9.2% vs paper; commit 6c167b5 |
| popularity_tieshuffle | test | 0.0095 [0.0082, 0.0107] | 0.0057 [0.0049, 0.0065] | 0.0155 [0.0138, 0.0171] | 0.0076 [0.0067, 0.0085] | shuffled same-day ties |
| sasrec_tieshuffle | test | 0.0596 [0.0565, 0.0626] | 0.0417 [0.0394, 0.0441] | 0.0842 [0.0808, 0.0878] | 0.0496 [0.0472, 0.0521] | shuffled ties, CE |
| sasrec_bce_tieshuffle | test | 0.0355 [0.0332, 0.0378] | 0.0231 [0.0215, 0.0249] | 0.0559 [0.0529, 0.0591] | 0.0297 [0.0280, 0.0316] | shuffled ties, BCE, seed 42; −6.6…−8.4% vs paper |
| **sasrec_tieshuffle ×3** | test | 0.0601 ± 0.0010 | 0.0420 ± 0.0003 | 0.0859 ± 0.0015 | 0.0503 ± 0.0006 | **main CE baseline**; seeds 42/43/44, mean ± std (ddof=1) |
| **sasrec_bce_tieshuffle ×3** | test | 0.0339 ± 0.0014 | 0.0220 ± 0.0009 | 0.0539 ± 0.0017 | 0.0285 ± 0.0011 | **paper check**; seeds 42/43/44; −10.5…−12.4% vs paper (within ±15%) |
| tiger_tieshuffle | valid | 0.0516 | 0.0338 | 0.0787 | 0.0426 | best.pt @ step 22k (subset valid N@10 0.0477) |
| **tiger_tieshuffle** | test | 0.0414 [0.0386, 0.0441] | 0.0270 [0.0252, 0.0290] | 0.0651 [0.0614, 0.0686] | 0.0346 [0.0326, 0.0367] | seed 42; run stopped by hand at 100k of 200k (overfitting); −8.9 / −15.8 / +0.5 / −9.8% vs paper TIGER |
| tiger − sasrec_tieshuffle | test | −0.0182 [−0.0212, −0.0153] | −0.0147 [−0.0167, −0.0125] | −0.0190 [−0.0228, −0.0156] | −0.0150 [−0.0170, −0.0129] | paired, vs CE seed 42 (seeds 43/44 similar, all CI < 0) |
| tiger − sasrec_bce_tieshuffle | test | +0.0059 [+0.0032, +0.0088] | +0.0039 [+0.0019, +0.0060] | +0.0092 [+0.0054, +0.0128] | +0.0049 [+0.0029, +0.0070] | paired, vs BCE seed 42 (seeds 43/44 larger, all CI > 0) |
| tiger − popularity_tieshuffle | test | +0.0319 [+0.0290, +0.0349] | +0.0213 [+0.0195, +0.0235] | +0.0496 [+0.0459, +0.0532] | +0.0270 [+0.0249, +0.0293] | paired |
| sasrec_coldstart | test | 0.0563 | 0.0398 | 0.0807 | 0.0477 | cold-start split, CE, seed 42; seen items only (not comparable to main runs) |
| tiger_coldstart | test | 0.0380 | 0.0253 | 0.0576 | 0.0316 | cold-start split, best.pt @ 28k of 30k; seen items only. Unseen targets: R@10 0 (see `results/coldstart_test.md`) |
| *paper SASRec* | test | 0.0387 | 0.0249 | 0.0605 | 0.0318 | reference |
| *paper TIGER* | test | 0.0454 | 0.0321 | 0.0648 | 0.0384 | reference |

## Open Issues / Decisions
- **Deviations from the TIGER paper (README must list these):**
  1. *Input standardization* (per dimension, mean/std over all 12,101 items, stored in the model): raw unit-norm Sentence-T5 vectors collapse the RQ-VAE to a single code. The reconstruction loss is measured in standardized space.
  2. *AdamW lr 1e-3, weight decay 0.01* instead of Adagrad lr 0.4: every Adagrad setting tried collapsed (see Phase 2 above).
  3. *Dead-code reset*: every 10 epochs until epoch 2,000, unused codes at each level move onto random current residuals (240 / 41 / 21 resets in total), followed by ≥1,000 epochs without resets. Final assignments come from this reset-free phase.
  4. *Early stop*: 3,000 epochs (plateau: 5 evals without 1% recon gain or higher min usage) instead of 20k.
  5. *TIGER training stopped at step ~100k of the paper's 200k* (manual stop, overfitting; best checkpoint at 22k). Model selection by NDCG@10 on a fixed 2,000-user valid subset (beam 30) — the paper does not say how it selects checkpoints.
  6. *Cold-start TIGER trained for 30k steps* (paper 200k), set from the main run's best step (22k). Cold-start RQ-VAE trained on seen items only; validation excludes the 913 users with an unseen valid target.
  - Not deviations, but choices the paper leaves open: Adafactor with parameter scaling (`scale_parameter: true`; the paper gives only lr 0.01 + inverse-sqrt schedule); MSE averaged over dimensions for the reconstruction and RQ terms; CPU training (deterministic); 4th token assigned 0, 1, 2, … in item-ID order within a collision group; purity uses the second category level. The paper settings remain runnable as `configs/rqvae_paper.yaml` (collapses).
- `data/processed` now also holds `item_emb.{npy,json}` and `semantic_ids.json`, so runs from now on record more files in `data.sha256`. Pairing uses `splits.json`, which is unchanged. Phase 3 reads Semantic IDs from `data/processed/semantic_ids.json` for the shuffled-ties runs too (same item IDs).
- `results/rqvae/` (model, log, chart) is gitignored like other run outputs; copy the chart into the README assets in Phase 5.
- **Original-order results carry the ASIN-order artifact.** In raw-file order, same-day items are in ASIN (= item ID) order: on all 9,719 same-day valid/test pairs the test item has the higher ID (4,760 / 9,719 with shuffled ties). Treat `popularity`, `sasrec` and `sasrec_bce` and `results/summary_test.md` as a sensitivity check only; the table is labeled.
- **The main-dataset switch was decided after seeing test numbers.** The reason is sound (raw-file order is an ASIN artifact), but the switch lowered SASRec-CE (R@10 −0.0056, paired), the baseline TIGER is compared against. The README must say this.
- **Only the shuffled-ties SASRec runs have 3 seeds.** Original-order runs are seed 42 only. The saved config of `sasrec` lacks `loss: ce` (added later; same default), so `train_sasrec.py` would retrain it rather than skip.
- **Phase 4 bucket note:** 64 test targets (shuffled ties; 138 on original order) never appear in any train sequence. They have 0 train interactions and go in the <=5 bucket (also in PROJECT_SPEC §6).
- **Timestamp ties — superseded (owner, 2026-09-30):** first decided to keep raw-file order; after finding the raw file is ASIN-sorted, the main dataset switched to shuffled ties (see above). README must explain both. (9,719 / 22,363 users have tied valid/test timestamps under raw order.)
- **SASRec gap — RESOLVED (owner, 2026-09-30):** cause is the loss (full CE vs original BCE). Decisions: (1) report both — SASRec-BCE = paper-reproduction check, SASRec-CE = main strong baseline; (2) main dataset = shuffled ties (`data/processed_tieshuffle`, tie-seed 0), original ASIN order kept as sensitivity check; TIGER on shuffled ties first, on original order only if time allows. PROJECT_SPEC §2, §4, §5 updated. Earlier read-only diagnostics:
  - Same-day valid/test users (43.5%): SASRec test R@10 0.1260 / N@10 0.0802 vs different-day 0.0618 / 0.0336 (different-day subgroup is close to the paper). The raw 5-core file is sorted by ASIN, so same-day items are in ASIN order.
  - History masking (valid): N@10 0.0701 masked vs 0.0519 unmasked; R@10 0.1137 vs 0.1037.
  - Loss: full-softmax CE vs original SASRec's BCE with one sampled negative — measured in (a), explains the gap.
  - Ruled out: exact score ties at target (0 in 2,000 valid users), repeated items (0), test used for selection (no).
- Main-dataset configs are the `*_tieshuffle.yaml` files (`popularity_tieshuffle`, `sasrec_tieshuffle` = CE main baseline, `sasrec_bce_tieshuffle` = paper check); the un-suffixed configs are the original-order sensitivity runs.
- README must document: history masking, tie-breaking by item ID, Popularity counts from train positions only.

## Next Step
- Phase 4 done (buckets, prefix analysis, cold start). Open choices for the owner: (a) whether to save the per-position log-prob diagnosis as a script + results file (the numbers above are not in `results/` yet); (b) more TIGER seeds (each ~3–4 h with `--max-steps 40000`); (c) whether to look into TIGER's overfitting (a further deviation from the paper); (d) Phase 5 README (copy `results/tiger/eval_curve.png`, `results/analysis/*.png`, `results/rqvae/first_code_categories.png`, `results/coldstart/fig5_coldstart.png` into README assets).
