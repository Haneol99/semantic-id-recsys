# PROJECT_SPEC.md — Semantic-ID Generative Retrieval Recommender

> Source of truth for this project. Claude Code: read this file (and STATUS.md) at the start of every session. Do not change scope, protocol, or targets here without the owner's approval.

## 1. Goal

Reimplement **TIGER** (Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023, arXiv:2305.05065) on Amazon Beauty, compare it against Popularity and SASRec under one shared evaluation pipeline, and analyze **where** generative retrieval helps (cold-start / infrequent items) with **bootstrap confidence intervals**.

Deliverables:
1. Reproducible GitHub repo (config-driven runs, fixed seeds, one command per experiment).
2. Results table (Recall@5/10, NDCG@5/10) with 95% bootstrap CIs.
3. Cold-start / item-frequency analysis and Semantic-ID quality visualization.
4. README suitable for a resume link.

Owner deadline: submit Google applications by ~2026-10-13 (hard stop: referral expires 2026-10-28). Only report numbers that were actually produced.

## 2. Data

- **Dataset:** Amazon Product Reviews 2014 (McAuley), category **Beauty**, 5-core.
  - Reviews: `reviews_Beauty_5.json.gz`
  - Metadata: `meta_Beauty.json.gz` (title, price, brand, categories). Note: the 2014 meta file is Python-dict-literal lines, not strict JSON — parse with `ast.literal_eval`.
- **Preprocessing (must match the paper):**
  - Build each user's sequence by sorting reviews by timestamp.
  - **Same-day ties (decided 2026-09-30):** timestamps have day resolution and the raw 5-core file is sorted by ASIN, so keeping raw-file order would order same-day items by ASIN. **Main dataset:** shuffle same-timestamp reviews per user with a fixed seed (`python -m recsys.data.preprocess --tie-seed 0 --out-dir data/processed_tieshuffle`). **Sensitivity check:** raw-file (ASIN) order in `data/processed/`. Both give the same counts; 6,554 users get a different test item.
  - Keep users with >= 5 interactions (5-core).
  - Remap item IDs to contiguous integers starting at 1 (0 = padding). Assign IDs in a **random or ASIN-sorted order, not in order of first appearance** (avoid sequential-ID leakage noted in TIGER Appendix D).
- **Sanity check (must pass):** 22,363 users, 12,101 items, mean sequence length ~8.87, median 6. If counts differ, stop and report.
- Raw and processed data live in `data/` and are **never committed** (gitignored).

## 3. Split & Evaluation Protocol (shared by all models)

- **Leave-one-out:** for each user, last item = test, second-to-last = validation, the rest = train.
- **Full ranking** over all items (no negative sampling).
- Mask items already in the user's input history when ranking (document this choice in README).
- Metrics: **Recall@5, Recall@10, NDCG@5, NDCG@10** (one relevant item per user, so Recall@K = HitRate@K).
- Save **per-user metric arrays** (`.npy`) for every run so bootstrap CIs and subgroup analysis can be computed later without retraining.
- Model selection uses validation only. Test is evaluated once per final config.

## 4. Models

| Model | Phase | Notes |
|---|---|---|
| Popularity | 1 | Rank items by train-set interaction count |
| SASRec (CE) | 1 | **Main strong baseline.** PyTorch, 2 causal self-attention blocks, hidden 64, 1 head, dropout 0.2, max_len 50, full-softmax cross-entropy over all items |
| SASRec (BCE) | 1 | **Paper-reproduction check.** Same model, original SASRec loss: BCE with one sampled negative per position |
| TIGER | 2–3 | Sentence-T5 embeddings -> RQ-VAE Semantic IDs -> encoder-decoder Transformer + beam search |

**Datasets per model (decided 2026-09-30):** all models run on the main dataset (shuffled ties) first. On the original-order dataset, Popularity and both SASRec variants already exist; TIGER runs there only if time allows.

### TIGER reference settings (from the paper)
- Item text: title, price, brand, categories -> Sentence-T5 (768-d).
- RQ-VAE: encoder 768->512->256->128->32 (ReLU), 3 levels x codebook 256 (dim 32), beta=0.25, k-means init on first batch, target codebook usage >= 80%. Append a 4th token to resolve collisions (always 0 if no collision) -> Semantic ID length 4.
- Seq2seq: 4 encoder + 4 decoder layers, 6 heads x 64, d_model 128, MLP 1024, dropout 0.1, ~13M params; 2000 hashed user-ID tokens prepended; history capped at 20 items; batch 256; lr 0.01 then inverse-sqrt decay.
- Decoding: beam search; filter invalid IDs (increase beam if needed).

## 5. Reference Numbers (paper, Amazon Beauty, full ranking)

| Model | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| SASRec | 0.0387 | 0.0249 | 0.0605 | 0.0318 |
| TIGER | 0.0454 | 0.0321 | 0.0648 | 0.0384 |

Targets: our SASRec **(BCE)** within roughly ±15% of the paper; if outside, investigate preprocessing/eval before moving on. SASRec (CE) is expected to exceed the paper's SASRec, since it uses a stronger loss than the original.

Status (Phase 1, test): SASRec (BCE) on shuffled ties is −6.6% to −8.4% vs paper for seed 42 and −10.5% to −12.4% for the 3-seed mean (seeds 42–44); on original order (seed 42 only) −7.3% to −9.2%. The check passes. SASRec (CE) on shuffled ties scores R@10 0.0859 ± 0.0015 / N@10 0.0503 ± 0.0006 (3 seeds), above the paper's TIGER too. TIGER is therefore compared against **both** SASRec variants, and the write-up must report honestly where TIGER stands relative to SASRec (CE).

## 6. Analysis (Phase 4)

- 95% bootstrap CIs over users (>= 1,000 resamples) for each metric and for paired differences between models.
- Item-frequency buckets (e.g., test items with <=5, 6–20, >20 train interactions) — compare SASRec vs TIGER per bucket.
  - Test targets that never appear in any train sequence (64 users on shuffled ties; 138 on original order) have 0 train interactions and go in the <=5 bucket. SASRec never sees those items as a positive or as an input (they get gradient only as negatives).
- Cold-start experiment: hold out ~5% of test items from training (paper Sec. 4.3); report retrieval of unseen items.
  - Status (done 2026-10-02, test, `results/coldstart_test.md`): 415 held-out items (5% of distinct test targets), 1,098 users with an unseen test target. RQ-VAE, SASRec-CE and TIGER (30k steps) retrained on the cold-start train split, one seed each. On unseen targets at eps 0.1, Recall@10 is TIGER (paper method) 0.0000, Semantic-KNN 0.0565, SASRec-CE + Semantic-KNN hybrid 0.0811. **The paper's cold-start result is not reproduced:** TIGER's beam never generates an unseen item's ID, and only 3 / 415 unseen items share a 3-code prefix with a seen item. Per-position log-probs (best.pt, `results/coldstart_logprob_test.md`) show why: TIGER predicts unseen targets' first code as well as seen ones (−5.14 vs −5.24) but gives their 2nd and 3rd codes very low probability (−7.93 / −12.52 vs −3.94 / −0.53). The write-up must report this honestly, together with the labeled TIGER variants (exact-scored unseen items, 2-/1-code matching) that are not the paper's method.
- Semantic-ID quality: category purity per first code; visualization.

## 7. Engineering Rules

- Python 3.11, PyTorch with MPS on Apple Silicon (fallback to CPU if unavailable; log the device).
- All runs driven by YAML configs in `configs/`; results written to `results/<run_name>/` (config copy, metrics.json, per-user .npy, seed, git hash).
- Fixed seeds; deterministic splits saved to disk.
- Small, tested modules; pytest for data/eval code.
- W&B / Hydra / serving are out of scope until Phase 3 numbers exist.

## 8. Repo Layout

```
configs/            YAML configs
data/               raw + processed (gitignored)
src/recsys/
  data/             download, preprocess, dataset classes
  eval/             metrics, full-ranking evaluator, bootstrap
  models/           popularity.py, sasrec.py, rqvae.py, tiger.py
  train/            training loops
scripts/            CLI entry points
tests/
results/            run outputs (gitignored except summary tables)
PROJECT_SPEC.md
STATUS.md
README.md
```

## 9. Phases

| Phase | Dates | Output |
|---|---|---|
| 1. Foundation | 9/30–10/2 | Data pipeline + eval + Popularity + SASRec numbers |
| 2. Semantic IDs | 10/3–10/5 | Item embeddings, RQ-VAE, Semantic IDs, code-quality check |
| 3. Generative model | 10/6–10/9 | TIGER training + beam search + eval |
| 4. Analysis | 10/10–10/11 | Bootstrap CIs, frequency/cold-start analysis, plots |
| 5. Write-up | 10/12–10/13 | README, resume bullet, apply |
