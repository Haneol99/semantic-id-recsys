### test — mean [95% bootstrap CI over 22,363 users]

> Main dataset: same-day ties shuffled (tie-seed 0). SASRec CIs and SASRec pairs are from the seed-42 runs; TIGER (one seed, 42; best.pt at step 22k of a run stopped at 100k) is paired with every SASRec seed.

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle | 0.0095 [0.0082, 0.0107] | 0.0057 [0.0049, 0.0065] | 0.0155 [0.0138, 0.0171] | 0.0076 [0.0067, 0.0085] |
| sasrec_tieshuffle | 0.0596 [0.0565, 0.0626] | 0.0417 [0.0394, 0.0441] | 0.0842 [0.0808, 0.0878] | 0.0496 [0.0472, 0.0521] |
| sasrec_bce_tieshuffle | 0.0355 [0.0332, 0.0378] | 0.0231 [0.0215, 0.0249] | 0.0559 [0.0529, 0.0591] | 0.0297 [0.0280, 0.0316] |
| tiger_tieshuffle | 0.0414 [0.0386, 0.0441] | 0.0270 [0.0252, 0.0290] | 0.0651 [0.0614, 0.0686] | 0.0346 [0.0326, 0.0367] |
| *paper tiger* | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| *paper sasrec* | 0.0387 | 0.0249 | 0.0605 | 0.0318 |

Across training seeds: mean ± std (sample std, ddof=1). SASRec training on MPS is not bit-for-bit deterministic; seed variation covers that noise.

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| sasrec_tieshuffle (seeds 42, 43, 44) | 0.0601 ± 0.0010 | 0.0420 ± 0.0003 | 0.0859 ± 0.0015 | 0.0503 ± 0.0006 |
| sasrec_bce_tieshuffle (seeds 42, 43, 44) | 0.0339 ± 0.0014 | 0.0220 ± 0.0009 | 0.0539 ± 0.0017 | 0.0285 ± 0.0011 |

Relative to paper tiger (single runs above):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle | -79.1% | -82.3% | -76.1% | -80.2% |
| sasrec_tieshuffle | +31.2% | +29.9% | +29.9% | +29.2% |
| sasrec_bce_tieshuffle | -21.9% | -27.9% | -13.7% | -22.6% |
| tiger_tieshuffle | -8.9% | -15.8% | +0.5% | -9.8% |

Relative to paper sasrec (single runs above):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle | -75.5% | -77.1% | -74.4% | -76.1% |
| sasrec_tieshuffle | +53.9% | +67.5% | +39.1% | +56.0% |
| sasrec_bce_tieshuffle | -8.4% | -7.1% | -7.6% | -6.6% |
| tiger_tieshuffle | +6.9% | +8.6% | +7.6% | +8.9% |

Paired difference a − b (paired by user): mean [95% CI] (fraction of resamples with diff ≤ 0):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| tiger_tieshuffle − popularity_tieshuffle | +0.0319 [+0.0290, +0.0349] (0.000) | +0.0213 [+0.0195, +0.0235] (0.000) | +0.0496 [+0.0459, +0.0532] (0.000) | +0.0270 [+0.0249, +0.0293] (0.000) |
| tiger_tieshuffle − sasrec_tieshuffle | -0.0182 [-0.0212, -0.0153] (1.000) | -0.0147 [-0.0167, -0.0125] (1.000) | -0.0190 [-0.0228, -0.0156] (1.000) | -0.0150 [-0.0170, -0.0129] (1.000) |
| tiger_tieshuffle − sasrec_tieshuffle_seed43 | -0.0181 [-0.0212, -0.0152] (1.000) | -0.0149 [-0.0171, -0.0128] (1.000) | -0.0216 [-0.0253, -0.0183] (1.000) | -0.0160 [-0.0182, -0.0140] (1.000) |
| tiger_tieshuffle − sasrec_tieshuffle_seed44 | -0.0199 [-0.0229, -0.0169] (1.000) | -0.0153 [-0.0174, -0.0131] (1.000) | -0.0216 [-0.0252, -0.0180] (1.000) | -0.0159 [-0.0181, -0.0138] (1.000) |
| tiger_tieshuffle − sasrec_bce_tieshuffle | +0.0059 [+0.0032, +0.0088] (0.000) | +0.0039 [+0.0019, +0.0060] (0.000) | +0.0092 [+0.0054, +0.0128] (0.000) | +0.0049 [+0.0029, +0.0070] (0.000) |
| tiger_tieshuffle − sasrec_bce_tieshuffle_seed43 | +0.0079 [+0.0052, +0.0107] (0.000) | +0.0056 [+0.0038, +0.0076] (0.000) | +0.0120 [+0.0084, +0.0155] (0.000) | +0.0069 [+0.0049, +0.0089] (0.000) |
| tiger_tieshuffle − sasrec_bce_tieshuffle_seed44 | +0.0085 [+0.0057, +0.0113] (0.000) | +0.0055 [+0.0035, +0.0075] (0.000) | +0.0123 [+0.0091, +0.0157] (0.000) | +0.0067 [+0.0047, +0.0088] (0.000) |
| sasrec_tieshuffle − popularity_tieshuffle | +0.0501 [+0.0469, +0.0535] (0.000) | +0.0360 [+0.0335, +0.0385] (0.000) | +0.0687 [+0.0650, +0.0724] (0.000) | +0.0420 [+0.0395, +0.0445] (0.000) |
| sasrec_bce_tieshuffle − popularity_tieshuffle | +0.0260 [+0.0233, +0.0287] (0.000) | +0.0174 [+0.0157, +0.0194] (0.000) | +0.0404 [+0.0371, +0.0437] (0.000) | +0.0221 [+0.0202, +0.0241] (0.000) |
| sasrec_tieshuffle − sasrec_bce_tieshuffle | +0.0241 [+0.0215, +0.0269] (0.000) | +0.0186 [+0.0166, +0.0207] (0.000) | +0.0283 [+0.0250, +0.0313] (0.000) | +0.0199 [+0.0179, +0.0219] (0.000) |

Data: popularity_tieshuffle → `data/processed_tieshuffle`, sasrec_tieshuffle → `data/processed_tieshuffle`, sasrec_bce_tieshuffle → `data/processed_tieshuffle`, tiger_tieshuffle → `data/processed_tieshuffle`
