### test — mean [95% bootstrap CI over 22,363 users]

> Main dataset: same-day ties shuffled (tie-seed 0). CIs and pairs are from the seed-42 runs.

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle | 0.0095 [0.0082, 0.0107] | 0.0057 [0.0049, 0.0065] | 0.0155 [0.0138, 0.0171] | 0.0076 [0.0067, 0.0085] |
| sasrec_tieshuffle | 0.0596 [0.0565, 0.0626] | 0.0417 [0.0394, 0.0441] | 0.0842 [0.0808, 0.0878] | 0.0496 [0.0472, 0.0521] |
| sasrec_bce_tieshuffle | 0.0355 [0.0332, 0.0378] | 0.0231 [0.0215, 0.0249] | 0.0559 [0.0529, 0.0591] | 0.0297 [0.0280, 0.0316] |
| *paper sasrec* | 0.0387 | 0.0249 | 0.0605 | 0.0318 |

Across training seeds: mean ± std (sample std, ddof=1). SASRec training on MPS is not bit-for-bit deterministic; seed variation covers that noise.

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| sasrec_tieshuffle (seeds 42, 43, 44) | 0.0601 ± 0.0010 | 0.0420 ± 0.0003 | 0.0859 ± 0.0015 | 0.0503 ± 0.0006 |
| sasrec_bce_tieshuffle (seeds 42, 43, 44) | 0.0339 ± 0.0014 | 0.0220 ± 0.0009 | 0.0539 ± 0.0017 | 0.0285 ± 0.0011 |

Relative to paper sasrec (single runs above):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle | -75.5% | -77.1% | -74.4% | -76.1% |
| sasrec_tieshuffle | +53.9% | +67.5% | +39.1% | +56.0% |
| sasrec_bce_tieshuffle | -8.4% | -7.1% | -7.6% | -6.6% |

Paired difference a − b (paired by user): mean [95% CI] (fraction of resamples with diff ≤ 0):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| sasrec_tieshuffle − popularity_tieshuffle | +0.0501 [+0.0469, +0.0535] (0.000) | +0.0360 [+0.0335, +0.0385] (0.000) | +0.0687 [+0.0650, +0.0724] (0.000) | +0.0420 [+0.0395, +0.0445] (0.000) |
| sasrec_bce_tieshuffle − popularity_tieshuffle | +0.0260 [+0.0233, +0.0287] (0.000) | +0.0174 [+0.0157, +0.0194] (0.000) | +0.0404 [+0.0371, +0.0437] (0.000) | +0.0221 [+0.0202, +0.0241] (0.000) |
| sasrec_tieshuffle − sasrec_bce_tieshuffle | +0.0241 [+0.0215, +0.0269] (0.000) | +0.0186 [+0.0166, +0.0207] (0.000) | +0.0283 [+0.0250, +0.0313] (0.000) | +0.0199 [+0.0179, +0.0219] (0.000) |

Data: popularity_tieshuffle → `data/processed_tieshuffle`, sasrec_tieshuffle → `data/processed_tieshuffle`, sasrec_bce_tieshuffle → `data/processed_tieshuffle`
