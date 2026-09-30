### test — mean [95% bootstrap CI over 22,363 users]

> Tie order effect: shuffled − original, same model and seed (42), paired by user ID. Test targets differ for 6,554 of 22,363 users. Single training seed per side.

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle | 0.0095 [0.0082, 0.0107] | 0.0057 [0.0049, 0.0065] | 0.0155 [0.0138, 0.0171] | 0.0076 [0.0067, 0.0085] |
| popularity | 0.0075 [0.0064, 0.0085] | 0.0041 [0.0035, 0.0048] | 0.0114 [0.0101, 0.0127] | 0.0054 [0.0047, 0.0061] |
| sasrec_tieshuffle | 0.0596 [0.0565, 0.0626] | 0.0417 [0.0394, 0.0441] | 0.0842 [0.0808, 0.0878] | 0.0496 [0.0472, 0.0521] |
| sasrec | 0.0635 [0.0606, 0.0666] | 0.0454 [0.0433, 0.0478] | 0.0897 [0.0863, 0.0935] | 0.0539 [0.0516, 0.0564] |
| sasrec_bce_tieshuffle | 0.0355 [0.0332, 0.0378] | 0.0231 [0.0215, 0.0249] | 0.0559 [0.0529, 0.0591] | 0.0297 [0.0280, 0.0316] |
| sasrec_bce | 0.0359 [0.0334, 0.0383] | 0.0228 [0.0212, 0.0245] | 0.0550 [0.0523, 0.0579] | 0.0290 [0.0273, 0.0306] |

Paired difference a − b (paired by user ID across data): mean [95% CI] (fraction of resamples with diff ≤ 0):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity_tieshuffle − popularity | +0.0020 [+0.0009, +0.0031] (0.001) | +0.0016 [+0.0009, +0.0022] (0.000) | +0.0041 [+0.0026, +0.0054] (0.000) | +0.0022 [+0.0015, +0.0029] (0.000) |
| sasrec_tieshuffle − sasrec | -0.0039 [-0.0071, -0.0010] (0.994) | -0.0037 [-0.0061, -0.0016] (1.000) | -0.0056 [-0.0093, -0.0021] (0.999) | -0.0043 [-0.0066, -0.0021] (1.000) |
| sasrec_bce_tieshuffle − sasrec_bce | -0.0004 [-0.0033, +0.0025] (0.601) | +0.0003 [-0.0017, +0.0024] (0.400) | +0.0009 [-0.0025, +0.0043] (0.298) | +0.0008 [-0.0014, +0.0027] (0.234) |

Data: popularity_tieshuffle → `data/processed_tieshuffle`, popularity → `data/processed`, sasrec_tieshuffle → `data/processed_tieshuffle`, sasrec → `data/processed`, sasrec_bce_tieshuffle → `data/processed_tieshuffle`, sasrec_bce → `data/processed`
