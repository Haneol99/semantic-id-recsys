### test — mean [95% bootstrap CI over 22,363 users]

> Sensitivity check only. Original raw-file order: same-day items are in ASIN (= item ID) order, so on all 9,719 same-day valid/test pairs the test item has the higher ID. These numbers carry that ASIN-order artifact.

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity | 0.0075 [0.0064, 0.0085] | 0.0041 [0.0035, 0.0048] | 0.0114 [0.0101, 0.0127] | 0.0054 [0.0047, 0.0061] |
| sasrec | 0.0635 [0.0606, 0.0666] | 0.0454 [0.0433, 0.0478] | 0.0897 [0.0863, 0.0935] | 0.0539 [0.0516, 0.0564] |
| sasrec_bce | 0.0359 [0.0334, 0.0383] | 0.0228 [0.0212, 0.0245] | 0.0550 [0.0523, 0.0579] | 0.0290 [0.0273, 0.0306] |
| *paper sasrec* | 0.0387 | 0.0249 | 0.0605 | 0.0318 |

Relative to paper sasrec (single runs above):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| popularity | -80.7% | -83.4% | -81.2% | -83.0% |
| sasrec | +64.0% | +82.5% | +48.3% | +69.5% |
| sasrec_bce | -7.3% | -8.3% | -9.2% | -9.0% |

Paired difference a − b (paired by user): mean [95% CI] (fraction of resamples with diff ≤ 0):

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| sasrec − popularity | +0.0560 [+0.0529, +0.0591] (0.000) | +0.0413 [+0.0391, +0.0436] (0.000) | +0.0783 [+0.0745, +0.0824] (0.000) | +0.0485 [+0.0460, +0.0510] (0.000) |
| sasrec_bce − popularity | +0.0284 [+0.0259, +0.0309] (0.000) | +0.0187 [+0.0170, +0.0205] (0.000) | +0.0436 [+0.0408, +0.0469] (0.000) | +0.0235 [+0.0218, +0.0254] (0.000) |
| sasrec − sasrec_bce | +0.0276 [+0.0249, +0.0306] (0.000) | +0.0226 [+0.0205, +0.0248] (0.000) | +0.0348 [+0.0315, +0.0382] (0.000) | +0.0249 [+0.0229, +0.0270] (0.000) |

Data: popularity → `data/processed`, sasrec → `data/processed`, sasrec_bce → `data/processed`
