# Phase 4 analysis — test split, main dataset (shuffled ties)

Per bucket: Popularity and TIGER (one seed) as mean [95% bootstrap CI]; SASRec as mean ± std over seeds 42/43/44. TIGER − SASRec: paired bootstrap over the bucket's users against the per-user mean over the three SASRec seeds, mean [95% CI]; **bold** = CI excludes 0. `seeds` = how many of the three single-seed pairings have a CI excluding 0 with the same sign. 1,000 resamples, seed 0.

## Recall@10

### Test-target train interactions

| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | TIGER − SASRec-BCE | seeds |
|---|---:|---|---|---|---|---|---|---|---|
| 0–5 | 5,114 | 0.0000 [0.0000, 0.0000] | 0.0096 ± 0.0015 | 0.0167 ± 0.0006 | 0.0047 [0.0029, 0.0066] | **-0.0120 [-0.0150, -0.0089]** | 3/3 | **-0.0049 [-0.0072, -0.0025]** | 3/3 |
| ↳ of which 0 (never in train) | 64 | 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | 0/3 | +0.0000 [+0.0000, +0.0000] | 0/3 |
| 6–20 | 8,620 | 0.0000 [0.0000, 0.0000] | 0.0256 ± 0.0038 | 0.0488 ± 0.0027 | 0.0254 [0.0220, 0.0285] | **-0.0234 [-0.0275, -0.0196]** | 3/3 | -0.0002 [-0.0035, +0.0027] | 1/3 |
| >20 | 8,629 | 0.0401 [0.0360, 0.0444] | 0.1085 ± 0.0008 | 0.1639 ± 0.0022 | 0.1406 [0.1332, 0.1476] | **-0.0233 [-0.0302, -0.0162]** | 3/3 | **+0.0321 [+0.0256, +0.0392]** | 3/3 |

### Test history length (items)

| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | TIGER − SASRec-BCE | seeds |
|---|---:|---|---|---|---|---|---|---|---|
| 4–5 | 11,383 | 0.0181 [0.0155, 0.0203] | 0.0477 ± 0.0008 | 0.0744 ± 0.0015 | 0.0574 [0.0533, 0.0616] | **-0.0170 [-0.0215, -0.0129]** | 3/3 | **+0.0096 [+0.0059, +0.0135]** | 3/3 |
| 6–10 | 7,433 | 0.0151 [0.0124, 0.0179] | 0.0515 ± 0.0010 | 0.0830 ± 0.0009 | 0.0638 [0.0580, 0.0696] | **-0.0192 [-0.0249, -0.0132]** | 3/3 | **+0.0123 [+0.0070, +0.0177]** | 3/3 |
| 11–20 | 2,528 | 0.0095 [0.0059, 0.0134] | 0.0671 ± 0.0070 | 0.1085 ± 0.0055 | 0.0783 [0.0680, 0.0882] | **-0.0302 [-0.0411, -0.0189]** | 3/3 | **+0.0112 [+0.0012, +0.0212]** | 2/3 |
| >20 | 1,019 | 0.0039 [0.0010, 0.0079] | 0.1079 ± 0.0068 | 0.1789 ± 0.0082 | 0.1286 [0.1099, 0.1511] | **-0.0504 [-0.0694, -0.0330]** | 3/3 | **+0.0206 [+0.0039, +0.0386]** | 2/3 |

### Valid/test pair

| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | TIGER − SASRec-BCE | seeds |
|---|---:|---|---|---|---|---|---|---|---|
| same day | 9,719 | 0.0218 [0.0190, 0.0247] | 0.0712 ± 0.0035 | 0.1145 ± 0.0029 | 0.0899 [0.0845, 0.0954] | **-0.0246 [-0.0304, -0.0190]** | 3/3 | **+0.0188 [+0.0134, +0.0237]** | 3/3 |
| different day | 12,644 | 0.0106 [0.0089, 0.0123] | 0.0407 ± 0.0020 | 0.0639 ± 0.0007 | 0.0460 [0.0421, 0.0496] | **-0.0178 [-0.0217, -0.0139]** | 3/3 | **+0.0054 [+0.0017, +0.0087]** | 1/3 |

## NDCG@10

### Test-target train interactions

| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | TIGER − SASRec-BCE | seeds |
|---|---:|---|---|---|---|---|---|---|---|
| 0–5 | 5,114 | 0.0000 [0.0000, 0.0000] | 0.0056 ± 0.0010 | 0.0094 ± 0.0002 | 0.0020 [0.0012, 0.0029] | **-0.0074 [-0.0093, -0.0053]** | 3/3 | **-0.0036 [-0.0050, -0.0022]** | 3/3 |
| ↳ of which 0 (never in train) | 64 | 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 [0.0000, 0.0000] | +0.0000 [+0.0000, +0.0000] | 0/3 | +0.0000 [+0.0000, +0.0000] | 0/3 |
| 6–20 | 8,620 | 0.0000 [0.0000, 0.0000] | 0.0147 ± 0.0023 | 0.0300 ± 0.0018 | 0.0140 [0.0120, 0.0162] | **-0.0160 [-0.0185, -0.0137]** | 3/3 | -0.0006 [-0.0025, +0.0010] | 1/3 |
| >20 | 8,629 | 0.0197 [0.0174, 0.0219] | 0.0558 ± 0.0002 | 0.0948 ± 0.0009 | 0.0746 [0.0703, 0.0789] | **-0.0202 [-0.0245, -0.0159]** | 3/3 | **+0.0188 [+0.0147, +0.0226]** | 3/3 |

### Test history length (items)

| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | TIGER − SASRec-BCE | seeds |
|---|---:|---|---|---|---|---|---|---|---|
| 4–5 | 11,383 | 0.0088 [0.0075, 0.0100] | 0.0258 ± 0.0006 | 0.0432 ± 0.0006 | 0.0303 [0.0278, 0.0328] | **-0.0129 [-0.0157, -0.0104]** | 3/3 | **+0.0045 [+0.0022, +0.0068]** | 3/3 |
| 6–10 | 7,433 | 0.0076 [0.0062, 0.0093] | 0.0272 ± 0.0006 | 0.0475 ± 0.0007 | 0.0337 [0.0303, 0.0371] | **-0.0138 [-0.0171, -0.0103]** | 3/3 | **+0.0065 [+0.0036, +0.0098]** | 3/3 |
| 11–20 | 2,528 | 0.0045 [0.0027, 0.0065] | 0.0340 ± 0.0032 | 0.0661 ± 0.0019 | 0.0434 [0.0373, 0.0495] | **-0.0227 [-0.0297, -0.0160]** | 3/3 | **+0.0094 [+0.0035, +0.0153]** | 2/3 |
| >20 | 1,019 | 0.0014 [0.0003, 0.0029] | 0.0536 ± 0.0068 | 0.1107 ± 0.0048 | 0.0682 [0.0565, 0.0818] | **-0.0425 [-0.0548, -0.0305]** | 3/3 | **+0.0146 [+0.0041, +0.0249]** | 2/3 |

### Valid/test pair

| Bucket | n | Popularity | SASRec-BCE ×3 | SASRec-CE ×3 | TIGER | TIGER − SASRec-CE | seeds | TIGER − SASRec-BCE | seeds |
|---|---:|---|---|---|---|---|---|---|---|
| same day | 9,719 | 0.0105 [0.0090, 0.0121] | 0.0386 ± 0.0016 | 0.0697 ± 0.0019 | 0.0492 [0.0457, 0.0524] | **-0.0205 [-0.0242, -0.0173]** | 3/3 | **+0.0106 [+0.0073, +0.0135]** | 3/3 |
| different day | 12,644 | 0.0054 [0.0044, 0.0063] | 0.0207 ± 0.0008 | 0.0353 ± 0.0006 | 0.0235 [0.0214, 0.0254] | **-0.0119 [-0.0141, -0.0097]** | 3/3 | **+0.0028 [+0.0007, +0.0047]** | 2/3 |

## Semantic-ID prefix analysis (missed test users)

Share of users whose target is not in the model's top 10 but whose top-10 list holds an item with the same first 1 / 2 / 3 Semantic-ID codes as the target. Chance: the same lists against a random other user's target. SASRec: mean over 3 seeds. Codes: 256 per level; first code ≈ product category (purity 0.915).

| Model | missed users | ≥ first code | ≥ first 2 | ≥ first 3 | chance ≥1 | chance ≥2 | chance ≥3 |
|---|---:|---|---|---|---|---|---|
| Popularity | 22,017 | 0.067 | 0.0015 | 0.0000 | 0.072 | 0.0018 | 0.0001 |
| SASRec-BCE | 21,157 | 0.153 | 0.0098 | 0.0013 | 0.042 | 0.0010 | 0.0000 |
| SASRec-CE | 20,443 | 0.153 | 0.010 | 0.0015 | 0.039 | 0.0008 | 0.0001 |
| TIGER | 20,907 | 0.163 | 0.013 | 0.0011 | 0.031 | 0.0009 | 0.0001 |

Paired, on users missed by TIGER and by all three seeds of the SASRec variant (SASRec share = mean over seeds), difference TIGER − SASRec [95% CI]:

| vs | users both missed | ≥ first code: TIGER / SASRec | diff | ≥ first 2: TIGER / SASRec | diff |
|---|---:|---|---|---|---|
| SASRec-CE | 19,239 | 0.147 / 0.141 | **+0.0064 [+0.0018, +0.0110]** | 0.010 / 0.0084 | **+0.0021 [+0.0008, +0.0034]** |
| SASRec-BCE | 19,504 | 0.150 / 0.134 | **+0.0152 [+0.0104, +0.0198]** | 0.011 / 0.0070 | **+0.0038 [+0.0024, +0.0054]** |

By test-target train interactions (missed users; share with ≥ first code / ≥ first 2 codes):

| Bucket | Popularity | SASRec-BCE | SASRec-CE | TIGER |
|---|---|---|---|---|
| 0–5 | 0.060 / 0.0008 | 0.138 / 0.0065 | 0.147 / 0.0078 | 0.158 / 0.011 |
| of which 0 (never in train) | 0.047 / 0.0000 | 0.156 / 0.016 | 0.125 / 0.016 | 0.188 / 0.016 |
| 6–20 | 0.063 / 0.0010 | 0.145 / 0.0094 | 0.153 / 0.011 | 0.164 / 0.013 |
| >20 | 0.074 / 0.0025 | 0.171 / 0.012 | 0.158 / 0.011 | 0.164 / 0.013 |

Checks: top-10 lists recomputed from best.pt agree with the saved per-user ranks for popularity_tieshuffle 1.0000, sasrec_bce_tieshuffle 1.0000, sasrec_bce_tieshuffle_seed43 1.0000, sasrec_bce_tieshuffle_seed44 1.0000, sasrec_tieshuffle 1.0000, sasrec_tieshuffle_seed43 1.0000, sasrec_tieshuffle_seed44 1.0000, tiger_tieshuffle 1.0000 of users.
