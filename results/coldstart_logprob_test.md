# Cold-start diagnosis — teacher-forced log-prob of the test target's codes (TIGER, `tiger_coldstart`)

Test split of `data/processed_coldstart`: 1,098 users with an unseen (held-out) target, 21,265 with a seen target. Decoder fed the target's own previous codes; full-vocab log-softmax as in beam search. Mean log-prob [95% bootstrap CI over users]. Uniform over one position's 256 codes: -5.55. Code 4 is the collision token (0 unless items share codes 1–3).

| checkpoint | step | targets | code 1 | code 2 | code 3 | code 4 | sum (codes 1–4) |
|---|---:|---|---|---|---|---|---|
| best.pt | 28,000 | unseen | -5.14 [-5.27, -5.01] | -7.93 [-8.10, -7.76] | -12.52 [-12.81, -12.24] | -0.15 [-0.23, -0.08] | -25.74 [-26.02, -25.46] |
| best.pt | 28,000 | seen | -5.24 [-5.27, -5.21] | -3.94 [-3.97, -3.91] | -0.53 [-0.55, -0.51] | -0.04 [-0.04, -0.03] | -9.75 [-9.80, -9.70] |
| last.pt | 30,000 | unseen | -5.33 [-5.47, -5.20] | -8.06 [-8.25, -7.89] | -12.77 [-13.07, -12.48] | -0.15 [-0.23, -0.08] | -26.31 [-26.60, -26.01] |
| last.pt | 30,000 | seen | -5.35 [-5.38, -5.32] | -4.00 [-4.03, -3.97] | -0.54 [-0.56, -0.52] | -0.04 [-0.04, -0.03] | -9.92 [-9.97, -9.87] |
