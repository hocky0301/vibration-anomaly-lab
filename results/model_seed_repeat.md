# Model-seed repeat: one fixed draw, 10 model seeds

Synthetic data, the default draw (the data file never changes between rows).  Only the
estimators' random_state and the leaky K-fold shuffle change.  Compare with
results/seed_sweep.md, where every row is a fresh draw of machines.

| protocol | trial F1@0.5 mean ± std | unit F1@0.5 mean ± std | unit AUC mean ± std |
|---|---|---|---|
| a | 0.907 ± 0.0143 | 1.000 ± 0.0000 | 1.000 ± 0.0000 |
| b | 0.455 ± 0.0000 | 0.000 ± 0.0000 | 1.000 ± 0.0000 |
| b_linear | 0.000 ± 0.0000 | 0.000 ± 0.0000 | 0.000 ± 0.0000 |
| b_noctx | 0.337 ± 0.0000 | 0.000 ± 0.0000 | 0.000 ± 0.0000 |
| c | 0.424 ± 0.0000 | 0.800 ± 0.0000 | 0.875 ± 0.0000 |
| c_meanstd | 0.395 ± 0.0000 | 0.500 ± 0.0000 | 0.875 ± 0.0000 |
| d | 0.286 ± 0.0000 | 0.000 ± 0.0000 | 0.938 ± 0.0000 |
| e | 0.852 ± 0.0000 | 1.000 ± 0.0000 | 1.000 ± 0.0000 |
| e_linear | 0.000 ± 0.0000 | 0.000 ± 0.0000 | 0.000 ± 0.0000 |

model seeds used: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
