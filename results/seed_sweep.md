# Seed sweep: 30 independent draws of the generator

Synthetic data.  Each row averages one protocol over all seeds; every seed is a fresh draw of the machines (fingerprints and fault placement) and a fresh generator noise stream.
Model settings: `--quick` = False (GBM max_iter 300), model seed 0.  Metric definitions: see `verify_claims.py`.

| protocol | split / space / model | mean trial F1@0.5 | mean unit F1@0.5 | seeds with unit F1@0.5 = 1.0 | mean unit AUC | mean unit F1 lowhalf@0.5 |
|---|---|---|---|---|---|---|
| a | StratifiedKFold(5) over trials / raw stats + setpoint + speed_rpm / HistGradientBoosting | 0.930 | 1.000 | 30/30 (1.00) | 1.000 | 1.000 |
| b | leave-one-unit-out / raw stats + setpoint + speed_rpm / HistGradientBoosting | 0.578 | 0.758 | 12/30 (0.40) | 0.950 | 0.967 |
| b_linear | leave-one-unit-out / raw stats + setpoint + speed_rpm / logistic regression | 0.000 | 0.000 | 0/30 (0.00) | 0.148 | 0.000 |
| b_noctx | leave-one-unit-out / raw stats only / HistGradientBoosting | 0.427 | 0.152 | 2/30 (0.07) | 0.688 | 0.809 |
| c | leave-one-unit-out / conditional robust-z (median/MAD) + setpoint / HistGradientBoosting | 0.562 | 0.772 | 9/30 (0.30) | 0.923 | 0.869 |
| c_meanstd | leave-one-unit-out / conditional z (mean/std) + setpoint / HistGradientBoosting | 0.543 | 0.751 | 7/30 (0.23) | 0.940 | 0.824 |
| d | leave-one-unit-out / conditional robust-z (median/MAD) + setpoint / logistic regression | 0.559 | 0.422 | 10/30 (0.33) | 0.923 | 0.877 |
| e | leave-one-unit-out / unit-relative log stats + setpoint / HistGradientBoosting | 0.882 | 1.000 | 30/30 (1.00) | 1.000 | 1.000 |
| e_linear | leave-one-unit-out / unit-relative log stats + setpoint / logistic regression | 0.000 | 0.000 | 0/30 (0.00) | 0.000 | 0.000 |

unit identifiability (5-fold accuracy of a GBM predicting unit_id from raw trial features, chance 0.100): mean 0.859 / min 0.688 / max 0.957

b vs c by unit AUC: wins/ties/losses = 4/18/8 (a win = conditional robust-z (c) strictly above raw (b) on that seed)

seeds used (30): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
