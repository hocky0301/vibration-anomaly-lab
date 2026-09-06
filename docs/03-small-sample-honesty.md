# Small-Sample Honesty

This document describes the synthetic experiment. Measurements come from
`results/default_draw.json`, `results/seed_sweep.json` and
`results/model_seed_repeat.json`; arithmetic is given below.

## Repeated measurements are not new machines

There are 720 trials from 10 machines, with 2 faulty and 8 healthy machines.
The unit-level decision aggregates trial probabilities with a median and uses
the fixed threshold 0.5. Trial measurements inform the decision, but they do
not create additional independently observed faulty machines.

The simulator fixes the number of faulty machines in each draw. Its repeated
seeds redraw machine effects and noise under that design. Real data may add
shared batches, sites, time trends and calibration events, so machine grouping
alone need not establish independence.

## Exact binomial intervals are an illustration with assumptions

For a detector fixed before evaluation, evaluated on independent faulty
machines with a common detection probability, 2 detections out of 2 gives a
two-sided 95% Clopper–Pearson interval of [0.158, 1.000]. With no false alarms
in 8 independent healthy machines, the corresponding false-positive-rate
interval is [0.000, 0.369]. They are intervals for binomial rates, not for F1.

These calculations show how little a perfect small sample can establish.
They are **not formal coverage guarantees for this experiment's pooled OOF
predictions**: folds share training data, different models make the decisions,
and model or threshold selection can use the same labels.

For a perfect independent record of n detections, the lower bound is
`(0.025)**(1/n)`. This gives:

| detected / evaluated | two-sided 95% lower bound |
|---|---|
| 1 / 1 | 0.025 |
| 2 / 2 | 0.158 |
| 3 / 3 | 0.292 |
| 5 / 5 | 0.478 |
| 10 / 10 | 0.692 |
| 20 / 20 | 0.832 |

The corresponding upper bound for no false positives is
`1 - (0.025)**(1/n)`. These are prospective illustrations, not extra observed
machines or results.

## Discrete metrics need careful interpretation

Enumerating every binary prediction on the fixed 2-of-10 label vector gives
15 possible unit F1 values:

```text
0.000  0.182  0.200  0.222  0.250  0.286  0.333  0.364
0.400  0.444  0.500  0.571  0.667  0.800  1.000
```

An F1 value alone need not identify a confusion matrix. In particular, 0.667
can result from one detected fault and no false positives, or two detected
faults and two false positives. Use the per-unit scores and a threshold to
recover the actual decisions.

AUC averages comparisons across 16 positive–negative pairs. Without score
ties it moves in steps of 1/16. Ties receive half credit, so values on a 1/32
grid are possible. The reported AUC also pools predictions from different
leave-one-unit-out models: cross-fold probability scales can affect the ranking.

Across 30 generator draws, c vs b has AUC wins/ties/losses of 4/18/8;
unit F1@0.5 has 10/8/12. Mean AUC favours b (0.950 vs 0.923), while mean
unit F1 favours c (0.772 vs 0.758). These mixed descriptive results do not
establish a consistent winner. Discreteness alone does not make comparison
impossible: enough independent simulator draws can estimate effects **within
that simulator**. It cannot establish real-world performance.

![Variation of trial and unit metrics across generator draws](../figures/05-seed-sweep.png)

## Two random seeds answer different questions

- A **generator seed** redraws machines and observations. Variation across it
  describes this simulation's distribution, including its machine effects.
- An **estimator seed** holds the data fixed and changes the estimators and the
  leaky split's randomisation. In the recorded 10-seed experiment, the
  leave-one-unit-out metrics do not move at the reported precision, while
  protocol a's trial F1 standard deviation is 0.0143.

Neither randomisation reproduces manufacturing changes, unmodelled failure
modes or deployment conditions absent from the simulator. Deterministic
computation is compatible with poor predictions on new machines.

## A threshold is a fitted parameter if labels choose it

The `best (t)` column maximises F1 over 37 thresholds from 0.05 through 0.95
on the same pooled OOF labels used to report the value. It is optimistic by
construction. It is the best **on that grid**, not an upper bound over all
thresholds; even a perfect ranking can require a threshold outside the grid.

The fixed 0.5 columns avoid that particular selection step. They do not undo
choices of model, features, aggregation region or simulator made after seeing
results. The low-half aggregation was defined from the fault's construction;
it is a declared modelling assumption, not a label-free discovery.

`ProductionModel` selects members and a threshold using OOF labels. A clean
separating gap is still a label-selected gap, so its threshold remains
optimistic. The class is an educational illustration, not deployment approval.

## Selection bias and its remedies

Selecting the largest noisy validation estimate tends to select favourable
noise as well as a good candidate. This is often called the winner's curse or
selection-induced optimism. Reporting that same maximum as performance is the
problem. Feature selection, hyperparameter search and threshold search can all
produce it.

A sealed test set estimates a chosen model without reusing selection labels.
Properly nested cross-validation separates selection inside each outer
training fold from evaluation on the outer test fold, estimating the whole
selection procedure. Neither creates more machines or removes small-sample
uncertainty; both address reuse of evaluation information. Selecting again
among outer-CV results reintroduces selection bias.

With this small set, a rich nested comparison is poorly supported. The
repository therefore reports fixed protocols and discloses exploratory
quantities rather than treating them as validated deployment performance.

## Sources of numbers

- Dataset dimensions and default measurements: `results/default_draw.json` and `data/make_synthetic_data.py`.
- Repeated-draw metrics and pairwise counts: `results/seed_sweep.json` and `results/seed_sweep.md`.
- Estimator randomisation: `results/model_seed_repeat.json` and `results/model_seed_repeat.md`.
- F1 ladder: exhaustive binary predictions, `F1 = 2 TP / (2 TP + FP + FN)`.
- AUC grids: 2 positive × 8 negative pairs, with half credit for ties.
- Binomial bounds: endpoint formulae given above under independent fixed-detector assumptions.
- Threshold grid: `verify_claims.py:GRID`.
