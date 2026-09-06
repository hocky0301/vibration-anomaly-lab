# vibration-anomaly-lab

**The unit of independence is the machine, not the row.** In industrial
condition monitoring one machine contributes thousands of rows and the health
label belongs to the machine, so a random split over rows or trials puts the
same machine on both sides of the fold and the model only has to recognise it.
This repository makes that failure measurable on a fully synthetic dataset:
a generator writes ten machines with individual fingerprints and two faulty
ones, a standalone script re-fits nine named evaluation protocols on that data,
and a seed sweep repeats the whole thing on 30 independent draws. It then
measures where the usual fix -- operating-point-conditional standardisation --
stops helping.

[日本語](README.ja.md) · [Reproduction guide](docs/05-reproduction.md) · [Limitations](docs/03-small-sample-honesty.md) · [Validation record](docs/06-validation-record.md)

Everything below is synthetic. Evaluation results come from `results/`;
generator settings and figure diagnostics come from the generator and
`scripts/make_figures.py`. This is a teaching experiment, not a deployable detector.
Machine-level grouping matches this simulator's design; real machines may still
share batch, site, temporal or calibration effects.

## Start here

Use Python 3.11 and the pinned reference environment to reproduce the published
numbers. The reference verification command writes fresh output separately and compares
it with the stored result, including every per-unit score.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
make data
make verify-reference
make test
```

Then read the default-draw table and the repeated-draw table together.
See [the reproduction guide](docs/05-reproduction.md) for supported environments,
full repeated runs, and the distinction between recomputing and updating evidence.

## What is measured

`verify_claims.py` fits nine protocols. Each is a (split, feature space,
model) triple; the trial features are 21 amplitude statistics (rms, std, mad,
iqr, p90, absmean, ptp on three axes) plus, where stated, the operating point.
GBM = `HistGradientBoostingClassifier(max_iter=300)`; linear = standardised
logistic regression, `C=0.005`.

| id | split | feature space | model |
|---|---|---|---|
| a | StratifiedKFold(5) over trials **[leaky]** | raw stats + setpoint + speed_rpm | GBM |
| b | leave-one-unit-out | raw stats + setpoint + speed_rpm | GBM |
| b_linear | leave-one-unit-out | raw stats + setpoint + speed_rpm | linear |
| b_noctx | leave-one-unit-out | raw stats only (no operating point) | GBM |
| c | leave-one-unit-out | conditional robust-z (median/MAD) + setpoint | GBM |
| c_meanstd | leave-one-unit-out | conditional z (mean/std) + setpoint | GBM |
| d | leave-one-unit-out | conditional robust-z (median/MAD) + setpoint | linear |
| e | leave-one-unit-out | unit-relative log stats + setpoint (transductive) | GBM |
| e_linear | leave-one-unit-out | unit-relative log stats + setpoint (transductive) | linear |

In c / c_meanstd / d the reference distribution of each fold is built from
training-fold trials labelled healthy, never from the held-out unit. Protocol
e subtracts each unit's own median log statistic (label-free, but it needs the
unit's whole sweep). Decisions are taken at a fixed threshold of 0.5; the unit
score is the median of the out-of-fold trial probabilities of that unit.

### 30 seeds

30 independent draws of the generator (seeds 0-29), fault placement uniform
at random among the ten units, no constant tuned. Copied from
[`results/seed_sweep.md`](results/seed_sweep.md):

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

`lowhalf`: the unit score is the median over the trials in the low-setpoint
half of the sweep (setpoint at or below the median setpoint), the region in
which the generator places the fault. Unit identifiability across the 30
seeds: mean 0.859 / min 0.688 / max 0.957 (chance 0.100). Pairwise, b vs c by
unit AUC: wins/ties/losses = 4/18/8 (a win = c strictly above b on that seed).

![unit F1 and trial F1 across 30 seeds, per protocol](figures/05-seed-sweep.png)

*Figure 05. One dot per seed: a and e are at 1.000 on every seed; b and
c range from 0.000 to 1.000 at the unit level; c_meanstd ranges from 0.400
to 1.000, and d is bimodal.*

### The default draw, as a worked example

`make verify` on the default seed (2026): 10 units, 720 trials, faulty units
U03 and U06. Copied from
[`results/default_draw.json`](results/default_draw.json). `best (t)` is the
maximum unit F1 over 37 thresholds, chosen on the labels it is scored on --
optimistic by construction.

| id | trial F1@0.5 | unit F1@0.5 | unit AUC | unit F1 lowhalf@0.5 | best (t) |
|---|---|---|---|---|---|
| a | 0.908 | 1.000 | 1.000 | 1.000 | 1.000 (0.05) |
| b | 0.455 | 0.000 | 1.000 | 1.000 | 0.667 (0.05) |
| b_linear | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 (0.05) |
| b_noctx | 0.337 | 0.000 | 0.000 | 1.000 | 0.000 (0.05) |
| c | 0.424 | 0.800 | 0.875 | 0.800 | 0.800 (0.15) |
| c_meanstd | 0.395 | 0.500 | 0.875 | 0.800 | 0.800 (0.15) |
| d | 0.286 | 0.000 | 0.938 | 0.667 | 0.667 (0.28) |
| e | 0.852 | 1.000 | 1.000 | 1.000 | 1.000 (0.05) |
| e_linear | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 (0.05) |

Per-unit out-of-fold median score on the same draw (truth 1 = faulty):

| unit | truth | a | b | c | e |
|---|---|---|---|---|---|
| U01 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| U02 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| U03 | 1 | 1.000 | 0.000 | 0.955 | 1.000 |
| U04 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| U05 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| U06 | 1 | 1.000 | 0.204 | 0.978 | 0.999 |
| U07 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| U08 | 0 | 0.000 | 0.000 | 0.130 | 0.000 |
| U09 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| U10 | 0 | 0.000 | 0.000 | 0.995 | 0.000 |

Unit identifiability on this draw: a GBM predicts `unit_id` from raw trial
features with 5-fold accuracy 0.904 (chance 0.100). This is the leak.

![per-unit scores for protocols a, b, c, e on the default draw](figures/04-protocol-unit-scores.png)

*Figure 04. Protocol b ranks both faulty units first (pooled unit AUC 1.000)
but neither crosses 0.5; protocol c crosses on both and on healthy U10.
Some b scores are too small to distinguish on this linear scale; their full
precision is recorded in `results/default_draw.json`.*

## How to read it

1. **The leak is not fragile.** Protocol a reaches unit F1 1.000 on 30 of 30
   seeds and trial F1 0.930 on average; the same model under leave-one-unit-out
   (b) averages 0.758 and reaches 1.000 on 12 seeds. The mechanism is visible
   without any model: a classifier recovers `unit_id` from the 21 raw
   statistics plus `setpoint` and `speed_rpm` with accuracy 0.859 on average (0.904 on the default draw) against
   a chance level of 0.100. Figure 03 shows why -- at one setpoint, units sit on
   separate levels in both amplitude and speed response. Since the label is
   constant within a unit, recognising the unit is knowing the label.
   [`docs/01`](docs/01-why-random-cv-lies.md).

2. **The grain changes the story.** Row b on the default draw: trial F1 0.455,
   unit F1 0.000, unit AUC 1.000, unit F1 over the low half 1.000. All four are
   the same out-of-fold probabilities. The trial-level number says the model is
   mediocre; the unit-level F1 says it detects nothing; the unit AUC says it
   ranks both faulty units above every healthy one; the low-half F1 says the
   fixed threshold works once the score is taken where the fault is. Report the
   grain the decision is made at, and say which aggregation produced it.

3. **The operating point dominates amplitude.** Removing the two
   operating-point columns from the raw space (b_noctx) drops the 30-seed mean
   unit AUC from 0.950 to 0.688 and the mean unit F1 from 0.758 to 0.152. Figure
   02 shows the reason: healthy amplitude grows across the sweep by more than
   the gap between healthy and faulty units at most setpoints, so an absolute
   amplitude without its setpoint mostly measures the setpoint. What survives
   is the bottom of the sweep, where a faulty unit is quieter than anything
   healthy: b_noctx still reaches a mean unit F1 of 0.809 over the low half.

4. **The conditioned representation helps the tested linear pipeline.** With a linear model, raw
   features give unit AUC 0.148 (b_linear) and conditional robust-z gives 0.923
   (d): the tested linear pipeline ranks units better. With a GBM, the two
   tested feature spaces give mixed results: c vs b
   by unit AUC is 4 wins, 18 ties, 8 losses over 30 seeds, and the means are
   0.923 (c) against 0.950 (b). These results do not establish a general advantage for c. The comparison
   also drops `speed_rpm` in c, so it does not isolate the effect of conditioning. Figure 04 shows the price on the default draw: c lifts the
   faulty units over 0.5 and also lifts healthy U10 to 0.995.
   [`docs/02`](docs/02-conditioning-not-models.md).

5. **Where the fault lives decides the aggregation, and unit-relative
   normalisation is the robust fix here -- with a cost.** The fault is
   concentrated in the low-setpoint half (the generator's knee sits at sweep
   position 0.55, just above the seventh of twelve setpoints, with a
   transition width of 0.22), so a unit score taken as the median over the
   whole sweep dilutes it: the linear model d goes from mean unit F1 0.422 (whole
   sweep) to 0.877 (low half). Protocol e normalises each unit against its own
   median log level and keeps the setpoint; a GBM on that space reaches unit F1
   1.000 on 30 of 30 seeds with mean trial F1 0.882, while the linear model on
   the same space scores 0.000 on every seed (e_linear): what survives the
   normalisation is a sweep *shape*, an interaction with the setpoint. The
   cost is that e is transductive -- it needs the new unit's whole sweep before
   it can score any trial of it.

6. **Two positives per draw give coarse measurements.** With two faulty units per draw, unit AUC
   takes a handful of values and c vs b ends in a tie on 18 of 30 seeds; unit
   F1 at 0.5 can take only 15 distinct values ([`docs/03`](docs/03-small-sample-honesty.md)). These measurements do not establish deployment superiority. Repeated draws
   can estimate differences within this fixed simulator, but Figure 05 does
   not estimate error on real machines.
   [`docs/03`](docs/03-small-sample-honesty.md).

## Honesty notes

- The decision threshold is fixed at 0.5 everywhere except the `best (t)`
  column, which is the maximum over 37 thresholds chosen on the scored labels
  and is optimistic by construction. It is shown so that the fixed-threshold
  numbers can be compared with an explicitly optimistic grid-search result.
  The grid is finite: it is not an upper bound over every possible threshold.
- n = 2 faulty units per draw. Every unit-level number is an existence proof
  about this generator. Thirty seeds show whether an ordering survives redraws
  of the same simulator; they do not estimate variance over real machines.
  The converse is in [`results/model_seed_repeat.md`](results/model_seed_repeat.md):
  on one fixed draw, changing only the estimators' random seed moves no
  leave-one-unit-out number at all (std 0.0000 on every row; the estimators
  are deterministic at this size) and moves the leaky protocol a by 0.014 in
  trial F1. Estimator-seed variation is not uncertainty over future machines.
- All data is synthetic and produced by `data/make_synthetic_data.py`.
- Unit AUC pools predictions from different leave-one-unit-out models. Its
  ranking is descriptive: changes in fold-specific probability scales and
  training class balance can change it. It is not the AUC of one fitted model
  on an independent test set.
- `production_model.py` is a teaching API. Its model/threshold selection uses
  out-of-fold labels and remains optimistic even when a separating gap exists.
  It returns provisional labels with review flags; a caller must honour
  those flags before taking an automatic action. Review flags are not a
  validation of safety or performance.
- Median/MAD and mean/std produce different decisions on some draws; neither
  establishes a consistent advantage across the reported metrics: mean unit F1 0.772 (c) against 0.751 (c_meanstd), mean unit AUC
  0.923 against 0.940, unit AUC tied on 23 of 30 seeds. The within-trial robust amplitude feature
  is less variable -- the trial-to-trial coefficient of variation of
  `vib_y_mad` within healthy (unit, setpoint) cells is 0.054 against 0.145 for
  `vib_y_rms` and `vib_y_std` and 0.387 for `vib_y_ptp` (Figure 06) -- but this measures variation of input features, not uncertainty of the
  fitted cross-unit conditioning denominator.
- The `unit F1 lowhalf@0.5` column aggregates over a region declared from
  the generator's definition of the fault. It is a modelling assumption, not a
  label-free discovery; a real deployment would have to declare its region
  from domain knowledge before scoring, or the region becomes another
  threshold fitted on the labels.
- The generator was not tuned toward any of these numbers. Where a number
  disappoints, it is reported.

## More commands

Run these in the activated environment from the start of this README:

```bash
make verify          # refit the protocols -> .work/default_draw.json
make verify-reference # refit and compare every field with results/default_draw.json
make sweep           # full generator sweep -> .work/sweep/
make model-seeds     # fixed data, estimator-seed repeat -> .work/model-seeds/
make figures         # refresh committed figures from data/ and results/
make demo            # run the educational package demonstrations
make check           # inspect the public file tree
```

`make data SEED=7` draws a different simulated population. The reference check
expects the default seed; use `make verify` to inspect a deliberate alternate
draw without rewriting the published evidence. `make sweep SEEDS=5 JOBS=2`
runs a shorter sweep. Runtime depends on the machine and dependency versions;
OpenMP and BLAS threads are capped to prevent oversubscription.

## What the generator embeds

`data/make_synthetic_data.py` simulates a test bench: each **unit** (machine)
runs 6 ascending **sweeps** of 12 setpoints (2.0 to 8.6), and each dwell is one
**trial** of 384 triaxial samples. Default: 10 labelled units, 2 faulty; a
holdout of 5 disjoint units, 1 faulty; seed 2026. Four properties are written
into the code, and the documentation reports what they produce:

1. **The operating point dominates amplitude.** Log amplitude grows linearly
   with sweep position at a per-axis exponent of 1.45 / 1.60 / 1.80, i.e.
   healthy amplitude multiplies by roughly exp(1.45) to exp(1.80) across a
   sweep. On the default draw the healthy mean `vib_y_rms` at setpoint 8.6 is
   4.84 times its value at 2.0 (5.66 for `vib_z_rms`).
2. **Units have fingerprints.** Each unit draws a level offset, per-axis
   offsets and slopes, a speed gain and droop, a noise floor, harmonic weights
   and a burst gain, fixed for its lifetime. This is the leak source.
3. **Healthy behaviour is heavy-tailed.** Student-t noise (2.4 degrees of
   freedom) and, in a quarter of trials, an impulsive burst, both scaled with
   the local amplitude. Outlier-sensitive statistics (std, ptp) show greater trial-to-trial
   variation than the measured robust statistics (mad, iqr, p90) on the default
   draw (Figure 06).
4. **The fault is a one-sided deficit concentrated in the low-setpoint half.** A faulty
   unit's log amplitude is reduced by a term that is close to its full size at
   the bottom of the sweep and fades to nearly nothing at the top, plus a small
   constant. Analytically, for the y axis at the middle of the drawn range, the
   faulty/healthy ratio is about 0.32 at the bottom and 1.08 at the top; on the
   default draw the measured median-`vib_y_rms` ratios are 0.30 (U03) and 0.37
   (U06) at setpoint 2.0, and 1.09 and 1.31 at 8.6 (Figure 02). The
   sweep-averaged *log* level (geometric-mean amplitude) of a faulty unit is
   therefore *lower* than healthy, not equal to it; on the arithmetic mean of
   `vib_y_rms` over the sweep the gap is smaller (0.78 for U03, 0.96 for U06
   relative to the healthy sweep mean on the default draw).

Which units are faulty is drawn uniformly at random after the fingerprints,
without looking at them. The fault term was chosen for its statistical shape;
it is not derived from a physical mechanism. A squared-speed forcing law alone
would not determine the measured amplitude or the faulty/healthy ratio without
a model of structural response and the healthy baseline.

![healthy and faulty amplitude versus setpoint on the default draw](figures/02-amplitude-vs-setpoint.png)

*Figure 02. Faulty units are far below the healthy band at low setpoints and
at or above it at the top; the ratio panel is the fault's shape.*

## Repository layout

```
vibration-anomaly-lab/
├── README.md, README.ja.md         this file and its Japanese version
├── LICENSE                         MIT
├── Makefile                        setup, setup-dev, data, verify, sweep, figures, test, demo, check, clean, all
├── CHANGELOG.md
├── pyproject.toml                  package metadata (pip install -e .)
├── requirements.txt                numpy, pandas, scikit-learn
├── requirements-dev.txt            + pytest, matplotlib, ruff
├── verify_claims.py                the nine protocols; standalone, imports nothing from the package
├── data/
│   ├── README.md                   schema, provenance, the shape of the fault
│   ├── make_synthetic_data.py      the generator
│   └── synthetic/                  train.csv, holdout.csv, holdout_labels.csv (generated, git-ignored)
├── vibration_anomaly_lab/          the same ideas as an importable package
│   ├── features.py                 trial-level featurisation (parity-tested against verify_claims.py)
│   ├── conditioning.py             ConditionalRobustZ with a fit / transform split
│   ├── validation.py               leave-one-unit-out, unit-level scoring
│   ├── production_model.py         a panel model that flags every unit for review when no member separates
│   └── metric_design.py            how many bits a returned score leaks about hidden group labels
├── scripts/
│   ├── sweep_seeds.py              verify_claims on many generator seeds -> results/
│   ├── model_seed_repeat.py        verify_claims on ONE draw, many model seeds -> results/
│   ├── make_figures.py             figures/*.png from data/ and results/
│   └── check_public.sh             anonymisation and tracked-file gate
├── results/
│   ├── default_draw.json           make verify on the default seed
│   ├── seed_sweep.json             30 per-seed documents
│   ├── seed_sweep.md               the 30-seed table quoted above
│   ├── seed_sweep_per_seed.md      one row per seed
│   └── model_seed_repeat.md        one draw, ten model seeds: reproducibility noise only
├── figures/                        01 sweep structure, 02 amplitude vs setpoint, 03 fingerprints,
│                                   04 protocol unit scores, 05 seed sweep, 06 robust scale, 07 leakage
├── docs/
│   ├── 01-why-random-cv-lies.md            anatomy of the leak on this data
│   ├── 02-conditioning-not-models.md       conditioning comparisons and confounds
│   ├── 03-small-sample-honesty.md          what n = 2 can and cannot say
│   └── 04-metric-design-and-disclosure.md  the returned score as an information channel
├── tests/                          generator, conditioning, validation, production model, metric design,
│                                   verify CLI, package/script parity, public hygiene
└── .github/workflows/ci.yml        tests, demos, verify, hygiene, drift check against committed results
```

`verify_claims.py` imports nothing from `vibration_anomaly_lab/`, so a broken
module cannot turn the verification into a tautology; `tests/test_parity.py`
asserts that the package's featurisation and conditioning agree with it. The
holdout file exists so that a pipeline can be scored once on machines that
never entered any fold; with five units and one fault it is a format check,
not an evaluation.

## What this is NOT

- **Not real data.** Every sample comes from `data/make_synthetic_data.py`.
  Nothing here is a measurement of any machine, product or fleet.
- **Not a field-performance estimate.** These are descriptive performance
  measurements under a particular simulator, with two faulty units per draw.
  They do not estimate detection performance on real equipment.
- **Not a pure conditioning ablation.** The compared feature spaces also
  differ in their speed column. The observed c-vs-b AUC record is 4/18/8;
  it does not isolate a causal effect of the transformation.
- **Not a calibrated physical model.** The fault shape is stipulated rather
  than derived from a validated failure mechanism.
- **Not threshold-honest in the `best (t)` column.** That column maximises a
  finite threshold grid on the scored labels. The other columns use fixed 0.5.
- **Not validation of spectral defect diagnosis.** The generator includes
  harmonic tones, noise and bursts, but no validated bearing or gear defect
  physics. The published feature extractor uses amplitude statistics.
  These measurements do not settle the usefulness of spectral features on
  real data; that depends on the mechanism, bandwidth and acquisition chain.
- **Not a claim that group-wise CV is sufficient.** Leave-one-unit-out fixes
  the leak modelled here. Temporal ordering, shared calibration events, batch
  effects and label definitions derived from future information are not
  modelled.

## License

MIT. See [`LICENSE`](LICENSE).
