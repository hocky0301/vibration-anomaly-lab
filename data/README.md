# Data

## This repository contains no real data

There is no measured, recorded, licensed, or third-party data anywhere in this
repository, and there never will be. The only dataset this project uses is
**synthetic**, and it is produced on your own machine by
[`make_synthetic_data.py`](make_synthetic_data.py).

Nothing under `data/synthetic/` is committed (see `.gitignore`). The generator
is deterministic given its `--seed`, so the CSVs are build products, not
sources: re-running the generator with the same seed reproduces byte-identical
files.

```sh
make data                                  # writes data/synthetic/ with the default seed
python3 data/make_synthetic_data.py --seed 7   # a different draw
```

Generation is the generator's only job. Everything that *measures* the data
lives in `verify_claims.py` at the repository root.

## Files

Written into `data/synthetic/`:

| file | contents |
| --- | --- |
| `train.csv` | labelled units, one row per sample |
| `holdout.csv` | the same schema **without** `label`, from units disjoint from `train.csv` |
| `holdout_labels.csv` | `unit_id,label`, one row per holdout unit — the answer key, meant to be consulted once, at the end |

With the default settings and seed, `train.csv` holds 10 units (2 of them
faulty) and `holdout.csv` holds 5 further units (1 of them faulty). Unit ids
never overlap between the two files, and `sample_id` is unique across both.
The 10-unit / 2-faulty default is the regime this repository studies; it can be
changed with `--units` / `--anomalous` (and `--holdout-units` /
`--holdout-anomalous` for the holdout file).

Which units are faulty is decided **uniformly at random** among the units of
each file, after their individual fingerprints have been drawn and without
looking at them: nothing about a unit's level, speed response, or noise makes
it more or less likely to receive the fault. The default seed happens to place
the training faults on `U03` and `U06`; a different seed places them
elsewhere.

## Columns

Every row is one sample of a triaxial vibration recording.

| column | type | meaning | unit |
| --- | --- | --- | --- |
| `unit_id` | string | Identity of the machine the recording came from. **This is the grouping key for cross-validation.** | — |
| `trial_id` | string | One stationary block: the machine held at one setpoint while a short window was recorded. Prefixed with `unit_id`, so a trial belongs to exactly one unit. | — |
| `setpoint` | float | Commanded operating point of the sweep. A dial position, not a calibrated physical quantity. | dimensionless |
| `speed_rpm` | float | Measured shaft speed for this sample. Responds to `setpoint` with a per-unit gain and a load droop, so it is not a deterministic function of `setpoint`. | revolutions per minute |
| `vib_x`, `vib_y`, `vib_z` | float | Triaxial vibration sample, one value per axis. Amplitude-like, in arbitrary "g"-ish units; the absolute scale carries no physical meaning, only ratios and distributions do. | arbitrary (acceleration-like) |
| `sample_id` | int | Unique row identifier, assigned in generation order. Bookkeeping only — it must never be used as a feature. | — |
| `label` | int | `1` = faulty unit, `0` = healthy unit. Present in `train.csv` only. | — |

Structure of the default draw, per unit: 12 setpoints from 2.0 to 8.6 in steps
of 0.6, swept 6 times, giving 72 trials per unit; each trial is 384 consecutive
samples, sampled at 1 kHz by the generator (about 0.38 s of signal). That is
27,648 rows per unit — 276,480 rows in `train.csv` and 138,240 in
`holdout.csv`.

## The unit is the unit of independence

This is the single most important property of the schema, and the reason this
project exists:

> **`label` is an attribute of the *unit*, not of the row and not of the trial.**

Every one of a unit's 27,648 rows carries the same label, because the label
describes the machine. Rows within a unit are therefore not independent
observations of the label — they are one observation, repeated.

Three consequences follow, and all three are load-bearing:

1. **Group your cross-validation splits by `unit_id`.** A random split over
   rows or trials puts samples from the same machine on both sides of the
   split. The model can then recognise the machine — units have distinct
   fingerprints in offset, per-axis gain, speed response, and noise floor — and
   recognising the machine is equivalent to knowing the label. The resulting
   score measures memorisation, not generalisation. `verify_claims.py`
   quantifies exactly this gap.
2. **Report the decision at the unit level.** The question asked is "is this
   machine faulty", so trial-level metrics answer a question nobody posed and
   flatter the model by treating 72 correlated trials as 72 independent
   verdicts.
3. **The effective sample size is the number of units, not the number of
   rows.** The default `train.csv` has 276,480 rows and *two* positive
   examples. Two. Any result obtained here — including the ones printed by
   `verify_claims.py` — is an existence proof that a method can work on this
   generator, never a performance guarantee. An F1 from two positive machines is highly uncertain and is not itself a
   binomial proportion. [docs/03](../docs/03-small-sample-honesty.md) illustrates
   binomial intervals for detection rates under independent fixed-detector
   assumptions; they are not formal intervals for this selected OOF experiment.

`setpoint` deserves a related warning. Vibration amplitude grows strongly with
the setpoint for healthy units too: pooled over the healthy units of the
default draw, the RMS computed over all pooled `vib_z` samples is about 5.3x larger at setpoint 8.6 than at
setpoint 2.0. This pools samples before computing RMS; the main README
averages per-trial RMS values, a different aggregation. An absolute amplitude threshold therefore measures the operating
point, not the health state. Comparisons only mean something *at the same
setpoint*.

## The shape of the fault

The fault is a **one-sided deficit**, not a symmetric tilt and not an overall
gain. For a faulty unit the generator subtracts from the log amplitude a term
that is large at the bottom of the sweep and fades to nearly nothing at the
top, then adds a small constant. In ratio terms (faulty over healthy, same
setpoint), the constants in `make_synthetic_data.py` put a faulty unit at
roughly 0.3x at the lowest setpoint and roughly 1.1x at the highest; averaged
in log over the whole sweep (geometric mean), a faulty unit is therefore
*quieter* than a healthy one, not indistinguishable from it. The exact per-setpoint ratio depends on the
draw, because unit fingerprints add their own spread on top.

Two consequences for anyone building a detector:

- The signal is an **interaction** between amplitude and setpoint — a sweep
  *shape* — so it is most legible to a model that can see both, or to features
  that condition on the operating point.
- The signal is **concentrated in the low-setpoint half** of the sweep (the
  fade has its knee at sweep position 0.55 with a transition width of 0.22, so
  the deficit is not confined to that half). A per-unit score
  aggregated over all setpoints dilutes it; aggregating over the low-setpoint
  region where the fault is declared to live does not.

The generator claims no physical mechanism for the fault; the shape was
chosen as a statistical test case. Forcing laws alone do not determine the
measured response or the faulty/healthy ratio without a structural-response
model and a healthy baseline.

## What this data is not

It is a simulation written to expose one specific failure mode of evaluation
design. It is not a physical model of any machine, it is not calibrated against
any measurement, and its fault term was chosen for its statistical shape rather
than derived from a mechanism. Methods that work here have been shown to work
*here*; that is the whole of the claim.
