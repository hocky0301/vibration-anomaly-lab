#!/usr/bin/env python3
"""
make_synthetic_data.py -- synthetic vibration data for group-leakage research.

This repository studies a failure mode that is common in industrial condition
monitoring: **the unit of independence is the machine, not the row.**  All data
here is 100% synthetic and produced by this file, so every claim in the README
can be re-derived from scratch on any machine.

What the generator reproduces
-----------------------------
A test bench runs a *setpoint sweep*: an operating parameter is stepped up in
discrete levels, and at each level the bench dwells for a while and records a
short stationary block of triaxial vibration.  One such block is a **trial**.
Several complete sweeps are repeated per machine.  A machine is called a
**unit**, and health labels are a property of the unit, so every trial from one
unit shares one label.

Four structural facts are baked into the simulator.  They are stated here as
what the code below does, not as what one might wish it did:

1. **The operating point dominates amplitude.**  Log amplitude grows linearly
   with the normalised sweep position ``u`` at a per-axis exponent
   ``SETPOINT_EXPONENT`` (1.45 / 1.60 / 1.80 for x / y / z), so a healthy unit
   has roughly ``exp(1.45)`` to ``exp(1.80)`` -- about 4x to 6x -- the
   amplitude at the top of the sweep that it has at the bottom.  An absolute threshold on RMS therefore
   measures the setpoint, not the health state.  Raw amplitude features
   without an operating-point column lose most of their whole-sweep unit-level
   signal (``results/seed_sweep.md``, protocol b_noctx: mean unit AUC 0.688,
   mean unit F1 0.152 over 30 draws) but keep a low-setpoint signal, because a
   faulty unit at the bottom of the sweep is quieter than anything a healthy
   unit produces there (b_noctx, unit F1 over the low half: 0.809).

2. **Units have individual fingerprints.**  Each unit draws its own overall
   level offset, per-axis offsets, per-axis sweep slopes, a small healthy tilt,
   a speed gain and droop, a noise floor, a noise-to-tone ratio, harmonic
   weights and phases, and a burst gain (``draw_unit``).  These are fixed for
   the life of the unit and are the same for every one of its trials.  A
   flexible model can therefore identify ``unit_id`` from the raw trial
   statistics plus ``setpoint`` and ``speed_rpm``, and since the label is constant within a unit, recognising the unit
   is equivalent to knowing the label.  That is what a random split over trials
   exploits, and it is what makes such a split a leak rather than a test.

3. **Heavy-tailed healthy behaviour.**  Every trial carries Student-t noise
   (``T_DF`` = 2.4 degrees of freedom) and, with probability ``BURST_PROB``, an
   impulsive burst, both scaled with the local amplitude.  Statistics driven by
   extreme samples -- the standard deviation and the peak-to-peak in particular
   -- are therefore noisy from trial to trial; rank-based statistics such as the
   median absolute deviation are far steadier.

4. **The fault is a one-sided deficit concentrated in the low-setpoint half.**  For a faulty
   unit, ``log_amplitude`` adds

       ANOM_LEVEL_SHIFT - anom_tilt * decay(u) * ANOM_AXIS_WEIGHT

   where ``decay(u) = 0.5 * (1 - tanh((u - ANOM_KNEE) / ANOM_WIDTH))`` is close
   to 1 at the bottom of the sweep and close to 0 at the top, ``anom_tilt`` is
   drawn uniformly from ``ANOM_TILT_RANGE`` = (1.15, 1.35), and
   ``ANOM_LEVEL_SHIFT`` = 0.10.  The analytic consequences, for the y axis
   (axis weight 1.0) and the midpoint of the tilt range:

       bottom of sweep (u = 0):  decay = 0.993, ratio = exp(0.10 - 1.25 * 0.993) ~ 0.32
       top of sweep    (u = 1):  decay = 0.016, ratio = exp(0.10 - 1.25 * 0.016) ~ 1.08

   So a faulty unit is markedly *quieter* than a healthy unit at low setpoints,
   returns to slight excess at the top, and its sweep-averaged *log* level
   (geometric-mean amplitude) is *lower* than healthy, not equal to it: the
   log-mean of the fault term over the 12-point grid is -0.58 (ratio 0.56).
   The deficit is concentrated in, not confined to, the low-setpoint half:
   ``ANOM_KNEE`` = 0.55 sits just above the seventh of the twelve setpoints
   (u = 6/11 = 0.545, where ``decay`` = 0.51), and with ``ANOM_WIDTH`` = 0.22
   the analytic y-axis ratio is still 0.58 there and 0.75 at the next setpoint
   (u = 7/11).  Two consequences for detection: the signal is an interaction
   between amplitude and setpoint (a sweep *shape*), and any per-unit aggregate
   taken over the whole sweep dilutes it.

   The sign is the opposite of what a mass-imbalance story would predict.  A
   rotating imbalance produces a forcing that grows with the square of the
   shaft speed, so its gap would widen towards the top of the sweep; here the
   gap is largest at the bottom and closes towards the top.  This term was
   chosen for its statistical shape only.  No physical mechanism is claimed for
   it, and none should be inferred.

The two evaluation protocols this file is built to contrast:

    (a) random K-fold over trials         -> the leak: units are shared across
                                             folds, the fingerprint is memorised
    (b) leave-one-unit-out                -> the honest number: every score is
                                             on a unit the model never saw

``verify_claims.py`` at the repository root measures these and seven further
protocols (feature spaces and model families under leave-one-unit-out) on the
data this file writes; see the README for the table.

Usage
-----
    python3 make_synthetic_data.py                    # write the default draw
    python3 make_synthetic_data.py --seed 7           # a different draw
    python3 make_synthetic_data.py --units 20 --anomalous 4

Outputs (into ``synthetic/``, git-ignored):
    train.csv           labelled, N_UNITS units
    holdout.csv         unlabelled, disjoint units
    holdout_labels.csv  unit_id,label for the holdout

Columns:
    unit_id     machine identity -- the correct CV grouping key
    trial_id    one stationary block at one setpoint
    setpoint    commanded operating point of the sweep
    speed_rpm   measured shaft speed (responds to setpoint, unit-specific gain)
    vib_x/y/z   triaxial vibration sample
    sample_id   unique row id
    label       1 = faulty unit, 0 = healthy unit (train.csv only)

Requires numpy + pandas only.  Faulty units are chosen uniformly at random
among the units of each file; nothing about a unit's fingerprint influences
whether it receives the fault.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Simulation constants.  These are the mechanism; they are not tuned to any
# target number, and the documentation reports whatever they produce.
# --------------------------------------------------------------------------- #

AXES = ("x", "y", "z")

N_UNITS = 10             # labelled units in train.csv
N_ANOMALOUS = 2          # faulty units among them (label is per-unit)
N_HOLDOUT_UNITS = 5      # extra, disjoint units for a final one-shot check
N_HOLDOUT_ANOMALOUS = 1

N_SETPOINTS = 12         # levels in one sweep
SETPOINT_LO = 2.0
SETPOINT_STEP = 0.6      # -> 2.0 .. 8.6
N_SWEEPS = 6             # complete ascending sweeps per unit (replicates)
ROWS_PER_TRIAL = 384     # samples in one stationary block
FS = 1000.0              # sampling rate [Hz]

RPM_PER_SETPOINT = 320.0 # nominal speed response
RPM_JITTER = 1.5         # per-sample speed measurement noise [rpm]
RPM_TRIAL_SD = 0.004     # trial-to-trial speed variation (load, warm-up)
UNIT_DROOP_MEAN = 0.020  # speed droop under load -> rpm/setpoint is not constant
UNIT_DROOP_SD = 0.010

# Baseline log-amplitude per axis (arbitrary "g"-like units).
BASE_LOG_AMP = np.log(np.array([0.020, 0.026, 0.033]))
# Setpoint exponent per axis: amplitude multiplies by exp(P) across the sweep.
SETPOINT_EXPONENT = np.array([1.45, 1.60, 1.80])

# --- unit-to-unit variability (the fingerprint that identifies a unit)
UNIT_OFFSET_SD = 0.10        # overall level, log scale
UNIT_AXIS_OFFSET_SD = 0.06   # per-axis level (sensor mounting)
UNIT_SLOPE_SD = 0.06         # per-axis sweep slope
UNIT_TILT_SD = 0.04          # healthy units also have a small sweep tilt
UNIT_RPM_GAIN_SD = 0.045
UNIT_NOISE_FLOOR_LOG_SD = 0.08

# --- trial-to-trial variability
TRIAL_JITTER_SD = 0.022      # shared across axes (load, temperature drift)
TRIAL_AXIS_JITTER_SD = 0.016

# --- heavy tails: why std and peak-to-peak are noisy here
T_DF = 2.4                   # Student-t degrees of freedom for the noise floor
NOISE_REL = 0.55             # noise weight, in MAD units relative to the tone
UNIT_NOISE_REL_LOG_SD = 0.08 # per-unit noise-to-tone ratio (sweep-flat fingerprint)
BURST_PROB = 0.25            # probability that a trial contains an impulsive burst
BURST_GAIN_RANGE = (3.0, 9.0)
BURST_LEN_RANGE = (4, 24)
BURST_FREQ_RANGE = (120.0, 320.0)

# --- the fault: a one-sided log-amplitude deficit at low setpoints that fades
#     out towards the top of the sweep (see log_amplitude and the docstring)
ANOM_LEVEL_SHIFT = 0.10          # residual level at the top of the sweep (ratio ~1.1)
ANOM_TILT_RANGE = (1.15, 1.35)   # log-amplitude deficit at the bottom (ratio ~0.3)
ANOM_KNEE = 0.55                 # sweep position where the deficit has half-faded
ANOM_WIDTH = 0.22                # transition width of the fade
ANOM_AXIS_WEIGHT = np.array([0.85, 1.00, 1.15])

# Harmonic structure of the rotating tone.
HARMONICS = np.array([1.0, 2.0, 3.3])
HARMONIC_WEIGHTS = np.array([1.00, 0.42, 0.22])


def _t_mad(df: float, n: int = 400_000) -> float:
    """MAD of a Student-t, computed once with a fixed seed (avoids a scipy dep)."""
    s = np.random.default_rng(20240101).standard_t(df, size=n)
    return float(np.median(np.abs(s - np.median(s))))


T_MAD = _t_mad(T_DF)


# --------------------------------------------------------------------------- #
# Unit parameters
# --------------------------------------------------------------------------- #

@dataclass
class UnitParams:
    """Everything that makes one machine physically itself."""
    unit_id: str
    label: int
    offset: float                       # overall log-amplitude offset
    axis_offset: np.ndarray             # per-axis log offset
    slope: np.ndarray                   # per-axis setpoint exponent
    tilt: float                         # healthy sweep tilt (log, at the sweep ends)
    anom_tilt: float                    # 0 for healthy units
    rpm_gain: float
    droop: float                        # load droop, makes the speed curve bend
    noise_floor: np.ndarray             # per-axis additive sensor floor
    phase: np.ndarray                   # per-axis tone phases (3 harmonics each)
    harm_weight: np.ndarray             # per-axis harmonic weights
    noise_rel: np.ndarray = field(          # per-axis noise-to-tone ratio (waveform shape)
        default_factory=lambda: np.full(3, NOISE_REL))
    burst_axis_gain: np.ndarray = field(default_factory=lambda: np.ones(3))


def draw_unit(rng: np.random.Generator, unit_id: str, label: int) -> UnitParams:
    """Draw one unit.  Healthy and faulty units share the *same* fingerprint
    priors; the fault term is added afterwards by ``build_units``.
    """
    n_ax = len(AXES)
    return UnitParams(
        unit_id=unit_id,
        label=label,
        offset=float(rng.normal(0.0, UNIT_OFFSET_SD)),
        axis_offset=rng.normal(0.0, UNIT_AXIS_OFFSET_SD, size=n_ax),
        slope=SETPOINT_EXPONENT + rng.normal(0.0, UNIT_SLOPE_SD, size=n_ax),
        tilt=float(rng.normal(0.0, UNIT_TILT_SD)),
        anom_tilt=0.0,
        rpm_gain=float(rng.normal(1.0, UNIT_RPM_GAIN_SD)),
        droop=float(max(0.0, rng.normal(UNIT_DROOP_MEAN, UNIT_DROOP_SD))),
        noise_floor=np.exp(np.log(0.0006) + rng.normal(0.0, UNIT_NOISE_FLOOR_LOG_SD, size=n_ax)),
        phase=rng.uniform(0.0, 2.0 * np.pi, size=(n_ax, len(HARMONICS))),
        harm_weight=HARMONIC_WEIGHTS * np.exp(rng.normal(0.0, 0.50, size=(n_ax, len(HARMONICS)))),
        noise_rel=NOISE_REL * np.exp(rng.normal(0.0, UNIT_NOISE_REL_LOG_SD, size=n_ax)),
        burst_axis_gain=np.exp(rng.normal(0.0, 0.35, size=n_ax)),
    )


def log_amplitude(p: UnitParams, u: float, rng: np.random.Generator) -> np.ndarray:
    """Log amplitude of one trial, per axis.

    u in [0, 1] is the normalised position in the sweep.  Additive terms:

      * ``slope * u``            the operating-point response (healthy units too)
      * ``tilt * (2u - 1)``      a small healthy unit-specific tilt
      * the fault                ``ANOM_LEVEL_SHIFT - anom_tilt * decay(u) * ANOM_AXIS_WEIGHT``
                                 for faulty units, zero otherwise.  ``decay`` is
                                 near 1 at the bottom of the sweep and near 0 at
                                 the top, so this is a one-sided deficit: the
                                 unit is markedly quieter at low setpoints and
                                 slightly (ratio ~exp(ANOM_LEVEL_SHIFT)) above
                                 parity at the top.  Sweep-averaged, it is below
                                 healthy.
      * trial jitter             one shared and one per-axis Gaussian term

    The fault is deliberately NOT proportional to speed^2.  A mass-imbalance
    story would predict the gap widening with speed; here it does the opposite,
    which is why this repository claims no physical mechanism for it.
    """
    shape = p.tilt * (2.0 * u - 1.0)
    if p.anom_tilt > 0.0:
        decay = 0.5 * (1.0 - np.tanh((u - ANOM_KNEE) / ANOM_WIDTH))
        fault = ANOM_LEVEL_SHIFT - p.anom_tilt * decay * ANOM_AXIS_WEIGHT
    else:
        fault = np.zeros(len(AXES))
    jitter = rng.normal(0.0, TRIAL_JITTER_SD) + rng.normal(0.0, TRIAL_AXIS_JITTER_SD, size=len(AXES))
    return BASE_LOG_AMP + p.offset + p.axis_offset + p.slope * u + shape + fault + jitter


def synth_trial(p: UnitParams, setpoint: float, u: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate one stationary block: (rpm[n], vib[n, 3])."""
    n = ROWS_PER_TRIAL
    t = np.arange(n) / FS

    # Speed responds to the setpoint with a unit-specific gain and a load droop,
    # so rpm/setpoint is a good but imperfect fingerprint of the machine.
    rpm_mean = (p.rpm_gain * RPM_PER_SETPOINT * setpoint
                * (1.0 - p.droop * setpoint ** 0.5)
                * (1.0 + rng.normal(0.0, RPM_TRIAL_SD)))
    rpm = rpm_mean + rng.normal(0.0, RPM_JITTER, size=n)
    f_rot = rpm_mean / 60.0

    amp = np.exp(log_amplitude(p, u, rng))          # (3,)

    vib = np.empty((n, len(AXES)))
    for a in range(len(AXES)):
        tone = np.zeros(n)
        for k, h in enumerate(HARMONICS):
            tone += p.harm_weight[a, k] * np.sin(2.0 * np.pi * h * f_rot * t + p.phase[a, k])
        # Heavy-tailed component, normalised to unit MAD so NOISE_REL is meaningful.
        heavy = rng.standard_t(T_DF, size=n) / T_MAD
        vib[:, a] = amp[a] * (tone + p.noise_rel[a] * heavy) + p.noise_floor[a] * rng.normal(size=n)

    # Impulsive burst: a mechanical knock, correlated across axes, scaled with the
    # local amplitude so it carries no information about the overall level.
    if rng.random() < BURST_PROB:
        length = int(rng.integers(*BURST_LEN_RANGE))
        start = int(rng.integers(0, max(1, n - length)))
        gain = float(rng.uniform(*BURST_GAIN_RANGE))
        f_b = float(rng.uniform(*BURST_FREQ_RANGE))
        idx = np.arange(start, start + length)
        window = np.hanning(length + 2)[1:-1]
        ring = np.sin(2.0 * np.pi * f_b * (idx - start) / FS)
        for a in range(len(AXES)):
            vib[idx, a] += gain * p.burst_axis_gain[a] * amp[a] * window * ring

    return rpm, vib


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #

def setpoint_grid() -> np.ndarray:
    return SETPOINT_LO + SETPOINT_STEP * np.arange(N_SETPOINTS)


def build_units(rng: np.random.Generator, n_units: int, n_anomalous: int,
                prefix: str) -> list[UnitParams]:
    """Draw every unit from one healthy population, then make some of them faulty.

    The faulty units are chosen uniformly at random among the ``n_units``
    drawn, *after* their fingerprints exist and without looking at them, so a
    unit's identity carries no information about its label beyond the fault
    term itself.  Each chosen unit receives its own ``anom_tilt`` drawn from
    ``ANOM_TILT_RANGE``.
    """
    units = [draw_unit(rng, f"{prefix}{i + 1:02d}", 0) for i in range(n_units)]
    for i in rng.choice(n_units, size=n_anomalous, replace=False):
        units[i].label = 1
        units[i].anom_tilt = float(rng.uniform(*ANOM_TILT_RANGE))
    return units


def generate(units: list[UnitParams], rng: np.random.Generator,
             with_label: bool, sample_id_start: int = 0) -> tuple[pd.DataFrame, int]:
    """Run every unit through ``N_SWEEPS`` ascending setpoint sweeps."""
    grid = setpoint_grid()
    u_grid = np.arange(N_SETPOINTS) / (N_SETPOINTS - 1)

    chunks = []
    sid = sample_id_start
    for p in units:
        trial_no = 0
        for _sweep in range(N_SWEEPS):
            for si, setpoint in enumerate(grid):
                rpm, vib = synth_trial(p, float(setpoint), float(u_grid[si]), rng)
                n = len(rpm)
                block = {
                    "unit_id": np.repeat(p.unit_id, n),
                    "trial_id": np.repeat(f"{p.unit_id}_T{trial_no:03d}", n),
                    "setpoint": np.repeat(round(float(setpoint), 2), n),
                    "speed_rpm": np.round(rpm, 3),
                    "vib_x": np.round(vib[:, 0], 6),
                    "vib_y": np.round(vib[:, 1], 6),
                    "vib_z": np.round(vib[:, 2], 6),
                    "sample_id": np.arange(sid, sid + n),
                }
                if with_label:
                    block["label"] = np.repeat(p.label, n)
                chunks.append(pd.DataFrame(block))
                sid += n
                trial_no += 1
    return pd.concat(chunks, ignore_index=True), sid


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(here, "synthetic"))
    ap.add_argument("--units", type=int, default=N_UNITS)
    ap.add_argument("--anomalous", type=int, default=N_ANOMALOUS)
    ap.add_argument("--holdout-units", type=int, default=N_HOLDOUT_UNITS)
    ap.add_argument("--holdout-anomalous", type=int, default=N_HOLDOUT_ANOMALOUS)
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args(argv)

    if args.units < 2:
        ap.error(f"--units must be at least 2 (got {args.units})")
    if not 0 <= args.anomalous <= args.units:
        ap.error(f"--anomalous must satisfy 0 <= anomalous <= units "
                 f"(got anomalous={args.anomalous}, units={args.units})")
    if not args.no_holdout:
        if args.holdout_units < 1:
            ap.error(f"--holdout-units must be at least 1 unless --no-holdout is given "
                     f"(got {args.holdout_units})")
        if not 0 <= args.holdout_anomalous <= args.holdout_units:
            ap.error(f"--holdout-anomalous must satisfy 0 <= holdout_anomalous <= holdout_units "
                     f"(got holdout_anomalous={args.holdout_anomalous}, "
                     f"holdout_units={args.holdout_units})")

    # Generate everything first; write only once nothing can fail any more, so
    # an error never leaves a half-written data directory behind.
    rng = np.random.default_rng(args.seed)

    train_units = build_units(rng, args.units, args.anomalous, prefix="U")
    train_df, next_sid = generate(train_units, rng, with_label=True)

    hold_df = labels = None
    if not args.no_holdout:
        hold_units = build_units(rng, args.holdout_units, args.holdout_anomalous, prefix="H")
        hold_df, _ = generate(hold_units, rng, with_label=True, sample_id_start=next_sid)
        labels = hold_df.groupby("unit_id", sort=True)["label"].first().reset_index()

    os.makedirs(args.out, exist_ok=True)

    train_path = os.path.join(args.out, "train.csv")
    train_df.to_csv(train_path, index=False)
    n_trials = train_df["trial_id"].nunique()
    pos_units = [p.unit_id for p in train_units if p.label == 1]
    print(f"wrote {train_path}")
    print(f"  {len(train_df):,} rows | {args.units} units | {n_trials} trials "
          f"| {ROWS_PER_TRIAL} samples/trial")
    print(f"  faulty units: {', '.join(pos_units) if pos_units else '(none)'} "
          f"({len(pos_units)}/{args.units})")

    if hold_df is not None:
        hold_path = os.path.join(args.out, "holdout.csv")
        lab_path = os.path.join(args.out, "holdout_labels.csv")
        hold_df.drop(columns=["label"]).to_csv(hold_path, index=False)
        labels.to_csv(lab_path, index=False)
        print(f"wrote {hold_path}  ({len(hold_df):,} rows, {args.holdout_units} unseen units)")
        print(f"wrote {lab_path}   (score once, at the very end)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
