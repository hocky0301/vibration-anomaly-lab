"""
features.py -- trial-level featurisation.

A *trial* is one stationary block of triaxial vibration recorded at one commanded
setpoint.  A *unit* is a machine; a unit runs several complete setpoint sweeps,
so one unit contributes many trials.  Health labels are a property of the unit,
never of the row and never of the trial.

This module is the importable form of the featurisation used by
``verify_claims.py`` at the repository root.  The definitions are identical --
if you change one, change both, or the verification script stops verifying
anything.  ``verify_claims.py`` is deliberately kept standalone (it re-derives
every README number without importing this package) so that a reader can audit
the claims without trusting the library.

Seven statistics per axis, chosen so the robust/non-robust split is visible:

    non-robust : rms, std, ptp, absmean     (dominated by a few samples)
    robust     : mad, iqr, p90              (barely move under an impulsive burst)

The synthetic generator injects Student-t noise and occasional impulsive bursts
into *healthy* trials, so the non-robust statistics carry a large nuisance
variance that has nothing to do with health.  Keeping both families in the
feature set makes that contrast measurable rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AXES: tuple[str, ...] = ("vib_x", "vib_y", "vib_z")
STATS: tuple[str, ...] = ("rms", "std", "mad", "iqr", "p90", "absmean", "ptp")

#: Columns that describe the operating point rather than the vibration itself.
CONTEXT_COLS: tuple[str, ...] = ("setpoint", "speed_rpm")

#: Columns that identify the row and must never be fed to a model.
ID_COLS: tuple[str, ...] = ("trial_id", "unit_id")


def _axis_stats(v: np.ndarray) -> dict[str, float]:
    """The seven per-axis statistics of one trial."""
    m = np.median(v)
    return {
        "rms": float(np.sqrt((v ** 2).mean())),
        "std": float(v.std()),
        "mad": float(np.median(np.abs(v - m))),
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
        "p90": float(np.percentile(np.abs(v), 90)),
        "absmean": float(np.abs(v).mean()),
        "ptp": float(v.max() - v.min()),
    }


def trial_features(df: pd.DataFrame, axes: tuple[str, ...] = AXES) -> pd.DataFrame:
    """Collapse the raw sample table into one row per trial.

    Parameters
    ----------
    df
        Raw samples with columns ``unit_id, trial_id, setpoint, speed_rpm,
        vib_x, vib_y, vib_z`` and, for labelled data, ``label``.

    Returns
    -------
    DataFrame
        One row per trial: ``trial_id, unit_id, [label,] setpoint, speed_rpm``
        plus ``{axis}_{stat}`` for every axis/statistic pair.  ``label`` is
        present only when the input carries it, so the same call works on the
        unlabelled holdout.

    Trial order follows first appearance in ``df``; nothing downstream depends
    on it, but keeping it stable makes diffs readable.
    """
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("axes must contain unique column names")
    required = ["trial_id", "unit_id", *CONTEXT_COLS, *axes]
    missing = set(required) - set(df.columns)
    if missing or not df.columns.is_unique:
        raise ValueError(f"missing or duplicate columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("sample table is empty")
    if df[["trial_id", "unit_id"]].isna().any().any():
        raise ValueError("trial_id and unit_id must not be missing")
    numeric = df[[*CONTEXT_COLS, *axes]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("sample features must be finite")
    groups = df.groupby("trial_id", sort=False)
    if (groups[["unit_id", "setpoint"]].nunique() != 1).any().any():
        raise ValueError("each trial must belong to one unit and one setpoint")
    has_label = "label" in df.columns
    if has_label:
        unit_labels(df)
    rows: list[dict[str, float | str | int]] = []
    for trial_id, g in df.groupby("trial_id", sort=False):
        rec: dict[str, float | str | int] = {
            "trial_id": trial_id,
            "unit_id": g["unit_id"].iloc[0],
            "setpoint": float(g["setpoint"].iloc[0]),
            "speed_rpm": float(g["speed_rpm"].mean()),
        }
        if has_label:
            rec["label"] = int(g["label"].iloc[0])
        for a in axes:
            for k, v in _axis_stats(g[a].to_numpy()).items():
                rec[f"{a}_{k}"] = v
        rows.append(rec)

    out = pd.DataFrame(rows)
    front = ["trial_id", "unit_id"] + (["label"] if has_label else []) + list(CONTEXT_COLS)
    return out[front + [c for c in out.columns if c not in front]]


def vibration_features(F: pd.DataFrame, axes: tuple[str, ...] = AXES) -> list[str]:
    """The vibration statistics only -- the columns that get conditioned."""
    return [c for c in F.columns if any(c.startswith(a + "_") for a in axes)]


def raw_features(F: pd.DataFrame, axes: tuple[str, ...] = AXES) -> list[str]:
    """The naive feature set: absolute statistics plus the operating point.

    Absolute amplitude can encode unit identity (mounting, alignment, sensor
    gain).  Since labels are constant within units, random trial folds can reward
    recognition of training units.  Transfer depends on the model and draw;
    raw features with operating context perform well on many generator seeds.
    """
    return vibration_features(F, axes) + list(CONTEXT_COLS)


def unit_labels(F: pd.DataFrame) -> pd.Series:
    """One label per unit, indexed by ``unit_id``.

    Raises if a unit carries more than one distinct label -- that would mean the
    data contradicts the premise that health is a property of the machine.
    """
    if F.empty or "unit_id" not in F or "label" not in F:
        raise ValueError("nonempty unit_id and label columns are required")
    if F["unit_id"].isna().any() or not F["label"].isin([0, 1]).all():
        raise ValueError("unit ids must be present and labels must be binary 0/1")
    g = F.groupby("unit_id")["label"]
    if (g.nunique() > 1).any():
        bad = sorted(g.nunique()[g.nunique() > 1].index)
        raise ValueError(f"units with inconsistent labels: {bad}")
    return g.first()
