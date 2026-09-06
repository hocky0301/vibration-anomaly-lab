"""Split and score at the machine level for unseen-machine evaluation.

Random trial folds can share a unit's fingerprint and label across train and
validation. They answer a different question from transfer to an unseen unit;
leave-one-unit-out matches the latter deployment target. Grouping avoids this
particular overlap, but does not guarantee an independent or representative
sample, nor prevent adaptive model/threshold selection on the same OOF labels.

Aggregate trial scores to one score per unit before computing the unit metric.
Repeated measurements can improve a unit estimate without adding independent
machines. Cross-fold AUC also compares scores from different fitted models,
whose calibration need not agree.

best_unit_f1 selects and scores a threshold on the same labels: it is an
optimistically selected diagnostic. max_margin_threshold is another fitted
threshold rule, not an independently validated deployment recommendation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from .features import unit_labels

#: Default threshold grid.  Coarse on purpose: a finer grid would only let the
#: sweep exploit gaps between ten unit scores more aggressively.
DEFAULT_GRID = np.linspace(0.05, 0.95, 37)


# ----------------------------------------------------------------- splitting --
def leave_one_unit_out(F: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-unit-out folds: every machine is the validation set once.

    Every unit is held out once. The training reference and estimator must
    additionally be fitted using that fold only. The resulting OOF scores
    support diagnostics for the sampled units, not a field guarantee.
    """
    unit_labels(F)
    y = F["label"].to_numpy()
    groups = F["unit_id"].to_numpy()
    return list(LeaveOneGroupOut().split(F, y, groups))


def random_kfold_over_trials(
    F: pd.DataFrame, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A random split over trials.  **Leaky.  For demonstration only.**

    Trials of the same machine can land on both sides, allowing recognition
    of familiar units. This is an intentionally mismatched baseline when the
    target is unseen-machine transfer.
    """
    unit_labels(F)
    y = F["label"].to_numpy()
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(F, y))


# --------------------------------------------------------------- aggregation --
def unit_scores(
    F: pd.DataFrame,
    trial_scores: np.ndarray,
    how: str = "median",
    subset: np.ndarray | None = None,
) -> pd.Series:
    """Collapse per-trial scores to one score per unit.

    ``how="median"`` by default: a unit runs many trials across the sweep, the
    fault is visible only in part of that sweep, and a mean is dragged around by
    the trials where nothing is visible plus the occasional impulsive burst.
    ``subset`` restricts the aggregation to a slice of the sweep (e.g. the
    low-setpoint half), which is a legitimate analysis of *where* a fault lives
    but must be declared, not tuned silently.
    """
    scores = np.asarray(trial_scores, dtype=float)
    if scores.shape != (len(F),) or not np.isfinite(scores).all():
        raise ValueError("trial_scores must be a finite vector aligned with the rows")
    if F["unit_id"].isna().any():
        raise ValueError("unit ids must not be missing")
    s = pd.DataFrame({"unit_id": F["unit_id"].to_numpy(), "score": scores})
    if subset is not None:
        subset = np.asarray(subset)
        if subset.dtype != np.bool_ or subset.shape != (len(F),):
            raise ValueError("subset must be a Boolean vector aligned with the rows")
        s = s[subset]
    if how not in {"median", "mean", "max"}:
        raise ValueError(f"unknown aggregation: {how}")
    return s.groupby("unit_id")["score"].agg(how)


def unit_truth(F: pd.DataFrame) -> pd.Series:
    """One label per unit, indexed by ``unit_id``."""
    return unit_labels(F)


# -------------------------------------------------------------------- scoring --
def unit_f1_at(
    F: pd.DataFrame,
    trial_scores: np.ndarray,
    threshold: float,
    how: str = "median",
    subset: np.ndarray | None = None,
) -> float:
    """Unit-level F1 at a given threshold.

    This function does not select the threshold; callers must disclose how it
    was chosen. Reusing these labels elsewhere can still make it optimistic.

    Each machine counts once, whatever number of trials it happens to have run.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    u = unit_scores(F, trial_scores, how=how, subset=subset)
    truth = unit_truth(F)
    pred = (u > threshold).astype(int).reindex(truth.index).fillna(0).astype(int)
    return float(f1_score(truth.to_numpy(), pred.to_numpy(), zero_division=0))


def best_unit_f1(
    F: pd.DataFrame,
    trial_scores: np.ndarray,
    grid: np.ndarray = DEFAULT_GRID,
    how: str = "median",
    subset: np.ndarray | None = None,
) -> tuple[float, float]:
    """Best unit-level F1 over a threshold sweep, and the threshold achieving it.

    **Optimistic by construction** -- see the module docstring.  Returned as a
    pair so that callers are forced to look at the threshold that produced the
    score and ask whether it would have been guessable in advance.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError("grid must be a nonempty finite vector")
    best_f1, best_t = 0.0, float(grid[0])
    for t in grid:
        f = unit_f1_at(F, trial_scores, float(t), how=how, subset=subset)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_f1, best_t


def max_margin_threshold(
    scores: pd.Series, truth: pd.Series, default: float = 0.5
) -> float:
    """Midpoint between the highest healthy score and the lowest faulty score.

    Falls back to ``default`` when the two classes overlap, because in that case
    no threshold separates them and picking the F1-maximising edge would be
    fitting noise with two positives.  Not unbiased -- the same labels are still
    involved -- just less brittle than sitting on the boundary.
    """
    if (not scores.index.is_unique or not truth.index.is_unique
            or not truth.isin([0, 1]).all() or not np.isfinite(default)):
        raise ValueError("unique unit indices, binary truth, and finite default required")
    s = scores.reindex(truth.index)
    if not np.isfinite(s.to_numpy(dtype=float)).all():
        raise ValueError("every truth unit needs a finite score")
    healthy, faulty = s[truth == 0], s[truth == 1]
    if len(healthy) == 0 or len(faulty) == 0:
        return default
    hi, lo = float(healthy.max()), float(faulty.min())
    return (hi + lo) / 2.0 if lo > hi else default


def unit_report(
    F: pd.DataFrame, trial_scores: np.ndarray, threshold: float, how: str = "median"
) -> pd.DataFrame:
    """Per-unit score, decision and truth -- ten rows you can read by eye.

    With this sample size, reading the ten numbers is more informative than any
    summary statistic computed from them.
    """
    u = unit_scores(F, trial_scores, how=how)
    truth = unit_truth(F)
    out = pd.DataFrame({"score": u, "decision": (u > threshold).astype(int)})
    out["truth"] = truth
    return out.sort_values("score", ascending=False)
