"""Unit-level scoring helpers and the leave-one-unit-out split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibration_anomaly_lab.features import unit_labels
from vibration_anomaly_lab.validation import (
    best_unit_f1,
    leave_one_unit_out,
    max_margin_threshold,
    unit_f1_at,
    unit_scores,
)


def _frame(n_per_unit: int = 4) -> pd.DataFrame:
    units = ["A", "B", "C", "D"]
    labels = {"A": 0, "B": 0, "C": 1, "D": 1}
    rows = []
    for u in units:
        for i in range(n_per_unit):
            rows.append({"unit_id": u, "label": labels[u], "setpoint": float(i)})
    return pd.DataFrame(rows)


def test_unit_f1_at_counts_absent_unit_as_healthy() -> None:
    F = _frame()
    scores = np.where(F["label"].to_numpy() == 1, 0.9, 0.1)
    assert unit_f1_at(F, scores, 0.5) == 1.0
    # Drop every trial of faulty unit C from the aggregation: C is then absent
    # from the unit scores and must be predicted healthy -> one false negative.
    subset = (F["unit_id"] != "C").to_numpy()
    assert "C" not in unit_scores(F, scores, subset=subset).index
    assert unit_f1_at(F, scores, 0.5, subset=subset) == pytest.approx(2 / 3)


def test_best_unit_f1_is_at_least_f1_at_half() -> None:
    F = _frame()
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores = rng.random(len(F))
        best, t = best_unit_f1(F, scores)
        assert best >= unit_f1_at(F, scores, 0.5)
        assert 0.05 <= t <= 0.95
        assert best == pytest.approx(unit_f1_at(F, scores, t))


def test_max_margin_threshold_midpoint_and_fallback() -> None:
    truth = pd.Series({"A": 0, "B": 0, "C": 1, "D": 1})
    sep = pd.Series({"A": 0.1, "B": 0.3, "C": 0.7, "D": 0.9})
    assert max_margin_threshold(sep, truth) == pytest.approx(0.5)
    sep2 = pd.Series({"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.9})
    assert max_margin_threshold(sep2, truth) == pytest.approx(0.4)
    overlap = pd.Series({"A": 0.1, "B": 0.8, "C": 0.7, "D": 0.9})
    assert max_margin_threshold(overlap, truth) == 0.5
    assert max_margin_threshold(overlap, truth, default=0.42) == 0.42
    # No positives at all -> fallback.
    assert max_margin_threshold(sep, pd.Series({"A": 0, "B": 0, "C": 0, "D": 0})) == 0.5


def test_unit_labels_raises_on_inconsistent_labels() -> None:
    F = _frame()
    assert unit_labels(F).to_dict() == {"A": 0, "B": 0, "C": 1, "D": 1}
    F.loc[F.index[0], "label"] = 1  # unit A now carries both labels
    with pytest.raises(ValueError, match="A"):
        unit_labels(F)


def test_leave_one_unit_out_one_fold_per_unit() -> None:
    F = _frame()
    folds = leave_one_unit_out(F)
    units = F["unit_id"].to_numpy()
    assert len(folds) == F["unit_id"].nunique()
    held = []
    for tr, va in folds:
        assert set(units[va]).isdisjoint(set(units[tr]))
        assert len(set(units[va])) == 1
        assert len(tr) + len(va) == len(F)
        assert not (set(tr) & set(va))
        held.append(units[va][0])
    assert sorted(held) == sorted(F["unit_id"].unique())
