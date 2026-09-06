"""ProductionModel end to end on the tiny synthetic fleet."""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from vibration_anomaly_lab import production_model
from vibration_anomaly_lab.features import trial_features
from vibration_anomaly_lab.production_model import ProductionModel, UnitDecision


@pytest.fixture(scope="module")
def F_train(tiny_data: Path) -> pd.DataFrame:
    return trial_features(pd.read_csv(tiny_data / "train.csv"))


@pytest.fixture(scope="module")
def F_hold(tiny_data: Path) -> pd.DataFrame:
    return trial_features(pd.read_csv(tiny_data / "holdout.csv"))


def test_fit_refuses_fewer_than_two_faulty_units(F_train: pd.DataFrame) -> None:
    faulty = sorted(F_train.loc[F_train["label"] == 1, "unit_id"].unique())
    assert len(faulty) == 2
    one = F_train.copy()
    one.loc[one["unit_id"] == faulty[0], "label"] = 0
    with pytest.raises(ValueError):
        ProductionModel().fit(one)
    none = F_train.copy()
    none["label"] = 0
    with pytest.raises(ValueError):
        ProductionModel().fit(none)


def test_fit_requires_labels(F_train: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        ProductionModel().fit(F_train.drop(columns=["label"]))


def test_end_to_end_on_tiny_fleet(F_train: pd.DataFrame, F_hold: pd.DataFrame) -> None:
    model = ProductionModel(seed=0).fit(F_train)
    assert model.primary_ in production_model.MEMBER_ORDER
    assert 0.0 <= model.threshold_ <= 1.0
    assert isinstance(model.summary(), str)

    decisions = model.predict_units(F_hold)
    hold_units = set(F_hold["unit_id"].unique())
    assert len(decisions) == len(hold_units) == 3
    assert {d.unit_id for d in decisions} == hold_units
    for d in decisions:
        assert isinstance(d, UnitDecision)
        assert 0.0 <= d.probability <= 1.0
        assert d.decision in (0, 1)
        assert set(d.members) == set(production_model.MEMBER_ORDER)
        assert all(0.0 <= v <= 1.0 for v in d.members.values())
        assert d.disagreement >= 0.0
        assert d.evidence, "each decision carries sigma-level evidence"
        assert d.n_trials == int((F_hold["unit_id"] == d.unit_id).sum())
        assert isinstance(d.describe(), str)


def test_holdout_labels_never_read_by_the_model() -> None:
    src = inspect.getsource(production_model)
    assert "holdout_labels" not in src
