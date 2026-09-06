"""Regression contracts found by review; no protocol-ranking expectations."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT, import_by_path, run_generator
from vibration_anomaly_lab.conditioning import ConditionalRobustZ
from vibration_anomaly_lab.features import trial_features, unit_labels
from vibration_anomaly_lab.validation import best_unit_f1, unit_f1_at, unit_scores, unit_truth
from vibration_anomaly_lab import production_model as pm
from vibration_anomaly_lab.metric_design import (
    enumerate_candidates, leakage_bound_bits, leakage_budget,
    mitigation_public_private_split, row_f1_from_groups,
)


@pytest.mark.parametrize("space", ["condz_mad", "condz_std"])
def test_standalone_reference_excludes_validation_rows(vc, space):
    frame = pd.DataFrame({"unit_id": np.repeat(["A", "B", "C", "D"], 4),
                          "label": np.repeat([0, 0, 1, 1], 4),
                          "setpoint": np.tile([1., 1., 2., 2.], 4),
                          "vib_x_mad": np.arange(16, dtype=float) + 1})
    train = np.flatnonzero(frame.unit_id != "A")
    base = vc._design(frame, space, ["vib_x_mad"], frame.label.to_numpy(), train)
    scale = "mad" if space == "condz_mad" else "std"
    reference = frame.iloc[train].loc[lambda df: df.label == 0]
    np.testing.assert_allclose(base, ConditionalRobustZ(["vib_x_mad"], scale=scale)
                               .fit(reference).transform(frame), rtol=1e-12, atol=1e-12)
    altered = frame.copy()
    altered.loc[altered.unit_id == "A", "vib_x_mad"] += 1e6
    altered.loc[altered.unit_id == "A", "label"] = 1
    changed = vc._design(altered, space, ["vib_x_mad"], altered.label.to_numpy(), train)
    np.testing.assert_array_equal(base[train], changed[train])


def test_unit_metric_parity_including_subset_ties_and_nondefault_index(vc):
    frame = pd.DataFrame({"unit_id": list("AABBCCDD"), "label": [0, 0, 0, 0, 1, 1, 1, 1]},
                         index=np.arange(8) * 7)
    scores = np.array([0.5, 0.5, 0, 1, 1, 1, 0.3, 0.4])
    subset = np.array([True, True, True, False, False, False, True, True])
    for mask in [None, subset, np.zeros(8, dtype=bool)]:
        pd.testing.assert_series_equal(unit_scores(frame, scores, subset=mask),
                                       vc._unit_scores(frame, scores, mask), check_names=False)
        for threshold in [0., 0.5, 1.]:
            assert unit_f1_at(frame, scores, threshold, subset=mask) == vc._unit_f1(
                frame, scores, threshold, mask)


def test_raw_trial_identity_and_labels_are_validated():
    frame = pd.DataFrame({"trial_id": ["t", "t"], "unit_id": ["A", "A"],
                          "setpoint": [1., 1.], "speed_rpm": [100., 101.],
                          "vib_x": [1., 2.], "label": [0, 0]})
    assert len(trial_features(frame, axes=("vib_x",))) == 1
    for column, value in [("unit_id", "B"), ("setpoint", 2.), ("label", 1),
                          ("label", np.nan), ("vib_x", np.inf)]:
        bad = frame.copy()
        bad.loc[1, column] = value
        with pytest.raises(ValueError):
            trial_features(bad, axes=("vib_x",))


def test_conditioner_rejects_nan_and_undefined_std():
    frame = pd.DataFrame({"setpoint": [1.], "vib_x_mad": [1.]})
    with pytest.raises(ValueError, match="two reference"):
        ConditionalRobustZ(["vib_x_mad"], scale="std").fit(frame)
    ref = ConditionalRobustZ(["vib_x_mad"]).fit(frame)
    frame.loc[0, "vib_x_mad"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ref.transform(frame)
    with pytest.raises(ValueError):
        ConditionalRobustZ(["vib_x_mad"], clip=-1)


def test_unit_truth_and_scores_do_not_silently_drop_invalid_rows():
    bad = pd.DataFrame({"unit_id": ["A", "A"], "label": [0, 1]})
    with pytest.raises(ValueError):
        unit_truth(bad)
    bad.label = [0, 0]
    with pytest.raises(ValueError):
        unit_scores(bad, [0.1, np.nan])
    with pytest.raises(ValueError):
        unit_scores(bad, [0.1, 0.2], subset=[1, 0])
    with pytest.raises(ValueError):
        best_unit_f1(bad, [0.1, 0.2], grid=[])
    bad.label = [0, 2]
    with pytest.raises(ValueError):
        unit_labels(bad)


@pytest.mark.parametrize("separates", [False, True])
def test_panel_selection_flags_optimism_and_abstains_without_separation(monkeypatch, separates):
    class Member:
        def __init__(self, name):
            self.name = name
        def fit(self, X, y):
            return self
        def predict_proba1(self, X):
            return X[:, -1] / 10 if separates else np.full(len(X), 0.2)
    monkeypatch.setattr(pm, "_build_members", lambda *a: [Member(n) for n in pm.MEMBER_ORDER])
    frame = pd.DataFrame({"unit_id": list("AABBCCDD"), "label": [0, 0, 0, 0, 1, 1, 1, 1],
                          "setpoint": np.repeat([1., 2., 8., 9.], 2),
                          "vib_x_mad": np.arange(8, dtype=float) + 1})
    model = pm.ProductionModel().fit(frame)
    assert model.threshold_is_optimistic_ is True
    assert model.no_member_separates_ is (not separates)
    assert model.primary_ == pm.MEMBER_ORDER[0]
    decisions = model.predict_units(frame.drop(columns="label"))
    assert all(d.review is (not separates) for d in decisions)
    if not separates:
        assert all("no member separates" in d.describe() for d in decisions)
    else:
        # A split remains reviewable even when a model passed the selection veto.
        model.members_[1].predict_proba1 = lambda X: 1 - X[:, -1] / 10
        assert all(d.review and "panel split" in d.review_reasons for d in model.predict_units(frame))


def test_rank_member_requires_a_healthy_reference():
    with pytest.raises(ValueError, match="healthy"):
        pm.RobustDeviation(1).fit(np.ones((2, 1)), np.ones(2))


def test_entropy_bound_is_not_a_pointwise_candidate_bound():
    # A positive public singleton identifies the truth among four 1-positive
    # assignments, including inference that all three private labels are zero.
    result = leakage_budget(4, [1, 1, 1, 1], public_rows_per_group=[1, 0, 0, 0],
                            known_n_positive=1, truth=[1, 0, 0, 0], n_submissions=1,
                            strategy="one_hot")
    assert result.bits_resolved == 2
    assert result.analytic_bound_bits == 1
    # Averaging over the entire declared prior does respect the entropy bound.
    all_results = [leakage_budget(4, [1] * 4, public_rows_per_group=[1, 0, 0, 0],
                                 known_n_positive=1, truth=t, n_submissions=1,
                                 strategy="one_hot") for t in enumerate_candidates(4, 1)]
    assert np.mean([r.bits_resolved for r in all_results]) <= result.analytic_bound_bits


@pytest.mark.parametrize("digits", [None, 0, 1])
def test_exact_prior_average_respects_channel_bound(digits):
    results = [leakage_budget(4, [1, 2, 3, 4], truth=t, displayed_digits=digits,
                             n_submissions=1, strategy="greedy", rng=np.random.default_rng(4))
               for t in enumerate_candidates(4)]
    assert np.mean([r.bits_resolved for r in results]) <= leakage_bound_bits(4, digits, 1) + 1e-12


def test_private_split_boundaries_and_invalid_truth_are_rejected():
    for by in ["row", "group"]:
        np.testing.assert_array_equal(mitigation_public_private_split(
            np.array([0, 3]), 0, np.random.default_rng(0), by=by), [0, 0])
        np.testing.assert_array_equal(mitigation_public_private_split(
            np.array([0, 3]), 1, np.random.default_rng(0), by=by), [0, 3])
    for kwargs in [{"truth": [0, 0], "known_n_positive": 1},
                   {"truth": [2, 0]}, {"public_rows_per_group": [2, 0]},
                   {"strategy": "typo"}, {"n_submissions": -1}]:
        with pytest.raises(ValueError):
            leakage_budget(2, [1, 1], **kwargs)
    with pytest.raises(ValueError):
        enumerate_candidates(60, 30)
    with pytest.raises(ValueError):
        row_f1_from_groups([1, 0], [0, 1], [1.5, 2])


def test_reference_comparator_checks_unit_scores_types_and_nonfinite_numbers():
    module = import_by_path("compare_results", REPO_ROOT / "scripts/compare_results.py")
    expected = {"unit_scores": {"A": 0.5}, "quick": False, "metadata": None}
    actual = copy.deepcopy(expected)
    actual["unit_scores"]["A"] += 0.01
    assert module.differences(expected, actual)
    assert module.differences(False, 0)
    assert module.differences(float("nan"), float("nan"))
    assert not module.differences(expected, expected)


def test_default_generator_byte_determinism_and_value_contract(default_data: Path, tmp_path, gen):
    result = run_generator(tmp_path / "again")
    assert result.returncode == 0, result.stderr
    for name in ["train.csv", "holdout.csv", "holdout_labels.csv"]:
        assert (default_data / name).read_bytes() == (tmp_path / "again" / name).read_bytes()
    frame = pd.read_csv(default_data / "train.csv")
    assert frame.unit_id.nunique() == 10
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
    assert set(frame.label.unique()) == {0, 1}
    # The sweep uses the published operating grid, not arbitrary numeric levels.
    assert sorted(frame.setpoint.unique()) == pytest.approx(gen.SETPOINT_LO + gen.SETPOINT_STEP * np.arange(gen.N_SETPOINTS))
    assert (frame.groupby("trial_id").setpoint.nunique() == 1).all()
