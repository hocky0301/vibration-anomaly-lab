"""
verify_claims.py is standalone on purpose.  These tests are the only place
where it and the package are checked against each other, so that a reader can
trust that the numbers the script re-derives are the numbers the library
computes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vibration_anomaly_lab.conditioning import ConditionalRobustZ, conditional_robust_z
from vibration_anomaly_lab.features import AXES, STATS, trial_features, vibration_features


@pytest.fixture(scope="module")
def train_df(tiny_data: Path) -> pd.DataFrame:
    return pd.read_csv(tiny_data / "train.csv")


def test_trial_features_agree(train_df: pd.DataFrame, vc) -> None:
    A = vc.trial_features(train_df).sort_values("trial_id").reset_index(drop=True)
    B = trial_features(train_df).sort_values("trial_id").reset_index(drop=True)

    expected_stats = [f"{a}_{s}" for a in AXES for s in STATS]
    assert len(expected_stats) == 21
    assert set(expected_stats) <= set(A.columns)
    assert set(expected_stats) <= set(B.columns)
    assert set(A.columns) == set(B.columns)

    # CONTRACT column order for the standalone script.
    assert list(A.columns[:5]) == ["trial_id", "unit_id", "label", "setpoint", "speed_rpm"]
    assert list(A.columns[5:]) == [
        f"{a}_{s}" for a in ("vib_x", "vib_y", "vib_z") for s in
        ("rms", "std", "mad", "iqr", "p90", "absmean", "ptp")
    ]

    assert A["trial_id"].tolist() == B["trial_id"].tolist()
    assert A["unit_id"].tolist() == B["unit_id"].tolist()
    assert A["label"].tolist() == B["label"].tolist()
    num = ["setpoint", "speed_rpm"] + expected_stats
    np.testing.assert_allclose(A[num].to_numpy(), B[num].to_numpy(), rtol=1e-12, atol=0)

    assert vc.vibration_columns(A) == vibration_features(B)


def test_conditional_robust_z_agree(train_df: pd.DataFrame, vc) -> None:
    F = vc.trial_features(train_df)
    feats = vc.vibration_columns(F)
    units = F["unit_id"].to_numpy()
    held_out = sorted(set(units))[0]
    ref_mask = (units != held_out) & (F["label"].to_numpy() == 0)
    assert ref_mask.sum() > 0

    Z_vc = vc.conditional_robust_z(F, feats, ref_mask, scale="mad")
    assert Z_vc.shape == (len(F), len(feats) + 1)
    assert np.all(np.abs(Z_vc[:, :-1]) <= 8.0 + 1e-12)

    # 1. Fully fair comparison: same context, same clip.
    ref = ConditionalRobustZ(feats, context=("setpoint",), clip=8.0).fit(F[ref_mask])
    Z_pkg = ref.transform(F)
    np.testing.assert_allclose(Z_vc, Z_pkg, rtol=1e-9, atol=1e-9)

    # 2. The package's mask form.  Compare the z-columns with the same clip
    #    applied, whatever context columns that form appends; when it appends
    #    exactly the setpoint (the CONTRACT), the whole matrix must agree.
    Z_mask = conditional_robust_z(F, feats, ref_mask)
    assert Z_mask.shape[0] == len(F) and Z_mask.shape[1] >= len(feats) + 1
    np.testing.assert_allclose(
        Z_vc[:, : len(feats)], np.clip(Z_mask[:, : len(feats)], -8.0, 8.0),
        rtol=1e-9, atol=1e-9,
    )
    if Z_mask.shape[1] == len(feats) + 1:
        np.testing.assert_allclose(Z_vc, Z_mask, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(Z_vc[:, -1], F["setpoint"].to_numpy())


def test_conditional_robust_z_std_scale_is_finite(train_df: pd.DataFrame, vc) -> None:
    F = vc.trial_features(train_df)
    feats = vc.vibration_columns(F)
    ref_mask = F["label"].to_numpy() == 0
    Z = vc.conditional_robust_z(F, feats, ref_mask, scale="std")
    assert Z.shape == (len(F), len(feats) + 1)
    assert np.isfinite(Z).all()
    assert np.all(np.abs(Z[:, :-1]) <= 8.0 + 1e-12)
    with pytest.raises((ValueError, KeyError, AssertionError)):
        vc.conditional_robust_z(F, feats, ref_mask, scale="nonsense")


def test_unit_relative_is_label_free_and_centred(train_df: pd.DataFrame, vc) -> None:
    F = vc.trial_features(train_df)
    feats = vc.vibration_columns(F)
    Z = vc.unit_relative(F, feats)
    assert Z.shape == (len(F), len(feats) + 1)
    assert np.isfinite(Z).all()
    # Per-unit median of every log-stat column is zero by construction.
    units = F["unit_id"].to_numpy()
    for u in np.unique(units):
        med = np.median(Z[units == u, : len(feats)], axis=0)
        np.testing.assert_allclose(med, 0.0, atol=1e-12)
    # Labels never enter: flipping them changes nothing.
    G = F.copy()
    G["label"] = 1 - G["label"]
    np.testing.assert_array_equal(Z, vc.unit_relative(G, feats))
    np.testing.assert_allclose(Z[:, -1], F["setpoint"].to_numpy())
