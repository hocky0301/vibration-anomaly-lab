"""Fold safety and edge cases of ConditionalRobustZ."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vibration_anomaly_lab.conditioning import ConditionalRobustZ
from vibration_anomaly_lab.features import trial_features, vibration_features


@pytest.fixture(scope="module")
def F(tiny_data: Path) -> pd.DataFrame:
    return trial_features(pd.read_csv(tiny_data / "train.csv"))


def _perturbed(F: pd.DataFrame, unit: str, feats: list[str], delta: float = 1e3) -> pd.DataFrame:
    G = F.copy()
    m = G["unit_id"] == unit
    G.loc[m, feats] = G.loc[m, feats] + delta
    return G


def test_fold_safe_reference_ignores_held_out_unit(F: pd.DataFrame) -> None:
    feats = vibration_features(F)
    units = sorted(F["unit_id"].unique())
    held_out = units[-1]
    others = F["unit_id"] != held_out

    def z_of_others(frame: pd.DataFrame) -> np.ndarray:
        ref = ConditionalRobustZ(feats).fit(frame[others & (frame["label"] == 0)])
        return ref.transform(frame)[others.to_numpy()]

    base = z_of_others(F)
    pert = z_of_others(_perturbed(F, held_out, feats))
    np.testing.assert_array_equal(base, pert)


def test_leaky_reference_moves_with_held_out_unit(F: pd.DataFrame) -> None:
    feats = vibration_features(F)
    units = sorted(F["unit_id"].unique())
    # Use a healthy unit so the deliberately leaky reference actually includes it.
    healthy_units = sorted(F.loc[F["label"] == 0, "unit_id"].unique())
    held_out = healthy_units[-1]
    others = (F["unit_id"] != held_out).to_numpy()
    assert len(units) >= 3

    def z_of_others(frame: pd.DataFrame) -> np.ndarray:
        ref = ConditionalRobustZ(feats).fit(frame[frame["label"] == 0])  # LEAKY
        return ref.transform(frame)[others]

    base = z_of_others(F)
    pert = z_of_others(_perturbed(F, held_out, feats))
    assert not np.array_equal(base, pert)


def test_unseen_setpoint_falls_back_to_global_stats(F: pd.DataFrame) -> None:
    feats = vibration_features(F)
    ref = ConditionalRobustZ(feats, clip=None).fit(F[F["label"] == 0])
    probe = F.iloc[:5].copy()
    probe["setpoint"] = 999.0
    Z = ref.transform(probe)
    assert Z.shape == (5, len(feats) + 1)
    assert np.isfinite(Z).all()
    expected = (probe[feats].to_numpy() - ref.global_median_.to_numpy()) / (
        1.4826 * ref.global_mad_.to_numpy() + 1e-9
    )
    np.testing.assert_allclose(Z[:, : len(feats)], expected, rtol=1e-12)


def test_clip_bounds_respected(F: pd.DataFrame) -> None:
    feats = vibration_features(F)
    ref = ConditionalRobustZ(feats, clip=3.0).fit(F[F["label"] == 0])
    wild = F.copy()
    wild[feats] = wild[feats] * 1e6
    Z = ref.transform(wild)
    assert np.all(Z[:, : len(feats)] <= 3.0)
    assert np.all(Z[:, : len(feats)] >= -3.0)
    # Context columns are never clipped.
    np.testing.assert_array_equal(Z[:, -1], wild["setpoint"].to_numpy())
    unclipped = ConditionalRobustZ(feats, clip=None).fit(F[F["label"] == 0]).transform(wild)
    assert np.abs(unclipped[:, : len(feats)]).max() > 3.0


def test_fit_on_empty_raises(F: pd.DataFrame) -> None:
    feats = vibration_features(F)
    with pytest.raises(ValueError):
        ConditionalRobustZ(feats).fit(F.iloc[0:0])


def test_transform_before_fit_raises(F: pd.DataFrame) -> None:
    feats = vibration_features(F)
    with pytest.raises(RuntimeError):
        ConditionalRobustZ(feats).transform(F)
    with pytest.raises(RuntimeError):
        ConditionalRobustZ(feats).z_frame(F)
