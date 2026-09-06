"""The synthetic generator: determinism, schema, and CLI guards."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import TINY_ARGS, run_generator

TRAIN_COLUMNS = [
    "unit_id", "trial_id", "setpoint", "speed_rpm",
    "vib_x", "vib_y", "vib_z", "sample_id", "label",
]
HOLDOUT_COLUMNS = [c for c in TRAIN_COLUMNS if c != "label"]
FILES = ("train.csv", "holdout.csv", "holdout_labels.csv")


def _read(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(d / "train.csv"),
        pd.read_csv(d / "holdout.csv"),
        pd.read_csv(d / "holdout_labels.csv"),
    )


def test_same_seed_is_byte_identical(tiny_data: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    r = run_generator(again, *TINY_ARGS)
    assert r.returncode == 0, r.stderr
    for name in FILES:
        assert (tiny_data / name).read_bytes() == (again / name).read_bytes(), name


def test_different_seed_differs(tiny_data: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    args = list(TINY_ARGS)
    args[args.index("--seed") + 1] = "2"
    r = run_generator(other, *args)
    assert r.returncode == 0, r.stderr
    assert (tiny_data / "train.csv").read_bytes() != (other / "train.csv").read_bytes()


def test_schema_columns(tiny_data: Path) -> None:
    train, hold, labels = _read(tiny_data)
    assert list(train.columns) == TRAIN_COLUMNS
    assert list(hold.columns) == HOLDOUT_COLUMNS
    assert list(labels.columns) == ["unit_id", "label"]


def test_label_constant_within_unit(tiny_data: Path) -> None:
    train, _, _ = _read(tiny_data)
    assert (train.groupby("unit_id")["label"].nunique() == 1).all()
    assert set(train["label"].unique()) <= {0, 1}
    assert train.groupby("unit_id")["label"].first().sum() == 2


def test_trial_id_starts_with_unit_id(tiny_data: Path) -> None:
    train, hold, _ = _read(tiny_data)
    for df in (train, hold):
        pairs = df[["unit_id", "trial_id"]].drop_duplicates()
        assert all(t.startswith(u) for u, t in pairs.itertuples(index=False))


def test_sample_id_unique_and_units_disjoint(tiny_data: Path) -> None:
    train, hold, labels = _read(tiny_data)
    ids = pd.concat([train["sample_id"], hold["sample_id"]])
    assert ids.is_unique
    assert not (set(train["unit_id"]) & set(hold["unit_id"]))
    assert set(labels["unit_id"]) == set(hold["unit_id"])


def test_sweep_geometry(tiny_data: Path, gen) -> None:
    train, hold, _ = _read(tiny_data)
    for df in (train, hold):
        assert df["setpoint"].nunique() == 12 == gen.N_SETPOINTS
        rows_per_trial = df.groupby("trial_id").size()
        assert (rows_per_trial == 384).all()
        assert gen.ROWS_PER_TRIAL == 384
        trials_per_unit = df.groupby("unit_id")["trial_id"].nunique()
        assert (trials_per_unit == gen.N_SWEEPS * gen.N_SETPOINTS).all()
    assert train["unit_id"].nunique() == 4
    assert hold["unit_id"].nunique() == 3


def test_holdout_labels_one_row_per_unit(tiny_data: Path) -> None:
    _, hold, labels = _read(tiny_data)
    assert len(labels) == hold["unit_id"].nunique() == 3
    assert labels["unit_id"].is_unique
    assert labels["label"].sum() == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--units", "3", "--anomalous", "4", "--no-holdout", "--seed", "1"],
        ["--units", "1", "--anomalous", "0", "--no-holdout", "--seed", "1"],
    ],
    ids=["anomalous_gt_units", "units_lt_2"],
)
def test_cli_rejects_impossible_fleets(tmp_path: Path, args: list[str]) -> None:
    r = run_generator(tmp_path / "bad", *args)
    assert r.returncode != 0
    assert (r.stderr + r.stdout).strip(), "expected an error message"
    assert not (tmp_path / "bad" / "train.csv").exists()
