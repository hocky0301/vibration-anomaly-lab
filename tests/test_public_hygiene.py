"""The repository ships code only, and nothing that identifies its origin."""
from __future__ import annotations

import subprocess

import pytest

from conftest import CHECK_PUBLIC, REPO_ROOT

DATA_SUFFIXES = (".csv", ".parquet", ".npy", ".npz", ".pkl", ".pickle", ".ipynb", ".h5", ".feather")


def test_check_public_passes() -> None:
    assert CHECK_PUBLIC.exists()
    r = subprocess.run(["bash", str(CHECK_PUBLIC)], capture_output=True, text=True,
                       cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "public hygiene: OK" in r.stdout


def test_no_tracked_data_files() -> None:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # An unpacked copy outside any git work tree has nothing tracked to
        # check; check_public.sh skips this part the same way.
        pytest.skip(f"not inside a git work tree: {r.stderr.strip()}")
    tracked = [p for p in r.stdout.splitlines() if p]
    offenders = [p for p in tracked
                 if p.lower().endswith(DATA_SUFFIXES) or "__pycache__" in p]
    assert not offenders, offenders
