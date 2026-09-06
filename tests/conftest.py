"""
Shared fixtures for the vibration-anomaly-lab test suite.

Everything the tests touch is synthetic and generated on the fly into
pytest's temporary directories, so the suite never depends on the contents of
``data/synthetic/`` and never writes into the repository.

The repository root is put on ``sys.path`` so the package imports as
``vibration_anomaly_lab`` without a ``pip install``.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

# The suite fits many tiny gradient-boosting models.  On such small problems
# OpenMP threads spend far more time synchronising than computing (measured:
# the quick verify run on the 4-unit fleet takes ~4 s single-threaded and
# ~30-50 s with 2-4 threads), so pin every BLAS/OpenMP pool to one thread
# before scikit-learn is imported anywhere, and again in the child processes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "data" / "make_synthetic_data.py"
VERIFY_CLAIMS = REPO_ROOT / "verify_claims.py"
SWEEP_SEEDS = REPO_ROOT / "scripts" / "sweep_seeds.py"
CHECK_PUBLIC = REPO_ROOT / "scripts" / "check_public.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Generator arguments for the smallest configuration the suite exercises.
TINY_ARGS = [
    "--units", "4", "--anomalous", "2",
    "--holdout-units", "3", "--holdout-anomalous", "1",
    "--seed", "1",
]


def subprocess_env() -> dict[str, str]:
    """Environment for child processes: single-threaded pools, repo on the path."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def run_generator(out_dir: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``data/make_synthetic_data.py`` by subprocess into ``out_dir``."""
    cmd = [sys.executable, str(GENERATOR), "--out", str(out_dir), *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, env=subprocess_env(), cwd=str(REPO_ROOT)
    )


def import_by_path(name: str, path: Path) -> ModuleType:
    """Import a standalone script as a module, registering it in ``sys.modules``.

    Registration before ``exec_module`` matters for files that use dataclasses
    with postponed annotations: the dataclass machinery looks the module up by
    name while the module body is still executing.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4 train units (2 faulty) + 3 holdout units (1 faulty), seed 1."""
    out = tmp_path_factory.mktemp("tiny_data")
    r = run_generator(out, *TINY_ARGS)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="session")
def default_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The generator at its default seed and default fleet sizes."""
    out = tmp_path_factory.mktemp("default_data")
    r = run_generator(out)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="session")
def vc() -> ModuleType:
    """``verify_claims.py`` imported by path (it is standalone by design)."""
    return import_by_path("verify_claims", VERIFY_CLAIMS)


@pytest.fixture(scope="session")
def gen() -> ModuleType:
    """``data/make_synthetic_data.py`` imported by path, for its constants."""
    return import_by_path("make_synthetic_data", GENERATOR)
