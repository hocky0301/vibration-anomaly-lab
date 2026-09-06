"""verify_claims.py and scripts/sweep_seeds.py as command-line tools.

Structure only: which protocol comes out ahead is a property of the draw, not
of the code, so nothing here asserts a relationship between protocols.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, SWEEP_SEEDS, VERIFY_CLAIMS, subprocess_env

PROTOCOLS = ["a", "b", "b_linear", "b_noctx", "c", "c_meanstd", "d", "e", "e_linear"]
METRICS = [
    "trial_f1_at_0_5", "trial_auc",
    "unit_f1_at_0_5", "unit_auc",
    "unit_f1_lowhalf_at_0_5",
    "unit_f1_best", "unit_f1_best_threshold",
]


@pytest.fixture(scope="module")
def verify_json(tiny_data: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("verify") / "doc.json"
    cmd = [sys.executable, str(VERIFY_CLAIMS),
           "--data", str(tiny_data / "train.csv"), "--quick", "--json", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env(),
                       cwd=str(REPO_ROOT), timeout=600)
    assert r.returncode == 0, r.stderr
    assert "unit AUC" in r.stdout
    assert out.exists()
    return json.loads(out.read_text())


def test_verify_json_document_shape(verify_json: dict) -> None:
    doc = verify_json
    assert doc["n_units"] == 4
    assert doc["n_trials"] == 4 * 72
    assert len(doc["faulty_units"]) == 2
    assert 0.0 <= doc["unit_identifiability_acc"] <= 1.0
    assert doc["quick"] is True
    assert doc["model_seed"] == 0
    assert list(doc["protocols"]) == PROTOCOLS


def test_verify_json_protocol_metrics(verify_json: dict) -> None:
    units = {"U01", "U02", "U03", "U04"}
    for pid, p in verify_json["protocols"].items():
        for key in ("label", "split", "space", "model"):
            assert isinstance(p[key], str) and p[key], (pid, key)
        for m in METRICS:
            assert 0.0 <= p[m] <= 1.0, (pid, m, p[m])
        assert p["unit_f1_best"] >= p["unit_f1_at_0_5"], pid
        assert set(p["unit_scores"]) == units, pid
        assert all(0.0 <= v <= 1.0 for v in p["unit_scores"].values()), pid


def test_verify_protocol_subset(tiny_data: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub.json"
    cmd = [sys.executable, str(VERIFY_CLAIMS), "--data", str(tiny_data / "train.csv"),
           "--quick", "--protocols", "a,e", "--model-seed", "3", "--json", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env(),
                       cwd=str(REPO_ROOT), timeout=600)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert list(doc["protocols"]) == ["a", "e"]
    assert doc["model_seed"] == 3


def test_sweep_seeds_smoke(tmp_path: Path) -> None:
    assert SWEEP_SEEDS.exists(), "scripts/sweep_seeds.py is missing"
    out = tmp_path / "sw"
    cmd = [sys.executable, str(SWEEP_SEEDS), "--seeds", "2", "--start", "0",
           "--jobs", "1", "--quick", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env(),
                       cwd=str(REPO_ROOT), timeout=900)
    assert r.returncode == 0, r.stderr
    for name in ("seed_sweep.json", "seed_sweep.md", "seed_sweep_per_seed.md"):
        assert (out / name).exists(), name
    docs = json.loads((out / "seed_sweep.json").read_text())
    assert isinstance(docs, list) and len(docs) == 2
    assert sorted(d["seed"] for d in docs) == [0, 1]
    for d in docs:
        assert list(d["protocols"]) == PROTOCOLS
        assert d["quick"] is True
    md = (out / "seed_sweep.md").read_text()
    assert "b vs c by unit AUC" in md
    assert "|" in md
