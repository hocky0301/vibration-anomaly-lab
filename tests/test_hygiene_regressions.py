"""Exercise rejection paths without placing real identities in the source tree."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT


@pytest.fixture
def tree(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ["check_public.sh", "check_public.py"]:
        shutil.copyfile(REPO_ROOT / "scripts" / name, scripts / name)
    (scripts / "forbidden.local").write_text("private-marker\n")
    return tmp_path


def check(tree, *args):
    return subprocess.run(["sh", str(tree / "scripts/check_public.sh"), *args],
                          capture_output=True, text=True)


def test_gate_rejects_without_disclosing_matched_tokens_or_lines(tree):
    (tree / "note.md").write_text("the private-marker must never appear in logs")
    result = check(tree)
    assert result.returncode == 1
    assert "note.md" in result.stdout
    assert "private-marker" not in result.stdout + result.stderr
    assert "must never appear" not in result.stdout + result.stderr


@pytest.mark.parametrize("content", ["/" + "Users/" + "example/file", "person" + "@" + "example.test"])
def test_gate_rejects_builtin_identifiers(tree, content):
    (tree / "note.md").write_text(content)
    assert check(tree).returncode == 1


def git(tree: Path, *args):
    result = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def test_gate_checks_tracked_data_and_full_identity_domain(tree):
    git(tree, "init", "-q")
    git(tree, "config", "user.name", "Example")
    git(tree, "config", "user.email", "example" + "@" + "users.noreply.github.com.invalid")
    (tree / "record.csv").write_text("synthetic fixture")
    git(tree, "add", "record.csv")
    git(tree, "commit", "-qm", "fixture")
    result = check(tree, "--identity")
    assert result.returncode == 1
    assert "tracked data" in result.stdout
    assert "noreply address" in result.stdout
    assert "@" not in result.stdout


def test_gate_accepts_noreply_identity_without_printing_it(tree):
    git(tree, "init", "-q")
    git(tree, "config", "user.name", "Example")
    git(tree, "config", "user.email", "example" + "@" + "users.noreply.github.com")
    (tree / "note.md").write_text("public")
    git(tree, "add", "note.md")
    git(tree, "commit", "-qm", "fixture")
    assert check(tree, "--identity").returncode == 0
