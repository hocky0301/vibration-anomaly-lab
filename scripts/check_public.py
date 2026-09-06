#!/usr/bin/env python3
"""Reject identifying content or tracked data; never print matched secret text.

Optional scripts/forbidden.local: one token per line; # starts a comment.
Built-in checks cover home paths and emails. This local content gate is a
heuristic, not a substitute for manual image review or a hosted secret scanner.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
           ".venv", "venv", "env", "ENV", ".work", "build", "dist"}
SUFFIXES = {".csv", ".parquet", ".npy", ".npz", ".pkl", ".pickle", ".ipynb",
            ".h5", ".feather", ".pyc"}
BUILTINS = [re.compile(p, re.IGNORECASE) for p in (
    r"/Users/[A-Za-z]", r"/home/[A-Za-z]", r"C:\\Users\\",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}",
)]


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args()
    patterns = []
    local = ROOT / "scripts/forbidden.local"
    if local.exists():
        for raw in local.read_text().splitlines():
            token = raw.split("#", 1)[0].strip()
            if token:
                pattern = re.escape(token)
                if re.fullmatch(r"[0-9.]+", token):
                    pattern = r"\b" + pattern + r"\b"
                patterns.append(re.compile(pattern, re.IGNORECASE))
    else:
        print("warning: no local token list; built-in patterns only")
    failed = []
    paths = []
    for parent, directories, files in os.walk(ROOT, followlinks=False):
        relative = Path(parent).relative_to(ROOT)
        directories[:] = [d for d in directories if d not in IGNORED
                          and not d.endswith(".egg-info")
                          and (relative / d).as_posix() != "data/synthetic"]
        for filename in files:
            path = Path(parent) / filename
            if path == local:
                continue
            if path.is_symlink():
                failed.append(f"symlink requires manual review: {path.relative_to(ROOT)}")
                continue
            paths.append(path)
            try:
                data = path.read_bytes()
            except OSError:
                failed.append(f"unreadable file: {path.relative_to(ROOT)}")
                continue
            # Binary image data is reviewed separately. Text with invalid UTF-8
            # still gets an ASCII-compatible scan rather than silently skipped.
            if b"\x00" in data:
                continue
            text = data.decode("utf-8", errors="replace")
            if any(p.search(text) for p in BUILTINS + patterns):
                failed.append(f"identifying content: {path.relative_to(ROOT)}")

    tracked = git("ls-files", "-z")
    if tracked.returncode == 0:
        names = [p for p in tracked.stdout.split("\x00") if p]
        for name in names:
            p = Path(name)
            if (p.suffix.lower() in SUFFIXES or "__pycache__" in p.parts
                    or name == "scripts/forbidden.local"):
                failed.append(f"tracked data/cache/private token file: {name}")
        if args.identity:
            identities = git("log", "--format=%an <%ae>%n%cn <%ce>")
            if identities.returncode:
                failed.append("cannot inspect commit identities (history may be empty)")
            else:
                for identity in identities.stdout.splitlines():
                    if any(p.search(identity) for p in patterns):
                        failed.append("commit identity contains a forbidden token")
                    # Check the full email domain, not a matching substring.
                    email = re.search(r"<([^<>]+)>$", identity)
                    if not email or not re.fullmatch(
                            r"[^@\s]+@users\.noreply\.github\.com", email.group(1)):
                        failed.append("commit identity does not use a GitHub noreply address")
    else:
        print("warning: no git index; checking local file extensions")
        for path in paths:
            if path.suffix.lower() in SUFFIXES:
                failed.append(f"data file outside generated data: {path.relative_to(ROOT)}")
        if args.identity:
            failed.append("--identity requires an accessible git history")
    if failed:
        print("public hygiene: FAILED")
        print("\n".join(sorted(set(failed))))
        return 1
    print("public hygiene: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
