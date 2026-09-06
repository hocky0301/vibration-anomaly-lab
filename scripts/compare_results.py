#!/usr/bin/env python3
"""Compare complete JSON artifacts without modifying either input."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def differences(expected, actual, path: str = "$", tolerance: float = 1e-6) -> list[str]:
    """Find structural/value drift, including per-unit scores and metadata."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{path}: keys differ: {sorted(set(expected) ^ set(actual))}"]
        return [d for key in expected
                for d in differences(expected[key], actual[key], f"{path}/{key}", tolerance)]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: list lengths differ"]
        return [d for i, (x, y) in enumerate(zip(expected, actual))
                for d in differences(x, y, f"{path}[{i}]", tolerance)]
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    if numeric(expected) and numeric(actual):
        equal = (math.isfinite(expected) and math.isfinite(actual)
                 and math.isclose(expected, actual, rel_tol=0.0, abs_tol=tolerance))
    else:
        equal = type(expected) is type(actual) and expected == actual
    return [] if equal else [f"{path}: expected={expected!r}, actual={actual!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    drift = differences(json.loads(args.expected.read_text()), json.loads(args.actual.read_text()))
    if drift:
        print("\n".join(drift))
        return 1
    print("All JSON fields match (absolute tolerance 1e-6, including unit_scores).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
