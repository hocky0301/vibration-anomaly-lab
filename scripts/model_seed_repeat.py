#!/usr/bin/env python3
"""Repeat verify_claims on ONE fixed draw while varying only the model seed.

Companion to scripts/sweep_seeds.py, which varies the generator seed (a fresh
population of machines each time).  Here the data never changes; only the
random_state of the estimators and of the leaky K-fold split does.  The spread
this produces is *reproducibility* noise.  Read next to results/seed_sweep.md
it shows the difference between "the same data, re-run" and "a different draw
of machines", which is the difference between a seed std and a generalisation
variance.

Usage:
    python3 scripts/model_seed_repeat.py --seeds 10 --out results
Writes results/model_seed_repeat.md and results/model_seed_repeat.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic", "train.csv"))
    ap.add_argument("--seeds", type=int, default=10, help="model seeds 0..N-1")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.exists(args.data):
        print(f"{args.data} not found -- run 'make data' first", file=sys.stderr)
        return 1
    vc = _import_by_path("verify_claims", os.path.join(ROOT, "verify_claims.py"))
    F = vc.trial_features(pd.read_csv(args.data))
    docs = []
    for s in range(args.seeds):
        d = vc.run_all(F, quick=args.quick, model_seed=s)
        d["model_seed"] = s
        docs.append(d)
        print(f"model seed {s}: done", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model_seed_repeat.json"), "w") as fh:
        json.dump(docs, fh, indent=1)
    metrics = ["trial_f1_at_0_5", "unit_f1_at_0_5", "unit_auc"]
    lines = [
        f"# Model-seed repeat: one fixed draw, {args.seeds} model seeds\n",
        "Synthetic data, the default draw (the data file never changes between rows).  Only the",
        "estimators' random_state and the leaky K-fold shuffle change.  Compare with",
        "results/seed_sweep.md, where every row is a fresh draw of machines.\n",
        "| protocol | trial F1@0.5 mean ± std | unit F1@0.5 mean ± std | unit AUC mean ± std |",
        "|---|---|---|---|",
    ]
    for pid in docs[0]["protocols"]:
        cells = []
        for m in metrics:
            v = np.array([d["protocols"][pid][m] for d in docs], dtype=float)
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=0):.4f}")
        lines.append(f"| {pid} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"model seeds used: {', '.join(str(d['model_seed']) for d in docs)}")
    with open(os.path.join(args.out, "model_seed_repeat.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
