#!/usr/bin/env python3
"""
sweep_seeds.py -- run verify_claims.py on many independent draws of the generator.

One draw of ten machines is one sample.  Every headline number in this
repository is therefore a single observation, and the only way to know whether
a protocol comparison is a property of the mechanism or an accident of the draw
is to redraw.  For each seed this script

  1. generates a fresh dataset with data/make_synthetic_data.py
     (``build_units`` + ``generate`` with ``numpy.random.default_rng(seed)``,
     written to a temporary directory and read back exactly as the CLI would),
  2. runs ``verify_claims.run_all`` on it -- the same code, the same protocols,
     the same metrics as the default draw,
  3. keeps the full per-seed document.

Outputs (into ``--out``):
    seed_sweep.json           list of per-seed verify_claims documents (+ "seed")
    seed_sweep.md             one row per protocol, averaged over seeds
    seed_sweep_per_seed.md    one row per seed, compact

Usage
    python3 scripts/sweep_seeds.py --seeds 30 --start 0 --jobs 8 --out results
    python3 scripts/sweep_seeds.py --seeds 2 --jobs 2 --quick --out /tmp/x   # smoke test

Nothing is tuned here: the generator constants are whatever the generator says
they are, and a disappointing number is reported, not moved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GEN_PATH = os.path.join(REPO, "data", "make_synthetic_data.py")
VC_PATH = os.path.join(REPO, "verify_claims.py")

PROTOCOL_ORDER = ["a", "b", "b_linear", "b_noctx", "c", "c_meanstd", "d", "e", "e_linear"]
_MODS: dict = {}


def _load(name: str, path: str):
    """Import a module by path.  The module is registered in sys.modules *before*
    exec, because the generator uses dataclasses with postponed annotations."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _init_worker() -> None:
    # One thread per worker: the seeds are the parallelism.  Must be set before
    # scikit-learn (OpenMP) is imported, which happens inside verify_claims.
    os.environ["OMP_NUM_THREADS"] = "1"
    _MODS["gen"] = _load("make_synthetic_data", GEN_PATH)
    _MODS["vc"] = _load("verify_claims", VC_PATH)


def run_seed(args: tuple[int, bool, int]) -> dict:
    seed, quick, model_seed = args
    if not _MODS:
        _init_worker()
    gen, vc = _MODS["gen"], _MODS["vc"]
    import numpy as np
    import pandas as pd

    t0 = time.time()
    rng = np.random.default_rng(seed)
    units = gen.build_units(rng, gen.N_UNITS, gen.N_ANOMALOUS, prefix="U")
    df, _ = gen.generate(units, rng, with_label=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "train.csv")
        df.to_csv(path, index=False)
        F = vc.trial_features(pd.read_csv(path))
    doc = vc.run_all(F, quick=quick, model_seed=model_seed)
    doc["seed"] = int(seed)
    doc["seconds"] = round(time.time() - t0, 1)
    return doc


# ------------------------------------------------------------------ reports --
def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def summary_markdown(docs: list[dict], quick: bool) -> str:
    seeds = [d["seed"] for d in docs]
    lines = [f"# Seed sweep: {len(docs)} independent draws of the generator",
             "",
             "Synthetic data.  Each row averages one protocol over all seeds; every seed is a fresh "
             "draw of the machines (fingerprints and fault placement) and a fresh generator noise stream.",
             f"Model settings: `--quick` = {quick} (GBM max_iter {'60' if quick else '300'}), "
             f"model seed {docs[0]['model_seed']}.  Metric definitions: see `verify_claims.py`.",
             "",
             "| protocol | split / space / model | mean trial F1@0.5 | mean unit F1@0.5 "
             "| seeds with unit F1@0.5 = 1.0 | mean unit AUC | mean unit F1 lowhalf@0.5 |",
             "|---|---|---|---|---|---|---|"]
    for pid in PROTOCOL_ORDER:
        rows = [d["protocols"][pid] for d in docs if pid in d["protocols"]]
        if not rows:
            continue
        r0 = rows[0]
        perfect = sum(1 for r in rows if r["unit_f1_at_0_5"] is not None and r["unit_f1_at_0_5"] >= 1.0)
        lines.append(
            f"| {pid} | {r0['split']} / {r0['space']} / {r0['model']} "
            f"| {_f(_mean([r['trial_f1_at_0_5'] for r in rows]))} "
            f"| {_f(_mean([r['unit_f1_at_0_5'] for r in rows]))} "
            f"| {perfect}/{len(rows)} ({perfect / len(rows):.2f}) "
            f"| {_f(_mean([r['unit_auc'] for r in rows]))} "
            f"| {_f(_mean([r['unit_f1_lowhalf_at_0_5'] for r in rows]))} |")
    ident = [d["unit_identifiability_acc"] for d in docs]
    n_units = docs[0]["n_units"]
    lines += ["",
              f"unit identifiability (5-fold accuracy of a GBM predicting unit_id from raw trial features, "
              f"chance {1 / n_units:.3f}): mean {_f(_mean(ident))} / min {_f(min(ident))} / max {_f(max(ident))}",
              ""]
    if all("b" in d["protocols"] and "c" in d["protocols"] for d in docs):
        pairs = [(d["protocols"]["b"]["unit_auc"], d["protocols"]["c"]["unit_auc"]) for d in docs]
        pairs = [(b, c) for b, c in pairs if b is not None and c is not None]
        wins = sum(1 for b, c in pairs if c > b + 1e-9)
        losses = sum(1 for b, c in pairs if c < b - 1e-9)
        ties = len(pairs) - wins - losses
        lines += [f"b vs c by unit AUC: wins/ties/losses = {wins}/{ties}/{losses} "
                  "(a win = conditional robust-z (c) strictly above raw (b) on that seed)", ""]
    lines += [f"seeds used ({len(seeds)}): {', '.join(str(s) for s in seeds)}", ""]
    return "\n".join(lines)


def per_seed_markdown(docs: list[dict]) -> str:
    pids = [p for p in PROTOCOL_ORDER if all(p in d["protocols"] for d in docs)]
    lines = ["# Seed sweep, per seed (synthetic data)", "",
             "Cell = unit F1@0.5 / unit AUC.  `ident` = unit identifiability accuracy.", "",
             "| seed | faulty | ident | " + " | ".join(pids) + " |",
             "|---|---|---|" + "---|" * len(pids)]
    for d in docs:
        cells = [f"{_f(d['protocols'][p]['unit_f1_at_0_5'])} / {_f(d['protocols'][p]['unit_auc'])}"
                 for p in pids]
        lines.append(f"| {d['seed']} | {' '.join(d['faulty_units'])} | {_f(d['unit_identifiability_acc'])} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", type=int, default=30, help="number of seeds")
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO, "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model-seed", type=int, default=0)
    args = ap.parse_args(argv)

    seeds = list(range(args.start, args.start + args.seeds))
    os.makedirs(args.out, exist_ok=True)
    jobs = [(s, args.quick, args.model_seed) for s in seeds]
    docs: list[dict] = []
    t0 = time.time()
    with Pool(min(args.jobs, len(jobs)), initializer=_init_worker) as pool:
        for d in pool.imap_unordered(run_seed, jobs):
            docs.append(d)
            P = d["protocols"]
            got = lambda p: _f(P[p]["unit_f1_at_0_5"]) if p in P else "-"
            print(f"[{len(docs):>3}/{len(seeds)}] seed={d['seed']:<4} faulty={' '.join(d['faulty_units'])}  "
                  f"unit F1@.5  a={got('a')} b={got('b')} c={got('c')} e={got('e')}  "
                  f"ident={_f(d['unit_identifiability_acc'])}  ({d['seconds']}s)", flush=True)
    docs.sort(key=lambda d: d["seed"])

    with open(os.path.join(args.out, "seed_sweep.json"), "w") as fh:
        json.dump(docs, fh, indent=1)
    with open(os.path.join(args.out, "seed_sweep.md"), "w") as fh:
        fh.write(summary_markdown(docs, args.quick))
    with open(os.path.join(args.out, "seed_sweep_per_seed.md"), "w") as fh:
        fh.write(per_seed_markdown(docs))
    print(f"\nwrote seed_sweep.json, seed_sweep.md, seed_sweep_per_seed.md into {args.out} "
          f"({len(docs)} seeds, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
