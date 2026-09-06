# Reproducing the Evidence

The repository contains synthetic data generation code and recorded results.
The generated CSV files are intentionally excluded from version control.

## Reference environment

Use Python 3.11 and the pinned versions in `requirements-repro.txt`. Create a
fresh virtual environment rather than installing into your system Python:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
make data
make verify-reference
make test
```

`make verify-reference` writes fresh output outside `results/` and checks every JSON
field against `results/default_draw.json`, including per-unit probabilities.
The committed result remains the comparison target. A mismatch is a finding:
inspect the differing fields and environment before changing any expectation.
Do not tune the generator to recover a preferred score.

Loose dependencies in `requirements.txt` support ordinary installation.
Compatibility with newer packages and exact reproduction of historical
numbers are different checks. The pinned environment is the numerical
reference; compatibility tests do not certify cross-version numerical parity.

## Full generator and estimator repetitions

After activating the reference environment, write all regenerated evidence
outside the committed directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/sweep_seeds.py --seeds 30 --jobs 3 --out .reproduction/sweep
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/model_seed_repeat.py --seeds 10 --out .reproduction/model-seeds
python scripts/make_figures.py --data data/synthetic/train.csv \
  --results results --out .reproduction/figures
```

The full sweep refits all published protocols. `--quick` reduces model
iterations and is for a smoke test only; it is not reproduction of the stored
full-run measurements. Changing `--model-seed` and changing the generator seed
answer different questions, as explained in [small-sample honesty](03-small-sample-honesty.md).

To compare stored repeated results with new files:

```bash
python - <<'PY'
import json
from pathlib import Path
def scientific(value):
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value

for reference, fresh in [
    ("results/seed_sweep.json", ".reproduction/sweep/seed_sweep.json"),
    ("results/model_seed_repeat.json", ".reproduction/model-seeds/model_seed_repeat.json"),
]:
    left = json.loads(Path(reference).read_text())
    right = json.loads(Path(fresh).read_text())
    print(reference, "EQUAL" if scientific(left) == scientific(right) else "DIFF")
PY
```

Only `seconds`, execution timing rather than a scientific measurement,
is excluded from this repeated-run comparison. No score or metric is excluded.
Inspect every reported difference; do not copy a fresh file over its reference
as a way of making a comparison pass. Updating evidence is a separate,
reviewable source change with its environment and rationale recorded.

## What reproduction does not establish

A matching result validates this computation under the stated environment.
It does not validate a physical fault model, establish performance on real
machines, or turn a pooled OOF AUC into an independent-test measurement.
The package's panel model is a demonstration of selection and abstention;
its name does not imply production readiness.

## Numeric sources

- Environment versions: `requirements-repro.txt`.
- Protocol and dataset metadata: `results/default_draw.json`.
- Repetition sizes and results: `results/seed_sweep.json` and `results/model_seed_repeat.json`.
- Thread and job counts above are execution settings, not measured performance claims.
