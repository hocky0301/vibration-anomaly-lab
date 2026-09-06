# vibration-anomaly-lab -- every reproducible step, one target each.
#
#   make setup      install the runtime dependencies (numpy, pandas, scikit-learn)
#   make setup-dev  runtime dependencies plus pytest, matplotlib, ruff
#   make setup-repro install the Python 3.11 reference artifact stack
#   make verify-reference compare fresh JSON against every committed field
#   make data       regenerate the synthetic dataset into data/synthetic/
#   make verify     refit every protocol; writes .work/default_draw.json
#   make sweep      repeat verify over SEEDS generator seeds; writes .work/sweep/seed_sweep.*
#   make figures    render figures/*.png from data/ and results/
#   make test       run the test suite (python3 -m pytest -q)
#   make demo       run the two package demos (production_model, metric_design)
#   make check      run scripts/check_public.sh (anonymisation / hygiene gate)
#   make clean      remove generated data, caches and packaging side effects
#   make all        setup-dev -> data -> verify -> test
#
# Run these from the repository root (or use `make -C /path/to/repo <target>`).
# No real-world data is involved at any step: `make data` writes the only data
# this project ever sees, and it writes it from a seeded generator.
#
# Knobs (override on the command line, e.g. `make sweep SEEDS=5 JOBS=2`):
#   PYTHON          interpreter to use                       (default python3)
#   SEED            generator seed for `make data`           (default: generator's own)
#   SEEDS           number of seeds for `make sweep`         (default 30)
#   JOBS            worker processes for `make sweep`        (default 4)
#   OMP_NUM_THREADS BLAS / scikit-learn thread cap, exported (default 1)

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
SEEDS  ?= 30
MODEL_SEEDS ?= 10
JOBS   ?= 4
OMP_NUM_THREADS ?= 1
export OMP_NUM_THREADS
OPENBLAS_NUM_THREADS ?= 1
MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS MKL_NUM_THREADS

TRAIN_CSV := data/synthetic/train.csv

.PHONY: model-seeds all setup setup-dev setup-repro data verify verify-reference sweep figures test demo check clean
.NOTPARALLEL:            # the targets are a pipeline: setup -> data -> verify -> ...

all: setup-dev data verify test

## Install the runtime dependencies into the active environment.
## Use a virtualenv first if you do not want to touch the system interpreter.
setup:
	$(PIP) install -r requirements.txt

## Runtime dependencies plus the development tools (pytest, matplotlib, ruff).
setup-dev:
	$(PIP) install -r requirements-dev.txt

## Python 3.11 environment that reproduces the published reference artifacts.
setup-repro:
	$(PIP) install -r requirements-repro.txt

## Generate train.csv, holdout.csv and holdout_labels.csv (a few seconds).
## Deterministic: pass SEED to explore a different draw of units, e.g.
##   make data SEED=7
## Overwrites whatever is already in data/synthetic/.
data:
	$(PYTHON) data/make_synthetic_data.py $(if $(SEED),--seed $(SEED),)

## Refit every protocol on the default draw and print the comparison table.
## Nothing is cached or hard-coded; the full document goes to
## .work/default_draw.json. Published results remain unchanged.
verify: $(TRAIN_CSV)
	mkdir -p .work
	$(PYTHON) verify_claims.py --json .work/default_draw.json

## Numerical reference check requires the reference stack and default data seed.
verify-reference: verify
	$(PYTHON) scripts/compare_results.py results/default_draw.json .work/default_draw.json

## The same protocols on SEEDS fresh generator draws, JOBS at a time.
## Writes new artifacts under .work/sweep, preserving published results.
sweep:
	$(PYTHON) scripts/sweep_seeds.py --seeds $(SEEDS) --jobs $(JOBS) --out .work/sweep

## Repeat verify_claims on the default draw with model seeds 0..N-1 (data unchanged).
## The spread is reproducibility noise; compare with results/seed_sweep.md.
model-seeds:
	@test -f $(TRAIN_CSV) || { echo "$(TRAIN_CSV) not found -- run 'make data' first." >&2; exit 1; }
	$(PYTHON) scripts/model_seed_repeat.py --seeds $(MODEL_SEEDS) --out .work/model-seeds

## Render figures/*.png (needs matplotlib: `make setup-dev`), from the data
## and from results/default_draw.json + results/seed_sweep.json.
figures: $(TRAIN_CSV)
	$(PYTHON) scripts/make_figures.py

## Run the test suite.
test:
	$(PYTHON) -m pytest -q

## Run both package demos end to end on the generated data.
demo: $(TRAIN_CSV)
	$(PYTHON) -m vibration_anomaly_lab.production_model
	$(PYTHON) -m vibration_anomaly_lab.metric_design

## Anonymisation and hygiene gate for the public tree.
check:
	sh scripts/check_public.sh

## Remove generated data, caches and packaging side effects.  Never touches
## results/ or figures/, which are committed.
clean:
	rm -rf data/synthetic .work .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

## Guard: every target that reads the data says so when it is missing.
$(TRAIN_CSV):
	@echo "$(TRAIN_CSV) not found -- run 'make data' first." >&2; exit 1
