# Changelog

All notable changes to this project are recorded here.  The project is a
synthetic-data teaching repository; version numbers track the claims the
documentation makes, not an API contract.

## 0.3.0

Publication audit. The default generator and all stored protocol measurements
are unchanged; the reference computation was independently reproduced.

- Corrected optimistic threshold selection and no-separation abstention in
  the educational panel model; strengthened input validation.
- Distinguished expected information bounds from realised candidate reduction
  in the metric-feedback demonstration.
- Added regression checks for identified failures and all-field reference JSON
  comparison, including per-unit scores. Verification and seed repetitions
  write outside the committed evidence directory.
- Added pinned numerical-reference dependencies and separate compatibility CI.
- Corrected documentation on MAD versus standard deviation, feature-column
  confounding, pooled OOF AUC, binomial interval assumptions, AUC ties, finite
  threshold grids and selection bias. Corrected the Figure 05 range caption.
- Added a reproduction guide and tightened the public-file hygiene checks.

Earlier entries record the interpretation at the time of each version. Where
wording conflicts, the current documentation and this audit supersede it.

## 0.2.1

- `scripts/check_public.sh` no longer carries the list of identifying tokens
  inside the repository: the list lives in the git-ignored
  `scripts/forbidden.local` (template: `scripts/forbidden.example`), CI supplies
  it through a secret, and `--identity` additionally checks every git
  author/committer against it.
- `scripts/model_seed_repeat.py` / `make model-seeds`: the same draw re-run
  under ten model seeds, written to `results/model_seed_repeat.md`, so the
  two kinds of seed variance can be read side by side (docs/03).
- `metric_design.leakage_bound_bits` applies the visibility cap also when the
  positive count is published (it previously could exceed the visible bits).
- README: the Nyquist bullet now says what the generator does and does not
  simulate.

## 0.2.0

A multi-seed audit of the 0.1 release found that its documentation described a
mechanism the generator did not implement, and that its headline table came
from a single draw that was not representative.  Specifically, 0.1 described
the fault as a zero-mean tilt of the amplitude-versus-setpoint curve, described
the between-unit spread as larger than the fault effect, and described
operating-point conditioning as recovering a signal that raw features could not
see.  None of the three held across seeds with the generator as written: the
fault is a one-sided deficit in the low-setpoint half, and for a
gradient-boosted tree that already sees the setpoint column, conditional
robust-z does not beat raw features (4 wins / 18 ties / 8 losses by unit AUC
over 30 seeds).  The generator's constants were not tuned; the documentation
was rewritten to what the generator measures.

Changes:

- Generator: removed a fault-placement bias (faulty units are now placed
  uniformly at random among the units).  No constant was changed.
- Verification: the generator's `--self-test` was removed and replaced by a
  standalone `verify_claims.py` that runs nine named protocols (leaky random
  K-fold; leave-one-unit-out on raw / no-operating-point / conditional
  robust-z (median/MAD and mean/std) / unit-relative feature spaces; tree and
  linear models), reports trial-level and unit-level metrics at a fixed 0.5
  threshold plus a labelled-optimistic best-threshold column, measures unit
  identifiability, and writes a JSON document (metrics rounded to 6 decimals).
- Seed sweep: `scripts/sweep_seeds.py` repeats the protocols over many
  generator seeds and writes `results/seed_sweep.json`, `results/seed_sweep.md`
  and `results/seed_sweep_per_seed.md`; the docs quote those files.
- Results and figures: `results/default_draw.json` and the 30-seed sweep are
  committed; `scripts/make_figures.py` renders seven figures from the data and
  the results.  Continuous integration re-derives the default draw and fails on
  drift from the committed JSON.
- Tests added: generator, conditioning, validation, production model, metric
  design, the verify CLI, package/script parity, and the public-hygiene gate
  (`scripts/check_public.sh`).
- Documentation rewritten to the measured mechanism.  README and the four
  design notes copy every number from a file under `results/` or from a
  command shown in the text; `docs/02` is retitled "Conditioning is
  legibility, not information"; `docs/04` corrects the per-query information
  accounting (at most 3 outcomes per fixed submission on 2-of-10, so at least
  4 adaptive queries, not 2).
- Packaging: the package directory is `vibration_anomaly_lab` (was `src`),
  installable with `pip install -e .`; development extras moved to
  `requirements-dev.txt`.
- `ProductionModel.fit` raises `ValueError` when the training fleet holds fewer
  than two faulty units; the demos resolve their default data path from the
  package location and fail loudly when the data has not been generated.

## 0.1.0

Initial release: synthetic generator, conditioning / validation /
production-model package, single-draw verification script, and the four
design notes.
