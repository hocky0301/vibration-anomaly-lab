# Validation Record

The original evidence was independently rerun in a fresh Python environment
on the same host. This is a reproduction of the computation, not an independent
lab replication or validation against real equipment.

| Check | Observed result |
|---|---|
| Default generator, repeated twice | Generated CSV bytes identical |
| Default protocol refit | Every JSON field exactly matches, including unit scores |
| Generator seeds 0–29, full models | Every scientific field exactly matches |
| Generator-sweep Markdown summaries | Byte-identical |
| Fixed data, estimator seeds 0–9 | Every JSON field and Markdown summary matches |
| Figures regenerated in reference environment | All seven reproduced; final implementation gives identical renders |
| Regression tests in reference Python 3.11 | 61 passed |
| Regression tests in Python 3.12 with newer dependencies | 61 passed |
| Published numeric table audit | 304 cells checked; no mismatches |

Only each sweep record's `seconds` field is excluded from the scientific
comparison: elapsed execution time is expected to change. No metric or unit
probability is excluded. The original generator and recorded protocol results
remain unchanged.

The numerical reference uses `requirements-repro.txt`. The newer locally
checked stack was Python 3.12.14, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0
and SciPy 1.18.1. It passes the API/regression tests but **does not reproduce
all reference scores**: for example, the default c_meanstd unit F1@0.5 changes
from 0.500 to 0.666667. This is why current-dependency compatibility and pinned
numerical reference checks are separate CI jobs. Upstream deprecation warnings
in the historical scikit-learn/SciPy stack do not invalidate its matching output.

[Hosted CI](https://github.com/hocky0301/vibration-anomaly-lab/actions) provides
the additional Linux checks; consult the exact commit's run for its status.
The reproduction commands are in [the reproduction guide](05-reproduction.md).

## Preserved evidence fingerprints

| Evidence | SHA-256 |
|---|---|
| `data/make_synthetic_data.py` | `1da557f251fc3057f663e441083d408f1c5a0fe9ac53413a3726dea0b1014936` |
| `results/default_draw.json` | `69c40c4044ff21b94761a8662be3b00f7883ce97fcf1c64a95c1c60e59602310` |
| `results/seed_sweep.json` | `94a8201c053758c00c601dfb31c15be83248d4983b02bae3828113816d217be7` |
| `results/model_seed_repeat.json` | `13b85604339051d6e7a216180dd771422c605415b66a7dca345cc3ff2f96b73c` |

## Findings corrected in this release

The audit corrected threshold-optimism metadata, no-separation review flags,
invalid-input handling, and misleading information-bound terminology. It also
corrected claims about identical MAD/std decisions, the c_meanstd range,
feature-column confounding, pooled OOF AUC, binomial assumptions, spectral
content, label grouping and selection bias. These corrections narrow claims;
they do not tune the simulator to recover a desired outcome.

## Remaining scientific limits

The simulator's fault shape is stipulated. The raw/conditioned comparison also
changes the speed column, so conditioning alone has not been isolated. Pooled
OOF probability scales can differ across models. Repeated seeds evaluate this
simulator, and neither the panel model nor its review flags certify deployment
performance. No real-data results were independently rerun.

## Numeric sources

- Reference results and fingerprints: the files named above, recomputed and compared.
- Repetition sizes: CLI arguments in the reproduction guide and result metadata.
- Test counts and compatibility versions: executed local test logs and environment metadata.
- Table audit count: the executed field-by-field comparison of the bilingual README and documentation.
- Hosted run status: GitHub Actions for the relevant commit.
