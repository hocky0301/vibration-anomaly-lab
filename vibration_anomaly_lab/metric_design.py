"""
metric_design.py -- how much does *your* leaderboard tell people about the answer?

Purpose
-------
This module is a tool for the people who **design** an evaluation system, not
for the people who compete in one.  Its job is to answer one question about a
scoring setup you are about to publish:

    "If I show a score for every submission, how many bits of the hidden
     ground truth does that score hand back, and what do my defences buy me?"

Everything here runs on the synthetic ``units`` of this repository.  No real
competition, dataset, score, or ground truth appears anywhere in this file,
and none is needed: the effect is a property of the *metric*, not of any
particular contest.

The structural fact
-------------------
This repository is about problems where the label is a property of the group
(here: the machine, ``unit_id``), not of the row.  In that world a sensible
submission is **constant within a group** -- every row of a unit gets the same
prediction, because that is what the problem actually asks for.

Write ``S`` for the set of groups the submission marks positive and ``T`` for
the set of groups that truly are positive.  Both are unions of whole groups, so
the row-wise F1 of the submission collapses to a closed algebraic form:

                       2 * rows(S and T)         2 * sum_{g in S and T} n_g
    F1(S, T)  =  -------------------------  =  -----------------------------
                    rows(S) + rows(T)          sum_{g in S} n_g + sum_{g in T} n_g

where ``n_g`` is the number of rows in group ``g``.  Nothing about the model,
the features, or the data survives into this expression.  The displayed score
is a function of two integers: a submission-controlled one and a truth-carrying
one.  A scorer that returns such a number is also a **query oracle** over the
hidden label vector, and a sequence of submissions is an adaptive search.

That is not an accusation against anyone.  It is a design constraint: if you
publish this metric on grouped data, you have published an oracle, and you
should know its bandwidth before you decide whether you care.

What this module measures
-------------------------
``leakage_budget`` simulates the honest worst case.  It enumerates every label
assignment consistent with what the leaderboard has displayed so far, and
reports how fast that candidate set shrinks as submissions are made.  The
headline number is *bits resolved*:

    bits_resolved = log2(|candidates before|) - log2(|candidates after|)

with ``log2(|C_0|) = n_groups`` bits when nothing about the positives is known.

The main act: mitigations
-------------------------
Three design levers are implemented as first-class functions, and the demo
measures what each one actually buys:

    ``mitigation_submission_limit``     cap the number of scored submissions
    ``mitigation_public_private_split`` score the display on a subset only
    ``mitigation_round_score``          round the displayed score

The point of the table at the bottom is that these three are *not*
interchangeable, and one of them has a failure mode that looks like a defence
and is not: splitting public/private **by row** leaves every group visible in
the public score and barely reduces the bandwidth at all, while splitting **by
group** hard-caps total leakage at the number of public groups, no matter how
many submissions are made.

Limits of this analysis (please read before quoting any number)
---------------------------------------------------------------
* The analytic bound concerns expected information in *the metric channel only*.
  A real evaluation leaks
  through other paths this file does not model (submission timing, correlated
  scores across metrics, public data statistics).
* The simulated attacker uses greedy one-step-lookahead query selection.
  Its expected information is achievable, not necessarily optimal.  Individual
  runs can exceed the entropy bound; ``leakage_bound_bits`` bounds expectation
  over the uniform prior, not each realized candidate reduction.
* Candidate spaces are enumerated exactly, so ``n_groups`` must stay small
  (<= 20).  This computational limit does not establish security at larger
  group counts; specialized strategies need not enumerate every assignment.
* The demo runs with a handful of groups.  Small-count results are an
  existence proof about the mechanism, not a performance guarantee about any
  particular contest.

Usage
-----
    python3 -m vibration_anomaly_lab.metric_design            # run the full demo
    python3 -m vibration_anomaly_lab.metric_design --seed 7
    python3 -m vibration_anomaly_lab.metric_design --nominal  # no data needed

Requires numpy + pandas.  Reads ``data/synthetic/train.csv`` (resolved relative
to this file; run ``make data`` first) and exits non-zero if it is absent.
``--nominal`` runs instead on a nominal layout of equally sized synthetic
groups, with a banner saying so.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Defaults for the demo.  Small on purpose: exact enumeration, no sampling.
# --------------------------------------------------------------------------- #

MAX_CANDIDATES = 1 << 20      # refuse to enumerate a bigger candidate space
QUERY_POOL = 64               # random submissions considered per greedy step
UNROUNDED_DIGITS = 12         # "unrounded" simulation compares 12-decimal buckets


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _rows(values: Sequence[int] | np.ndarray, size: int | None = None) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1 or not len(a) or (size is not None and len(a) != size):
        raise ValueError("row counts must be a nonempty vector aligned with groups")
    if a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError("row counts must be nonnegative integers")
    # All products below use masks; twice the total is the largest denominator.
    if sum(map(int, a)) > np.iinfo(np.int64).max // 2:
        raise ValueError("row counts exceed safe int64 arithmetic")
    return a.astype(np.int64)


def _binary(values: Sequence[int] | np.ndarray, size: int) -> np.ndarray:
    a = np.asarray(values)
    if a.shape != (size,) or not np.isin(a, [0, 1]).all():
        raise ValueError("labels and submissions must be aligned binary vectors")
    return a.astype(np.int8)


def _digits(displayed_digits: int | None) -> int:
    d = UNROUNDED_DIGITS if displayed_digits is None else _integer(
        displayed_digits, "displayed_digits")
    if d > UNROUNDED_DIGITS:
        raise ValueError(f"displayed_digits must be at most {UNROUNDED_DIGITS}")
    return d


# --------------------------------------------------------------------------- #
# 1. The closed form
# --------------------------------------------------------------------------- #

def row_f1_from_groups(
    submission: np.ndarray,
    truth: np.ndarray,
    rows_per_group: np.ndarray,
) -> float:
    """Row-wise F1 of a group-constant submission, computed from row counts only.

    ``submission`` and ``truth`` are 0/1 masks over groups; ``rows_per_group``
    holds the number of rows each group contributes to the scored set.  A group
    with zero scored rows is invisible to this score -- that is exactly how the
    public/private split is represented later on.

    Convention: an empty prediction scores 0.0, including the degenerate case
    where the truth is also empty (the positive-class F1 is undefined there, and
    0.0 matches what common implementations return).
    """
    n = _rows(rows_per_group)
    s = _binary(submission, len(n))
    t = _binary(truth, len(n))
    inter = int(np.sum(s * t * n))
    denom = int(np.sum(s * n) + np.sum(t * n))
    if denom == 0:
        return 0.0
    return 2.0 * inter / denom


def _row_f1_bruteforce(pred_rows: np.ndarray, true_rows: np.ndarray) -> float:
    """Ordinary row-level positive-class F1, computed the slow honest way."""
    tp = int(np.sum((pred_rows == 1) & (true_rows == 1)))
    fp = int(np.sum((pred_rows == 1) & (true_rows == 0)))
    fn = int(np.sum((pred_rows == 0) & (true_rows == 1)))
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2.0 * tp / (2 * tp + fp + fn)


def verify_closed_form(
    rows_per_group: np.ndarray,
    rng: np.random.Generator,
    n_trials: int = 200,
) -> float:
    """Check the algebra against a real row-by-row F1.  Returns max abs error.

    This is the load-bearing claim of the module, so it is measured rather than
    asserted in prose: expand every group into its actual rows, score a random
    group-constant submission the ordinary way, and compare with
    ``row_f1_from_groups``.
    """
    n = np.asarray(rows_per_group, dtype=np.int64)
    group_of_row = np.repeat(np.arange(len(n)), n)
    worst = 0.0
    for _ in range(n_trials):
        s = rng.integers(0, 2, size=len(n))
        t = rng.integers(0, 2, size=len(n))
        slow = _row_f1_bruteforce(s[group_of_row], t[group_of_row])
        fast = row_f1_from_groups(s, t, n)
        worst = max(worst, abs(slow - fast))
    return worst


# --------------------------------------------------------------------------- #
# 2. The three mitigations, as functions
# --------------------------------------------------------------------------- #

def mitigation_submission_limit(n_submissions: int, cap: int | None) -> int:
    """Cap the number of *scored* submissions.

    The cheapest lever and the weakest one on its own: it multiplies the
    per-submission bandwidth by a smaller number but does not change it.  Its
    value is that it composes -- combined with rounding it is what turns a
    linear search into an unfinished one.
    """
    n_submissions = _integer(n_submissions, "n_submissions")
    if cap is None:
        return n_submissions
    return min(n_submissions, _integer(cap, "cap"))


def mitigation_public_private_split(
    rows_per_group: np.ndarray,
    public_fraction: float,
    rng: np.random.Generator,
    by: str = "group",
) -> np.ndarray:
    """Return the rows each group contributes to the *displayed* score.

    ``by="group"``  whole groups are assigned to public or private.  Private
                    groups contribute zero scored rows, so their labels never
                    enter the displayed number and cannot be resolved by any
                    number of submissions.  This is the mitigation that
                    actually bounds total leakage on grouped data.

    ``by="row"``    a random fraction of the rows of *every* group is public.
                    This looks like the same defence and is not: every group
                    still moves the displayed score, so every group stays
                    identifiable.  The demo measures the difference.

    Returns a vector of public row counts, aligned with ``rows_per_group``.
    """
    n = _rows(rows_per_group)
    if by not in {"group", "row"}:
        raise ValueError(f"by must be 'group' or 'row', got {by!r}")
    if not np.isfinite(public_fraction) or not 0 <= public_fraction <= 1:
        raise ValueError("public_fraction must be finite and in [0, 1]")
    if public_fraction == 0:
        return np.zeros_like(n)
    if public_fraction == 1:
        return n.copy()
    if by == "group":
        k = max(1, round(public_fraction * len(n)))
        public = np.zeros(len(n), dtype=np.int64)
        chosen = rng.choice(len(n), size=k, replace=False)
        public[chosen] = n[chosen]
        return public
    if by == "row":
        # Independent Bernoulli row assignment, including possible empty groups.
        return rng.binomial(n, public_fraction).astype(np.int64)
    raise ValueError(f"by must be 'group' or 'row', got {by!r}")


def mitigation_round_score(scores: np.ndarray, displayed_digits: int | None) -> np.ndarray:
    """Quantize the displayed score, returning integer bucket ids.

    ``displayed_digits=None`` means "show the raw float"; it is represented as
    comparison after rounding to 12 decimal places. This is an approximation
    used for these small examples; it can merge sufficiently close scores.

    Rounding is the only one of the three levers that reduces the information
    *per* submission.  Its effect is bounded above by log2(10**d + 1) bits per
    submission, since a score in [0, 1] shown to d decimals has at most
    10**d + 1 distinguishable values.

    That bound is often slack, and the demo measures a case where it is very
    slack: when all groups have the same number of rows, the score can only take
    a handful of values anyway, so quantising it removes close to nothing.
    Rounding pays off where the row counts are distinct enough to make many
    scores reachable.  Do not assume it helps -- measure it.
    """
    d = _digits(displayed_digits)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite and in [0, 1]")
    return np.rint(scores * (10.0 ** d)).astype(np.int64)


# --------------------------------------------------------------------------- #
# 3. Candidate space and the analytic ceiling
# --------------------------------------------------------------------------- #

def enumerate_candidates(n_groups: int, known_n_positive: int | None = None) -> np.ndarray:
    """All label assignments an outside observer cannot yet rule out.

    ``known_n_positive`` models a design choice that is easy to overlook:
    publishing the number (or rate) of positives in the evaluation set shrinks
    the prior candidate space from 2**G to C(G, k) before a single submission is
    scored.  Leave it ``None`` for the conservative "nothing published" case.
    """
    n_groups = _integer(n_groups, "n_groups", minimum=1)
    if n_groups > 20:
        raise ValueError("exact enumeration supports at most 20 groups")
    if known_n_positive is None:
        size = 1 << n_groups
        if size > MAX_CANDIDATES:
            raise ValueError(
                f"2**{n_groups} candidates exceeds the exact-enumeration cap "
                f"({MAX_CANDIDATES}); this module is for small group counts."
            )
        idx = np.arange(size, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n_groups)[None, :]) & 1).astype(np.int8)
        return bits
    known_n_positive = _integer(known_n_positive, "known_n_positive")
    if known_n_positive > n_groups:
        raise ValueError("known_n_positive exceeds n_groups")
    size = math.comb(n_groups, known_n_positive)
    if size > MAX_CANDIDATES:
        raise ValueError("candidate space exceeds the exact-enumeration cap")
    out = np.zeros((size, n_groups), dtype=np.int8)
    for i, c in enumerate(itertools.combinations(range(n_groups), known_n_positive)):
        out[i, list(c)] = 1
    return out


def leakage_bound_bits(
    n_groups: int,
    displayed_digits: int | None,
    n_submissions: int,
    n_public_groups: int | None = None,
    known_n_positive: int | None = None,
) -> float:
    """Upper bound on EXPECTED bits resolved under a uniform candidate prior.

    Two independent caps, whichever binds first:

      * **channel cap** -- each submission returns one displayed value, so it
        carries at most log2(distinct displayable values) bits.  Rounding to
        ``d`` decimals gives at most 10**d + 1 of them.
      * **visibility cap** -- a group with no scored rows never moves the
        displayed number, so at most ``n_public_groups`` bits exist to be found
        in the first place, however many submissions are made.

    This entropy bound applies after averaging over every prior truth (and any
    independent query randomness).  It does NOT bound an individual run's
    log2(prior/posterior): a rare reply can remove far more candidates.  Even a
    finite Monte Carlo mean can exceed the expectation bound through sampling
    error.  A published positive count can also reveal private labels by
    inference; private bits are independent only without such side information.
    """
    n_groups = _integer(n_groups, "n_groups", minimum=1)
    n_submissions = _integer(n_submissions, "n_submissions")
    _digits(displayed_digits)
    if n_public_groups is not None:
        n_public_groups = _integer(n_public_groups, "n_public_groups")
        if n_public_groups > n_groups:
            raise ValueError("n_public_groups exceeds n_groups")
    if known_n_positive is not None:
        known_n_positive = _integer(known_n_positive, "known_n_positive")
        if known_n_positive > n_groups:
            raise ValueError("known_n_positive exceeds n_groups")
    if displayed_digits is None:
        per_query = float(n_groups)          # cannot exceed the whole secret
    else:
        per_query = math.log2(10 ** int(displayed_digits) + 1)
    visible = n_groups if n_public_groups is None else n_public_groups
    total_secret = float(visible)
    if known_n_positive is not None:
        # A published positive count shrinks the prior, but it cannot make
        # more entropy than the visible vector: the expectation cap still binds.
        total_secret = min(total_secret, math.log2(math.comb(n_groups, known_n_positive)))
    return float(min(total_secret, n_submissions * per_query))


# --------------------------------------------------------------------------- #
# 4. The simulation
# --------------------------------------------------------------------------- #

@dataclass
class LeakageResult:
    """Outcome of one adaptive querying run against one hidden truth."""
    n_groups: int
    n_public_groups: int
    displayed_digits: int | None
    submissions_allowed: int
    submissions_used: int
    prior_candidates: int
    posterior_candidates: int
    bits_resolved: float
    uniquely_identified: bool
    analytic_bound_bits: float
    trace: list[tuple[int, float, int]] = field(default_factory=list)
    #   trace entries: (submission index, displayed score, candidates remaining)

    @property
    def prior_bits(self) -> float:
        return math.log2(self.prior_candidates)

    @property
    def residual_bits(self) -> float:
        return math.log2(self.posterior_candidates)


def _scores_matrix(
    candidates: np.ndarray,
    queries: np.ndarray,
    public_rows: np.ndarray,
) -> np.ndarray:
    """F1 of every query against every candidate truth.  Shape (n_cand, n_query).

    Uses the closed form directly: with 0/1 masks, the intersection row count is
    a single matmul, ``C @ (public_rows * s)``.
    """
    n = public_rows.astype(np.float64)
    inter = candidates.astype(np.float64) @ (queries.astype(np.float64) * n).T
    q_rows = candidates.astype(np.float64) @ n            # rows(T) per candidate
    p_rows = queries.astype(np.float64) @ n               # rows(S) per query
    denom = p_rows[None, :] + q_rows[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-12), 0.0)
    return f1


def _partition_entropy(buckets: np.ndarray) -> float:
    """Entropy in bits of the partition a query induces on the candidate set."""
    _, counts = np.unique(buckets, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _propose_queries(n_groups: int, rng: np.random.Generator, pool: int) -> np.ndarray:
    """A pool of candidate submissions: every singleton, the full set, and noise.

    Singletons are included because they are the obvious baseline strategy;
    random subsets are included because they are usually much better.
    """
    singles = np.eye(n_groups, dtype=np.int8)
    full = np.ones((1, n_groups), dtype=np.int8)
    extra = max(0, pool - n_groups - 1)
    rand = rng.integers(0, 2, size=(extra, n_groups)).astype(np.int8)
    return np.vstack([singles, full, rand])


def leakage_budget(
    n_groups: int,
    rows_per_group: Sequence[int] | np.ndarray,
    displayed_digits: int | None = None,
    n_submissions: int = 10,
    truth: Sequence[int] | np.ndarray | None = None,
    public_rows_per_group: Sequence[int] | np.ndarray | None = None,
    known_n_positive: int | None = None,
    strategy: str = "greedy",
    rng: np.random.Generator | None = None,
    query_pool: int = QUERY_POOL,
) -> LeakageResult:
    """Simulate how fast the candidate space collapses under repeated scoring.

    Parameters
    ----------
    n_groups, rows_per_group
        The evaluation set: how many groups, and how many rows each contributes.
    displayed_digits
        Decimals shown on the leaderboard; ``None`` for an unrounded float.
    n_submissions
        How many scored submissions the observer is allowed.
    truth
        The hidden label vector to simulate against.  Random if omitted.
    public_rows_per_group
        Rows that count toward the *displayed* score, from
        ``mitigation_public_private_split``.  Defaults to all rows.
    known_n_positive
        Set it if the design publishes the number of positives.
    strategy
        ``"greedy"``   pick the submission that maximally splits the candidate
                       set (one-step lookahead over a random pool);
        ``"one_hot"``  probe one group per submission -- the naive baseline.

    Returns
    -------
    LeakageResult with the prior/posterior candidate counts, bits resolved, and
    a per-submission trace.  ``bits_resolved`` is the realized reduction for
    this strategy and truth.  ``analytic_bound_bits`` bounds its expectation
    under the uniform prior, not this individual value.  Greedy query selection
    optimizes expected one-step information gain, not every truth's outcome.
    """
    rng = np.random.default_rng() if rng is None else rng
    candidates = enumerate_candidates(n_groups, known_n_positive)
    n = _rows(rows_per_group, n_groups)
    n_submissions = _integer(n_submissions, "n_submissions")
    query_pool = _integer(query_pool, "query_pool", minimum=1)
    _digits(displayed_digits)
    if strategy not in {"greedy", "one_hot"}:
        raise ValueError("strategy must be greedy or one_hot")
    public = n.copy() if public_rows_per_group is None else _rows(
        public_rows_per_group, n_groups)
    if np.any(public > n):
        raise ValueError("public row counts cannot exceed total row counts")
    prior = len(candidates)

    if truth is None:
        truth = candidates[rng.integers(0, prior)]
    truth = _binary(truth, n_groups)
    if known_n_positive is not None and int(truth.sum()) != known_n_positive:
        raise ValueError("truth contradicts known_n_positive")

    pool = (np.eye(n_groups, dtype=np.int8) if strategy == "one_hot"
            else _propose_queries(n_groups, rng, query_pool))

    trace: list[tuple[int, float, int]] = []
    for i in range(n_submissions):
        if len(candidates) <= 1:
            break
        if strategy == "one_hot":
            if i >= n_groups:
                break
            query = pool[i]
        else:
            scores = _scores_matrix(candidates, pool, public)
            buckets = mitigation_round_score(scores, displayed_digits)
            gains = [_partition_entropy(buckets[:, j]) for j in range(pool.shape[0])]
            query = pool[int(np.argmax(gains))]

        observed = row_f1_from_groups(query, truth, public)
        obs_bucket = mitigation_round_score(np.array([observed]), displayed_digits)[0]

        cand_scores = _scores_matrix(candidates, query[None, :], public)[:, 0]
        keep = mitigation_round_score(cand_scores, displayed_digits) == obs_bucket
        candidates = candidates[keep]
        if len(candidates) == 0:
            raise RuntimeError("candidate elimination removed the supplied truth")
        displayed = obs_bucket / (10.0 ** _digits(displayed_digits))
        trace.append((i + 1, float(displayed), len(candidates)))

    posterior = len(candidates)
    n_public_groups = int(np.sum(public > 0))
    return LeakageResult(
        n_groups=n_groups,
        n_public_groups=n_public_groups,
        displayed_digits=displayed_digits,
        submissions_allowed=n_submissions,
        submissions_used=len(trace),
        prior_candidates=prior,
        posterior_candidates=posterior,
        bits_resolved=math.log2(prior) - math.log2(posterior),
        uniquely_identified=posterior == 1,
        # The expectation bound uses the allowed budget, not a stopping time
        # chosen from this truth's replies. It does not bound individual runs.
        analytic_bound_bits=leakage_bound_bits(
            n_groups, displayed_digits, n_submissions, n_public_groups, known_n_positive),
        trace=trace,
    )


def average_leakage(
    n_groups: int,
    rows_per_group: np.ndarray,
    n_truths: int,
    rng: np.random.Generator,
    **kwargs,
) -> dict[str, float]:
    """Average ``leakage_budget`` over random hidden truths.

    Averaging matters: a single run can look reassuring purely because that
    particular truth happened to be resolved late.
    """
    n_truths = _integer(n_truths, "n_truths", minimum=1)
    bits, unique, bound, residual = [], [], [], []
    for _ in range(n_truths):
        # The public split is redrawn per truth so the reported number is not an
        # artefact of one lucky assignment of groups to public/private.
        kw = dict(kwargs)
        maker = kw.pop("public_split_fn", None)
        if maker is not None:
            kw["public_rows_per_group"] = maker(rows_per_group, rng)
        res = leakage_budget(n_groups, rows_per_group, rng=rng, **kw)
        bits.append(res.bits_resolved)
        unique.append(float(res.uniquely_identified))
        bound.append(res.analytic_bound_bits)
        residual.append(float(res.posterior_candidates))
    return {
        "bits_resolved": float(np.mean(bits)),
        "frac_unique": float(np.mean(unique)),
        "bound_bits": float(np.mean(bound)),
        "median_residual_candidates": float(np.median(residual)),
    }


# --------------------------------------------------------------------------- #
# 5. Demo
# --------------------------------------------------------------------------- #

#: Nominal layout used only under ``--nominal``: ten equally sized groups of
#: 12 setpoints x 6 sweeps x 384 rows, mirroring the generator's defaults.
NOMINAL_N_GROUPS = 10
NOMINAL_ROWS_PER_GROUP = 12 * 6 * 384


def nominal_units() -> tuple[list[str], np.ndarray, None]:
    """The ``--nominal`` layout: equal-sized groups, no labels."""
    ids = [f"U{i:02d}" for i in range(1, NOMINAL_N_GROUPS + 1)]
    return ids, np.full(NOMINAL_N_GROUPS, NOMINAL_ROWS_PER_GROUP, dtype=np.int64), None


def _default_data_path() -> str:
    """``<repo>/data/synthetic/train.csv``, resolved from this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "synthetic", "train.csv")


def load_units(path: str) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """Group ids, row counts and (if present) labels from the synthetic CSV.

    Raises ``FileNotFoundError`` when the file is absent; the CLI turns that
    into a non-zero exit unless ``--nominal`` was passed explicitly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    import pandas as pd
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in ("unit_id", "label") if c in cols]
    df = pd.read_csv(path, usecols=use)
    from .features import unit_labels
    if "unit_id" not in df or df.empty or df["unit_id"].isna().any():
        raise ValueError("nonempty unit_id column without missing values required")
    sizes = df.groupby("unit_id", sort=True).size()
    ids = list(sizes.index)
    rows = sizes.to_numpy(dtype=np.int64)
    labels = None
    if "label" in df.columns:
        labels = unit_labels(df).reindex(ids).to_numpy(dtype=np.int8)
    return ids, rows, labels


def _demo_header(text: str) -> None:
    print("\n" + text)
    print("-" * max(62, len(text)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=_default_data_path(),
                    help="path to train.csv; default: <repo>/data/synthetic/train.csv")
    ap.add_argument("--nominal", action="store_true",
                    help="ignore --data and run on a nominal layout of "
                         f"{NOMINAL_N_GROUPS} equally sized synthetic groups")
    ap.add_argument("--truths", type=int, default=32,
                    help="random hidden truths to average each table row over")
    args = ap.parse_args()
    if args.truths < 1:
        ap.error("--truths must be positive")
    rng = np.random.default_rng(args.seed)

    if args.nominal:
        print("=" * 72)
        print("NOMINAL LAYOUT: --nominal was passed.  No data file is being read;")
        print(f"the groups below are {NOMINAL_N_GROUPS} equally sized synthetic")
        print("groups.  Numbers from this mode describe that layout only.")
        print("=" * 72)
        ids, rows, labels = nominal_units()
    else:
        try:
            ids, rows, labels = load_units(args.data)
        except FileNotFoundError:
            print(f"error: {args.data} not found -- run `make data` first, "
                  f"or pass --nominal to run on a nominal layout.",
                  file=sys.stderr)
            sys.exit(2)
    G = len(ids)

    print("evaluation set under audit (synthetic units from this repository)")
    print(f"  groups           : {G}")
    print(f"  rows per group   : min={rows.min()} max={rows.max()} total={rows.sum()}")
    print(f"  prior candidates : 2**{G} = {1 << G}  ({G} bits of hidden labels)")

    # -- 1. the algebra is real ------------------------------------------------
    _demo_header("1. the closed form, checked against row-by-row F1")
    err = verify_closed_form(rows, rng, n_trials=200)
    print("max |row-level F1  -  2*rows(S&T)/(rows(S)+rows(T))| over 200 random")
    print(f"group-constant submissions: {err:.2e}   ->  {'PASS' if err < 1e-12 else 'FAIL'}")
    print("The model never enters the expression.  The score is arithmetic on")
    print("row counts, which is what makes it usable as a query oracle.")

    # -- 2. one worked run, no mitigations -------------------------------------
    _demo_header("2. undefended scorer: unrounded score, unlimited submissions")
    truth = labels if labels is not None else rng.integers(0, 2, size=G).astype(np.int8)
    if labels is not None:
        print(f"hidden truth = the synthetic labels ({int(truth.sum())} positive of {G})")
    res = leakage_budget(G, rows, displayed_digits=None, n_submissions=G,
                         truth=truth, strategy="greedy", rng=rng)
    print(f"{'submission':>11}{'displayed F1':>15}{'candidates left':>18}{'bits resolved':>15}")
    for i, score, left in res.trace:
        print(f"{i:>11}{score:>15.6f}{left:>18}{math.log2(res.prior_candidates / left):>15.2f}")
    print(f"\nuniquely identified after {res.submissions_used} submissions: "
          f"{res.uniquely_identified}")
    print("Note the equal group sizes here: they are a real (accidental) defence,")
    print("because a score then reveals only *how many* groups were hit, not which.")

    # -- 3. group sizes are part of the design ---------------------------------
    _demo_header("3. why equal group sizes matter (a design lever, not a fix)")
    print("median submissions to pin the label vector down exactly "
          f"(censored at {G + 5})\n")
    print(f"{'row counts per group':<34}{'raw float':>12}{'2 digits':>12}{'1 digit':>12}")
    size_variants = {
        "equal (this repo's units)": rows,
        "mildly unequal (+/-30%)": (rows * rng.uniform(0.7, 1.3, size=G)).astype(np.int64),
        "distinct powers of two": (1000 * (2 ** np.arange(G))).astype(np.int64),
    }
    for name, variant in size_variants.items():
        cells = []
        for digits in (None, 2, 1):
            needed = []
            for _ in range(args.truths):
                r = leakage_budget(G, variant, displayed_digits=digits,
                                   n_submissions=G + 4, strategy="greedy", rng=rng)
                needed.append(r.submissions_used if r.uniquely_identified else G + 5)
            cells.append(np.median(needed))
        print(f"{name:<34}{cells[0]:>12.1f}{cells[1]:>12.1f}{cells[2]:>12.1f}")
    print("\nUnequal row counts can yield more distinct hit-row sums;")
    print("injectivity depends on the sizes. Equal sizes")
    print("leave only the count of hits and cost several.  Rounding is what pulls")
    print("those distinct sums back into the same displayed bucket, which is why it")
    print("interacts with group sizes rather than replacing them.")
    print("Caveat on the last row: heavily unequal groups do resist a coarse score,")
    print("but for a bad reason -- a 1-digit score barely moves when a small group")
    print("flips, i.e. the metric has stopped measuring those groups at all.  That")
    print("is an evaluation-validity problem, not a leakage defence.")

    # -- 4. the main act: the three mitigations --------------------------------
    _demo_header("4. what each mitigation actually buys")
    print(f"averaged over {args.truths} random hidden truths; "
          f"prior = {G} bits\n")
    print(f"{'evaluation design':<44}{'subs':>6}{'bits':>7}{'ceil':>7}"
          f"{'%uniq':>7}{'left':>7}")
    print("-" * 78)

    def row(name: str, **kw) -> None:
        st = average_leakage(G, rows, args.truths, rng, **kw)
        subs = kw.get("n_submissions", G)
        print(f"{name:<44}{subs:>6}{st['bits_resolved']:>7.2f}"
              f"{st['bound_bits']:>7.2f}{100 * st['frac_unique']:>7.0f}"
              f"{st['median_residual_candidates']:>7.0f}")

    row("no mitigation (raw float, many subs)",
        displayed_digits=None, n_submissions=2 * G)
    row("B. rounding only (2 digits, many subs)",
        displayed_digits=2, n_submissions=2 * G)
    row("B. rounding only (1 digit, many subs)",
        displayed_digits=1, n_submissions=2 * G)
    row("A. submission limit only (3 subs)",
        displayed_digits=None,
        n_submissions=mitigation_submission_limit(2 * G, 3))
    row("A+B. 3 subs AND 1-digit rounding",
        displayed_digits=1,
        n_submissions=mitigation_submission_limit(2 * G, 3))
    row("A. 3 subs, naive one-group-per-sub probe",
        displayed_digits=None, strategy="one_hot",
        n_submissions=mitigation_submission_limit(2 * G, 3))
    row("C. public/private split BY ROW (50%)",
        displayed_digits=None, n_submissions=2 * G,
        public_split_fn=lambda n, r: mitigation_public_private_split(n, 0.5, r, by="row"))
    row("C. public/private split BY GROUP (50%)",
        displayed_digits=None, n_submissions=2 * G,
        public_split_fn=lambda n, r: mitigation_public_private_split(n, 0.5, r, by="group"))
    row("A+B+C (3 subs, 1 digit, group split)",
        displayed_digits=1,
        n_submissions=mitigation_submission_limit(2 * G, 3),
        public_split_fn=lambda n, r: mitigation_public_private_split(n, 0.5, r, by="group"))
    row("published positive count, no other fix",
        displayed_digits=None, n_submissions=2 * G,
        known_n_positive=int(truth.sum()))

    print("\ncolumns: subs = submissions allowed; bits = simulated bits of the label")
    print("vector resolved for this strategy, averaged over sampled truths;")
    print("ceil = upper bound on EXPECTED information, not each run or sample mean;")
    print("%uniq = share of")
    print("hidden truths pinned down exactly; left = median candidates remaining.")

    # -- 5. reading of the table ----------------------------------------------
    _demo_header("5. what to take away, and what not to")
    print("* Row-wise F1 on group-constant submissions is arithmetic on row counts.")
    print("  If your labels are per-group, your leaderboard is a query oracle.")
    print("* Compare the three levers in the tables above; their effects depend on")
    print("  group sizes, prior information, precision and allowed submissions.")
    print("    - Rounding can matter little when equal groups already leave only a")
    print("      few reachable scores. Unequal groups can produce more score values.")
    print("    - A submission cap is the cheapest real reduction, but it only slows")
    print("      the search -- every extra scored submission keeps paying out.")
    print("    - Only a BY-GROUP public/private split removes bits from existence:")
    print("      without a known positive count, residual candidates are at least")
    print("      2**(private groups); side information can correlate private labels.")
    print("      Splitting BY ROW generally leaves")
    print("      every group moving the public score and defended nothing here.")
    print("* Publishing the positive count is a design decision with a measurable")
    print("  cost: it shrinks the prior before anyone submits anything.")
    print("* The ceiling bounds expected information under the stated uniform prior;")
    print("  it does not bound individual runs or finite-sample averages.")
    print("* Everything above is the metric channel alone, with a greedy attacker,")
    print("  on a handful of synthetic groups.  It demonstrates that the mechanism")
    print("  exists and that the levers differ; it is not a security guarantee, and")
    print("  the numbers do not transfer to any other evaluation set unchanged.")


if __name__ == "__main__":
    main()
