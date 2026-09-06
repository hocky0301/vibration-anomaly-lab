"""The closed-form row F1 and the mitigations in metric_design."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from vibration_anomaly_lab.metric_design import (
    enumerate_candidates,
    leakage_bound_bits,
    leakage_budget,
    mitigation_public_private_split,
    row_f1_from_groups,
)


def test_row_f1_from_groups_matches_sklearn_on_unequal_groups() -> None:
    rng = np.random.default_rng(123)
    for _ in range(50):
        G = int(rng.integers(3, 12))
        rows = rng.integers(1, 40, size=G)
        group_of_row = np.repeat(np.arange(G), rows)
        S = rng.integers(0, 2, size=G)
        T = rng.integers(0, 2, size=G)
        fast = row_f1_from_groups(S, T, rows)
        slow = f1_score(T[group_of_row], S[group_of_row], zero_division=0)
        assert fast == pytest.approx(float(slow), abs=1e-12)


def test_fixed_submission_has_at_most_three_outcomes_over_two_of_ten_truths() -> None:
    G = 10
    rows = np.full(G, 2304, dtype=np.int64)          # equal group sizes
    truths = enumerate_candidates(G, known_n_positive=2)
    assert len(truths) == 45
    rng = np.random.default_rng(7)
    for _ in range(30):
        S = rng.integers(0, 2, size=G)
        outcomes = {round(row_f1_from_groups(S, T, rows), 12) for T in truths}
        assert len(outcomes) <= 3


def test_by_group_split_leaves_private_groups_unresolved() -> None:
    G = 8
    rows = 1000 * (2 ** np.arange(G))                # distinct sizes: maximally leaky
    rng = np.random.default_rng(11)
    for _ in range(5):
        public = mitigation_public_private_split(rows, 0.5, rng, by="group")
        n_private = int(np.sum(public == 0))
        assert n_private >= 1
        res = leakage_budget(
            G, rows, displayed_digits=None, n_submissions=4 * G,
            public_rows_per_group=public, strategy="greedy", rng=rng,
        )
        assert res.n_public_groups == G - n_private
        # Private groups never move the displayed score: every assignment of
        # their labels survives, whatever the submissions were.
        assert res.posterior_candidates >= 2 ** n_private
        assert res.posterior_candidates % (2 ** n_private) == 0
        assert res.analytic_bound_bits <= G - n_private


def test_no_mitigation_on_distinct_sizes_is_fully_resolved() -> None:
    G = 8
    rows = 1000 * (2 ** np.arange(G))
    rng = np.random.default_rng(3)
    res = leakage_budget(G, rows, displayed_digits=None, n_submissions=4 * G,
                         strategy="greedy", rng=rng)
    assert res.prior_candidates == 2 ** G
    assert res.posterior_candidates == 1
    assert res.uniquely_identified


def test_leakage_bound_never_exceeds_visibility_cap() -> None:
    """A published positive count shrinks the prior but cannot reveal invisible groups."""
    without = leakage_bound_bits(10, None, 20, n_public_groups=5)
    with_count = leakage_bound_bits(10, None, 20, n_public_groups=5, known_n_positive=2)
    assert without == pytest.approx(5.0)
    assert with_count <= without
    # With every group visible the count does bind: log2(C(10, 2)) < 10.
    assert leakage_bound_bits(10, None, 20, known_n_positive=2) == pytest.approx(5.4918530963)
