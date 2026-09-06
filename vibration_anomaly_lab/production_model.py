"""Educational panel with unit-level review flags and feature deviations.

This is an experimental decision-format demonstration, not a validated
production system. Each member is evaluated using leave-one-unit-out folds;
healthy reference statistics and supervised fits exclude the held-out unit.
The first member in a declared preference order whose OOF unit scores separate
the labels becomes the deciding member. A threshold is then selected using
those same OOF labels. BOTH choices are adaptive and their apparent performance
is optimistic, even when all observed units are separated.

The panel mixes a healthy-reference rank, logistic scores and boosted-tree
scores. These are not mutually calibrated probabilities. Their spread is a
heuristic reason to request review, not a confidence interval, and agreement
cannot certify safety. When no member separates the validation classes, every
unit is sent for review and any binary decision is provisional.

Feature evidence uses clipped robust z-scores; these describe deviations from
training reference features, not Gaussian tail probabilities or causal fault
explanations. Training/future-fleet stability, calibration, threshold transfer,
missed-fault costs and review workflow need independent validation before use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .conditioning import ConditionalRobustZ
from .features import trial_features, vibration_features
from .validation import (
    leave_one_unit_out,
    max_margin_threshold,
    unit_scores,
    unit_truth,
)

#: Spread across panel members (unit-level) above which a machine is sent to a
#: human instead of being decided automatically.
REVIEW_DISAGREEMENT = 0.40

#: Fallback decision threshold when the panel's leave-one-unit-out scores do not
#: separate the classes.
DEFAULT_THRESHOLD = 0.5


# ------------------------------------------------------------ panel members --
class RobustDeviation:
    """Healthy-reference member: deviations ranked against healthy trials.

    Scores a trial by the mean of its ``top_k`` largest absolute z-scores, then
    converts that to a probability-like number by its empirical rank among the
    healthy reference trials. Positive examples do not fit its score function,
    but panel selection and threshold fitting still use their OOF labels.
    The rank is not a calibrated probability of a fault.
    """

    name = "robust_deviation"

    def __init__(self, n_z: int, top_k: int = 3) -> None:
        if not isinstance(n_z, int) or n_z < 1 or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("n_z and top_k must be positive integers")
        self.n_z = n_z
        self.top_k = top_k
        self.reference_scores_: np.ndarray | None = None

    def _score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < self.n_z or not np.isfinite(X).all():
            raise ValueError("finite matrix with at least n_z columns required")
        A = np.abs(X[:, : self.n_z])
        k = min(self.top_k, A.shape[1])
        return np.sort(A, axis=1)[:, -k:].mean(axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RobustDeviation:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if (X.ndim != 2 or X.shape[1] < self.n_z or y.shape != (len(X),)
                or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all()):
            raise ValueError("finite feature matrix and aligned binary labels required")
        if not (y == 0).any():
            raise ValueError("RobustDeviation requires healthy reference trials")
        healthy = X[y == 0]
        self.reference_scores_ = np.sort(self._score(healthy))
        return self

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        if self.reference_scores_ is None:
            raise RuntimeError("call fit() first")
        r = np.searchsorted(self.reference_scores_, self._score(X), side="right")
        return r / (len(self.reference_scores_) + 1.0)


class LinearMember:
    """Strongly regularised logistic regression on the conditioned features."""

    name = "conditional_linear"

    def __init__(self, C: float = 0.005, seed: int = 0) -> None:
        self.C = C
        self.seed = seed
        self.model_: LogisticRegression | None = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearMember:
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        self.model_ = LogisticRegression(max_iter=5000, C=self.C, random_state=self.seed)
        self.model_.fit((X - self.mu_) / self.sd_, y)
        return self

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba((X - self.mu_) / self.sd_)[:, 1]


class GBMMember:
    """Gradient-boosted trees on the conditioned features.

    The most flexible member, and therefore the one most able to memorise two
    faulty machines.  Present for its inductive bias, not for its confidence.
    """

    name = "conditional_gbm"

    def __init__(self, max_iter: int = 300, seed: int = 0) -> None:
        self.max_iter = max_iter
        self.seed = seed
        self.model_: HistGradientBoostingClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> GBMMember:
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter, random_state=self.seed
        )
        self.model_.fit(X, y)
        return self

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1]


#: Declared in increasing order of capacity to memorise two positive examples:
#: a member that never sees a positive label, then a strongly regularised linear
#: model, then boosted trees.  Selection walks this list and stops at the first
#: eligible member, so the order *is* the preference for the humbler model.
MEMBER_ORDER = ("robust_deviation", "conditional_linear", "conditional_gbm")


def _build_members(n_z: int, seed: int = 0) -> list:
    return [RobustDeviation(n_z), LinearMember(seed=seed), GBMMember(seed=seed)]


# ------------------------------------------------------------------- output --
@dataclass
class Evidence:
    """One feature's deviation from healthy machines at the same operating point."""

    feature: str
    sigma: float          # signed median robust z across the unit's trials
    worst_setpoint: float # operating point where the deviation is largest
    worst_sigma: float

    def describe(self) -> str:
        d = "above" if self.sigma > 0 else "below"
        return (f"{self.feature} {abs(self.sigma):.1f} sigma {d} healthy "
                f"(worst at setpoint {self.worst_setpoint:g}: "
                f"{self.worst_sigma:+.1f} sigma)")


@dataclass
class UnitDecision:
    """Provisional unit result; probability is an uncalibrated score.

    A true review flag requests abstention from automatic use of decision.
    """

    unit_id: str
    probability: float
    decision: int
    review: bool
    disagreement: float
    members: dict[str, float]
    evidence: list[Evidence] = field(default_factory=list)
    n_trials: int = 0
    review_reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        tag = "FAULTY" if self.decision else "healthy"
        rv = ("  [REVIEW: " + ", ".join(self.review_reasons or ["panel split"]) + "]"
              if self.review else "")
        ev = "; ".join(e.describe() for e in self.evidence)
        mem = ", ".join(f"{k}={v:.2f}" for k, v in self.members.items())
        return (f"{self.unit_id}: p={self.probability:.2f} -> {tag}{rv}\n"
                f"    panel: {mem}  (spread {self.disagreement:.2f})\n"
                f"    why:   {ev}")


# ---------------------------------------------------------------- the model --
class ProductionModel:
    """Educational panel of three with review flags and robust-z evidence.

    Usage::

        m = ProductionModel().fit(trial_features(train_df))
        for d in m.predict_units(trial_features(new_df)):
            print(d.describe())

    ``fit`` requires at least **two** faulty units in the training fleet and
    raises ``ValueError`` otherwise.  The veto in stage 2 compares every
    held-out faulty machine against every held-out healthy one; with a single
    faulty machine the leave-one-unit-out fold that holds it out has no
    positive left to train the supervised members on, and "separates" would be
    a statement about one example.  A fleet with one labelled fault is a
    fleet for the unsupervised member alone (:class:`RobustDeviation` on
    :class:`ConditionalRobustZ` features), not for this selection rule.
    """

    def __init__(
        self,
        review_disagreement: float = REVIEW_DISAGREEMENT,
        seed: int = 0,
        aggregate: str = "median",
    ) -> None:
        if not np.isfinite(review_disagreement) or not 0 <= review_disagreement <= 1:
            raise ValueError("review_disagreement must be finite and in [0, 1]")
        if aggregate not in {"median", "mean", "max"}:
            raise ValueError("aggregate must be median, mean, or max")
        self.review_disagreement = review_disagreement
        self.seed = seed
        self.aggregate = aggregate
        self.features_: list[str] = []
        self.reference_: ConditionalRobustZ | None = None
        self.members_: list = []
        self.primary_: str = ""
        self.threshold_: float = DEFAULT_THRESHOLD
        self.threshold_is_optimistic_: bool = False
        self.no_member_separates_: bool = False
        self.selection_: pd.DataFrame | None = None
        self.oof_units_: pd.DataFrame | None = None

    # ---------------------------------------------------------------- fit --
    def fit(self, F: pd.DataFrame) -> ProductionModel:
        """Fit the panel, choose the deciding member, choose the threshold.

        Three stages, in this order, because each depends on the previous:

        1. Leave-one-unit-out over the training machines, every panel member
           scored on machines it has never seen.  The reference distribution for
           each fold is rebuilt from that fold's healthy training units only, so
           the held-out machine influences neither the model nor the scale it is
           measured against.
        2. Veto, then choose.  Members whose held-out faulty machines do not sit
           strictly above every held-out healthy machine are dropped: two
           positives are a limited selection sample.  Of the survivors,
           the first in :data:`MEMBER_ORDER` -- the least able to have memorised
           two examples -- decides.  If nothing survives the veto, the humblest
           member supplies a provisional score and every output requires human
           review.  That is an abstention from automatic use, not validation.
        3. Choose the threshold from the deciding member's held-out unit scores,
           midway between the classes.  This is less brittle than sitting on the
           edge of separation, but it is not unbiased: the same ten machines
           supplied the scores and the labels.  ``threshold_is_optimistic_``
           is always true after fitting; ``no_member_separates_`` separately
           records when even the midpoint was unavailable.

        Then every member is refitted on all the training machines.
        """
        if "label" not in F.columns:
            raise ValueError("fit() needs labelled trial features")
        truth = unit_truth(F)
        n_faulty = int((truth == 1).sum())
        n_healthy = int((truth == 0).sum())
        if n_healthy < 2:
            raise ValueError("ProductionModel.fit needs at least 2 healthy units for fold references")
        if n_faulty < 2:
            raise ValueError(
                f"ProductionModel.fit needs at least 2 faulty units in the "
                f"training fleet, found {n_faulty}: the panel veto compares "
                f"held-out faulty machines against held-out healthy ones, and "
                f"with fewer than two faulty machines every leave-one-unit-out "
                f"fold that holds a faulty machine out has no positive left to "
                f"train on.  Use RobustDeviation on ConditionalRobustZ features "
                f"directly for a fleet with a single labelled fault."
            )
        self.features_ = vibration_features(F)
        y = F["label"].to_numpy()

        # --- 1. leave-one-unit-out over the training fleet -------------------
        n_z = len(self.features_)
        oof = {name: np.zeros(len(F)) for name in MEMBER_ORDER}
        for tr, va in leave_one_unit_out(F):
            tr_rows = F.iloc[tr]
            ref = ConditionalRobustZ(self.features_).fit(tr_rows[tr_rows["label"] == 0])
            Xtr, Xva = ref.transform(tr_rows), ref.transform(F.iloc[va])
            for m in _build_members(n_z, self.seed):
                m.fit(Xtr, y[tr])
                oof[m.name][va] = m.predict_proba1(Xva)

        truth = unit_truth(F)
        oof_units = pd.DataFrame(
            {name: unit_scores(F, p, how=self.aggregate) for name, p in oof.items()}
        )
        oof_units["truth"] = truth
        self.oof_units_ = oof_units

        # --- 2. veto on the two positives, then take the humblest survivor ---
        rows = []
        for rank, name in enumerate(MEMBER_ORDER):
            col = oof_units[name]
            healthy, faulty = col[truth == 0], col[truth == 1]
            margin = float(faulty.min() - healthy.max())
            rows.append(
                {
                    "member": name,
                    "capacity_rank": rank,
                    "separates": bool(margin > 0.0),
                    # Diagnostics.  Reported, never ranked on: with two positives
                    # a margin is an anecdote and a recall is a coin flip.
                    "margin_n2": round(margin, 3),
                    "specificity_at_0.5": float((healthy <= DEFAULT_THRESHOLD).mean()),
                    "recall_n2": float((faulty > DEFAULT_THRESHOLD).mean()),
                }
            )
        sel = pd.DataFrame(rows)
        self.selection_ = sel
        eligible = sel[sel["separates"]]
        self.no_member_separates_ = bool(eligible.empty)
        chosen = (sel if eligible.empty else eligible).iloc[0]
        self.primary_ = str(chosen["member"])

        # --- 3. threshold from the deciding member's held-out unit scores ----
        prim = oof_units[self.primary_]
        self.threshold_ = max_margin_threshold(prim, truth, default=DEFAULT_THRESHOLD)
        # Both member selection and threshold selection reuse OOF labels.
        # Separation does not turn their evaluation into an unbiased estimate.
        self.threshold_is_optimistic_ = True

        # --- refit on the whole training fleet -------------------------------
        self.reference_ = ConditionalRobustZ(self.features_).fit(F[F["label"] == 0])
        X = self.reference_.transform(F)
        self.members_ = [m.fit(X, y) for m in _build_members(n_z, self.seed)]
        return self

    # ------------------------------------------------------------ predict --
    def predict_trials(self, F: pd.DataFrame) -> pd.DataFrame:
        """Per-trial uncalibrated score from every panel member.

        The reference distribution is the one fitted at training time: these
        machines are scored against healthy *training* machines at the same
        operating point, never against each other.  A batch of new machines that
        happened to be mostly faulty would otherwise normalise its own fault
        away.
        """
        if self.reference_ is None:
            raise RuntimeError("call fit() first")
        X = self.reference_.transform(F)
        out = pd.DataFrame({"unit_id": F["unit_id"].to_numpy()}, index=F.index)
        for m in self.members_:
            out[m.name] = m.predict_proba1(X)
        return out

    def predict_units(self, F: pd.DataFrame) -> list[UnitDecision]:
        """One decision per machine, with panel spread and sigma-level evidence."""
        trials = self.predict_trials(F)
        per_unit = trials.groupby("unit_id")[list(MEMBER_ORDER)].agg(self.aggregate)
        counts = F.groupby("unit_id").size()

        decisions = []
        for unit_id, row in per_unit.iterrows():
            members = {k: float(row[k]) for k in MEMBER_ORDER}
            p = members[self.primary_]
            spread = float(max(members.values()) - min(members.values()))
            decisions.append(
                UnitDecision(
                    unit_id=str(unit_id),
                    probability=p,
                    decision=int(p > self.threshold_),
                    review=bool(self.no_member_separates_ or spread > self.review_disagreement),
                    review_reasons=(
                        (["no member separates validation classes"] if self.no_member_separates_ else [])
                        + (["panel split"] if spread > self.review_disagreement else [])
                    ),
                    disagreement=spread,
                    members=members,
                    evidence=self.explain_unit(F[F["unit_id"] == unit_id]),
                    n_trials=int(counts[unit_id]),
                )
            )
        return sorted(decisions, key=lambda d: -d.probability)

    # ------------------------------------------------------------ explain --
    def explain_unit(self, F_unit: pd.DataFrame, top: int = 3) -> list[Evidence]:
        """Why this machine looks the way it does, in robust sigma.

        For each feature, the median signed z across the machine's trials -- how
        far it sits from healthy training machines *at the same operating
        point*, in units of the healthy spread there.  The largest single
        deviation and the setpoint at which it occurs are reported alongside,
        because a fault that lives in one part of the sweep is a different
        physical story from one that is present throughout, and the person
        reading this is the one who can tell which.
        """
        if self.reference_ is None:
            raise RuntimeError("call fit() first")
        Z = self.reference_.z_frame(F_unit)
        med = Z.median(axis=0)
        order = med.abs().sort_values(ascending=False).index[:top]
        sp = F_unit["setpoint"].to_numpy()
        out = []
        for f in order:
            col = Z[f].to_numpy()
            j = int(np.argmax(np.abs(col)))
            out.append(
                Evidence(
                    feature=str(f),
                    sigma=float(med[f]),
                    worst_setpoint=float(sp[j]),
                    worst_sigma=float(col[j]),
                )
            )
        return out

    # ------------------------------------------------------------- report --
    def summary(self) -> str:
        """Human-readable account of what fit() decided, and on what evidence."""
        assert self.selection_ is not None and self.oof_units_ is not None
        n_pos = int((self.oof_units_["truth"] == 1).sum())
        lines = [
            "panel selection (leave-one-unit-out over the training fleet)",
            self.selection_.to_string(index=False),
            "",
            (f"deciding member : {self.primary_}"
             "   (humblest member that separates the held-out classes)"),
            f"threshold       : {self.threshold_:.3f}"
            + ("   [NO MEMBER SEPARATES -- do not deploy]"
               if self.no_member_separates_ else "   [midpoint; selected using OOF labels]"),
            "selection and threshold evaluation are optimistic on these same labels",
            (f"review rule     : panel spread > {self.review_disagreement:.2f}"
             " -> human review"),
            "",
            f"NOTE: margin_n2 and recall_n2 rest on n={n_pos} faulty machines.  They",
            "      are noisy diagnostics used for an adaptive selection veto.  A run that",
            "      separates is an existence proof, not a performance estimate.",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------- demo --
def _demo(data_dir: str) -> None:
    """Fit on the labelled fleet, then decide on machines never seen before."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    F = trial_features(train)
    model = ProductionModel().fit(F)
    print(model.summary())

    holdout_path = os.path.join(data_dir, "holdout.csv")
    if not os.path.exists(holdout_path):
        return
    G = trial_features(pd.read_csv(holdout_path))
    print(f"\nunseen machines ({G['unit_id'].nunique()} units, {len(G)} trials)")
    print("-" * 72)
    for d in model.predict_units(G):
        print(d.describe())
    print(
        "\nThis is a demonstration of the output format on machines the model has\n"
        "never seen.  The holdout contains a single-digit number of machines, so\n"
        "it can show that the pipeline runs end to end and cannot show how well\n"
        "it performs."
    )


def _default_data_dir() -> str:
    """``<repo>/data/synthetic``, resolved from this file, not from the cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data", "synthetic")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data",
        default=_default_data_dir(),
        help=("directory holding train.csv (and optionally holdout.csv); "
              "default: <repo>/data/synthetic"),
    )
    args = ap.parse_args(argv)
    train_path = os.path.join(args.data, "train.csv")
    if not os.path.exists(train_path):
        msg = (
            f"error: {train_path} not found -- run `make data` first "
            "(or `python3 data/make_synthetic_data.py`)."
        )
        print(msg, file=sys.stderr)
        return 2
    _demo(args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
