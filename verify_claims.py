#!/usr/bin/env python3
"""
verify_claims.py -- re-derive the published protocol metrics from scratch.

Run after ``python3 data/make_synthetic_data.py``.  Nothing here is cached or
hard-coded: every cell of the results table is produced by actually fitting the
models on the CSV you point it at, and ``--json`` writes the document that
results/default_draw.json and the seed sweep (scripts/sweep_seeds.py) are built
from. Seed-sweep summaries, plots and interval calculations have separate
reproduction commands; this entry point covers one generator draw.

The script is deliberately standalone (numpy, pandas, scikit-learn only; it
imports nothing from the package) so the claims can be audited without trusting
the library.  Featurisation and conditioning are meant to be identical to the
package's; tests/ asserts that they agree.

Protocols  (split / feature space / model)
    a          random 5-fold over trials [LEAKY] / raw stats + operating point / GBM
    b          leave-one-unit-out (LOUO)         / raw stats + operating point / GBM
    b_linear   LOUO / raw stats + operating point                      / logistic
    b_noctx    LOUO / raw stats only, no operating-point columns        / GBM
    c          LOUO / setpoint-conditional robust-z (median/MAD)        / GBM
    c_meanstd  LOUO / setpoint-conditional z (mean/std)                 / GBM
    d          LOUO / setpoint-conditional robust-z (median/MAD)        / logistic
    e          LOUO / unit-relative log amplitude (label-free)          / GBM
    e_linear   LOUO / unit-relative log amplitude                       / logistic

In c / c_meanstd / d the reference distribution of each fold is built from the
training-fold trials whose label is 0 -- never from the held-out unit.

Usage
    python3 verify_claims.py                       # table on data/synthetic/train.csv
    python3 verify_claims.py --json results/default_draw.json
    python3 verify_claims.py --data other.csv --protocols a,b,c --quick
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score

AXES = ("vib_x", "vib_y", "vib_z")
STATS = ("rms", "std", "mad", "iqr", "p90", "absmean", "ptp")
CONTEXT = ("setpoint", "speed_rpm")      # operating-point columns of the raw space
CLIP = 8.0                               # symmetric clip on z-columns
MAD_TO_SIGMA = 1.4826
GRID = np.linspace(0.05, 0.95, 37)       # threshold grid for the optimistic "best" F1
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "synthetic", "train.csv")

# id: (label, split, space, model)
PROTOCOLS = {
    "a":         ("random K-fold over trials [LEAKY]", "kfold", "raw", "gbm"),
    "b":         ("leave-one-unit-out, raw", "louo", "raw", "gbm"),
    "b_linear":  ("leave-one-unit-out, raw, linear", "louo", "raw", "linear"),
    "b_noctx":   ("leave-one-unit-out, raw without operating point", "louo", "raw_noctx", "gbm"),
    "c":         ("leave-one-unit-out, conditional robust-z", "louo", "condz_mad", "gbm"),
    "c_meanstd": ("leave-one-unit-out, conditional z (mean/std)", "louo", "condz_std", "gbm"),
    "d":         ("leave-one-unit-out, conditional robust-z, linear", "louo", "condz_mad", "linear"),
    "e":         ("leave-one-unit-out, unit-relative", "louo", "unit_relative", "gbm"),
    "e_linear":  ("leave-one-unit-out, unit-relative, linear", "louo", "unit_relative", "linear"),
}
SPLIT_NAMES = {"kfold": "StratifiedKFold(5) over trials", "louo": "leave-one-unit-out"}
SPACE_NAMES = {"raw": "raw stats + setpoint + speed_rpm", "raw_noctx": "raw stats only",
               "condz_mad": "conditional robust-z (median/MAD) + setpoint",
               "condz_std": "conditional z (mean/std) + setpoint",
               "unit_relative": "unit-relative log stats + setpoint"}
MODEL_NAMES = {"gbm": "HistGradientBoosting", "linear": "logistic regression"}


# ------------------------------------------------------------------ features --
def trial_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trial: trial_id, unit_id, [label,] setpoint, speed_rpm, 21 stats."""
    has_label = "label" in df.columns
    rows = []
    for trial_id, g in df.groupby("trial_id", sort=False):
        rec = {"trial_id": trial_id, "unit_id": g["unit_id"].iloc[0]}
        if has_label:
            rec["label"] = int(g["label"].iloc[0])
        rec["setpoint"] = float(g["setpoint"].iloc[0])
        rec["speed_rpm"] = float(g["speed_rpm"].mean())
        for a in AXES:
            v = g[a].to_numpy()
            m = np.median(v)
            rec[f"{a}_rms"] = float(np.sqrt((v ** 2).mean()))
            rec[f"{a}_std"] = float(v.std())
            rec[f"{a}_mad"] = float(np.median(np.abs(v - m)))
            rec[f"{a}_iqr"] = float(np.subtract(*np.percentile(v, [75, 25])))
            rec[f"{a}_p90"] = float(np.percentile(np.abs(v), 90))
            rec[f"{a}_absmean"] = float(np.abs(v).mean())
            rec[f"{a}_ptp"] = float(v.max() - v.min())
        rows.append(rec)
    return pd.DataFrame(rows)


def vibration_columns(F: pd.DataFrame) -> list[str]:
    """The 21 amplitude statistics -- the columns that get conditioned."""
    return [c for c in F.columns if any(c.startswith(a + "_") for a in AXES)]


# -------------------------------------------------------------- conditioning --
def conditional_robust_z(F: pd.DataFrame, feats: list[str], ref_mask: np.ndarray,
                         scale: str = "mad") -> np.ndarray:
    """z-scores of ``feats`` against the reference rows at the same setpoint.

    Reference = rows where ``ref_mask`` is True (callers pass training-fold
    healthy rows).  Bins are the setpoint rounded to 3 decimals; a setpoint
    absent from the reference falls back to the global reference statistics.
    ``scale='mad'`` uses median / 1.4826*MAD, ``scale='std'`` mean / std.
    Output: clipped z-columns followed by the setpoint (speed_rpm is left out:
    it carries a per-unit gain, i.e. a fingerprint).
    """
    if scale not in ("mad", "std"):
        raise ValueError(f"scale must be 'mad' or 'std', got {scale!r}")
    R = F.loc[np.asarray(ref_mask, dtype=bool)]
    if len(R) == 0:
        raise ValueError("empty reference: no healthy training trials given")
    b, rb = F["setpoint"].round(3), R["setpoint"].round(3)
    if scale == "mad":
        centre = R.groupby(rb)[feats].median()
        spread = R.groupby(rb)[feats].apply(lambda g: (g - g.median()).abs().median()) * MAD_TO_SIGMA
        g_centre = R[feats].median()
        g_spread = (R[feats] - g_centre).abs().median() * MAD_TO_SIGMA
    else:
        centre, spread = R.groupby(rb)[feats].mean(), R.groupby(rb)[feats].std()
        g_centre, g_spread = R[feats].mean(), R[feats].std()
    M = centre.reindex(b).fillna(g_centre).to_numpy()
    D = spread.reindex(b).fillna(g_spread).to_numpy()
    Z = np.clip((F[feats].to_numpy() - M) / (D + 1e-9), -CLIP, CLIP)
    return np.column_stack([Z, F[["setpoint"]].to_numpy()])


def unit_relative(F: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """log(stat) minus the unit's own median log(stat), then the setpoint.

    Label-free and reference-free, but transductive: it needs the whole sweep of
    the unit being scored.  What survives is the *shape* of the sweep response.
    """
    L = np.log(F[feats].to_numpy() + 1e-12)
    units = F["unit_id"].to_numpy()
    for u in np.unique(units):
        m = units == u
        L[m] -= np.median(L[m], axis=0)
    return np.column_stack([L, F[["setpoint"]].to_numpy()])


# ------------------------------------------------------------------- scoring --
def _unit_truth(F: pd.DataFrame) -> pd.Series:
    return F.groupby("unit_id")["label"].first()


def _unit_scores(F: pd.DataFrame, p: np.ndarray, subset: np.ndarray | None = None) -> pd.Series:
    s = pd.DataFrame({"unit_id": F["unit_id"].to_numpy(), "p": p})
    if subset is not None:
        s = s[subset]
    return s.groupby("unit_id")["p"].median()


def _unit_f1(F: pd.DataFrame, p: np.ndarray, t: float, subset: np.ndarray | None = None) -> float:
    """Unit-level F1 at threshold t; a unit with no trials in ``subset`` counts as healthy."""
    truth = _unit_truth(F)
    pred = (_unit_scores(F, p, subset) > t).astype(int).reindex(truth.index).fillna(0).astype(int)
    return float(f1_score(truth.to_numpy(), pred.to_numpy(), zero_division=0))


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def _summarize(F: pd.DataFrame, p: np.ndarray) -> dict:
    y = F["label"].to_numpy()
    truth = _unit_truth(F)
    u = _unit_scores(F, p).reindex(truth.index)
    low = (F["setpoint"] <= F["setpoint"].median()).to_numpy()
    best = max(((_unit_f1(F, p, float(t)), float(t)) for t in GRID), key=lambda v: v[0])
    r = lambda v: None if v is None else round(v, 6)
    return {
        "trial_f1_at_0_5": r(float(f1_score(y, (p > 0.5).astype(int), zero_division=0))),
        "trial_auc": r(_auc(y, p)),
        "unit_f1_at_0_5": r(_unit_f1(F, p, 0.5)),
        "unit_auc": r(_auc(truth.to_numpy(), u.to_numpy())),
        "unit_f1_lowhalf_at_0_5": r(_unit_f1(F, p, 0.5, low)),
        "unit_f1_best": r(best[0]),
        "unit_f1_best_threshold": r(best[1]),
        "unit_scores": {str(k): round(float(v), 6) for k, v in u.items()},
    }


# ----------------------------------------------------------------- protocols --
def _design(F: pd.DataFrame, space: str, feats: list[str], y: np.ndarray, tr: np.ndarray) -> np.ndarray:
    if space == "raw":
        return F[feats + list(CONTEXT)].to_numpy()
    if space == "raw_noctx":
        return F[feats].to_numpy()
    if space == "unit_relative":
        return unit_relative(F, feats)
    ref = np.zeros(len(F), dtype=bool)
    ref[tr] = y[tr] == 0                      # training-fold healthy rows only
    return conditional_robust_z(F, feats, ref, scale="mad" if space == "condz_mad" else "std")


def _fit_predict(X, y, tr, va, model: str, quick: bool, seed: int) -> np.ndarray:
    if len(np.unique(y[tr])) < 2:
        # Holding out the only faulty unit leaves a one-class training fold (tiny
        # datasets); a model that has never seen a fault predicts the prior.
        return np.full(len(va), float(y[tr].mean()))
    if model == "linear":
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=0.005, max_iter=5000).fit((X[tr] - mu) / sd, y[tr])
        return m.predict_proba((X[va] - mu) / sd)[:, 1]
    m = HistGradientBoostingClassifier(max_iter=60 if quick else 300, random_state=seed)
    return m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]


def run_protocol(F: pd.DataFrame, protocol_id: str, quick: bool = False, model_seed: int = 0) -> dict:
    """Out-of-fold trial probabilities for one protocol, summarised (see _summarize)."""
    label, split, space, model = PROTOCOLS[protocol_id]
    y, groups, feats = F["label"].to_numpy(), F["unit_id"].to_numpy(), vibration_columns(F)
    folds = (StratifiedKFold(5, shuffle=True, random_state=model_seed).split(F, y) if split == "kfold"
             else LeaveOneGroupOut().split(F, y, groups))
    p = np.zeros(len(y))
    for tr, va in folds:
        X = _design(F, space, feats, y, tr)
        p[va] = _fit_predict(X, y, tr, va, model, quick, model_seed)
    out = {"label": label, "split": SPLIT_NAMES[split], "space": SPACE_NAMES[space],
           "model": MODEL_NAMES[model]}
    out.update(_summarize(F, p))
    return out


def unit_identifiability(F: pd.DataFrame, quick: bool = False, model_seed: int = 0) -> float:
    """5-fold accuracy of a GBM predicting unit_id from raw features (chance = 1/n_units)."""
    X = F[vibration_columns(F) + list(CONTEXT)].to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=60 if quick else 100, random_state=model_seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=model_seed)
    return round(float(cross_val_score(clf, X, F["unit_id"].to_numpy(), cv=cv).mean()), 6)


def run_all(F: pd.DataFrame, quick: bool = False, model_seed: int = 0,
            protocols: list[str] | None = None) -> dict:
    ids = protocols or list(PROTOCOLS)
    return {
        "n_units": int(F["unit_id"].nunique()),
        "n_trials": int(len(F)),
        "faulty_units": sorted(str(u) for u in F.loc[F["label"] == 1, "unit_id"].unique()),
        "unit_identifiability_acc": unit_identifiability(F, quick, model_seed),
        "protocols": {pid: run_protocol(F, pid, quick, model_seed) for pid in ids},
        "quick": bool(quick),
        "model_seed": int(model_seed),
    }


# ------------------------------------------------------------------- console --
def _fmt(v) -> str:
    return "   n/a" if v is None else f"{v:6.3f}"


def print_report(doc: dict) -> None:
    P = doc["protocols"]
    print(f"units={doc['n_units']}  trials={doc['n_trials']}  faulty units={doc['faulty_units']}  "
          f"=> effective positive sample size n={len(doc['faulty_units'])}"
          + ("   [--quick]" if doc["quick"] else "") + "\n")
    head = (f"{'id':<10}{'split / space / model':<46}{'trial F1@.5':>12}{'unit F1@.5':>12}"
            f"{'unit AUC':>10}{'unit F1 lowhalf@.5':>20}{'best (t)':>14}")
    print(head)
    print("-" * len(head))
    for pid, r in P.items():
        _, split, space, model = PROTOCOLS[pid]
        desc = f"{split} / {space} / {model}"
        print(f"{pid:<10}{desc:<46}{_fmt(r['trial_f1_at_0_5']):>12}{_fmt(r['unit_f1_at_0_5']):>12}"
              f"{_fmt(r['unit_auc']):>10}{_fmt(r['unit_f1_lowhalf_at_0_5']):>20}"
              f"{_fmt(r['unit_f1_best']):>8} ({r['unit_f1_best_threshold']:.2f})")
    print("\n(a) vs (b): a random split over trials puts every machine on both sides of the fold, so the"
          "\n            model recognises the machine from its absolute fingerprint and reads the label off"
          "\n            the training set; leave-one-unit-out asks what happens on a machine never seen."
          "\n(b) vs (c): conditioning on the setpoint makes the fault linearly legible (compare d with"
          "\n            b_linear). These spaces also differ in speed_rpm inclusion; use the matched-input\n            ablation to isolate conditioning. A deterministic transform cannot add information."
          "\n(e):        the fault is a sweep SHAPE, so a unit normalised against its own median log level"
          "\n            exposes it to a tree and not to a linear model -- label-free, but the new unit's"
          "\n            whole sweep must be available (transductive)."
          "\nlowhalf:    unit score = median over the declared low-setpoint half of the sweep, where the fault"
          "\n            lives.  best (t) is OPTIMISTIC: the threshold is chosen on the labels it is scored on.")
    shown = [pid for pid in ("a", "b", "c", "e") if pid in P]
    if shown:
        units = list(P[shown[0]]["unit_scores"])
        truth = {u: (1 if u in doc["faulty_units"] else 0) for u in units}
        print(f"\nper-unit OOF median score\n{'unit':<8}{'truth':>6}" + "".join(f"{pid:>10}" for pid in shown))
        for u in units:
            print(f"{u:<8}{truth[u]:>6}" + "".join(f"{P[pid]['unit_scores'][u]:>10.3f}" for pid in shown))
    print(f"\nunit identifiability: a GBM predicts unit_id from raw trial features with 5-fold accuracy "
          f"{doc['unit_identifiability_acc']:.3f} (chance {1 / doc['n_units']:.3f}) -- this demonstrates a route for shared-unit recognition.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-derive protocol metrics from the synthetic data.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--json", default=None, help="write the full result document here")
    ap.add_argument("--quick", action="store_true", help="max_iter 60 instead of 300 (tests)")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--protocols", default=None, help="comma-separated subset, e.g. a,b,c")
    args = ap.parse_args(argv)
    ids = args.protocols.split(",") if args.protocols else None
    unknown = set(ids or []) - set(PROTOCOLS)
    if unknown:
        ap.error(f"unknown protocol id(s): {sorted(unknown)}; known: {list(PROTOCOLS)}")
    F = trial_features(pd.read_csv(args.data))
    doc = run_all(F, quick=args.quick, model_seed=args.model_seed, protocols=ids)
    print_report(doc)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(doc, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
