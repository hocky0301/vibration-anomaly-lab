"""
conditioning.py -- operating-point-conditional robust z-scores.

The question this module answers is never "how loud is this trial?" but "how far
is this trial from what healthy machines do *at the same operating point*?".

Two properties of the data motivate this framing:

* The setpoint dominates amplitude.  Vibration grows several-fold from the
  bottom to the top of a sweep for healthy machines too, so any absolute
  threshold on RMS measures the setpoint, not the health state.
* Each machine has its own fingerprint (mounting, alignment, sensor gain).
  Absolute features are an excellent way to recognise a machine the model
  *has* seen, which is the leak that a random split over trials exploits.

What the transform does and does not do (measured, ``results/seed_sweep.md``,
30 draws of the generator under leave-one-unit-out):

* It makes the fault *linearly legible*.  A logistic regression on raw
  statistics plus the operating point reaches a mean unit AUC of 0.148
  (protocol b_linear); the same model on the conditional robust-z space
  reaches 0.923 (protocol d).  In the raw space "quiet relative to what is
  normal at this setpoint" is not a direction a single hyperplane can express;
  in the conditioned space it is a negative shift along one axis. This
  historical comparison also removes speed_rpm; the matched-input ablation
  isolates that feature-space difference from conditioning.
* It does **not** add information that a tree with the setpoint column could
  not already find.  A gradient-boosted model on raw statistics plus the
  operating point (protocol b) averages unit AUC 0.950; on the conditioned
  space (protocol c) it averages 0.923, and seed by seed c is strictly above b
  on 4 draws, tied on 18 and below on 8.  Absolute features *do* transfer to
  an unseen machine for a flexible model on this generator; the v0.1 text of
  this module said otherwise and was wrong.

Why MAD and not the standard deviation
--------------------------------------
The scale of the reference distribution is estimated with the median absolute
deviation (times 1.4826, so it agrees with the standard deviation for Gaussian
data).  This is a robustness default, not a measured win:

1. *The healthy population has heavy tails.*  Healthy trials contain Student-t
   noise and occasional impulsive bursts, so a trial's standard deviation and
   peak-to-peak are set by a handful of samples; the MAD of the same trials is
   about three times steadier from trial to trial (README, Figure 06).  On
   this generator that steadiness does not reach the decision: median/MAD
   (protocol c) and mean/std (protocol c_meanstd) tie on unit AUC on 23 of the
   30 draws (mean 0.923 against 0.940) and their mean unit F1 at the fixed
   threshold is 0.772 against 0.751.
2. *The reference itself is not guaranteed clean.*  The reference is built from
   training-fold units labelled healthy, and in any real fleet some of them are
   mislabelled or drifting.  The MAD has a 50% breakdown point: contaminating a
   minority of the reference cannot drive a nondegenerate location estimate
   arbitrarily far, although it can change it substantially, whereas
   the standard deviation is moved by a single outlier without bound.  This
   argument is about a failure mode the synthetic data does not contain, and
   it is the reason the default stays at MAD.

Fold safety
-----------
The reference distribution is fitted and applied through two separate calls:

    ref = ConditionalRobustZ(feats).fit(train_healthy_frame)
    Z_train, Z_valid = ref.transform(F_train), ref.transform(F_valid)

``fit`` takes *a frame*, not the full table plus a boolean mask.  That is a
deliberate API choice: a mask argument invites the mistake of passing the whole
table and trusting that the mask was right, and a mask built from the full
table's labels is exactly how validation rows leak into the reference.  Handing
``fit`` a frame you have already restricted makes the restriction visible at the
call site, and ``transform`` is a pure function of the fitted statistics -- it
cannot see the rows it is scoring as a population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826
_EPS = 1e-9


class ConditionalRobustZ:
    """Robust z-scores against healthy trials at the same operating point.

    Parameters
    ----------
    features
        Columns to condition.  Normally :func:`features.vibration_features`.
    condition_on
        Column defining the operating point.  Values are rounded to
        ``round_to`` decimals to form bins, which is exact for a commanded
        sweep grid and tolerant of float noise.
    context
        Columns appended, unmodified, to the transformed matrix.  Defaults to
        the operating point alone.  ``speed_rpm`` is deliberately *not* a
        default: absolute speed carries a per-unit gain and can expose unit
        identity.  Whether raw features transfer is measured per protocol.
    clip
        Symmetric clip applied to the z-columns only (never to the context
        columns).  Bins where the reference MAD is degenerate can otherwise
        produce enormous scores that dominate a linear model.
    scale
        ``"mad"`` (default): centre on the per-bin median, scale by
        1.4826 * MAD.  ``"std"``: centre on the per-bin mean, scale by the
        standard deviation.  The non-robust variant exists so the two can be
        compared on equal footing; it is not the recommended setting.
    """

    def __init__(
        self,
        features: list[str],
        condition_on: str = "setpoint",
        context: tuple[str, ...] = ("setpoint",),
        clip: float | None = 8.0,
        round_to: int = 3,
        scale: str = "mad",
    ) -> None:
        if scale not in {"mad", "std"}:
            raise ValueError(f"scale must be 'mad' or 'std', got {scale!r}")
        if not features or len(set(features)) != len(features):
            raise ValueError("features must contain unique column names")
        if clip is not None and (not np.isfinite(clip) or clip <= 0):
            raise ValueError("clip must be finite and positive, or None")
        if not isinstance(round_to, int) or isinstance(round_to, bool):
            raise ValueError("round_to must be an integer")
        self.features = list(features)
        self.condition_on = condition_on
        self.context = tuple(context)
        self.clip = clip
        self.round_to = round_to
        self.scale = scale
        self.median_: pd.DataFrame | None = None
        self.mad_: pd.DataFrame | None = None
        self.global_median_: pd.Series | None = None
        self.global_mad_: pd.Series | None = None
        self.n_reference_: int = 0

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        columns = list(dict.fromkeys([self.condition_on, *self.features]))
        if not frame.columns.is_unique or not set(columns) <= set(frame.columns):
            raise ValueError("conditioning columns are missing or duplicated")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError("conditioning inputs must be finite")

    # ------------------------------------------------------------------ fit --
    def fit(self, reference: pd.DataFrame) -> ConditionalRobustZ:
        """Learn the healthy reference distribution.

        ``reference`` must already contain *only* the rows you are willing to
        call the reference: in cross-validation, the trials of training-fold
        units that are labelled healthy.  Nothing else in this class ever looks
        at a label, so this call is the single place where fold safety is
        decided.
        """
        if len(reference) == 0:
            raise ValueError("empty reference: no healthy training trials given")
        self._validate_frame(reference)
        if self.scale == "std" and len(reference) < 2:
            raise ValueError("std scale requires at least two reference trials")
        R = reference
        b = R[self.condition_on].round(self.round_to)
        if self.scale == "mad":
            self.median_ = R.groupby(b)[self.features].median()
            self.mad_ = R.groupby(b)[self.features].apply(
                lambda g: (g - g.median()).abs().median()
            )
            # Fallbacks for operating points absent from the reference.
            self.global_median_ = R[self.features].median()
            self.global_mad_ = (R[self.features] - self.global_median_).abs().median()
        else:
            # Non-robust variant: the attribute names are kept so that
            # ``z_frame`` is one code path; ``median_`` holds means and
            # ``mad_`` holds sample standard deviations (ddof=1, the pandas
            # default, as in verify_claims.py) divided by MAD_TO_SIGMA.
            self.median_ = R.groupby(b)[self.features].mean()
            self.mad_ = R.groupby(b)[self.features].std() / MAD_TO_SIGMA
            self.global_median_ = R[self.features].mean()
            self.global_mad_ = R[self.features].std() / MAD_TO_SIGMA
        self.n_reference_ = len(R)
        return self

    # -------------------------------------------------------------- transform --
    def z_frame(self, F: pd.DataFrame) -> pd.DataFrame:
        """Signed z-scores as a named DataFrame (context columns not appended).

        This is the form to read: ``z_frame(F)["vib_y_mad"]`` is "how many
        robust sigma is this trial's y-axis MAD away from healthy machines at
        this setpoint", sign included.
        """
        if self.median_ is None:
            raise RuntimeError("call fit() before transform()")
        self._validate_frame(F)
        b = F[self.condition_on].round(self.round_to)
        M = self.median_.reindex(b).fillna(self.global_median_).to_numpy()
        D = self.mad_.reindex(b).fillna(self.global_mad_).to_numpy()
        Z = (F[self.features].to_numpy() - M) / (MAD_TO_SIGMA * D + _EPS)
        if self.clip is not None:
            Z = np.clip(Z, -self.clip, self.clip)
        return pd.DataFrame(Z, index=F.index, columns=self.features)

    def transform(self, F: pd.DataFrame) -> np.ndarray:
        """Model matrix: clipped z-columns followed by the context columns."""
        Z = self.z_frame(F).to_numpy()
        if not self.context:
            return Z
        if not set(self.context) <= set(F.columns):
            raise ValueError("context columns are missing")
        context = F[list(self.context)].to_numpy(dtype=float)
        if not np.isfinite(context).all():
            raise ValueError("context inputs must be finite")
        return np.column_stack([Z, context])

    def fit_transform(self, reference: pd.DataFrame, F: pd.DataFrame | None = None) -> np.ndarray:
        """Fit on ``reference`` and transform ``F`` (default: the reference).

        Provided for convenience only.  In cross-validation, prefer the explicit
        two-call form so the reference restriction is visible.
        """
        self.fit(reference)
        return self.transform(reference if F is None else F)


# ------------------------------------------------------------------ helpers --
def healthy_reference(F: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """The rows that may form the reference for one fold.

    Training-fold rows whose unit is labelled healthy -- and nothing else.  The
    validation rows are not merely excluded from the *labels* used here, they
    are excluded from the population whose median and MAD define the scale.
    """
    tr = F.iloc[train_idx]
    return tr[tr["label"] == 0]


def conditional_robust_z(
    F: pd.DataFrame, feats: list[str], ref_mask: np.ndarray, scale: str = "mad"
) -> np.ndarray:
    """Mask-based form, matching ``verify_claims.conditional_robust_z``.

    Returns the z-columns for ``feats`` clipped to [-8, 8], followed by the
    ``setpoint`` column; the reference is the rows where ``ref_mask`` is True.
    Kept so the root verification script and this package can be checked
    against each other (``tests/test_parity.py`` asserts they agree).  New code
    should use :class:`ConditionalRobustZ`, whose ``fit``/``transform`` split
    makes it harder to build the reference from the wrong rows.
    """
    mask = np.asarray(ref_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(F),):
        raise ValueError("ref_mask must be a Boolean vector aligned with the rows")
    ref = ConditionalRobustZ(
        feats, context=("setpoint",), clip=8.0, scale=scale
    ).fit(F[mask])
    return ref.transform(F)
