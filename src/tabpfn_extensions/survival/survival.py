#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import torch
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_y_survival

# If you use the Prior Labs telemetry extension, keep this import+decorator;
# otherwise you can safely remove them.
try:
    from tabpfn_common_utils.telemetry import set_extension
except Exception:  # pragma: no cover
    def set_extension(_name):
        def deco(cls): return cls
        return deco

# TabPFN front-ends (assumed available in your environment)
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor


@set_extension("survival")
class SurvivalTabPFN(SurvivalAnalysisMixin):
    """
    A lightweight survival “risk scorer” built from two TabPFN heads:
    1) a classifier for P(event | X),
    2) a distributional regressor for T | (event, X).

    At prediction time we can:
      - produce horizon CDFs  F(t|X) = P(T <= t | X)  by mixing
        P(event|X) with the regressor CDF learned on event times.
      - return a single scalar risk:
          * With horizons:  sum_k w_k * F(t_k | X).
          * Without horizons:  P(event|X) / E[T | event, X].

    This is intended for **ranking** (C-index) or simple horizon risk
    scoring, not for full survival calibration.

    Parameters
    ----------
    cls_model : TabPFNClassifier | None, default=None
        Optionally pass a pre-configured classifier instance.

    reg_model : TabPFNRegressor | None, default=None
        Optionally pass a pre-configured regressor instance.

    time_transform : {"none","log1p"}, default="log1p"
        Optional monotone transform applied to target times *before*
        training the regressor. If "log1p", we train on log1p(time).
        Note: the regressor's returned distribution then lives in that
        transformed (raw) space; we handle conversions when querying CDF.

    default_horizons : iterable of float | None, default=None
        If provided, `predict(X)` will return the weighted horizon risk.
        If None, `predict(X)` falls back to P(event)/E[T|event].

    horizon_weights : iterable of float | None, default=None
        Weights for `default_horizons`. If None, weights are uniform.

    eps : float, default=1e-9
        Small epsilon to stabilize divisions.

    Attributes
    ----------
    _predict_risk_score : bool
        Flag for scikit-survival: higher score means higher risk.
    """

    def __init__(
        self,
        *,
        cls_model: Optional[TabPFNClassifier] = None,
        reg_model: Optional[TabPFNRegressor] = None,
        time_transform: Literal["none", "log1p"] = "log1p",
        default_horizons: Optional[Iterable[float]] = None,
        horizon_weights: Optional[Iterable[float]] = None,
        eps: float = 1e-9,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.cls_model = cls_model or TabPFNClassifier(random_state=random_state)
        self.reg_model = reg_model or TabPFNRegressor(random_state=random_state)
        self.time_transform = time_transform
        self.default_horizons = None if default_horizons is None else list(default_horizons)
        self.horizon_weights = None if horizon_weights is None else list(horizon_weights)
        self.eps = float(eps)

    # --- scikit-survival convention: larger = higher risk
    @property
    def _predict_risk_score(self) -> bool:  # noqa: D401
        """Indicates that `predict(X)` returns a risk score (higher = riskier)."""
        return True

    # ------------------------------ helpers ------------------------------

    def _to_reg_space(self, t: np.ndarray | float) -> np.ndarray:
        """Map physical time(s) to the regressor 'raw' target space."""
        if self.time_transform == "log1p":
            return np.log1p(np.asarray(t, dtype=float))
        return np.asarray(t, dtype=float)

    def _from_reg_space(self, z: np.ndarray | float) -> np.ndarray:
        """Inverse-map from regressor 'raw' space back to physical time(s)."""
        if self.time_transform == "log1p":
            return np.expm1(np.asarray(z, dtype=float))
        return np.asarray(z, dtype=float)

    # ------------------------------ fit ------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | list | dict) -> "TabPFNSurvival":
        """
        Fit the classifier on event indicators, and the regressor on *event times only*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : structured array / mapping / iterable of (event, time)

        Returns
        -------
        self
        """
        y_event, y_time = check_y_survival(y)

        # Basic sanity checks
        if np.sum(y_event) < 2:
            raise ValueError("Need at least two events to learn time distribution.")
        if np.any(np.asarray(y_time) < 0):
            raise ValueError("Times must be non-negative.")

        # 1) Event classifier on ALL samples
        self.cls_model.fit(X, y_event)

        # 2) Time regressor on EVENT samples only
        X_ev = X[y_event]
        t_ev = np.asarray(y_time, dtype=float)[y_event]
        t_ev_reg = self._to_reg_space(t_ev)
        self.reg_model.fit(X_ev, t_ev_reg)

        return self

    # ------------------------------ distributions ------------------------------

    def predict_cdf_at(self, X: np.ndarray, t_grid: Iterable[float]) -> np.ndarray:
        """
        Return F(t|X) = P(T <= t | X) for each t in `t_grid`.

        We compute:
            P(T <= t | X) = P(event | X) * P(T <= t | event, X)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        t_grid : iterable of float
            Physical 'time' values.

        Returns
        -------
        F : ndarray, shape (n_samples, len(t_grid))
        """
        t_grid = np.asarray(list(t_grid), dtype=float)
        if t_grid.ndim != 1 or t_grid.size == 0:
            raise ValueError("t_grid must be a 1-D, non-empty sequence of times.")

        # 1) P(event|X)
        p_event = self.cls_model.predict_proba(X)[:, 1]  # (n,)

        # 2) P(T<=t | event, X) using regressor distribution
        #    We ask for full outputs to access logits + distribution object.
        full = self.reg_model.predict(X, output_type="full")  # type: ignore
        logits = full["logits"]            # torch.Tensor [n, n_bins] (post-processed)
        criterion = full["criterion"]      # FullSupportBarDistribution

        # Convert physical times to the regressor's target space
        t_reg = self._to_reg_space(t_grid)

        # Vectorized across grid (loop over m is fine; TabPFN inference is the hard part)
        cdfs = []
        with torch.no_grad():
            for tau in t_reg:
                # criterion.cdf(logits, tau) -> torch.Tensor [n,]
                cdfs.append(criterion.cdf(logits, float(tau)).cpu().numpy())
        F_event = np.column_stack(cdfs)  # (n, m)

        # 3) Mix with event probability
        F_mixed = p_event[:, None] * F_event  # (n, m)
        return F_mixed

    def predict_survival_at(self, X: np.ndarray, t_grid: Iterable[float]) -> np.ndarray:
        """
        Return S(t|X) = 1 - F(t|X) for each t in `t_grid`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        t_grid : iterable of float

        Returns
        -------
        S : ndarray, shape (n_samples, len(t_grid))
        """
        return 1.0 - self.predict_cdf_at(X, t_grid)

    # ------------------------------ scalar risks ------------------------------

    def predict_risk(
        self,
        X: np.ndarray,
        *,
        horizons: Optional[Iterable[float]] = None,
        weights: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """
        Scalar risk for ranking.

        If `horizons` provided:
            risk = sum_k w_k * P(T <= t_k | X).
        Else:
            risk = P(event|X) / E[T | event, X].

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        horizons : iterable of float | None
            Physical horizon times. If None, uses the fallback risk.
        weights : iterable of float | None
            Optional weights for horizons (defaults to uniform).

        Returns
        -------
        risk : ndarray of shape (n_samples,)
        """
        if horizons is not None:
            horizons = list(horizons)
            if len(horizons) == 0:
                raise ValueError("If provided, 'horizons' cannot be empty.")
            F = self.predict_cdf_at(X, horizons)  # (n, m)
            if weights is None:
                w = np.full(F.shape[1], 1.0 / F.shape[1], dtype=float)
            else:
                w = np.asarray(list(weights), dtype=float)
                if w.shape[0] != F.shape[1]:
                    raise ValueError("weights must match number of horizons.")
                s = w.sum()
                w = w / (s + self.eps)
            return (F @ w).astype(float)

        # Fallback: P(event)/E[T | event,X]
        p_event = self.cls_model.predict_proba(X)[:, 1]
        full = self.reg_model.predict(X, output_type="full")  # type: ignore
        logits = full["logits"]
        criterion = full["criterion"]
        with torch.no_grad():
            mu_reg = criterion.mean(logits).cpu().numpy()  # mean in reg space
        mu_time = self._from_reg_space(mu_reg)
        inv_mean_time = 1.0 / (mu_time + self.eps)
        return (p_event * inv_mean_time).astype(float)

    # ------------------------------ scikit API ------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return a single **risk score** per row (higher = riskier).

        If this instance was created with `default_horizons`, this returns
        the weighted horizon risk; otherwise it returns the fallback risk
        (P(event)/E[T|event]).
        """
        if self.default_horizons is not None:
            return self.predict_risk(X, horizons=self.default_horizons, weights=self.horizon_weights)
        return self.predict_risk(X, horizons=None)
