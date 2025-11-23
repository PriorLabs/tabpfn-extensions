#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

# INFO: The survival prediction work is work-in-progress and still unoptimized,
#  TabPFN survival may provide inconsistent improvements.

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import warnings
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_y_survival
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor


@set_extension("survival")
class SurvivalTabPFN(SurvivalAnalysisMixin, BaseEstimator):
    """Two-head TabPFN survival scorer:
      1) classifier -> P(event | X)
      2) distributional regressor -> T | (event, X)

    Scalar risk options (higher = riskier):
      - "weighted_cdf" (default): sum_j w_j * CIF_1(tau_j | X),
        where CIF_1(t | X) = P(T <= t, event=1 | X).
      - "avg_cdf": uniform average of CIF_1(tau_j | X) across horizons.
      - "p_over_mean": P(event|X) / E[T | event, X]  (legacy fallback).

    Also exposes CIF-based curves for event=1:
      - predict_cif_at(X, t_grid)
          -> CIF_1(t|X) = P(T <= t, event=1 | X)
      - predict_survival_from_cif_at(X, t_grid)
          -> 1 - CIF_1(t|X)

    Backwards-compatible aliases (deprecated):
      - predict_cdf_at(X, t_grid)
          -> same as predict_cif_at, returns CIF_1(t|X)
      - predict_survival_at(X, t_grid)
          -> same as predict_survival_from_cif_at

    Parameters
    ----------
    cls_model : TabPFNClassifier | None
        If None, created in fit with random_state.
    reg_model : TabPFNRegressor | None
        If None, created in fit with random_state.
    risk_strategy : {"weighted_cdf","avg_cdf","p_over_mean"}, default="weighted_cdf"
    default_horizons : iterable[float] | None
    n_auto_horizons : int, default=6
    auto_horizon_quantile_range : tuple[float,float], default=(0.10, 0.90)
    exp_weight_gamma : float, default=0.85
    eps : float, default=1e-9
    random_state : any, default=None
    """

    def __init__(
        self,
        *,
        cls_model: TabPFNClassifier | None = None,
        reg_model: TabPFNRegressor | None = None,
        risk_strategy: Literal[
            "weighted_cdf", "avg_cdf", "p_over_mean"
        ] = "weighted_cdf",
        default_horizons: Iterable[float] | None = None,
        n_auto_horizons: int = 6,
        auto_horizon_quantile_range: tuple[float, float] = (0.10, 0.90),
        exp_weight_gamma: float = 0.85,
        eps: float = 1e-9,
        random_state=None,
    ):
        # store constructor params (BaseEstimator will expose them to get/set_params)
        self.cls_model = cls_model
        self.reg_model = reg_model
        self.risk_strategy = risk_strategy
        self.default_horizons = (
            None if default_horizons is None else list(default_horizons)
        )
        self.n_auto_horizons = int(n_auto_horizons)
        self.auto_horizon_quantile_range = tuple(auto_horizon_quantile_range)
        self.exp_weight_gamma = float(exp_weight_gamma)
        self.eps = float(eps)
        self.random_state = random_state

        # fitted attributes (set in fit)
        self._auto_horizons_: np.ndarray | None = None
        self._cls_model: TabPFNClassifier | None = None
        self._reg_model: TabPFNRegressor | None = None
        self.n_features_in_: int | None = None

    @property
    def _predict_risk_score(self) -> bool:
        # scikit-survival convention: higher = higher risk
        return True

    # ------------------------------ fit ------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | list | dict) -> SurvivalTabPFN:
        y_event, y_time = check_y_survival(y)

        if np.sum(y_event) < 2:
            raise ValueError(f"Need at least 2 events to learn a time distribution, but found {np.sum(y_event)}.")
        if np.any(np.asarray(y_time) < 0):
            raise ValueError("Times must be non-negative.")

        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        # instantiate sub-models if not provided
        self._cls_model = self.cls_model or TabPFNClassifier(
            random_state=self.random_state
        )
        self._reg_model = self.reg_model or TabPFNRegressor(
            random_state=self.random_state
        )

        # 1) P(event|X)
        self._cls_model.fit(X, y_event)

        # 2) T | event, X
        X_ev = X[y_event]
        t_ev = np.asarray(y_time, dtype=float)[y_event]
        self._reg_model.fit(X_ev, t_ev)

        # 3) Auto horizons (if not provided)
        if self.default_horizons is None:
            q_lo, q_hi = self.auto_horizon_quantile_range
            q_grid = np.linspace(q_lo, q_hi, num=self.n_auto_horizons)
            taus = np.quantile(t_ev, q_grid).astype(float)
            taus = np.unique(np.clip(taus, a_min=np.min(t_ev), a_max=np.max(t_ev)))
            self._auto_horizons_ = taus
        else:
            self._auto_horizons_ = np.asarray(self.default_horizons, dtype=float)

        return self

    # ------------------------------ distributions ------------------------------

    def predict_cif_at(self, X: np.ndarray, t_grid: Iterable[float]) -> np.ndarray:
        """Event-1 cumulative incidence CIF_1(t|X).

        Computes

            CIF_1(t|X) = P(T <= t, event=1 | X)

        via the factorization

            P(event=1 | X) * P(T <= t | event=1, X).
        """
        if self._cls_model is None or self._reg_model is None:
            raise RuntimeError("Estimator not fitted yet.")

        t_grid = np.asarray(list(t_grid), dtype=float)
        if t_grid.ndim != 1 or t_grid.size == 0:
            raise ValueError("t_grid must be a 1-D, non-empty sequence of times.")

        X = np.asarray(X)

        p_event = self._cls_model.predict_proba(X)[:, 1]  #  P(event|X), (n,)

        # P(T<=t | event, X) from regressor distribution
        full = self._reg_model.predict(X, output_type="full")  # type: ignore
        logits = torch.tensor(full["logits"])  # torch.Tensor [n, nbins]
        criterion = full["criterion"]  # FullSupportBarDistribution

        cifs = []
        with torch.no_grad():
            for tau in t_grid:
                ys = torch.as_tensor(
                    [float(tau)], device=logits.device, dtype=logits.dtype
                )
                cdf_tau = criterion.cdf(logits, ys).squeeze(-1).cpu().numpy()  # (n,)
                cifs.append(cdf_tau)
        F_event = np.column_stack(cifs)  # (n, m)

        # Mix with event probability
        F_mixed = p_event[:, None] * F_event
        return F_mixed

    def predict_survival_from_cif_at(
        self, X: np.ndarray, t_grid: Iterable[float]
    ) -> np.ndarray:
        """Complement 1 - CIF_1(t|X) based on :meth:`predict_cif_at`.

        Returns

            1 - CIF_1(t|X) = 1 - P(T <= t, event=1 | X),

        i.e. the probability that event 1 has not yet occurred by time ``t``.
        """

        return 1.0 - self.predict_cif_at(X, t_grid)

    # ------------------------------ scalar risk ------------------------------

    def _horizons_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (horizons, weights) according to config/auto, normalized weights."""
        if self._auto_horizons_ is None:
            raise RuntimeError("Model not fitted; horizons are undefined.")

        taus = self._auto_horizons_
        m = taus.shape[0]

        if self.risk_strategy == "avg_cdf":
            w = np.full(m, 1.0 / max(m, 1), dtype=float)
            return taus, w

        if self.risk_strategy == "weighted_cdf":
            # earliest horizons get largest weights: w_j ∝ gamma^(j), j=0..m-1
            j = np.arange(m, dtype=float)
            w = np.power(self.exp_weight_gamma, j)
            w = w / (w.sum() + self.eps)
            return taus, w

        # Fallback strategy uses no horizons
        return np.array([]), np.array([])

    def predict_risk(
        self,
        X: np.ndarray,
        *,
        horizons: Iterable[float] | None = None,
        weights: Iterable[float] | None = None,
    ) -> np.ndarray:
        if self._cls_model is None or self._reg_model is None:
            raise RuntimeError("Estimator not fitted yet.")

        strategy = self.risk_strategy

        # If user passes horizons explicitly, we override strategy to CIF-based
        if horizons is not None:
            taus = np.asarray(list(horizons), dtype=float)
            if taus.size == 0:
                raise ValueError("If provided, 'horizons' cannot be empty.")
            F = self.predict_cif_at(X, taus)  # (n, m)
            if weights is None:
                w = np.full(F.shape[1], 1.0 / F.shape[1], dtype=float)
            else:
                w = np.asarray(list(weights), dtype=float)
                if w.shape[0] != F.shape[1]:
                    raise ValueError("weights must match number of horizons.")
                w = w / (w.sum() + self.eps)
            return (F @ w).astype(float)

        # No horizons provided: use configured strategy (CIF-based when applicable)
        if strategy in ("weighted_cdf", "avg_cdf"):
            taus, w = self._horizons_and_weights()
            F = self.predict_cif_at(X, taus)  # (n, m)
            return (F @ w).astype(float)

        # Legacy fallback: P(event)/E[T | event, X]
        p_event = self._cls_model.predict_proba(np.asarray(X))[:, 1]
        full = self._reg_model.predict(np.asarray(X), output_type="full")  # type: ignore
        logits = full["logits"]
        criterion = full["criterion"]
        with torch.no_grad():
            mu_reg = criterion.mean(logits).cpu().numpy()
        return (p_event * (1.0 / (mu_reg + self.eps))).astype(float)

    # ------------------------------ scikit API ------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_risk(X, horizons=None)
