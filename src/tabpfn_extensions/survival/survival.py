#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import torch
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_y_survival

try:
    from tabpfn_common_utils.telemetry import set_extension
except Exception:  # pragma: no cover
    def set_extension(_name):
        def deco(cls): return cls
        return deco

from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor


@set_extension("survival")
class SurvivalTabPFN(SurvivalAnalysisMixin):
    """
    Two-head TabPFN survival scorer:
      1) classifier -> P(event | X)
      2) distributional regressor -> T | (event, X)

    Scalar risk options (higher = riskier):
      - 'weighted_cdf' (default): sum_j w_j * P(T <= tau_j | X), with
        auto-chosen horizons tau_j from event-time quantiles and
        exponential weights emphasizing early horizons.
      - 'avg_cdf': uniform average across horizons.
      - 'p_over_mean': P(event|X) / E[T | event, X]  (legacy fallback).

    Also exposes CDF/Survival curves:
      - predict_cdf_at(X, t_grid) -> F(t|X)
      - predict_survival_at(X, t_grid) -> S(t|X) = 1 - F(t|X)

    Parameters
    ----------
    cls_model : TabPFNClassifier | None
    reg_model : TabPFNRegressor | None
    time_transform : {"none","log1p"}, default="log1p"
        Transform applied to event times for the regressor target.
    risk_strategy : {"weighted_cdf","avg_cdf","p_over_mean"}, default="weighted_cdf"
    default_horizons : iterable[float] | None
        If given, use these horizons for CDF-based risks. Otherwise picked
        from train data (event-time quantiles).
    n_auto_horizons : int, default=6
        Number of horizons to auto-pick when default_horizons=None.
    auto_horizon_quantile_range : tuple[float,float], default=(0.10, 0.90)
        Range of quantiles to sample for horizons (inclusive).
    exp_weight_gamma : float, default=0.85
        Exponential decay for horizon weights (near 1.0 = flatter).
    eps : float, default=1e-9
        Numerical stability constant.
    random_state : any, default=None
        Passed into TabPFN models if you don’t pass custom ones.
    """

    def __init__(
        self,
        *,
        cls_model: Optional[TabPFNClassifier] = None,
        reg_model: Optional[TabPFNRegressor] = None,
        time_transform: Literal["none", "log1p"] = "log1p",
        risk_strategy: Literal["weighted_cdf", "avg_cdf", "p_over_mean"] = "weighted_cdf",
        default_horizons: Optional[Iterable[float]] = None,
        n_auto_horizons: int = 6,
        auto_horizon_quantile_range: tuple[float, float] = (0.10, 0.90),
        exp_weight_gamma: float = 0.85,
        eps: float = 1e-9,
        random_state=None,
    ):
        self.cls_model = cls_model or TabPFNClassifier(random_state=random_state)
        self.reg_model = reg_model or TabPFNRegressor(random_state=random_state)

        self.time_transform = time_transform
        self.risk_strategy = risk_strategy

        self.default_horizons = None if default_horizons is None else list(default_horizons)
        self.n_auto_horizons = int(n_auto_horizons)
        self.auto_horizon_quantile_range = tuple(auto_horizon_quantile_range)
        self.exp_weight_gamma = float(exp_weight_gamma)

        self.eps = float(eps)

        # filled after fit()
        self._auto_horizons_: Optional[np.ndarray] = None  # 1D array of floats

    @property
    def _predict_risk_score(self) -> bool:
        return True  # scikit-survival convention: higher = higher risk

    # ---------- transforms for regressor target ----------

    def _to_reg_space(self, t: np.ndarray | float) -> np.ndarray:
        if self.time_transform == "log1p":
            return np.log1p(np.asarray(t, dtype=float))
        return np.asarray(t, dtype=float)

    def _from_reg_space(self, z: np.ndarray | float) -> np.ndarray:
        if self.time_transform == "log1p":
            return np.expm1(np.asarray(z, dtype=float))
        return np.asarray(z, dtype=float)

    # ------------------------------ fit ------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | list | dict) -> "SurvivalTabPFN":
        y_event, y_time = check_y_survival(y)

        if np.sum(y_event) < 2:
            raise ValueError("Need at least two events to learn a time distribution.")
        if np.any(np.asarray(y_time) < 0):
            raise ValueError("Times must be non-negative.")

        # 1) P(event|X)
        self.cls_model.fit(X, y_event)

        # 2) T | event, X
        X_ev = X[y_event]
        t_ev = np.asarray(y_time, dtype=float)[y_event]
        t_ev_reg = self._to_reg_space(t_ev)
        self.reg_model.fit(X_ev, t_ev_reg)

        # 3) Auto horizons (if not provided)
        if self.default_horizons is None:
            q_lo, q_hi = self.auto_horizon_quantile_range
            q_grid = np.linspace(q_lo, q_hi, num=self.n_auto_horizons)
            # Use EVENT times to avoid censoring complication
            taus = np.quantile(t_ev, q_grid).astype(float)
            # Ensure strict monotonicity (dedup)
            taus = np.unique(np.clip(taus, a_min=np.min(t_ev), a_max=np.max(t_ev)))
            self._auto_horizons_ = taus
        else:
            self._auto_horizons_ = np.asarray(self.default_horizons, dtype=float)

        return self

    # ------------------------------ distributions ------------------------------

    def predict_cdf_at(self, X: np.ndarray, t_grid: Iterable[float]) -> np.ndarray:
        t_grid = np.asarray(list(t_grid), dtype=float)
        if t_grid.ndim != 1 or t_grid.size == 0:
            raise ValueError("t_grid must be a 1-D, non-empty sequence of times.")

        # P(event|X)
        p_event = self.cls_model.predict_proba(X)[:, 1]  # (n,)

        # P(T<=t | event, X) from regressor distribution
        full = self.reg_model.predict(X, output_type="full")  # type: ignore
        logits = full["logits"]            # torch.Tensor [n, nbins]
        criterion = full["criterion"]      # FullSupportBarDistribution

        t_reg = self._to_reg_space(t_grid)

        cdfs = []
        with torch.no_grad():
            for tau in t_reg:
                ys = torch.as_tensor([float(tau)], device=logits.device, dtype=logits.dtype)
                cdf_tau = criterion.cdf(logits, ys).squeeze(-1).cpu().numpy()  # (n,)
                cdfs.append(cdf_tau)
        F_event = np.column_stack(cdfs)  # (n, m)

        # Mix with event probability
        F_mixed = p_event[:, None] * F_event
        return F_mixed

    def predict_survival_at(self, X: np.ndarray, t_grid: Iterable[float]) -> np.ndarray:
        return 1.0 - self.predict_cdf_at(X, t_grid)

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
            # where taus is ascending; we want heavier weight on early taus -> low j
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
        horizons: Optional[Iterable[float]] = None,
        weights: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        strategy = self.risk_strategy

        # If user passes horizons explicitly, we override strategy to CDF-based
        if horizons is not None:
            taus = np.asarray(list(horizons), dtype=float)
            if taus.size == 0:
                raise ValueError("If provided, 'horizons' cannot be empty.")
            F = self.predict_cdf_at(X, taus)  # (n, m)
            if weights is None:
                w = np.full(F.shape[1], 1.0 / F.shape[1], dtype=float)
            else:
                w = np.asarray(list(weights), dtype=float)
                if w.shape[0] != F.shape[1]:
                    raise ValueError("weights must match number of horizons.")
                w = w / (w.sum() + self.eps)
            return (F @ w).astype(float)

        # No horizons provided: use configured strategy
        if strategy in ("weighted_cdf", "avg_cdf"):
            taus, w = self._horizons_and_weights()
            F = self.predict_cdf_at(X, taus)  # (n, m)
            return (F @ w).astype(float)

        # Legacy fallback: P(event)/E[T | event, X]
        p_event = self.cls_model.predict_proba(X)[:, 1]
        full = self.reg_model.predict(X, output_type="full")  # type: ignore
        logits = full["logits"]
        criterion = full["criterion"]
        with torch.no_grad():
            mu_reg = criterion.mean(logits).cpu().numpy()
        mu_time = self._from_reg_space(mu_reg)
        return (p_event * (1.0 / (mu_time + self.eps))).astype(float)

    # ------------------------------ scikit API ------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_risk(X, horizons=None)
