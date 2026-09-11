from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions.misc.sklearn_compat import validate_data
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

_QUANTILE_GRID = (
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)


class AutoHurdleRegressor(RegressorMixin, BaseEstimator):
    """Two-stage regression for non-negative targets with a point mass at zero.

    Args:
        classifier: Cloneable classifier with predict_proba. Defaults to TabPFN.
        regressor: Cloneable regressor with TabPFN's output_type and quantiles
            prediction arguments. Defaults to TabPFN.
        hurdle: Whether to split zero and positive targets. "auto" enables the
            split for non-negative targets whose zero rate exceeds zero_threshold.
            Otherwise, all rows go to the regressor.
        zero_threshold: Training zero-rate threshold for automatic selection.
        quantile_grid: Strictly increasing positive-stage quantile levels between
            0 and 1, with at least two entries. None uses the 21-point grid from
            0.01 to 0.99. Required levels outside the grid clamp to its endpoints.

    predict defaults to the mixture median, suitable for absolute error. With
    positive probability p, it is zero for p <= 0.5 and otherwise the positive
    distribution's (p - 0.5) / p quantile. Quantiles use linear interpolation on
    the supplied grid, clamping levels outside that grid. Negative
    positive-stage predictions are clipped to zero. Mean predictions multiply
    the clipped positive-stage mean by p.

    Fitted attributes include hurdle_, zero_rate_, classifier_, and regressor_.
    classifier_ is None when the hurdle is inactive. Both estimators are None
    for an all-zero target with the hurdle enabled.
    """

    def __init__(
        self,
        classifier: Any = None,
        regressor: Any = None,
        *,
        hurdle: Literal["auto"] | bool = "auto",
        zero_threshold: float = 0.5,
        quantile_grid: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.hurdle = hurdle
        self.zero_threshold = zero_threshold
        self.quantile_grid = quantile_grid

    def fit(self, X: Any, y: Any) -> AutoHurdleRegressor:
        """Fit cloned estimators, preserving DataFrame columns and dtypes."""
        self.__dict__.pop("is_fitted_", None)
        if self.hurdle not in ("auto", True, False):
            raise ValueError("hurdle must be 'auto', True, or False.")
        if not 0 <= self.zero_threshold <= 1:
            raise ValueError("zero_threshold must be between 0 and 1.")
        grid = np.array(
            _QUANTILE_GRID if self.quantile_grid is None else self.quantile_grid,
            dtype=float,
            copy=True,
        )
        if (
            grid.ndim != 1
            or grid.size < 2
            or not np.isfinite(grid).all()
            or np.any((grid <= 0) | (grid >= 1))
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError(
                "quantile_grid must contain at least two finite, strictly increasing "
                "levels strictly between 0 and 1."
            )
        self.quantile_grid_ = grid
        _, y = validate_data(
            self, X, y, dtype=None, ensure_all_finite=False, y_numeric=True
        )
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Targets must be finite.")
        self.zero_rate_ = float(np.mean(y == 0))
        self.hurdle_ = (
            bool(np.all(y >= 0) and self.zero_rate_ > self.zero_threshold)
            if self.hurdle == "auto"
            else bool(self.hurdle)
        )
        if self.hurdle_ and np.any(y < 0):
            raise ValueError("Hurdle modelling requires non-negative targets.")
        positive = y > 0
        if self.hurdle_ and positive.all():
            raise ValueError("Hurdle modelling requires zero and positive targets.")

        classifier = None
        regressor = None
        if not self.hurdle_ or positive.any():
            regressor = clone(
                self.regressor if self.regressor is not None else TabPFNRegressor()
            )
            if self.hurdle_:
                classifier = clone(
                    self.classifier
                    if self.classifier is not None
                    else TabPFNClassifier()
                )
                classifier.fit(X, positive.astype(int))
                regressor.fit(_safe_indexing(X, positive), y[positive])
            else:
                regressor.fit(X, y)
        self.classifier_ = classifier
        self.regressor_ = regressor
        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: Any,
        *,
        output_type: Literal["mean", "median", "quantiles"] = "median",
        quantiles: list[float] | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Predict a mean, median, or list of quantile arrays in target units."""
        check_is_fitted(self, "is_fitted_")
        validate_data(self, X, reset=False, dtype=None, ensure_all_finite=False)
        if output_type not in ("mean", "median", "quantiles"):
            raise ValueError("output_type must be 'mean', 'median', or 'quantiles'.")
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if not quantiles or any(not 0 < q < 1 for q in quantiles):
            raise ValueError("quantiles must be nonempty and strictly between 0 and 1.")
        if not self.hurdle_:
            return self.regressor_.predict(
                X, output_type=output_type, quantiles=quantiles
            )
        if self.regressor_ is None:
            zeros = np.zeros(len(X))
            return (
                [zeros.copy() for _ in quantiles]
                if output_type == "quantiles"
                else zeros
            )

        positive_class = np.flatnonzero(self.classifier_.classes_ == 1).item()
        p = np.clip(
            np.asarray(
                self.classifier_.predict_proba(X)[:, positive_class], dtype=float
            ),
            0.0,
            1.0,
        )
        if output_type == "mean":
            return p * np.maximum(self.regressor_.predict(X, output_type="mean"), 0.0)
        grid = np.asarray(
            self.regressor_.predict(
                X, output_type="quantiles", quantiles=self.quantile_grid_.tolist()
            ),
            dtype=float,
        )
        levels = quantiles if output_type == "quantiles" else [0.5]
        predictions = []
        for q in levels:
            prediction = np.zeros(len(p))
            active = p > 1 - q
            for row in np.flatnonzero(active):
                level = (p[row] - (1 - q)) / p[row]
                prediction[row] = np.interp(level, self.quantile_grid_, grid[:, row])
            predictions.append(np.maximum(prediction, 0.0))
        return predictions if output_type == "quantiles" else predictions[0]
