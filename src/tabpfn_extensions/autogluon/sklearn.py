from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# Re‑use the already provided AutoGluon model wrapper for TabPFN
from tabpfn_extensions.autogluon.model import TabPFNV2Model

__all__ = [
    "AutogluonTabPFNClassifier",
    "AutogluonTabPFNRegressor",
]


class _BaseAutoGluonTabPFN:
    """Shared logic between classifier and regressor."""

    _is_classifier: bool = True  # overridden by subclass

    def __init__(
        self,
        *,
        max_time: int = 180,
        presets: str | None = "medium_quality",
        num_gpus: int = 0,
        **predictor_kwargs,
    ) -> None:
        self.max_time = max_time
        self.presets = presets
        self.num_gpus = num_gpus
        self._predictor_kwargs = predictor_kwargs
        self._predictor: TabularPredictor | None = None

    # ---------------------------------------------------------------------
    # Public scikit‑style API
    # ---------------------------------------------------------------------
    def fit(self, X, y):  # noqa: D401 – keep scikit signature short
        """Train a single TabPFN model within *AutoGluon*."""
        df = pd.DataFrame(X).copy()
        df["_target_"] = y  # lightweight, avoids name clashes

        problem_type = (
            "binary"
            if self._is_classifier and len(np.unique(y)) == 2
            else ("multiclass" if self._is_classifier else "regression")
        )

        self._predictor = TabularPredictor(
            label="_target_",
            problem_type=problem_type,
            **self._predictor_kwargs,
        )

        # Single TabPFN with optional GPU resources
        hyperparameters = {
            TabPFNV2Model: [
                {
                    "ag_args_fit": {"num_gpus": self.num_gpus},
                }
            ]
        }

        self._predictor.fit(
            train_data=df,
            time_limit=self.max_time,
            presets=self.presets,
            hyperparameters=hyperparameters,
        )
        return self

    # NOTE: TabularPredictor already keeps pandas types; returning numpy keeps it
    # inline with scikit‑learn conventions.
    def predict(self, X):
        if self._predictor is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self._predictor.predict(pd.DataFrame(X))
        return preds.to_numpy()

    def predict_proba(self, X):
        if not self._is_classifier:
            raise AttributeError("predict_proba is only available for classifiers.")
        if self._predictor is None:
            raise RuntimeError("Model is not fitted yet.")
        proba = self._predictor.predict_proba(pd.DataFrame(X))
        return proba.to_numpy()


class AutogluonTabPFNClassifier(_BaseAutoGluonTabPFN):
    """Scikit‑style TabPFN classifier powered by AutoGluon."""

    _is_classifier = True


class AutogluonTabPFNRegressor(_BaseAutoGluonTabPFN):
    """Scikit‑style TabPFN regressor powered by AutoGluon."""

    _is_classifier = False
