from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor
from tabpfn_extensions.autogluon.model import TabPFNV2Model


class _BaseAutoGluonTabPFN:
    """Shared logic between classifier and regressor for TabPFN models powered by AutoGluon.

    This base class provides a scikit-learn compatible interface to train and predict
    using TabPFN models managed by AutoGluon's TabularPredictor. It leverages
    AutoGluon's robust ensemble capabilities and automatic feature engineering.
    """

    _is_classifier: bool = None  # overridden by subclass

    def __init__(
        self,
        *,
        max_time: int = 180,
        num_gpus: int = 0,
        presets: Literal["best_quality", "high_quality", "good_quality", "medium_quality"] = "medium_quality",
        tabpfn_model_type: Literal["dt_pfn", "single"] = "single",
        **predictor_kwargs,
    ) -> None:
        """Initializes the AutoGluonTabPFN wrapper.

        Parameters
        ----------
        max_time : int, default=180
            Maximum time in seconds for the AutoGluon `fit` method to run.
            AutoGluon's predictive performance generally improves with more time.
            Set this to the longest amount of time you are willing to wait.
        num_gpus : int, default=0
            Number of GPUs to use for training. Set to 0 for CPU-only training.
            TabPFN can leverage GPUs for faster training if available.
        presets : Literal['best_quality', 'high_quality', 'good_quality', 'medium_quality'], default="medium_quality"
            Presets for AutoGluon's `TabularPredictor.fit` method. These control the
            trade-off between model quality and training/inference speed.
            - 'best_quality': State-of-the-art accuracy, often utilizing stacking/bagging.
            - 'high_quality': Strong accuracy with faster inference.
            - 'good_quality': Good accuracy with very fast inference.
            - 'medium_quality': Competitive with other AutoML frameworks, ideal for prototyping.
            For serious usage, 'best_quality' or 'high_quality' are recommended.
        tabpfn_model_type : Literal["dt_pfn", "single"], default="dt_pfn"
            Specifies the type of TabPFN model to use within AutoGluon.
            - "dt_pfn": Uses a Random Forest TabPFN (ensemble of TabPFNs).
            - "single": Uses a single TabPFN model.
            This parameter is passed directly to the `TabPFNV2Model`'s hyperparameters.
        path : str, optional
            Path to directory where models and intermediate artifacts will be saved.
            If not specified, a default path will be used by AutoGluon.
        verbosity : int, default = 2
            Verbosity level for AutoGluon's logging. Controls how much information
            is printed during the fitting process.
            - 0: Only log exceptions.
            - 1: Only log warnings + exceptions.
            - 2: Standard logging.
            - 3: Verbose logging (e.g., log validation score every 50 iterations).
            - 4: Maximally verbose logging (e.g., log validation score every iteration).
            Passed directly to `TabularPredictor(verbosity=...)`.
        eval_metric : str, optional
            Metric by which predictions will be ultimately evaluated on test data.
            AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights,
            etc., in order to improve this metric on validation data. If `None`,
            it is automatically chosen based on `problem_type` (e.g., 'accuracy' for
            classification, 'root_mean_squared_error' for regression).
            Examples for classification: 'accuracy', 'balanced_accuracy', 'roc_auc', 'f1', 'log_loss'.
            Examples for regression: 'root_mean_squared_error', 'mean_absolute_error', 'r2'.
            Passed directly to `TabularPredictor(eval_metric=...)`.
        predictor_kwargs : dict, optional
            Additional keyword arguments to pass directly to the `TabularPredictor`
            constructor. This allows for fine-grained control over AutoGluon's
            behavior, such as `problem_type` (though inferred if not set),
            `sample_weight`, or `groups`.
            See AutoGluon's `TabularPredictor` documentation for full details.
            Example: `predictor_kwargs={'positive_class': 'positive_label'}`.
        fit_kwargs : dict, optional
            Additional keyword arguments to pass directly to the `TabularPredictor.fit`
            method. This provides extensive control over the training process,
            including aspects like `num_bag_folds`, `num_stack_levels`, `auto_stack`,
            `hyperparameters` (for other AutoGluon models), `calibrate_decision_threshold`,
            `infer_limit`, `infer_limit_batch_size`, etc.
            See AutoGluon's `TabularPredictor.fit` documentation for full details.
            Example: `fit_kwargs={'num_bag_folds': 5, 'auto_stack': True}
        """
        self.max_time = max_time
        self.presets = presets
        self.num_gpus = num_gpus
        self.tabpfn_model_type = tabpfn_model_type
        self._predictor_kwargs = predictor_kwargs
        self._predictor: TabularPredictor | None = None

    # ---------------------------------------------------------------------
    # Public scikit-style API
    # ---------------------------------------------------------------------
    def fit(self, X, y):
        """Train a single TabPFN model within *AutoGluon*."""
        training_df = pd.DataFrame(X).copy()
        training_df["_target_"] = y  # lightweight, avoids name clashes

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

        hyperparameters = {TabPFNV2Model: [{
            "ag_args_fit": {"num_gpus": self.num_gpus},
            "model_type": self.tabpfn_model_type,
        }]}

        self._predictor.fit(
            train_data=training_df,
            time_limit=self.max_time,
            presets=self.presets,
            hyperparameters=hyperparameters,
        )
        return self

    # NOTE: TabularPredictor already keeps pandas types; returning numpy keeps it
    # inline with scikit-learn conventions.
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
    """Scikit-style TabPFN classifier powered by AutoGluon."""

    _is_classifier = True


class AutogluonTabPFNRegressor(_BaseAutoGluonTabPFN):
    """Scikit-style TabPFN regressor powered by AutoGluon."""

    _is_classifier = False
