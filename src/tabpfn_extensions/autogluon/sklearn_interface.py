from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


from tabpfn_extensions.autogluon.model import TabPFNV2Model
from tabpfn_extensions.autogluon.utils import search_space_func

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
        presets: Literal[
            "best_quality", "high_quality", "good_quality", "medium_quality"
        ] = "medium_quality",
        num_random_configs: int = 200,
        random_state: int = 1234,
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
        num_random_configs : int, default=200
            Number of random TabPFN hyperparameter configurations to sample.
        random_state : int, default=1234
            Seed for the random number generator to ensure reproducibility.
        **predictor_kwargs : dict, optional
            Additional keyword arguments to pass directly to the `TabularPredictor`
            constructor. This allows for fine-grained control over its behavior.
            
            See the official AutoGluon `TabularPredictor` documentation for a
            complete list of available options.
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ImportError("AutoGluon is required but not installed")
        
        self.max_time = max_time
        self.presets = presets
        self.num_gpus = num_gpus
        self._predictor_kwargs = predictor_kwargs
        self._predictor: TabularPredictor | None = None

        self.num_random_configs = num_random_configs
        self.random_state = random_state

    # ---------------------------------------------------------------------
    # Public scikit-style API
    # ---------------------------------------------------------------------
    def fit(self, X, y):
        """Train a single TabPFN model within *AutoGluon*."""
        from autogluon.tabular import TabularPredictor

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

        task_type = "multiclass" if self._is_classifier else "regression"

        num_configs_to_generate = max(1, self.num_random_configs) # Ensure at least one config

        tabpfn_configs = search_space_func(
            task_type=task_type,
            num_random_configs=num_configs_to_generate,
            seed=self.random_state
        )

        hyperparameters = {
            TabPFNV2Model: tabpfn_configs
        }
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
