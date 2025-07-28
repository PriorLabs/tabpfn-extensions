#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Hyperparameter Optimization (HPO) for TabPFN models.

This module provides automatic tuning capabilities for TabPFN models using
Bayesian optimization via Hyperopt. It finds optimal hyperparameters for both
the TabPFN model and its inference configuration.

Key features:
- Optimized search spaces for classification and regression tasks
- Support for multiple evaluation metrics (accuracy, ROC-AUC, F1, RMSE, MAE)
- Proper handling of categorical features through automatic encoding
- Compatible with both TabPFN and TabPFN-client backends
- Implements scikit-learn's estimator interface for easy integration
- Built-in validation strategies for reliable performance estimation
- Configurable search algorithms (TPE, Random Search) and warm-start capabilities.

Example usage:
    ```python
    from tabpfn_extensions.hpo import TunedTabPFNClassifier

    # Create a tuned classifier with 50 optimization trials
    tuned_clf = TunedTabPFNClassifier(
        n_trials=50,                    # Number of hyperparameter configurations to try
        metric='accuracy',              # Metric to optimize
        categorical_feature_indices=[0, 2],  # Categorical features
        random_state=42                 # For reproducibility
    )

    # Fit will automatically find the best hyperparameters
    tuned_clf.fit(X_train, y_train)

    # Use like any scikit-learn estimator
    y_pred = tuned_clf.predict(X_test)
    ```
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, rand, tpe
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt
from tabpfn_extensions.misc.sklearn_compat import validate_data
from tabpfn_extensions.scoring.scoring_utils import (
    score_classification,
    score_regression,
)

# Import TabPFN models from extensions (which handles backend compatibility)
try:
    from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor
except ImportError:
    raise ImportError(
        "TabPFN extensions utils module not found. Please make sure tabpfn_extensions is installed correctly.",
    )

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Supported evaluation metrics for TabPFN hyperparameter tuning.

    This enum defines the metrics that can be used to evaluate and select
    the best hyperparameter configuration during optimization.

    Values:
        ACCURACY: Classification accuracy (proportion of correct predictions)
        ROC_AUC: Area under the ROC curve (for binary or multiclass problems)
        F1: F1 score (harmonic mean of precision and recall)
        RMSE: Root mean squared error (regression)
        MSE: Mean squared error (regression)
        MAE: Mean absolute error (regression)
    """

    ACCURACY = "accuracy"
    ROC_AUC = "roc_auc"
    F1 = "f1"
    RMSE = "rmse"
    MSE = "mse"
    MAE = "mae"


class TunedTabPFNBase(BaseEstimator):
    """Base class for tuned TabPFN models with proper categorical handling."""

    task_type = None

    def __init__(
        self,
        n_trials: int = 50,
        n_validation_size: float = 0.2,
        shuffle_data: bool = True,
        metric: MetricType = MetricType.ACCURACY,
        device: str = "auto",
        random_state: int | None = None,
        categorical_feature_indices: list[int] | None = None,
        verbose: bool = True,
        search_space: dict[str, Any] | None = None,
        objective_fn: Callable[[Any, np.ndarray, np.ndarray], float] | None = None,
        search_algorithm_type: str = "tpe",
        existing_trials: Trials | None = None,
    ):
        self.n_trials = n_trials
        self.n_validation_size = n_validation_size
        self.shuffle_data = shuffle_data
        self.metric = MetricType(metric)
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices
        self.verbose = verbose
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.search_algorithm_type = search_algorithm_type
        self.existing_trials = existing_trials

    def _optimize(self, X: np.ndarray, y: np.ndarray, task_type: str):
        """Optimize hyperparameters using hyperopt with proper data handling."""
        rng = check_random_state(self.random_state)

        # Set random seeds for reproducibility
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        np.random.seed(rng.randint(0, 2**31 - 1))

        # Check if stratification is possible
        use_stratification = False
        if task_type == "multiclass":
            # Check for classes with too few samples for stratification
            from collections import Counter

            from sklearn.utils.multiclass import type_of_target

            y_type = type_of_target(y)
            if y_type in ["binary", "multiclass"]:
                class_counts = Counter(y)
                # Need at least 2 samples per class for stratification
                use_stratification = all(count >= 2 for count in class_counts.values())

        # Split data for validation with error handling
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.n_validation_size,
                random_state=rng.randint(0, 2**31 - 1),
                stratify=y if use_stratification else None,
                shuffle=self.shuffle_data,
            )
        except ValueError:
            # Fall back to non-stratified split if stratification fails
            logger.warning(
                "Stratified split failed, falling back to random split. "
                "This may happen with very imbalanced data or small sample sizes.",
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.n_validation_size,
                random_state=rng.randint(0, 2**31 - 1),
                stratify=None,
                shuffle=self.shuffle_data,
            )

        # Use custom search space if provided, otherwise use default
        if hasattr(self, "search_space") and self.search_space is not None:
            # For test with simple search space (just a dict with lists of values)
            custom_space = {}
            for k, v_item in self.search_space.items():
                if isinstance(v_item, list):
                    custom_space[k] = hp.choice(k, v_item)
                elif isinstance(v_item, (int, float, bool, str)) or v_item is None:
                    custom_space[k] = v_item
                else:
                    custom_space[k] = v_item
            current_search_space = custom_space
        else:
            current_search_space = get_param_grid_hyperopt(
                "multiclass" if task_type in ["binary", "multiclass"] else "regression",
            )

        def objective(params):
            inference_config = {
                k.split("/")[-1]: v
                for k, v in params.items()
                if k.startswith("inference_config/") and (_k := k.split("/")[-1])
            }

            # Print and assert shapes for debugging
            assert (
                len(X_train.shape) == 2
            ), f"X_train shape is {X_train.shape}, should be 2D"
            assert len(X_val.shape) == 2, f"X_val shape is {X_val.shape}, should be 2D"
            assert (
                len(y_train.shape) == 1
            ), f"y_train shape is {y_train.shape}, should be 1D"
            assert len(y_val.shape) == 1, f"y_val shape is {y_val.shape}, should be 1D"

            # Clean up params
            model_params = {
                k: v for k, v in params.items() if not k.startswith("inference_config/")
            }
            model_params["inference_config"] = inference_config
            # Use device utility for automatic selection
            from tabpfn_extensions.utils import infer_device_and_type

            model_params["device"] = infer_device_and_type(self.device)
            model_params["random_state"] = rng.randint(0, 2**31 - 1)

            # Handle model type selection
            model_type = model_params.pop("model_type", "single")

            # Import model implementations based on type
            if model_type == "dt_pfn":
                # Import DecisionTreeTabPFN models
                try:
                    from tabpfn_extensions.rf_pfn import (
                        DecisionTreeTabPFNClassifier,
                        DecisionTreeTabPFNRegressor,
                    )
                except ImportError:
                    # If import fails, skip this trial
                    return {"loss": float("inf"), "status": STATUS_OK}

            try:
                # Extract decision tree specific parameters
                # and remove them from model_params so that
                # instances of TabPFNClassifier or
                # TabPFNRegressor can be initialized properly
                max_depth = model_params.pop("max_depth", 3)
                # Create and fit model based on model type
                if model_type == "dt_pfn":
                    if task_type in ["binary", "multiclass"]:
                        # Use TabPFNClassifier as the base model for DT
                        base_model = TabPFNClassifier(**model_params)
                        model = DecisionTreeTabPFNClassifier(
                            tabpfn=base_model,
                            max_depth=max_depth,
                        )
                    else:
                        # Use TabPFNRegressor as the base model for DT
                        base_model = TabPFNRegressor(**model_params)
                        model = DecisionTreeTabPFNRegressor(
                            tabpfn=base_model,
                            max_depth=max_depth,
                        )
                # Standard single model
                elif task_type in ["binary", "multiclass"]:
                    model = TabPFNClassifier(**model_params)
                else:
                    model = TabPFNRegressor(**model_params)

                model.fit(X_train, y_train)

                score = None
                if hasattr(self, "objective_fn") and self.objective_fn is not None:
                    # Use custom objective function if provided
                    # Custom objective should return a negative score (for minimization)
                    score = -self.objective_fn(model, X_val, y_val)
                elif task_type in ["binary", "multiclass"]:
                    y_pred = model.predict_proba(X_val)
                    if self.metric == MetricType.ACCURACY:
                        score = score_classification("accuracy", y_val, y_pred)
                    elif self.metric in [MetricType.ROC_AUC]:
                        score = score_classification("auroc", y_val, y_pred)
                    elif self.metric == MetricType.F1:
                        score = score_classification("f1", y_val, y_pred)
                else:  # Regression
                    y_pred = model.predict(X_val)
                    if self.metric == MetricType.RMSE:
                        score = score_regression(
                            "rmse",
                            y_val,
                            y_pred,
                        )  # returns negative value already - minimize
                    elif self.metric == MetricType.MSE:
                        score = score_regression(
                            "mse",
                            y_val,
                            y_pred,
                        )  # returns negative value already - minimize
                    elif self.metric == MetricType.MAE:
                        score = score_regression(
                            "mae",
                            y_val,
                            y_pred,
                        )  # returns negative value already - minimize
                    else:  # Default to R2 for regression if metric not MAE/MSE/RMSE
                        score = score_regression(
                            "r2",
                            y_val,
                            y_pred,
                        )  # returns positive value - maximize

                # Ensure score is not None before negating
                loss_value = -score if score is not None else float("inf")
                return {
                    "loss": loss_value,
                    "status": STATUS_OK,
                    "model": model,
                    "params": params,
                }

            except (
                ValueError,
                TypeError,
                RuntimeError,
                ImportError,
                torch.cuda.CudaError,
            ) as e:
                if self.verbose:
                    logger.warning(f"Trial failed with error: {e!s}")
                return {"loss": float("inf"), "status": STATUS_OK, "params": params}

        trials_obj = (
            self.existing_trials if self.existing_trials is not None else Trials()
        )

        if self.search_algorithm_type == "tpe":
            algo_fn = tpe.suggest
        elif self.search_algorithm_type == "random":
            algo_fn = rand.suggest
        else:
            raise ValueError(
                f"Unsupported search_algorithm_type: {self.search_algorithm_type}. "
                "Choose 'tpe' or 'random'."
            )

        best_hyperparams = fmin(
            fn=objective,
            space=current_search_space,
            algo=algo_fn,
            max_evals=self.n_trials,
            trials=trials_obj,
            verbose=self.verbose,
            rstate=np.random.default_rng(
                rng.randint(0, 2**31 - 1)
            ),  # for algo reproducibility
        )

        self.best_params_ = best_hyperparams
        try:
            # Ensure losses are numeric for min()
            valid_losses = [loss for loss in trials_obj.losses() if loss is not None]
            if not valid_losses:  # All trials might have failed or returned None loss
                self.best_score_ = -float("inf")  # or some other indicator of failure
            else:
                self.best_score_ = -min(valid_losses)
        except (
            TypeError
        ):  # Handle cases where losses might not be comparable (e.g. None)
            self.best_score_ = -float("inf")  # Or handle as appropriate

        # Retrieve the best model from the best trial's result
        if (
            trials_obj.best_trial
            and "result" in trials_obj.best_trial
            and trials_obj.best_trial["result"].get("status") == STATUS_OK
        ):
            self.best_model_ = trials_obj.best_trial["result"].get("model")
        else:
            self.best_model_ = None

        # Remove model objects from all trials' results to save memory, keeping params and loss
        for trial_info in trials_obj.trials:
            if "result" in trial_info and "model" in trial_info["result"]:
                del trial_info["result"]["model"]
        self.trials_ = trials_obj  # Store the modified trials history

        if self.best_model_ is None:
            raise ValueError(
                "No valid model was found during hyperparameter optimization. "
                "Check the trials for errors or issues with the search space.",
            )

    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        if self.task_type == "multiclass":
            tags.estimator_type = "classifier"
        else:
            tags.estimator_type = "regressor"
        return tags


class TunedTabPFNClassifier(TunedTabPFNBase, ClassifierMixin):
    """TabPFN Classifier with hyperparameter tuning and proper categorical handling."""

    task_type = "multiclass"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> TunedTabPFNClassifier:
        # Validate input
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._label_encoder = LabelEncoder()
        y_transformed = self._label_encoder.fit_transform(y)

        # Store classes
        self.classes_ = self._label_encoder.classes_.copy()

        # Determine task type and optimize
        task_type = "multiclass" if len(self.classes_) > 2 else "binary"
        self._optimize(X, y_transformed, task_type)

        # Mark as fitted for sklearn
        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "is_fitted_")
            and self.is_fitted_
            and hasattr(self, "best_model_")
            and self.best_model_ is not None  # Ensure best_model_ is not None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise ValueError(
                "This TunedTabPFNClassifier instance is not fitted yet or fitting failed. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = validate_data(
            self,
            X,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )

        # Check if best_model_ itself is fitted (e.g. if default model fitting failed)
        if (
            not hasattr(self.best_model_, "classes_")
            and not hasattr(self.best_model_, "_get_tags")
            and not getattr(self.best_model_, "is_fitted_", True)
        ):  # Heuristic check
            raise ValueError("The underlying best_model_ is not properly fitted.")

        return self._label_encoder.inverse_transform(self.best_model_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise ValueError(
                "This TunedTabPFNClassifier instance is not fitted yet or fitting failed. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = validate_data(
            self,
            X,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )

        # Check if best_model_ itself is fitted
        if (
            not hasattr(self.best_model_, "classes_")
            and not hasattr(self.best_model_, "_get_tags")
            and not getattr(self.best_model_, "is_fitted_", True)
        ):  # Heuristic check
            raise ValueError("The underlying best_model_ is not properly fitted.")

        return self.best_model_.predict_proba(X)


class TunedTabPFNRegressor(TunedTabPFNBase, RegressorMixin):
    """TabPFN Regressor with hyperparameter tuning and proper categorical handling."""

    task_type = "regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> TunedTabPFNRegressor:
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )

        self.n_features_in_ = X.shape[1]
        self._optimize(X, y, self.task_type)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return (
            hasattr(self, "is_fitted_")
            and self.is_fitted_
            and hasattr(self, "best_model_")
            and self.best_model_ is not None  # Ensure best_model_ is not None
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.__sklearn_is_fitted__():
            raise ValueError(
                "This TunedTabPFNRegressor instance is not fitted yet or fitting failed. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )

        X = validate_data(
            self,
            X,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )

        # Check if best_model_ itself is fitted
        # Regressors might not have 'classes_', so check for a common fit attribute or tag.
        if not hasattr(self.best_model_, "_get_tags") and not getattr(
            self.best_model_, "is_fitted_", True
        ):  # Heuristic check for scikit-learn like estimators
            raise ValueError("The underlying best_model_ is not properly fitted.")

        return self.best_model_.predict(X)
