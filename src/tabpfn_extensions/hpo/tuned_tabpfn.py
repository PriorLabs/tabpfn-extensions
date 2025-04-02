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
import warnings
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import check_random_state

from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt
from tabpfn_extensions.misc.sklearn_compat import check_array, check_X_y

# Import TabPFN models from extensions (which handles backend compatibility)
try:
    from tabpfn_extensions.utils.utils import TabPFNClassifier, TabPFNRegressor
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

    def __init__(
        self,
        n_trials: int = 50,
        n_validation_size: float = 0.2,
        metric: MetricType = MetricType.ACCURACY,
        device: str = "auto",
        random_state: int | None = None,
        categorical_feature_indices: list[int] | None = None,
        verbose: bool = True,
        search_space: dict[str, Any] | None = None,
        objective_fn: Callable[[Any, np.ndarray, np.ndarray], float] | None = None,
    ):
        self.n_trials = n_trials  # Set n_trials from the parameter
        self.n_validation_size = n_validation_size
        self.metric = MetricType(metric)  # Validate metric type
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = (
            categorical_feature_indices  # Don't modify the parameter
        )
        self.verbose = verbose
        self.search_space = search_space
        self.objective_fn = objective_fn

    def _setup_data_encoders(
        self,
        X: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ):
        """Set up categorical and label encoders."""
        # Use provided indices or instance attribute, create a working copy
        indices = (
            categorical_feature_indices
            if categorical_feature_indices is not None
            else self.categorical_feature_indices
        )

        # Use enhanced infer_categorical_features to detect categorical columns
        # including text and object columns
        from tabpfn_extensions.utils.utils import infer_categorical_features

        detected_indices = infer_categorical_features(X, indices)

        self._categorical_indices = detected_indices

        if not self._categorical_indices:
            logger.info(
                "No categorical features specified or detected. Using all features as numeric.",
            )

        # Create categorical feature encoder
        self._cat_encoder = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    self._categorical_indices,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
        )

    def _optimize(self, X: np.ndarray, y: np.ndarray, task_type: str):
        """Optimize hyperparameters using hyperopt with proper data handling."""
        rng = check_random_state(self.random_state)

        # Set random seeds for reproducibility
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        np.random.seed(rng.randint(0, 2**31 - 1))

        # Store original categorical feature indices before transformation
        original_categorical_indices = (
            self._categorical_indices.copy() if self._categorical_indices else []
        )

        # Check for TestData object and extract raw data
        X_data = X.data if hasattr(X, "data") else X

        # Fit transformers
        X_transformed = self._cat_encoder.fit_transform(X_data)

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
                X_transformed,
                y,
                test_size=self.n_validation_size,
                random_state=rng.randint(0, 2**31 - 1),
                stratify=y if use_stratification else None,
            )
        except ValueError:
            # Fall back to non-stratified split if stratification fails
            logger.warning(
                "Stratified split failed, falling back to random split. "
                "This may happen with very imbalanced data or small sample sizes.",
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_transformed,
                y,
                test_size=self.n_validation_size,
                random_state=rng.randint(0, 2**31 - 1),
                stratify=None,
            )

        # Use custom search space if provided, otherwise use default
        if hasattr(self, "search_space") and self.search_space is not None:
            # For test with simple search space (just a dict with lists of values)
            custom_space = {}
            for k, v in self.search_space.items():
                if isinstance(v, list):
                    custom_space[k] = hp.choice(k, v)
                elif isinstance(v, (int, float, bool, str)) or v is None:
                    # Handle simple values directly
                    custom_space[k] = v
                else:
                    # Try to handle hyperopt objects directly
                    custom_space[k] = v
            search_space = custom_space
        else:
            search_space = get_param_grid_hyperopt(
                "multiclass" if task_type in ["binary", "multiclass"] else "regression",
            )

        def objective(params):
            inference_config = {
                k.split("/")[-1]: v
                for k, v in params.items()
                if k.startswith("inference_config/") and (_k := k.split("/")[-1])
            }

            # Clean up params
            model_params = {
                k: v for k, v in params.items() if not k.startswith("inference_config/")
            }
            model_params["inference_config"] = inference_config
            # Use device utility for automatic selection
            from tabpfn_extensions.utils.utils import get_device

            model_params["device"] = get_device(self.device)
            model_params["random_state"] = rng.randint(0, 2**31 - 1)

            # Since we've already transformed the categorical features, don't specify them again
            # to avoid double-transformation, remove categorical_feature_indices from params
            if "categorical_feature_indices" in model_params:
                model_params.pop("categorical_feature_indices")
            if "categorical_features_indices" in model_params:
                model_params.pop("categorical_features_indices")

            if self.verbose and original_categorical_indices:
                logger.info(
                    f"Original categorical features {original_categorical_indices} already encoded",
                )

            # Handle special parameters
            n_ensemble_repeats = model_params.pop("n_ensemble_repeats", None)
            if n_ensemble_repeats is not None:
                model_params["n_estimators"] = n_ensemble_repeats

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
                # Create and fit model based on model type
                if model_type == "dt_pfn":
                    # Extract decision tree specific parameters
                    max_depth = model_params.pop("max_depth", 3)

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

                # Use custom objective function if provided
                if hasattr(self, "objective_fn") and self.objective_fn is not None:
                    # Custom objective should return a negative score (for minimization)
                    score = -self.objective_fn(model, X_val, y_val)
                # Evaluate based on metric
                elif task_type in ["binary", "multiclass"]:
                    if self.metric == MetricType.ACCURACY:
                        score = model.score(X_val, y_val)
                    elif self.metric in [MetricType.ROC_AUC]:
                        from sklearn.metrics import roc_auc_score

                        y_pred = model.predict_proba(X_val)
                        score = roc_auc_score(y_val, y_pred, multi_class="ovr")
                    elif self.metric == MetricType.F1:
                        from sklearn.metrics import f1_score

                        y_pred = model.predict(X_val)
                        score = f1_score(y_val, y_pred, average="weighted")
                else:
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import (
                        mean_absolute_error,
                        mean_squared_error,
                        r2_score,
                    )

                    # Initialize score variable
                    score = None

                    if self.metric in [MetricType.RMSE, MetricType.MSE]:
                        score = -mean_squared_error(
                            y_val,
                            y_pred,
                            squared=self.metric == MetricType.MSE,
                        )
                    elif self.metric == MetricType.MAE:
                        score = -mean_absolute_error(y_val, y_pred)
                    else:
                        # Default to R2 for any other metric or if not specified
                        score = r2_score(y_val, y_pred)

                # Ensure score is not None before negating
                loss_value = -score if score is not None else float("inf")
                return {"loss": loss_value, "status": STATUS_OK, "model": model}

            except (
                ValueError,
                TypeError,
                RuntimeError,
                ImportError,
                torch.cuda.CudaError,
            ) as e:
                if self.verbose:
                    logger.warning(f"Trial failed with error: {e!s}")
                return {"loss": float("inf"), "status": STATUS_OK}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials,
            verbose=self.verbose,
        )

        # Store results
        self.best_params_ = best
        self.best_score_ = -min(trials.losses())
        self.best_model_ = trials.best_trial["result"].get("model")

        # If all trials failed, create default model
        if self.best_model_ is None:
            warnings.warn(
                "All optimization trials failed. Creating default model.",
                stacklevel=2,
            )
            # Create a default model without specifying categorical features
            # since data has already been transformed
            if task_type in ["binary", "multiclass"]:
                # Check if TabPFN implementation supports categorical_features_indices
                try:
                    self.best_model_ = TabPFNClassifier(
                        device=self.device,
                        random_state=rng.randint(0, 2**31 - 1),
                        categorical_features_indices=None,  # Already encoded
                    )
                except (TypeError, ValueError):
                    # If not, just create without that parameter
                    self.best_model_ = TabPFNClassifier(
                        device=self.device,
                        random_state=rng.randint(0, 2**31 - 1),
                    )
            else:
                # Check if TabPFN implementation supports categorical_features_indices
                try:
                    self.best_model_ = TabPFNRegressor(
                        device=self.device,
                        random_state=rng.randint(0, 2**31 - 1),
                        categorical_features_indices=None,  # Already encoded
                    )
                except (TypeError, ValueError):
                    # If not, just create without that parameter
                    self.best_model_ = TabPFNRegressor(
                        device=self.device,
                        random_state=rng.randint(0, 2**31 - 1),
                    )
            self.best_model_.fit(X_transformed, y)


class TunedTabPFNClassifier(TunedTabPFNBase, ClassifierMixin):
    """TabPFN Classifier with hyperparameter tuning and proper categorical handling."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> TunedTabPFNClassifier:
        # Check if X is a TestData object with categorical_features
        if hasattr(X, "categorical_features") and not categorical_feature_indices:
            categorical_feature_indices = getattr(X, "categorical_features", [])
            if categorical_feature_indices and self.verbose:
                logger.info(
                    f"Using categorical features from TestData: {categorical_feature_indices}",
                )

        # Validate input
        X, y = check_X_y(
            X,
            y,
            ensure_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._setup_data_encoders(X, categorical_feature_indices)
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
        """Check if the model has been fitted."""
        return (
            hasattr(self, "is_fitted_")
            and self.is_fitted_
            and hasattr(self, "best_model_")
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simple fit check instead of check_is_fitted to avoid sklearn Tags issue
        if (
            not hasattr(self, "is_fitted_")
            or not self.is_fitted_
            or not hasattr(self, "best_model_")
        ):
            raise ValueError(
                "This TunedTabPFNClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            )

        # Check if X is a TestData object and extract raw data if needed
        X_data = X.data if hasattr(X, "data") else X

        X_data = check_array(X_data, ensure_all_finite="allow-nan", dtype=object)
        X_data = check_array(
            self._cat_encoder.transform(X_data),
            ensure_all_finite="allow-nan",
            dtype="numeric",
        )
        return self._label_encoder.inverse_transform(self.best_model_.predict(X_data))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Simple fit check instead of check_is_fitted to avoid sklearn Tags issue
        if (
            not hasattr(self, "is_fitted_")
            or not self.is_fitted_
            or not hasattr(self, "best_model_")
        ):
            raise ValueError(
                "This TunedTabPFNClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            )

        # Check if X is a TestData object and extract raw data if needed
        X_data = X.data if hasattr(X, "data") else X

        X_data = check_array(X_data, ensure_all_finite="allow-nan", dtype=object)
        X_data = check_array(
            self._cat_encoder.transform(X_data),
            ensure_all_finite="allow-nan",
            dtype="numeric",
        )
        return self.best_model_.predict_proba(X_data)

    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags


class TunedTabPFNRegressor(TunedTabPFNBase, RegressorMixin):
    """TabPFN Regressor with hyperparameter tuning and proper categorical handling."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> TunedTabPFNRegressor:
        # Check if X is a TestData object with categorical_features
        if hasattr(X, "categorical_features") and not categorical_feature_indices:
            categorical_feature_indices = getattr(X, "categorical_features", [])
            if categorical_feature_indices and self.verbose:
                logger.info(
                    f"Using categorical features from TestData: {categorical_feature_indices}",
                )

        # Validate input
        X, y = check_X_y(
            X,
            y,
            ensure_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._setup_data_encoders(X, categorical_feature_indices)

        # Optimize
        self._optimize(X, y, "regression")

        # Mark as fitted for sklearn
        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self):
        """Check if the model has been fitted."""
        return (
            hasattr(self, "is_fitted_")
            and self.is_fitted_
            and hasattr(self, "best_model_")
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simple fit check instead of check_is_fitted to avoid sklearn Tags issue
        if (
            not hasattr(self, "is_fitted_")
            or not self.is_fitted_
            or not hasattr(self, "best_model_")
        ):
            raise ValueError(
                "This TunedTabPFNRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            )

        # Check if X is a TestData object and extract raw data if needed
        X_data = X.data if hasattr(X, "data") else X

        X_data = check_array(X_data, ensure_all_finite="allow-nan", dtype=object)
        X_data = check_array(
            self._cat_encoder.transform(X_data),
            ensure_all_finite="allow-nan",
            dtype="numeric",
        )
        return self.best_model_.predict(X_data)

    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags
