#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import numpy as np
import torch
import warnings
from enum import Enum
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, check_X_y
from typing import Literal, Optional

from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
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
        device: str = "cpu",
        random_state: Optional[int] = None,
        categorical_feature_indices: Optional[list[int]] = None,
        verbose: bool = True,
    ):
        self.n_trials = n_trials
        self.n_validation_size = n_validation_size
        self.metric = MetricType(metric)  # Validate metric type
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices or []
        self.verbose = verbose

    def _setup_data_encoders(
        self, X: np.ndarray, categorical_feature_indices: Optional[list[int]] = None
    ):
        """Set up categorical and label encoders."""
        if categorical_feature_indices is not None:
            self.categorical_feature_indices = categorical_feature_indices

        if not self.categorical_feature_indices:
            logger.info(
                "No categorical features specified. Using all features as numeric."
            )
            self.categorical_feature_indices = []

        # Create categorical feature encoder
        self._cat_encoder = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    self.categorical_feature_indices,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
        )

    def _optimize(self, X: np.ndarray, y: np.ndarray, task_type: str):
        """Optimize hyperparameters using hyperopt with proper data handling."""
        rng = check_random_state(self.random_state)

        # Set random seeds for reproducibility
        torch.manual_seed(rng.randint(0, 2**32 - 1,dtype=np.int64))
        np.random.seed(rng.randint(0, 2**32 - 1,dtype=np.int64))

        # Fit transformers
        X = self._cat_encoder.fit_transform(X)
        if hasattr(self, "_label_encoder"):
            y = self._label_encoder.transform(y)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.n_validation_size,
            random_state=rng.randint(0, 2**32 - 1,dtype=np.int64),
            stratify=y if task_type == "multiclass" else None,
        )

        search_space = get_param_grid_hyperopt(
            "multiclass" if task_type in ["binary", "multiclass"] else "regression"
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
            model_params["device"] = self.device
            model_params["random_state"] = rng.randint(0, 2**32 - 1,dtype=np.int64)

            # Handle special parameters
            n_ensemble_repeats = model_params.pop("n_ensemble_repeats", None)
            if n_ensemble_repeats is not None:
                model_params["n_estimators"] = n_ensemble_repeats

            # Handle model type - skip RF-PFN for now
            model_type = model_params.pop("model_type", "single")
            if model_type == "dt_pfn":
                return {"loss": float("inf"), "status": STATUS_OK}

            try:
                # Create and fit model
                if task_type in ["binary", "multiclass"]:
                    model = TabPFNClassifier(**model_params)
                else:
                    model = TabPFNRegressor(**model_params)

                model.fit(X_train, y_train)

                # Evaluate based on metric
                if task_type in ["binary", "multiclass"]:
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
                    from sklearn.metrics import mean_squared_error, mean_absolute_error

                    if self.metric in [MetricType.RMSE, MetricType.MSE]:
                        score = -mean_squared_error(
                            y_val, y_pred, squared=self.metric == MetricType.MSE
                        )
                    elif self.metric == MetricType.MAE:
                        score = -mean_absolute_error(y_val, y_pred)

                return {"loss": -score, "status": STATUS_OK, "model": model}

            except Exception as e:
                if self.verbose:
                    logger.warning(f"Trial failed with error: {str(e)}")
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
            if task_type in ["binary", "multiclass"]:
                self.best_model_ = TabPFNClassifier(
                    device=self.device,
                    random_state=rng.randint(0, 2**32 - 1,dtype=np.int64),
                )
            else:
                self.best_model_ = TabPFNRegressor(
                    device=self.device,
                    random_state=rng.randint(0, 2**32 - 1,dtype=np.int64),
                )
            self.best_model_.fit(X, y)


class TunedTabPFNClassifier(TunedTabPFNBase, ClassifierMixin):
    """TabPFN Classifier with hyperparameter tuning and proper categorical handling."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: Optional[list[int]] = None,
    ) -> TunedTabPFNClassifier:
        # Validate input
        X, y = check_X_y(
            X,
            y,
            force_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._setup_data_encoders(X, categorical_feature_indices)
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)

        # Store classes
        self.classes_ = self._label_encoder.classes_

        # Determine task type and optimize
        task_type = "multiclass" if len(self.classes_) > 2 else "binary"
        self._optimize(X, y, task_type)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, force_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            force_all_finite="allow-nan",
            dtype="numeric",
        )
        return self._label_encoder.inverse_transform(self.best_model_.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, force_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            force_all_finite="allow-nan",
            dtype="numeric",
        )
        return self.best_model_.predict_proba(X)

    def _more_tags(self):
        return {"allow_nan": True}


class TunedTabPFNRegressor(TunedTabPFNBase, RegressorMixin):
    """TabPFN Regressor with hyperparameter tuning and proper categorical handling."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: Optional[list[int]] = None,
    ) -> TunedTabPFNRegressor:
        # Validate input
        X, y = check_X_y(
            X,
            y,
            force_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )

        # Store dimensions
        self.n_features_in_ = X.shape[1]

        # Set up encoders
        self._setup_data_encoders(X, categorical_feature_indices)

        # Optimize
        self._optimize(X, y, "regression")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, force_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            force_all_finite="allow-nan",
            dtype="numeric",
        )
        return self.best_model_.predict(X)

    def _more_tags(self):
        return {"allow_nan": True}
