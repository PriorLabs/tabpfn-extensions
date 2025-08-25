

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions.utils import infer_categorical_features, infer_device_and_type


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class AutoTabPFN_TS_Base(BaseEstimator):


    def __init__(
        self,
        *,
        max_time: int | None = 3600,
        eval_metric: str | None = None,
        presets: list[str] | str | None = "best_quality",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        n_ensemble_models: int = 20,
        n_estimators: int = 8,
        ignore_pretraining_limits: bool = False,
    ):
        self.max_time = max_time
        self.eval_metric = eval_metric
        self.presets = presets
        self.device = device
        if isinstance(random_state, np.random.Generator):
            random_state = random_state.integers(np.iinfo(np.int32).max)
        self.random_state = random_state
        self.phe_init_args = phe_init_args
        self.phe_fit_args = phe_fit_args
        self.n_ensemble_models = n_ensemble_models
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits

        self._is_classifier = False

    def _get_predictor_init_args(self) -> dict[str, Any]:
        """Constructs the initialization arguments for AutoGluon's TabularPredictor."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_args = {"verbosity": 1, "path": f"TabPFNModels/m-{timestamp}"}
        user_args = self.phe_init_args or {}
        return {**default_args, **user_args}

    def _get_predictor_fit_args(self) -> dict[str, Any]:
        """Constructs the fit arguments for AutoGluon's TabularPredictor."""
        default_args = {
            #"num_bag_folds": 8,
            #"fit_weighted_ensemble": True,
        }
        user_args = self.phe_fit_args or {}
        return {**default_args, **user_args}

    def _prepare_fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_feature_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | np.ndarray]:
        """Sets up the training environment and normalizes input data for fitting.

        This helper method performs the initial setup before the main training
        process. It determines the computation device (CPU/GPU), validates key
        model parameters, and ensures the feature matrix `X` is a Pandas
        DataFrame. If the input `X` is a NumPy array, it is converted to a
        DataFrame, using the provided `feature_names` or generating default names.
        Finally, it resolves the categorical feature indices to be used.
        """
        self.device_ = infer_device_and_type(self.device)
        if self.n_ensemble_models < 1:
            raise ValueError(
                f"n_ensemble_models must be >= 1, got {self.n_ensemble_models}"
            )
        if self.max_time is not None and self.max_time <= 0:
            raise ValueError("max_time must be a positive integer or None.")

        if not isinstance(X, pd.DataFrame):
            original_columns = feature_names or [f"f{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=original_columns)

        self.feature_names_in_ = X.columns.to_numpy(dtype=object)

        # Auto-detect if still not specified and store in a new "fitted" attribute
        if categorical_feature_indices is None:
            self.categorical_feature_indices_ = infer_categorical_features(X)
        else:
            self.categorical_feature_indices_ = categorical_feature_indices

        return X, y

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        """Fits the model by training an ensemble of TabPFN configurations using AutoGluon.
        This method should be called from the child class's fit method after validation.
        """
        #from autogluon.tabular import TabularPredictor
        from autogluon.timeseries import TimeSeriesPredictor
        #from autogluon.tabular.models import TabPFNV2Model
        # TODO: Get TabPFN from somewehre else

        from tabpfn_extensions.post_hoc_ensembles.utils import search_space_func

        if isinstance(X, pd.DataFrame):
            training_df = X.copy()
            self._column_names = X.columns.tolist()
        else:
            self._column_names = [f"f{i}" for i in range(X.shape[1])]
            training_df = pd.DataFrame(X, columns=self._column_names)

        training_df["_target_"] = y


        self.predictor_ = TimeSeriesPredictor(
            label="_target_",
            eval_metric=self.eval_metric,
            **self._get_predictor_init_args(),
        )

        # Generate hyperparameter configurations for TabPFN Ensemble

        task_type = "multiclass" if self._is_classifier else "regression"

        if self.n_ensemble_models > 1:
            rng = check_random_state(self.random_state)
            seed = rng.randint(np.iinfo(np.int32).max)

            tabpfn_configs = search_space_func(
                task_type=task_type,
                n_ensemble_models=self.n_ensemble_models,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
                seed=seed,
                **self.get_task_args_(),
            )
        else:
            tabpfn_configs = {
                "n_estimators": self.n_estimators,
                "ignore_pretraining_limits": self.ignore_pretraining_limits,
                **self.get_task_args_(),
            }
        hyperparameters = {TabPFNV2Model: tabpfn_configs}

        # Set GPU count
        num_gpus = 0
        if self.device_.type == "cuda":
            num_gpus = torch.cuda.device_count()

        self.predictor_.fit(
            train_data=training_df,
            time_limit=self.max_time,
            presets=self.presets,
            hyperparameters=hyperparameters,
            #num_gpus=num_gpus,
            **self._get_predictor_fit_args(),
        )

        # Set sklearn required attributes from the fitted predictor
        self.n_features_in_ = len(self.predictor_.features())

        return self

    def get_task_args_(self) -> dict[str, Any]:
        """Returns task-specific arguments for the TabPFN search space."""
        return {}

    def _more_tags(self):
        return {"allow_nan": True, "non_deterministic": True}


class AutoTabPFN_TS_Classifier(ClassifierMixin, AutoTabPFN_TS_Base):
    def __init__(
        self,
        *,
        max_time: int | None = 3600,
        eval_metric: str | None = None,
        presets: list[str] | str | None = "best_quality",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        n_ensemble_models: int = 20,
        n_estimators: int = 8,
        balance_probabilities: bool = False,
        ignore_pretraining_limits: bool = False,
    ):
        super().__init__(
            max_time=max_time,
            eval_metric=eval_metric,
            presets=presets,
            device=device,
            random_state=random_state,
            phe_init_args=phe_init_args,
            phe_fit_args=phe_fit_args,
            n_ensemble_models=n_ensemble_models,
            n_estimators=n_estimators,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )

        self.balance_probabilities = balance_probabilities
        self._is_classifier = True

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_feature_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> AutoTabPFNClassifier:
        X, y = self._prepare_fit(
            X,
            y,
            categorical_feature_indices=categorical_feature_indices,
            feature_names=feature_names,
        )

        # Encode labels to be 0-indexed and set self.classes_
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.n_features_in_ = X.shape[1]

        # Single class case - special handling
        if len(self.classes_) == 1:
            self.single_class_ = True
            self.single_class_value_ = self.classes_[0]
            self.n_features_in_ = X.shape[1]
            return self

        # Normal case - multiple classes with sufficient samples per class
        self.single_class_ = False
        super().fit(X, y_encoded)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        if hasattr(self, "single_class_") and self.single_class_:
            return np.full(X.shape[0], self.single_class_value_)

        preds = self.predictor_.predict(pd.DataFrame(X, columns=self._column_names))
        # Decode predictions back to original labels.
        return self.label_encoder_.inverse_transform(preds.to_numpy())

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        if hasattr(self, "single_class_") and self.single_class_:
            # Return correct (n_samples, n_classes) shape
            proba = np.zeros((X.shape[0], len(self.classes_)))
            proba[:, 0] = 1.0
            return proba

        # Re-align predict_proba output to match self.classes_
        proba_df = self.predictor_.predict_proba(
            pd.DataFrame(X, columns=self._column_names), as_pandas=True
        )
        original_cols = self.label_encoder_.inverse_transform(proba_df.columns)
        proba_df.columns = original_cols
        return proba_df.reindex(columns=self.classes_).to_numpy()

    def get_task_args_(self) -> dict[str, Any]:
        return {"balance_probabilities": self.balance_probabilities}


class AutoTabPFN_TS_Regressor(RegressorMixin, AutoTabPFN_TS_Base):


    def __init__(
        self,
        *,
        max_time: int | None = 3600,
        eval_metric: str | None = None,
        presets: list[str] | str | None = "best_quality",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        n_ensemble_models: int = 20,
        n_estimators: int = 8,
        ignore_pretraining_limits: bool = False,
    ):
        super().__init__(
            max_time=max_time,
            eval_metric=eval_metric,
            presets=presets,
            device=device,
            random_state=random_state,
            phe_init_args=phe_init_args,
            phe_fit_args=phe_fit_args,
            n_ensemble_models=n_ensemble_models,
            n_estimators=n_estimators,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )

        self._is_classifier = False

    def _more_tags(self) -> dict:
        return {"allow_nan": True, "non_deterministic": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_feature_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> AutoTabPFNRegressor:
        X, y = self._prepare_fit(
            X,
            y,
            categorical_feature_indices=categorical_feature_indices,
            feature_names=feature_names,
        )
        super().fit(X, y)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        preds = self.predictor_.predict(pd.DataFrame(X, columns=self._column_names))
        return preds.to_numpy()