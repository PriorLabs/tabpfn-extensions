#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""TabPFN implementation in AutoGluon taken from TabArena: A Living Benchmark for Machine Learning on Tabular Data,
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas,
Frank Hutter, Preprint., 2025.
"""

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


class AutoTabPFNBase(BaseEstimator):
    """An AutoGluon-powered scikit-learn wrapper for ensembling TabPFN models.

    This class serves as a base for creating powerful classification and regression
    models by building a post-hoc ensemble of multiple TabPFN configurations using
    AutoGluon. This approach leverages AutoGluon's robust ensembling strategies to
    combine predictions from various specialized TabPFN models, often leading to
    state-of-the-art performance on tabular datasets.

    The implementation is based on the methodology presented in the "TabArena" paper.

    Parameters
    ----------
    max_time : int | None, default=3600
        Maximum time in seconds to train the ensemble. If `None`, training will run until
        all models are fitted.
    eval_metric : str | None, default=None
        The evaluation metric for AutoGluon to optimize. If `None`, a default metric
        is chosen based on the problem type (e.g., 'accuracy' for classification).
        For a full list of options, see the AutoGluon documentation.
    presets : list[str] | str | None, default=None
        AutoGluon preset to control the quality-time trade-off.
    device : {"cpu", "cuda", "auto"}, default="auto"
        The device to use for training. "auto" will select "cuda" if available, otherwise "cpu".
    random_state : int | np.random.RandomState | None, default=None
        Controls the randomness for both base model training and the ensembling process.
    categorical_feature_indices : list[int] | None, default=None
        Indices of the categorical features in the input data. If `None`, they will be
        automatically inferred during `fit()`.
    phe_init_args : dict | None, default=None
        Advanced customization arguments passed directly to the `TabularPredictor`
        constructor in AutoGluon. See the AutoGluon documentation for details.
    phe_fit_args : dict | None, default=None
        Advanced customization arguments passed to the `TabularPredictor.fit()` method
        in AutoGluon. See the AutoGluon documentation for details.
    n_ensemble_models : int, default=5
        The number of random TabPFN configurations to generate and include in the
        AutoGluon model zoo for ensembling. TabArena used 200 models for their final
        evaluation. In the case of a single model, the model is not ensembled and the
        model is fitted with the default hyperparameters (with optional `n_estimators`
        and `ignore_pretraining_limits` parameters).
    n_estimators : int, default=8
        The number of internal transformers to ensemble within each individual TabPFN model.
        Higher values can improve performance but increase resource usage.
    ignore_pretraining_limits : bool, default=False
        If `True`, bypasses TabPFN's built-in limits on dataset size (10000 samples)
        and feature count (500). **Warning:** Use with caution, as performance is not
        guaranteed and may be poor when exceeding these limits.

    Attributes:
    ----------
    predictor_ : autogluon.tabular.TabularPredictor
        The fitted AutoGluon predictor object that manages the ensemble.
    categorical_feature_indices_ : list[int]
        The effective list of categorical feature indices used by the model.
    classes_ : np.ndarray
        For classifiers, an array of class labels known to the model.
    n_features_in_ : int
        The number of features seen during `fit()`.
    _column_names : list[str]
        Internal list of feature names used for prediction.
    """

    def __init__(
        self,
        *,
        max_time: int | None = 3600,
        eval_metric: str | None = None,
        presets: list[str] | str | None = None,
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
            "num_bag_folds": 8,
            "fit_weighted_ensemble": True,
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
        from autogluon.tabular import TabularPredictor
        from autogluon.tabular.models import TabPFNV2Model

        from tabpfn_extensions.post_hoc_ensembles.utils import search_space_func

        if isinstance(X, pd.DataFrame):
            training_df = X.copy()
            self._column_names = X.columns.tolist()
        else:
            self._column_names = [f"f{i}" for i in range(X.shape[1])]
            training_df = pd.DataFrame(X, columns=self._column_names)

        training_df["_target_"] = y

        problem_type = (
            TaskType.BINARY
            if self._is_classifier and len(np.unique(y)) == 2
            else (TaskType.MULTICLASS if self._is_classifier else TaskType.REGRESSION)
        )

        self.predictor_ = TabularPredictor(
            label="_target_",
            problem_type=problem_type,
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
            num_gpus=num_gpus,
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


class AutoTabPFNClassifier(ClassifierMixin, AutoTabPFNBase):
    """An AutoGluon-powered scikit-learn wrapper for ensembling TabPFN classifiers.

    This model creates a post-hoc ensemble of multiple TabPFN configurations using
    AutoGluon, leveraging its ensembling strategies for state-of-the-art performance.
    It is designed for binary and multi-class classification tasks.

    The implementation is based on the methodology from the "TabArena" paper.

    Parameters
    ----------
    max_time : int | None, default=3600
        Maximum time in seconds to train the ensemble.
    eval_metric : str | None, default=None
        Metric for AutoGluon to optimize. Defaults to 'accuracy'.
    presets : list[str] | str | None, default=None
        AutoGluon preset to control the quality-time trade-off.
    device : {"cpu", "cuda", "auto"}, default="auto"
        Device for training. "auto" selects "cuda" if available.
    random_state : int | np.random.RandomState | None, default=None
        Controls randomness for reproducibility.
    phe_init_args : dict | None, default=None
        Advanced arguments for AutoGluon's `TabularPredictor` constructor.
    phe_fit_args : dict | None, default=None
        Advanced arguments for AutoGluon's `TabularPredictor.fit()` method.
    n_ensemble_models : int, default=5
        The number of random TabPFN configurations to generate and include in the
        AutoGluon model zoo for ensembling. TabArena used 200 models for their final
        evaluation.
    n_estimators : int, default=8
        The number of internal transformers to ensemble within each individual TabPFN model.
        Higher values can improve performance but increase resource usage.
    balance_probabilities : bool, default=False
        Whether to balance the output probabilities from TabPFN. This can be beneficial
        for classification tasks with imbalanced classes.
    ignore_pretraining_limits : bool, default=False
        If `True`, bypasses TabPFN's built-in limits on dataset size (10000 samples)
        and feature count (500). **Warning:** Use with caution, as performance is not
        guaranteed and may be poor when exceeding these limits.

    Attributes:
    ----------
    predictor_ : autogluon.tabular.TabularPredictor
        The fitted AutoGluon predictor managing the ensemble.
    categorical_feature_indices_ : list[int]
        The effective list of categorical feature indices used by the model.
    classes_ : np.ndarray
        An array of class labels known to the classifier.
    n_features_in_ : int
        The number of features seen during `fit()`.
    """

    def __init__(
        self,
        *,
        max_time: int | None = 3600,
        eval_metric: str | None = None,
        presets: list[str] | str | None = None,
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


class AutoTabPFNRegressor(RegressorMixin, AutoTabPFNBase):
    """An AutoGluon-powered scikit-learn wrapper for ensembling TabPFN regressors.

    This model creates a post-hoc ensemble of multiple TabPFN configurations using
    AutoGluon, leveraging its ensembling strategies for state-of-the-art performance.
    It is designed for regression tasks.

    The implementation is based on the methodology from the "TabArena" paper.

    Parameters
    ----------
    max_time : int | None, default=3600
        Maximum time in seconds to train the ensemble.
    eval_metric : str | None, default=None
        Metric for AutoGluon to optimize. Defaults to 'root_mean_squared_error'.
    presets : list[str] | str | None, default=None
        AutoGluon preset to control the quality-time trade-off.
    device : {"cpu", "cuda", "auto"}, default="auto"
        Device for training. "auto" selects "cuda" if available.
    random_state : int | np.random.RandomState | None, default=None
        Controls randomness for reproducibility.
    phe_init_args : dict | None, default=None
        Advanced arguments for AutoGluon's `TabularPredictor` constructor.
    phe_fit_args : dict | None, default=None
        Advanced arguments for AutoGluon's `TabularPredictor.fit()` method.
    n_ensemble_models : int, default=5
        The number of random TabPFN configurations to generate and include in the
        AutoGluon model zoo for ensembling. TabArena used 200 models for their final
        evaluation.
    n_estimators : int, default=8
        The number of internal transformers to ensemble within each individual TabPFN model.
        Higher values can improve performance but increase resource usage.
    ignore_pretraining_limits : bool, default=False
        If `True`, bypasses TabPFN's built-in limits on dataset size (10000 samples)
        and feature count (500). **Warning:** Use with caution, as performance is not
        guaranteed and may be poor when exceeding these limits.

    Attributes:
    ----------
    predictor_ : autogluon.tabular.TabularPredictor
        The fitted AutoGluon predictor managing the ensemble.
    categorical_feature_indices_ : list[int]
        The effective list of categorical feature indices used by the model.
    n_features_in_ : int
        The number of features seen during `fit()`.
    """

    def __init__(
        self,
        *,
        max_time: int | None = 3600,
        eval_metric: str | None = None,
        presets: list[str] | str | None = None,
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
