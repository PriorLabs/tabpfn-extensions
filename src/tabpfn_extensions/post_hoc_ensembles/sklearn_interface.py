#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Implementation taken from TabArena: A Living Benchmark for Machine Learning on Tabular Data,
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas,
Frank Hutter, Preprint., 2025,

Original Code: https://github.com/autogluon/tabrepo/tree/main/tabrepo/benchmark/models/ag/tabpfnv2
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions.misc.sklearn_compat import validate_data
from tabpfn_extensions.utils import (
    TabPFNClassifier,
    get_device,
    infer_categorical_features,
)


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


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
    max_time : int | None, default=60*3
        Maximum time in seconds to train the ensemble. If `None`, training will run until
        all models are fitted.
    eval_metric : str | None, default=None
        The evaluation metric for AutoGluon to optimize. If `None`, a default metric
        is chosen based on the problem type (e.g., 'accuracy' for classification).
        For a full list of options, see the AutoGluon documentation.
    presets : {"best_quality", "high_quality", "good_quality", "medium_quality"}, default="medium_quality"
        AutoGluon preset to control the trade-off between training time and predictive accuracy.
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
    n_ensemble_models : int, default=200
        The number of random TabPFN configurations to generate and include in the
        AutoGluon model zoo for ensembling.
    n_estimators : int, default=16
        The number of internal transformers to ensemble within each individual TabPFN model.
        Higher values can improve performance but increase resource usage.
    balance_probabilities : bool, default=False
        Whether to balance the output probabilities from TabPFN. This can be beneficial
        for classification tasks with imbalanced classes.
    ignore_pretraining_limits : bool, default=False
        If `True`, bypasses TabPFN's built-in limits on dataset size (1024 samples)
        and feature count (100). **Warning:** Use with caution, as performance is not
        guaranteed and may be poor when exceeding these limits.

    Attributes
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
        max_time: int | None = 60 * 3,
        eval_metric: str | None = None,
        presets: Literal[
            "best_quality", "high_quality", "good_quality", "medium_quality"
        ] = "medium_quality",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        categorical_feature_indices: list[int] | None = None,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        n_ensemble_models: int = 200,
        n_estimators: int = 16,
        balance_probabilities: bool = False,
        ignore_pretraining_limits: bool = False,
    ):
        self.max_time = max_time
        self.eval_metric = eval_metric
        self.presets = presets
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices
        self.phe_init_args = phe_init_args
        self.phe_fit_args = phe_fit_args
        self.n_ensemble_models = n_ensemble_models
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.ignore_pretraining_limits = ignore_pretraining_limits

        assert n_ensemble_models >= 1, "n_ensemble_models must be >= 1"

    def _get_predictor_init_args(self) -> dict[str, Any]:
        """Constructs the initialization arguments for AutoGluon's TabularPredictor."""
        default_args = {"verbosity": 5}
        user_args = self.phe_init_args or {}
        return {**default_args, **user_args}

    def _get_predictor_fit_args(self) -> dict[str, Any]:
        """Constructs the fit arguments for AutoGluon's TabularPredictor."""
        default_args = {
            "num_bag_folds": 5,
            "num_bag_sets": 8,
            "num_stack_levels": 1,
            "fit_weighted_ensemble": True,
            "ag_args_ensemble": {"fit_strategy": "parallel"},
        }
        user_args = self.phe_fit_args or {}
        return {**default_args, **user_args}

    def _prepare_fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
        original_columns = None
        if isinstance(X, pd.DataFrame):
            original_columns = X.columns.tolist()

        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,
        )

        if original_columns is not None:
            X = pd.DataFrame(X, columns=original_columns)

        effective_cat_indices = categorical_feature_indices
        if effective_cat_indices is None:
            effective_cat_indices = self.categorical_feature_indices

        # Auto-detect if still not specified and store in a new "fitted" attribute
        if effective_cat_indices is None:
            self.categorical_feature_indices_ = infer_categorical_features(X)
        else:
            self.categorical_feature_indices_ = effective_cat_indices

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
        tabpfn_configs = search_space_func(
            task_type=task_type,
            n_ensemble_models=self.n_ensemble_models,
            seed=self.random_state,
            n_estimators=self.n_estimators,
            balance_probabilities=self.balance_probabilities,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        hyperparameters = {TabPFNV2Model: tabpfn_configs}

        # TODO: Code limited to 1 GPU, infer dynamically
        device = get_device(self.device)
        num_gpus = 1 if device == DeviceType.CUDA else 0

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
        if self._is_classifier:
            # Storing the classes_ in the order AutoGluon uses them
            self.classes_ = self.predictor_.class_labels

        return self

    def _more_tags(self):
        return {
            "allow_nan": True,
        }


class AutoTabPFNClassifier(ClassifierMixin, AutoTabPFNBase):
    """The implementation is based on the methodology presented in the "TabArena" paper.

    Parameters
    ----------
    max_time : int | None, default=60*3
        Maximum time in seconds to train the ensemble. If `None`, training will run until
        all models are fitted.
    eval_metric : str | None, default=None
        The evaluation metric for AutoGluon to optimize. If `None`, a default metric
        is chosen based on the problem type (e.g., 'accuracy' for classification).
        For a full list of options, see the AutoGluon documentation.
    presets : {"best_quality", "high_quality", "good_quality", "medium_quality"}, default="medium_quality"
        AutoGluon preset to control the trade-off between training time and predictive accuracy.
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
    n_ensemble_models : int, default=200
        The number of random TabPFN configurations to generate and include in the
        AutoGluon model zoo for ensembling.
    n_estimators : int, default=16
        The number of internal transformers to ensemble within each individual TabPFN model.
        Higher values can improve performance but increase resource usage.
    balance_probabilities : bool, default=False
        Whether to balance the output probabilities from TabPFN. This can be beneficial
        for classification tasks with imbalanced classes.
    ignore_pretraining_limits : bool, default=False
        If `True`, bypasses TabPFN's built-in limits on dataset size (1024 samples)
        and feature count (100). **Warning:** Use with caution, as performance is not
        guaranteed and may be poor when exceeding these limits.

    Attributes
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
        max_time: int | None = 60 * 3,
        eval_metric: str | None = None,
        presets: Literal[
            "best_quality", "high_quality", "good_quality", "medium_quality"
        ] = "medium_quality",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        categorical_feature_indices: list[int] | None = None,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        n_ensemble_models: int = 200,
        n_estimators: int = 16,
        balance_probabilities: bool = False,
        ignore_pretraining_limits: bool = False,
    ):
        self.max_time = max_time
        self.eval_metric = eval_metric
        self.presets = presets
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices
        self.phe_init_args = phe_init_args
        self.phe_fit_args = phe_fit_args
        self.n_ensemble_models = n_ensemble_models
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.ignore_pretraining_limits = ignore_pretraining_limits

        assert n_ensemble_models >= 1, "n_ensemble_models must be >= 1"

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
    ) -> AutoTabPFNClassifier:
        X, y = self._prepare_fit(X, y, categorical_feature_indices)

        # Single class case - special handling
        if len(unique_labels(y)) == 1:
            self.single_class_ = True
            self.classes_ = unique_labels(y)
            self.single_class_value_ = self.classes_[0]
            self.n_features_in_ = X.shape[1]
            return self

        # Check for extremely imbalanced classes - handle case with only 1 sample per class
        class_counts = np.bincount(y.astype(int))
        if np.min(class_counts[class_counts > 0]) < 2:
            self.single_class_ = False
            self.predictor_ = TabPFNClassifier(
                device=get_device(self.device),
                categorical_features_indices=self.categorical_feature_indices_,
            )
            self.predictor_.fit(X, y)
            # Store the classes
            self.classes_ = self.predictor_.classes_
            self.n_features_in_ = X.shape[1]
            return self

        # Normal case - multiple classes with sufficient samples per class
        self.single_class_ = False

        super().fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        if hasattr(self, "single_class_") and self.single_class_:
            # For single class, always predict that class
            return np.full(X.shape[0], self.single_class_value_)
        # Convert to pandas dataframe for AutoGluon
        preds = self.predictor_.predict(pd.DataFrame(X, columns=self._column_names))
        # Convert back to numpy array for sklearn
        return preds.to_numpy()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        if hasattr(self, "single_class_") and self.single_class_:
            # For single class, return probabilities of 1.0
            return np.ones((X.shape[0], 1))

        preds = self.predictor_.predict_proba(
            pd.DataFrame(X, columns=self._column_names)
        )
        return preds.to_numpy()


class AutoTabPFNRegressor(RegressorMixin, AutoTabPFNBase):
    """The implementation is based on the methodology presented in the "TabArena" paper.

    Parameters
    ----------
    max_time : int | None, default=60*3
        Maximum time in seconds to train the ensemble. If `None`, training will run until
        all models are fitted.
    eval_metric : str | None, default=None
        The evaluation metric for AutoGluon to optimize. If `None`, a default metric
        is chosen based on the problem type (e.g., 'accuracy' for classification).
        For a full list of options, see the AutoGluon documentation.
    presets : {"best_quality", "high_quality", "good_quality", "medium_quality"}, default="medium_quality"
        AutoGluon preset to control the trade-off between training time and predictive accuracy.
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
    n_ensemble_models : int, default=200
        The number of random TabPFN configurations to generate and include in the
        AutoGluon model zoo for ensembling.
    n_estimators : int, default=16
        The number of internal transformers to ensemble within each individual TabPFN model.
        Higher values can improve performance but increase resource usage.
    balance_probabilities : bool, default=False
        Whether to balance the output probabilities from TabPFN. This can be beneficial
        for classification tasks with imbalanced classes.
    ignore_pretraining_limits : bool, default=False
        If `True`, bypasses TabPFN's built-in limits on dataset size (1024 samples)
        and feature count (100). **Warning:** Use with caution, as performance is not
        guaranteed and may be poor when exceeding these limits.

    Attributes
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
        max_time: int | None = 60 * 3,
        eval_metric: str | None = None,
        presets: Literal[
            "best_quality", "high_quality", "good_quality", "medium_quality"
        ] = "medium_quality",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        categorical_feature_indices: list[int] | None = None,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        n_ensemble_models: int = 200,
        n_estimators: int = 16,
        balance_probabilities: bool = False,
        ignore_pretraining_limits: bool = False,
    ):
        self.max_time = max_time
        self.eval_metric = eval_metric
        self.presets = presets
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices
        self.phe_init_args = phe_init_args
        self.phe_fit_args = phe_fit_args
        self.n_ensemble_models = n_ensemble_models
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.ignore_pretraining_limits = ignore_pretraining_limits

        assert n_ensemble_models >= 1, "n_ensemble_models must be >= 1"

        self._is_classifier = False

    def _more_tags(self) -> dict:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> AutoTabPFNRegressor:
        X, y = self._prepare_fit(
            X, y, categorical_feature_indices=categorical_feature_indices
        )
        super().fit(X, y)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        preds = self.predictor_.predict(pd.DataFrame(X, columns=self._column_names))
        return preds.to_numpy()
