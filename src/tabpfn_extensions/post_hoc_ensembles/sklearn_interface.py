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

MAX_INT = int(np.iinfo(np.int32).max)


# TODO: Convert to Dataclass and use these
class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class AutoTabPFNBase(BaseEstimator):
    """Base class for AutoGluon-powered TabPFN scikit-learn wrappers.

    Parameters
    ----------
        max_time : int | None, default=None
            The maximum time to spend on fitting the post hoc ensemble.
        preset: {"default", "custom_hps", "avoid_overfitting"}, default="default"
            The preset to use for the post hoc ensemble.
        eval_metric : str, default=None
            The scoring string to use for the eval_emtric of Autogluon.
            If None it is automciatlly chosen based on problem type, here
            are the options, https://auto.gluon.ai/dev/api/autogluon.tabular.models.html
        device : {"cpu", "cuda"}, default="auto"
            The device to use for training and prediction.
        random_state : int, RandomState instance or None, default=None
            Controls both the randomness base models and the post hoc ensembling method.
        categorical_feature_indices: list[int] or None, default=None
            The indices of the categorical features in the input data. Can also be passed to `fit()`.
        ignore_pretraining_limits: bool, default=False
            Whether to ignore the pretraining limits of the TabPFN base models.
        phe_init_args : dict | None, default=None
            The initialization arguments for the post hoc ensemble predictor.
            See Autogluon TabularPredictor for more options and all details.
        phe_fit_args : dict | None, default=None
            The fit arguments for the post hoc ensemble predictor.
            See Autogluon TabularPredictor for more options and all details.

    Attributes:
    ----------
        predictor_ : TabularPredictor
            The predictor interface used to make predictions.
        phe_init_args_ : dict
            The optional initialization arguments used for the post hoc ensemble predictor.
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
        ignore_pretraining_limits: bool = False,
        phe_init_args: dict | None = None,
        phe_fit_args: dict | None = None,
        num_random_configs: int = 200,
    ):
        self.max_time = max_time
        self.eval_metric = eval_metric
        self.presets = presets
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.num_random_configs = num_random_configs

        self.phe_init_args = phe_init_args
        self.phe_fit_args = phe_fit_args
        self.use_ensemble_model = True

    def _get_predictor_init_args(self) -> dict[str, Any]:
        """Constructs the initialization arguments for AutoGluon's TabularPredictor."""
        default_args = {"verbosity": 5}
        user_args = self.phe_init_args or {}
        return {**default_args, **user_args}

    def _get_predictor_fit_args(self) -> dict[str, Any]:
        """Constructs the fit arguments for AutoGluon's TabularPredictor."""
        default_args = {
            "num_bag_folds": 5,
            "num_bag_sets": 5,
            "num_stack_levels": 1,
        }
        user_args = self.phe_fit_args or {}
        return {**default_args, **user_args}

    def _prepare_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,
        )

        if self.categorical_feature_indices is not None:
            self.categorical_feature_indices = self.categorical_feature_indices

        # Auto-detect categorical features including text columns
        if self.categorical_feature_indices is None:
            self.categorical_feature_indices = infer_categorical_features(X)

        return X, y

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # TODO: Add better typing for X and y
    # E.g. With numpy and then internally convert to Pandas
    # Or also allow pandas dataframes
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the model by training an ensemble of TabPFN configurations using AutoGluon.
        This method should be called from the child class's fit method after validation.
        """
        from autogluon.tabular import TabularPredictor
        from autogluon.tabular.models import TabPFNV2Model

        from tabpfn_extensions.post_hoc_ensembles.utils import search_space_func

        self._column_names = [f"f{i}" for i in range(X.shape[1])]
        training_df = pd.DataFrame(X.copy(), columns=self._column_names)
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
        num_configs = max(1, self.num_random_configs)
        tabpfn_configs = search_space_func(
            task_type=task_type,
            num_random_configs=num_configs,
        )
        hyperparameters = {TabPFNV2Model: tabpfn_configs}

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

        # TODO: Put the Sklearn required values for the classes
        # and number of features input feature back in here

        return self

    def _more_tags(self):
        return {
            "allow_nan": True,
        }


class AutoTabPFNClassifier(ClassifierMixin, AutoTabPFNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._is_classifier = True

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> AutoTabPFNClassifier:
        X, y = self._prepare_fit(X, y, categorical_feature_indices)

        # TODO: Make sure the logic below works as intended
        # Check for single class
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        # Single class case - special handling
        if len(self.classes_) == 1:
            self.single_class_ = True
            self.single_class_value_ = self.classes_[0]
            return self

        # Check for extremely imbalanced classes - handle case with only 1 sample per class
        class_counts = np.bincount(y.astype(int))
        # TODO: Re-Implement this
        if np.min(class_counts[class_counts > 0]) < 2:
            self.single_class_ = False
            self.predictor_ = TabPFNClassifier(
                device=get_device(self.device),
                categorical_features_indices=self.categorical_feature_indices,
            )
            self.predictor_.fit(X, y)
            # Store the classes
            self.classes_ = self.predictor_.classes_
            self.n_features_in_ = X.shape[1]
            return self

        # Normal case - multiple classes with sufficient samples per class
        self.single_class_ = False
        TaskType.MULTICLASS if len(self.classes_) > 2 else TaskType.BINARY

        super().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        if hasattr(self, "single_class_") and self.single_class_:
            # For single class, return probabilities of 1.0
            return np.ones((X.shape[0], 1))
        # Convert to pandas dataframe for AutoGluon
        preds = self.predictor_.predict_proba(
            pd.DataFrame(X, columns=self._column_names)
        )
        # Convert back to numpy array for sklearn
        return preds.to_numpy()


class AutoTabPFNRegressor(RegressorMixin, AutoTabPFNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> AutoTabPFNRegressor:
        X, y = self._prepare_fit(
            X, y, categorical_feature_indices=categorical_feature_indices
        )

        super().fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        preds = self.predictor_.predict(pd.DataFrame(X, columns=self._column_names))
        return preds.to_numpy()


if __name__ == "__main__":
    import os

    from sklearn.utils.estimator_checks import check_estimator

    os.environ["SK_COMPATIBLE_PRECISION"] = "True"
    raise_on_error = True
    nan_test = 9

    # Precision issues do not allow for such deterministic behavior as expected, thus retrying certain tests to show it can work.
    clf_non_deterministic_for_reasons = [
        31,
        30,
    ]
    reg_non_deterministic_for_reasons = [
        27,
        28,
    ]

    for est, non_deterministic in [
        (AutoTabPFNClassifier(device="cuda"), clf_non_deterministic_for_reasons),
        (AutoTabPFNRegressor(device="cuda"), reg_non_deterministic_for_reasons),
    ]:
        lst = []
        for i, x in enumerate(check_estimator(est, generate_only=True)):
            if (i == nan_test) and ("allow_nan" in x[0]._get_tags()):
                # sklearn test does not check for the tag!
                continue

            n_tests = 5
            while n_tests:
                try:
                    x[1](x[0])
                except Exception as e:
                    if i in non_deterministic:
                        n_tests -= 1
                        continue
                    if raise_on_error:
                        raise e
                    lst.append((i, x, e))
                break
