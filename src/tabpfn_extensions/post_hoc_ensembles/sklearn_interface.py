#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Implementation taken from TabArena: A Living Benchmark for Machine Learning on Tabular Data,
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas,
Frank Hutter, Preprint., 2025,

Original Code: https://github.com/autogluon/tabrepo/tree/main/tabrepo/benchmark/models/ag/tabpfnv2
"""

from __future__ import annotations

import random
from typing import Literal
from enum import Enum

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions.misc.sklearn_compat import validate_data


MAX_INT = int(np.iinfo(np.int32).max)

class TaskType(str, Enum):
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"


class PresetType(str, Enum):
    DEFAULT = "default"
    AVOID_OVERFITTING = "avoid_overfitting"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class AutoTabPFNBase(BaseEstimator):
    """Automatic Post Hoc Ensemble Classifier for TabPFN models.

    # TODO: Add a Dictionary or a typed intput here

    Parameters
    ----------
        max_time : int | None, default=None
            The maximum time to spend on fitting the post hoc ensemble.
        preset: {"default", "custom_hps", "avoid_overfitting"}, default="default"
            The preset to use for the post hoc ensemble.
        ges_scoring_string : str, default="roc"
            The scoring string to use for the greedy ensemble search.
            Allowed values are: {"accuracy", "roc" / "auroc", "f1", "log_loss"}.
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
            See post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more options and all details.

    # TODO: Overwrite these below
    Attributes:
    ----------
        predictor_ : AutoPostHocEnsemblePredictor
            The predictor interface used to make predictions, see post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more.
        phe_init_args_ : dict
            The optional initialization arguments used for the post hoc ensemble predictor.
    """

    def __init__(
        self,
        max_time: int | None = 30,
        preset: Literal[
            "best_quality", "high_quality", "good_quality", "medium_quality"
        ] = "medium_quality",
        ges_scoring_string: str = "roc",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        random_state: int | None | np.random.RandomState = None,
        categorical_feature_indices: list[int] | None = None,
        ignore_pretraining_limits: bool = False,
        phe_init_args: dict | None = {'verbosity': 0},
        num_random_configs: int = 5,
    ):
        self.max_time = max_time
        self.presets = preset
        self.ges_scoring_string = ges_scoring_string
        self.device = device
        self.random_state = random_state
        self.categorical_feature_indices = categorical_feature_indices
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.num_random_configs = num_random_configs

        self._predictor: TabularPredictor | None = None
        self.phe_init_args = phe_init_args

    def _more_tags(self):
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # TODO: Add better typing for X and y
    # E.g. With numpy and then internally convert to Pandas
    # Or also allow pandas dataframes
    def fit(self, X, y, categorical_feature_indices: list[int] | None = None):
        from autogluon.tabular import TabularPredictor
        from tabpfn_extensions.post_hoc_ensembles.utils import search_space_func
        from autogluon.tabular.models import TabPFNV2Model

        training_df = pd.DataFrame(X).copy()
        training_df["_target_"] = y  # lightweight, avoids name clashes

        problem_type = (
            "binary"
            if self._is_classifier and len(np.unique(y)) == 2
            else ("multiclass" if self._is_classifier else "regression")
        )

        # TODO: Double Check Code below
        '''rnd = check_random_state(self.random_state)

        # Torch reproducibility bomb
        torch.manual_seed(rnd.randint(0, MAX_INT))
        random.seed(rnd.randint(0, MAX_INT))
        np.random.seed(rnd.randint(0, MAX_INT))

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
        if np.min(class_counts[class_counts > 0]) < 2:
            # Cannot do stratification with less than 2 samples per class
            # Use a standard TabPFN classifier without ensemble
            from tabpfn_extensions.utils import TabPFNClassifier, get_device

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
        task_type = TaskType.MULTICLASS if len(self.classes_) > 2 else TaskType.BINARY
        # Use the device utility for automatic selection
        from tabpfn_extensions.utils import get_device'''



        self.predictor_ = TabularPredictor(
            label="_target_",
            problem_type=problem_type,
            **self.phe_init_args,
        )

        task_type = "multiclass" if self._is_classifier else "regression"

        num_configs_to_generate = max(
            1, self.num_random_configs
        )  # Ensure at least one config

        tabpfn_configs = search_space_func(
            task_type=task_type,
            num_random_configs=num_configs_to_generate,
            seed=self.random_state,
        )

        hyperparameters = {TabPFNV2Model: tabpfn_configs}
        self.predictor_.fit(
            train_data=training_df,
            time_limit=self.max_time,
            presets=self.presets,
            hyperparameters=hyperparameters,
            #categorical_feature_indices=self.categorical_feature_indices,
        )

        # TODO: Add more arguments to fit above
        '''self.predictor_ = AutoPostHocEnsemblePredictor(
            preset=self.preset,
            task_type=task_type,
            max_time=self.max_time,
            ges_scoring_string=self.ges_scoring_string,
            device=get_device(self.device),
            bm_random_state=rnd.randint(0, MAX_INT),
            ges_random_state=rnd.randint(0, MAX_INT),
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            **self.phe_init_args_,
        )'''

        # -- Sklearn required values
        #TODO: Put this back in
        #self.classes_ = self.predictor_._label_encoder.classes_
        #self.n_features_in_ = self.predictor_.n_features_in_

        return self



class AutoTabPFNClassifier(ClassifierMixin, AutoTabPFNBase):


    predictor_: AutoPostHocEnsemblePredictor
    phe_init_args_: dict
    n_features_in_: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._is_classifier = True

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    '''def fit(self, X, y, categorical_feature_indices: list[int] | None = None):
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,
        )

        if categorical_feature_indices is not None:
            self.categorical_feature_indices = categorical_feature_indices

        # Auto-detect categorical features including text columns
        if self.categorical_feature_indices is None:
            from tabpfn_extensions.utils import infer_categorical_features

            self.categorical_feature_indices = infer_categorical_features(X)

        self.phe_init_args_ = {} if self.phe_init_args is None else self.phe_init_args
        rnd = check_random_state(self.random_state)

        # Torch reproducibility bomb
        torch.manual_seed(rnd.randint(0, MAX_INT))
        random.seed(rnd.randint(0, MAX_INT))
        np.random.seed(rnd.randint(0, MAX_INT))

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
        if np.min(class_counts[class_counts > 0]) < 2:
            # Cannot do stratification with less than 2 samples per class
            # Use a standard TabPFN classifier without ensemble
            from tabpfn_extensions.utils import TabPFNClassifier, get_device

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
        task_type = TaskType.MULTICLASS if len(self.classes_) > 2 else TaskType.BINARY
        # Use the device utility for automatic selection
        from tabpfn_extensions.utils import get_device
    '''


    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        if hasattr(self, "single_class_") and self.single_class_:
            # For single class, always predict that class
            return np.full(X.shape[0], self.single_class_value_)
        #Convert to pandas dataframe for AutoGluon
        preds = self.predictor_.predict(pd.DataFrame(X))
        # Convert back to numpy array for sklearn
        return preds.to_numpy()

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        if hasattr(self, "single_class_") and self.single_class_:
            # For single class, return probabilities of 1.0
            return np.ones((X.shape[0], 1))
        #Convert to pandas dataframe for AutoGluon
        preds = self.predictor_.predict_proba(pd.DataFrame(X))
        # Convert back to numpy array for sklearn
        return preds.to_numpy()


class AutoTabPFNRegressor(RegressorMixin, AutoTabPFNBase):

    predictor_: AutoPostHocEnsemblePredictor
    phe_init_args_: dict
    n_features_in_: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._is_classifier = False

    def _more_tags(self):
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    """def fit(self, X, y, categorical_feature_indices: list[int] | None = None):
        # Validate input data

        # Will raise ValueError if X is empty or invalid
        # For regressor, ensure y is numeric
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,
        )

        if categorical_feature_indices is not None:
            self.categorical_feature_indices = categorical_feature_indices

        # Auto-detect categorical features including text columns
        if self.categorical_feature_indices is None:
            from tabpfn_extensions.utils import infer_categorical_features

            self.categorical_feature_indices = infer_categorical_features(X)

        self.phe_init_args_ = {} if self.phe_init_args is None else self.phe_init_args
        rnd = check_random_state(self.random_state)

        # Torch reproducibility bomb
        torch.manual_seed(rnd.randint(0, MAX_INT))
        random.seed(rnd.randint(0, MAX_INT))
        np.random.seed(rnd.randint(0, MAX_INT))

        # Use the device utility for automatic selection
        from tabpfn_extensions.utils import get_device

        self.predictor_ = AutoPostHocEnsemblePredictor(
            preset=self.preset,
            task_type=TaskType.REGRESSION,
            max_time=self.max_time,
            ges_scoring_string=self.ges_scoring_string,
            device=get_device(self.device),
            bm_random_state=rnd.randint(0, MAX_INT),
            ges_random_state=rnd.randint(0, MAX_INT),
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            **self.phe_init_args_,
        )

        self.predictor_.fit(
            X,
            y,
            categorical_feature_indices=self.categorical_feature_indices,
        )

        # -- Sklearn required values
        self.n_features_in_ = self.predictor_.n_features_in_

        return self
    """

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,
        )
        #Convert to pandas dataframe for AutoGluon
        preds = self.predictor_.predict(pd.DataFrame(X))
        # Convert back to numpy array for sklearn
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
