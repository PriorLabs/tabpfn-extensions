#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import numpy as np
import random
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from typing import Literal

from .pfn_phe import (
    AutoPostHocEnsemblePredictor,
    TaskType,
)


class AutoTabPFNClassifier(ClassifierMixin, BaseEstimator):
    """Automatic Post Hoc Ensemble Classifier for TabPFN models.

    Parameters
    ----------
        max_time : int | None, default=None
            The maximum time to spend on fitting the post hoc ensemble.
        preset: {"default", "custom_hps", "avoid_overfitting"}, default="default"
            The preset to use for the post hoc ensemble.
        ges_scoring_string : str, default="roc"
            The scoring string to use for the greedy ensemble search.
            Allowed values are: {"accuracy", "roc" / "auroc", "f1", "log_loss"}.
        device : {"cpu", "cuda"}, default="cuda"
            The device to use for training and prediction.
        random_state : int, RandomState instance or None, default=None
            Controls both the randomness base models and the post hoc ensembling method.
        categorical_feature_indices: list[int] or None, default=None
            The indices of the categorical features in the input data. Can also be passed to `fit()`.
        phe_init_args : dict | None, default=None
            The initialization arguments for the post hoc ensemble predictor.
            See post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more options and all details.

    Attributes:
    ----------
        predictor_ : AutoPostHocEnsemblePredictor
            The predictor interface used to make predictions, see post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more.
        phe_init_args_ : dict
            The optional initialization arguments used for the post hoc ensemble predictor.
    """

    predictor_: AutoPostHocEnsemblePredictor
    phe_init_args_: dict
    n_features_in_: int

    def __init__(
        self,
        max_time: int | None = 30,
        preset: Literal["default", "custom_hps", "avoid_overfitting"] = "default",
        ges_scoring_string: str = "roc",
        device: Literal["cpu", "cuda"] = "cpu",
        random_state: int | None | np.random.RandomState = None,
        categorical_feature_indices: list[int] | None = None,
        phe_init_args: dict | None = None,
    ):
        self.max_time = max_time
        self.preset = preset
        self.ges_scoring_string = ges_scoring_string
        self.device = device
        self.random_state = random_state
        self.phe_init_args = phe_init_args
        self.categorical_feature_indices = categorical_feature_indices

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y, categorical_feature_indices: list[int] | None = None):
        if categorical_feature_indices is not None:
            self.categorical_feature_indices = categorical_feature_indices

        self.phe_init_args_ = {} if self.phe_init_args is None else self.phe_init_args
        rnd = check_random_state(self.random_state)

        # Torch reproducibility bomb
        torch.manual_seed(rnd.randint(0, 2**32 - 1, dtype=np.int64))
        random.seed(rnd.randint(0, 2**32 - 1, dtype=np.int64))
        np.random.seed(rnd.randint(0, 2**32 - 1, dtype=np.int64))  # noqa: NPY002

        task_type = (
            TaskType.MULTICLASS if len(unique_labels(y)) > 2 else TaskType.BINARY
        )
        self.predictor_ = AutoPostHocEnsemblePredictor(
            preset=self.preset,
            task_type=task_type,
            max_time=self.max_time,
            ges_scoring_string=self.ges_scoring_string,
            device=self.device,
            bm_random_state=rnd.randint(0, 2**32 - 1, dtype=np.int64),
            ges_random_state=rnd.randint(0, 2**32 - 1, dtype=np.int64),
            **self.phe_init_args_,
        )

        self.predictor_.fit(
            X,
            y,
            categorical_feature_indices=self.categorical_feature_indices,
        )

        # -- Sklearn required values
        self.classes_ = self.predictor_._label_encoder.classes_
        self.n_features_in_ = self.predictor_.n_features_in_

        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.predictor_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.predictor_.predict_proba(X)

    def _more_tags(self):
        return {"allow_nan ": True}


class AutoTabPFNRegressor(RegressorMixin, BaseEstimator):
    """Automatic Post Hoc Ensemble Regressor for TabPFN models.

    Parameters
    ----------
        max_time : int | None, default=None
            The maximum time to spend on fitting the post hoc ensemble.
        preset: {"default", "custom_hps", "avoid_overfitting"}, default="default"
            The preset to use for the post hoc ensemble.
        ges_scoring_string : str, default="mse"
            The scoring string to use for the greedy ensemble search.
            Allowed values are: {"rmse", "mse", "mae"}.
        device : {"cpu", "cuda"}, default="cuda"
            The device to use for training and prediction.
        random_state : int, RandomState instance or None, default=None
            Controls both the randomness base models and the post hoc ensembling method.
        categorical_feature_indices: list[int] or None, default=None
            The indices of the categorical features in the input data. Can also be passed to `fit()`.
        phe_init_args : dict | None, default=None
            The initialization arguments for the post hoc ensemble predictor.
            See post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more options and all details.

    Attributes:
    ----------
        predictor_ : AutoPostHocEnsemblePredictor
            The predictor interface used to make predictions, see post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more.
        phe_init_args_ : dict
            The optional initialization arguments used for the post hoc ensemble predictor.
    """

    predictor_: AutoPostHocEnsemblePredictor
    phe_init_args_: dict
    n_features_in_: int

    def __init__(
        self,
        max_time: int | None = 30,
        preset: Literal["default", "custom_hps", "avoid_overfitting"] = "default",
        ges_scoring_string: str = "mse",
        device: Literal["cpu", "cuda"] = "cpu",
        random_state: int | None | np.random.RandomState = None,
        categorical_feature_indices: list[int] | None = None,
        phe_init_args: dict | None = None,
    ):
        self.max_time = max_time
        self.preset = preset
        self.ges_scoring_string = ges_scoring_string
        self.device = device
        self.random_state = random_state
        self.phe_init_args = phe_init_args
        self.categorical_feature_indices = categorical_feature_indices

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags

    def fit(self, X, y, categorical_feature_indices: list[int] | None = None):
        if categorical_feature_indices is not None:
            self.categorical_feature_indices = categorical_feature_indices

        self.phe_init_args_ = {} if self.phe_init_args is None else self.phe_init_args
        rnd = check_random_state(self.random_state)

        # Torch reproducibility bomb
        torch.manual_seed(rnd.randint(0, 2**32 - 1, dtype=np.int64))
        random.seed(rnd.randint(0, 2**32 - 1, dtype=np.int64))
        np.random.seed(rnd.randint(0, 2**32 - 1, dtype=np.int64))  # noqa: NPY002

        self.predictor_ = AutoPostHocEnsemblePredictor(
            preset=self.preset,
            task_type=TaskType.REGRESSION,
            max_time=self.max_time,
            ges_scoring_string=self.ges_scoring_string,
            device=self.device,
            bm_random_state=rnd.randint(0, 2**32 - 1, dtype=np.int64),
            ges_random_state=rnd.randint(0, 2**32 - 1, dtype=np.int64),
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

    def predict(self, X):
        check_is_fitted(self)
        return self.predictor_.predict(X)

    def _more_tags(self):
        return {"allow_nan ": True}


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
        print("Run for", est)
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
                    print("Error in test:", i)
                    print(x)
                    if i in non_deterministic:
                        print("Non deterministic test, retrying")
                        n_tests -= 1
                        continue
                    if raise_on_error:
                        raise e
                    lst.append((i, x, e))
                break

    print("\n\n\n\n\n\nFailed tests")
    print(len(lst))
    print(lst)
    print([x[0] for x in lst])
