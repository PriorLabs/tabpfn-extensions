#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import itertools
import logging
import warnings
from enum import Enum
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabpfn_extensions.misc.sklearn_compat import check_array, check_X_y
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

from .greedy_weighted_ensemble import (
    GreedyWeightedEnsembleClassifier,
    GreedyWeightedEnsembleRegressor,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AutoPostHocEnsemble")


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


class TabPFNBaseModelSource(str, Enum):
    CUSTOM = "custom"
    RANDOM_PORTFOLIO = "random_portfolio"


class AutoPostHocEnsemblePredictor(BaseEstimator):
    """A wrapper to effectively performing post hoc ensemble with TabPFN models."""

    # Task Config
    preset: PresetType
    task_type: TaskType
    device: DeviceType
    bm_random_state: int
    ges_random_state: int

    # Validation Data Config
    n_folds: int | None
    n_repeats: int
    validation_method: Literal["holdout", "cv"]
    holdout_fraction: float

    # GES config
    ges_max_time: int
    ges_n_iterations: int
    ges_scoring_string: str

    # Base Model config
    max_models: int | None
    tabpfn_base_model_source: TabPFNBaseModelSource
    custom_tabpfn_models: list[tuple[str, BaseEstimator]] | None
    additional_sklearn_models: list[tuple[str, BaseEstimator]] | None

    # Attributes
    _ens_model: (
        GreedyWeightedEnsembleClassifier | GreedyWeightedEnsembleRegressor
    )  # Ensemble model to use.
    _estimators: list[tuple[str, BaseEstimator]]  # Base models used in the ensemble.
    _cat_encoder: ColumnTransformer  # Encoder to use for categorical features before passing it to the base models.
    _label_encoder: LabelEncoder  # Label encoder to use for label features before passing it to the base models.

    def __init__(
        self,
        *,
        preset: PresetType,
        max_time: int | None,
        task_type: TaskType,
        ges_scoring_string: str,
        device: DeviceType,
        bm_random_state: int,
        ges_random_state: int,
        # -- Custom hyperparameters
        # - How to get base models
        tabpfn_base_model_source: TabPFNBaseModelSource = TabPFNBaseModelSource.RANDOM_PORTFOLIO,
        custom_tabpfn_models: list[tuple[str, BaseEstimator]] | None = None,
        additional_sklearn_models: list[tuple[str, BaseEstimator]] | None = None,
        # - How to use base models
        max_models: int | None = None,
        validation_method: Literal["holdout", "cv"] = "holdout",
        n_repeats: int = 80,
        n_folds: int | None = None,
        holdout_fraction: float = 0.33,
        ges_n_iterations: int = 25,
        ignore_pretraining_limits: bool = False,
    ) -> None:
        """Builds a PostHocEnsembleConfig with default values for the given parameters.

        Args:
            preset: The preset to use, which prompts changes to the method. Note, these
                - "default" use default hyperparameter. Does not override user-supplied parameters.
                - "avoid_overfitting" uses a preset optimized for avoiding overfitting (needs more training time!).
                    This will override certain parameters no-matter the user input.
            max_time: The maximum time to run the ensemble builder (only respected with an overhead).
            task_type: The task type of the dataset. Either "regression", "binary_classification", "multiclass_classification".
            ges_scoring_string: The scoring string to use for the greedy ensemble search.
                Supported values for classification: "accuracy", "roc" / "auroc", "f1", "log_loss".
                Supported values for regression: "rmse", "mse", "mae".
            device: The device to run the models on. Either "cpu" or "cuda".
            bm_random_state: The random state to use for the base models.
            ges_random_state: The random state to use for the greedy ensemble search.
            tabpfn_base_model_source: Where the TabPFN base models are sourced from.
                - `custom` uses models supplied via the custom_tabpfn_models parameter.
                - `random_portfolio` uses models from a random portfolio.
                - `zeroshot_portfolio` uses models from meta-learned portfolio, as defined in (the default) `zero_shot_portfolio_config`.
            custom_tabpfn_models: Custom TabPFN models to use for the post hoc ensemble.
                Only used if tabpfn_base_model_source is set to "custom".
            additional_sklearn_models: Additional sklearn models to use. If None, no additional models are used.
            max_models: The maximum number of base models to use. If None, all models are used.
            validation_method: The validation method to use. Either "holdout" or "cv".
            n_repeats: The number of repeats to use for holdout or cross-validation.
            n_folds: The number of folds to use for cross-validation. Pas None in the holdout setting.
            holdout_fraction: The fraction of the data to use for holdout validation.
            ges_n_iterations: The number of iterations to use for the greedy ensemble search.
            ignore_pretraining_limits: Whether to ignore the pretraining limits of the TabPFN models.
        """
        # Task Type and User Input
        self.preset = preset
        self.max_time = max_time
        self.task_type = task_type
        self.ges_scoring_string = ges_scoring_string
        self.device = device
        self.bm_random_state = bm_random_state
        self.ges_random_state = ges_random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits

        # Model Source
        self.tabpfn_base_model_source = tabpfn_base_model_source
        self.custom_tabpfn_models = custom_tabpfn_models
        self.additional_sklearn_models = additional_sklearn_models

        # Model Usage
        self.max_models = max_models
        self.validation_method = validation_method
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.holdout_fraction = holdout_fraction
        self.ges_n_iterations = ges_n_iterations

        # -- Input Validation
        if self.task_type not in TaskType:
            raise ValueError(f"Only supports {list(TaskType)}! Got: {task_type}")

        # -- Handle time
        if max_time in [None, -1, 0]:
            warnings.warn(
                "No max_time or max_time=-1 or max_time=0 given... ensembling two models and evaluate both with holdout validation!",
                stacklevel=2,
            )
            warnings.warn(
                "Set a time limit to train and ensemble more models!",
                stacklevel=2,
            )
            ges_max_time = float("inf")
            self.max_models = 2
            self.n_repeats = 1
            self.holdout_fraction = 0.2
            self.validation_method = "holdout"
        elif isinstance(max_time, int) and max_time > 0:
            ges_max_time = max_time
        else:
            raise NotImplementedError(
                f"max_time {max_time} not supported for TabPFN yet!",
            )
        self.ges_max_time = ges_max_time

        # -- Handle presets
        if self.preset == "default":
            logger.info("Using `default` preset for Post Hoc Ensemble.", stacklevel=2)
        elif self.preset == "avoid_overfitting":
            logger.info(
                "Using preset to avoid overfitting parameters. Enforcing cross-validation with 8 folds.",
                stacklevel=2,
            )
            self.validation_method = "cv"
            self.n_folds = 8  # AutoGluon value
        else:
            raise ValueError(f"preset {preset} not supported yet!")

        if (
            self.tabpfn_base_model_source == TabPFNBaseModelSource.CUSTOM
            and self.custom_tabpfn_models is None
        ):
            raise ValueError(
                "Custom models are specified as the source of TabPFN but not custom TabPFN models are supplied!",
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_feature_indices: list[int] | None = None,
    ) -> AutoPostHocEnsemblePredictor:
        """Fits the post hoc ensemble on the given data.

        Args:
            X: The input data to fit the ensemble on.
            y: The target values to fit the ensemble on.
            categorical_feature_indices: The indices of the categorical features in the data.
                If None, no categorical features are assumed to be present.
        """
        if categorical_feature_indices is None:
            logger.info(
                "No categorical_feature_indices given. Assuming no categorical features.",
                stacklevel=2,
            )
            categorical_feature_indices = []
        elif all(isinstance(x, (int, np.integer)) for x in categorical_feature_indices):
            logger.info(
                f"Using categorical_feature_indices: {categorical_feature_indices}",
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Invalid categorical_feature_indices: {categorical_feature_indices}",
            )

        # -- Ensure dtype encoding.
        X, y = check_X_y(
            X,
            y,
            ensure_all_finite="allow-nan",
            dtype=object,
            accept_sparse=False,
        )
        self.n_features_in_ = X.shape[1]

        # -- Data Cleaning to be usable by base models.
        self._cat_encoder = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    categorical_feature_indices,
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
        )
        X = self._cat_encoder.fit_transform(X)

        if self.task_type in [TaskType.BINARY, TaskType.MULTICLASS]:
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)

        # -- sanity check, no trust.
        X, y = check_X_y(
            X,
            y,
            ensure_all_finite="allow-nan",
            dtype="numeric",
            accept_sparse=False,
        )

        if self.task_type in [TaskType.BINARY, TaskType.MULTICLASS]:
            self._ens_model: GreedyWeightedEnsembleClassifier = (
                GreedyWeightedEnsembleClassifier
            )
            self._pred_func = "predict_proba"
        elif self.task_type == TaskType.REGRESSION:
            self._ens_model: GreedyWeightedEnsembleRegressor = (
                GreedyWeightedEnsembleRegressor
            )
            self._pred_func = "predict"
        else:
            raise ValueError(f"Unknown task type {self.task_type}!")
        logger.info(f"Using task type: {self.task_type}", stacklevel=2)

        if self.validation_method == "cv" and self.task_type in [
            "multiclass_classification",
            "binary_classification",
        ]:
            # n_folds sanity check
            _, counts = np.unique(y, return_counts=True)
            second_largest: int = int(np.partition(counts.flatten(), -2)[-2])
            if second_largest < self.n_folds:
                warnings.warn(
                    "Detected that `n_folds` is hardcoded but dataset has less minimal non-prune-able classes. "
                    "We adjust n_folds to avoid cross-validation bugs in that case to the minimal class count. "
                    f"Old value: {self.n_folds}, new value: {second_largest}.",
                    stacklevel=2,
                )
                self.n_folds = second_largest

        self._estimators, model_family_per_estimator = self._collect_base_models(
            categorical_feature_indices=categorical_feature_indices,
        )

        self._ens_model = self._ens_model(
            estimators=self._estimators,
            seed=self.ges_random_state,
            n_repeats=self.n_repeats,
            n_folds=self.n_folds,
            n_iterations=self.ges_n_iterations,
            score_metric=self.ges_scoring_string,
            time_limit=self.ges_max_time,
            validation_method=self.validation_method,
            holdout_fraction=self.holdout_fraction,
            model_family_per_estimator=model_family_per_estimator,
        )

        self._ens_model.fit(X, y)

        return self

    def _collect_base_models(
        self,
        categorical_feature_indices: list[int],
    ) -> tuple[list[tuple[str, object]], list[str]]:
        assert isinstance(categorical_feature_indices, list)

        if self.tabpfn_base_model_source == TabPFNBaseModelSource.CUSTOM:
            logger.info(
                "Obtaining TabPFN models from user-defined models.",
                stacklevel=2,
            )
            bm_list = self.custom_tabpfn_models
        elif self.tabpfn_base_model_source == TabPFNBaseModelSource.RANDOM_PORTFOLIO:
            logger.info(
                "Obtaining TabPFN models from a random portfolio.",
                stacklevel=2,
            )
            bm_list = _get_base_models_from_random_search(
                task_type=self.task_type,
                device=self.device,
                random_state=self.bm_random_state,
                categorical_indices=categorical_feature_indices,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )
        else:
            raise ValueError(
                f"Unknown base model source {self.tabpfn_base_model_source}!",
            )

        if self.additional_sklearn_models is not None:
            bm_list = _interleave_lists(bm_list, self.additional_sklearn_models)

        if self.max_models is not None:
            bm_list = bm_list[: self.max_models]

        logger.info(
            f"Using {len(bm_list)} base models: {[x[0] for x in bm_list]}",
            stacklevel=2,
        )

        model_family_per_bm = []
        for _, bm in bm_list:
            if isinstance(
                bm,
                (RandomForestTabPFNClassifier, RandomForestTabPFNRegressor),
            ):
                mf = "rf-pfn" + str(bm.tabpfn.model_path)
            elif isinstance(bm, (TabPFNClassifier, TabPFNRegressor)):
                mf = "pfn" + str(bm.model_path)
            elif isinstance(bm, BaseEstimator):
                mf = "sk-non-pfn"
            else:
                mf = "NAN"
            model_family_per_bm.append(mf)

        return bm_list, model_family_per_bm

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target values for the given data."""
        X = check_array(X, ensure_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            ensure_all_finite="allow-nan",
            dtype="numeric",
        )
        if self.task_type == "regression":
            return self._ens_model.predict(X)

        return self._label_encoder.inverse_transform(self._ens_model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target values for the given data."""
        X = check_array(X, ensure_all_finite="allow-nan", dtype=object)
        X = check_array(
            self._cat_encoder.transform(X),
            ensure_all_finite="allow-nan",
            dtype="numeric",
        )
        return self._ens_model.predict_proba(X)


def _get_base_models_from_random_search(
    *,
    task_type: str,
    device: str,
    random_state: int,
    categorical_indices: list[int],
    random_portfolio_size: int = 100,
    start_with_default_pfn: bool = True,
    ignore_pretraining_limits: bool = False,
) -> list[tuple[str, object]]:
    # Convert device if needed
    from tabpfn_extensions.utils import get_device

    resolved_device = get_device(device)
    # TODO: switch to config space to not depend on hyperopt
    from hyperopt.pyll import stochastic

    from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt

    _task_type = (
        "multiclass"
        if task_type in [TaskType.MULTICLASS, TaskType.BINARY]
        else task_type
    )
    rng = np.random.default_rng(4224)
    bm_model_rng = np.random.default_rng(random_state + 4224)
    search_space = get_param_grid_hyperopt(_task_type)
    model_base = TabPFNClassifier if _task_type == "multiclass" else TabPFNRegressor

    bm_list = []
    first_non_rf_pfn = None
    for i in range(random_portfolio_size):
        model_seed = int(bm_model_rng.integers(0, np.iinfo(np.int32).max))

        param = (
            {}
            if (start_with_default_pfn and i == 0)
            else stochastic.sample(search_space, rng=rng)
        )

        inference_config = {
            _k: v
            for k, v in param.items()
            if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            param["inference_config"] = inference_config
        for k in list(param.keys()):
            if k.startswith("inference_config/"):
                del param[k]

        param["device"] = resolved_device
        param["random_state"] = model_seed
        param["categorical_features_indices"] = categorical_indices
        param["ignore_pretraining_limits"] = ignore_pretraining_limits
        n_ensemble_repeats = param.pop("n_ensemble_repeats", None)
        model_is_rf_pfn = param.pop("model_type", "no") == "dt_pfn"

        # Remove max_depth if it exists (only used for decision tree models)
        max_depth = param.pop("max_depth", None)

        if model_is_rf_pfn:
            param["n_estimators"] = 1
            rf_model_base = (
                RandomForestTabPFNClassifier
                if _task_type == "multiclass"
                else RandomForestTabPFNRegressor
            )
            bm = rf_model_base(
                tabpfn=model_base(**param),
                categorical_features=categorical_indices,
                max_depth=max_depth
                if max_depth is not None
                else 3,  # Use removed max_depth here
            )
        else:
            if n_ensemble_repeats is not None:
                param["n_estimators"] = n_ensemble_repeats
            bm = model_base(**param)

        model_indicator = "rf_pfn" if model_is_rf_pfn else "tabpfn"
        model_prefix = "default" if (start_with_default_pfn and i == 0) else "random"
        bm_list.append((f"{model_prefix}_{model_indicator}_model_{i}", bm))

    if first_non_rf_pfn is not None:
        bm_list = [first_non_rf_pfn, *bm_list]

    return bm_list


def _interleave_lists(list1, list2):
    return [
        x
        for x in itertools.chain(*itertools.zip_longest(list1, list2))
        if x is not None
    ]


class TabPFNPostHocEnsemble:
    """Simple wrapper around AutoTabPFNClassifier for backward compatibility."""

    def __init__(self, n_models=5, device="auto", random_state=42):
        from .sklearn_interface import AutoTabPFNClassifier

        self.classifier = AutoTabPFNClassifier(
            max_time=30,
            device=device,
            random_state=random_state,
            phe_init_args={"max_models": n_models},
        )
        self.n_models = n_models
        self.device = device
        self.random_state = random_state
        self.estimator_type = "classifier"  # Add estimator_type attribute

    def fit(self, X, y):
        self.classifier.fit(X, y)
        # Set attributes to match test expectations
        self.base_models_ = self.classifier.predictor_._estimators
        self.ensemble_ = self.classifier.predictor_._ens_model
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
