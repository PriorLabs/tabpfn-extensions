from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from tabpfn.preprocessing import PreprocessorConfig


def get_params_from_config(c):
    return {}  # here you add things that you want to use from the config to do inference in transformer_predict


@dataclass
class EnsembleConfiguration:
    """
    Configuration for an ensemble member.

    Attributes:
        class_shift_configuration (torch.Tensor | None): Permutation to apply to classes. Only used for classification.
        feature_shift_configuration (int | None): Random seed for feature shuffling.
        preprocess_transform_configuration (PreprocessorConfig): Preprocessor configuration to use.
        styles_configuration (int | None): Styles configuration to use.
        subsample_samples_configuration (int | None): Indices of samples to use for this ensemble member.
    """

    class_shift_configuration: torch.Tensor | None = None
    feature_shift_configuration: int | None = None
    preprocess_transform_configuration: PreprocessorConfig = PreprocessorConfig("none")
    styles_configuration: int | None = None
    subsample_samples_configuration: int | None = None


@dataclass
class TabPFNConfig:
    """
    Configuration for TabPFN models.

    Check TabPFNBaseEstimator for more information on attributes.
    """

    task_type: str
    model_type: Literal[
        "best",
        "single",
        "ensemble",
        "single_fast",
        "stacking",
        "bagging",
        "rf_pfn",
        "rf_xgb_pfn",
        "weighted_average",
    ]
    paths_config: TabPFNModelPathsConfig
    task_type_config: (
        TabPFNClassificationConfig
        | TabPFNRegressionConfig
        | TabPFNSurvivalConfig
        | None
    ) = None
    model_type_config: (
        StackingConfig | TabPFNRFConfig | BaggingConfig | WeightedAverageConfig | None
    ) = None

    model_name: str = "tabpfn"  # This name will be tracked on neptune

    # Core TabPFNClassifier parameters
    n_estimators: int = 4
    categorical_features_indices: Optional[Tuple[int, ...]] = None
    softmax_temperature: float = 0.9
    average_before_softmax: bool = False
    ignore_pretraining_limits: bool = False
    inference_precision: Literal["auto", "autocast"] | torch.dtype = "auto"
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = (
        "fit_preprocessors"
    )
    memory_saving_mode: Union[bool, Literal["auto"], float, int] = "auto"
    random_state: Optional[int] = 0
    n_jobs: int = -1

    # TODO: pass directly as Interface config parameters
    # Interface config parameters
    subsample_samples: Optional[float] = None
    add_fingerprint_features: bool = True
    feature_shift_method: Literal["shuffle", "none", "local_shuffle", "rotate"] = (
        "shuffle"
    )
    use_poly_features: bool = False
    max_number_of_samples: Optional[int] = None
    max_number_of_features: Optional[int] = None
    max_number_of_classes: int = 10
    min_number_samples_for_categorical_inference: int = 100
    max_unique_for_categorical_features: int = 20
    min_unique_for_numerical_features: int = 5
    remove_outliers: Union[float, Literal["auto"]] = "auto"
    use_sklearn_16_decimal_precision: bool = False
    preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
        PreprocessorConfig("safepower", categorical_name="numeric"),
        PreprocessorConfig("power", categorical_name="numeric"),
    )
    batch_size_inference: int = 1
    save_peak_memory: Union[bool, Literal["auto"], str] = "auto"

    # Optional model configuration
    optimize_metric: Optional[str] = None
    model_config: Optional[Dict] = field(default_factory=dict)
    model: Optional[torch.nn.Module] = None

    def to_kwargs(self):
        """Convert config to kwargs for TabPFNClassifier initialization."""
        task_type_config = dataclasses.asdict(self.task_type_config)
        # Core classifier kwargs
        kwargs = {
            "n_estimators": self.n_estimators,
            "categorical_features_indices": self.categorical_features_indices,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
            "model_path": (
                self.paths_config.model_strings[0]
                if self.paths_config is not None
                else "auto"
            ),
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "inference_precision": self.inference_precision,
            "fit_mode": self.fit_mode,
            "memory_saving_mode": self.memory_saving_mode,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        if self.task_type == "multiclass":
            kwargs["balance_probabilities"] = task_type_config.pop(
                "balance_probabilities"
            )

        assert kwargs["model_path"] != "auto", (
            "model_path must be set"
        )  # TODO: might be relaxed at some point
        # Interface config kwargs
        if not self.use_poly_features:
            poly_features = "no"
        else:
            raise ValueError("Not supported yet during refactor")
        interface_config = {
            "SUBSAMPLE_SAMPLES": self.subsample_samples,
            "FINGERPRINT_FEATURE": self.add_fingerprint_features,
            "FEATURE_SHIFT_METHOD": self.feature_shift_method,
            "POLYNOMIAL_FEATURES": poly_features,
            # TODO: don't hardcode these
            "MAX_NUMBER_OF_SAMPLES": self.max_number_of_samples
            if self.max_number_of_samples is not None
            else 10_000,
            "MAX_NUMBER_OF_FEATURES": self.max_number_of_features
            if self.max_number_of_features is not None
            else 500,
            "MAX_NUMBER_OF_CLASSES": self.max_number_of_classes,
            "MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE": self.min_number_samples_for_categorical_inference,
            "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": self.max_unique_for_categorical_features,
            "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": self.min_unique_for_numerical_features,
            "OUTLIER_REMOVAL_STD": self.remove_outliers,
            "USE_SKLEARN_16_DECIMAL_PRECISION": self.use_sklearn_16_decimal_precision,
            "PREPROCESS_TRANSFORMS": self.preprocess_transforms,
        }
        if self.task_type == "multiclass":
            interface_config["CLASS_SHIFT_METHOD"] = task_type_config.pop(
                "multiclass_decoder"
            )
        if self.task_type == "regression":
            interface_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = (
                task_type_config.pop("regression_y_preprocess_transforms")
            )
            interface_config["FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM"] = (
                task_type_config.pop("cancel_nan_borders")
            )
        kwargs["inference_config"] = interface_config

        if task_type_config is not None:
            kwargs.update(task_type_config)

        return kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}: {repr(v)[:100]}" for k, v in asdict(self).items())
            + ")"
        )


@dataclass
class TabPFNModelPathsConfig:
    """
    paths: list of model paths, e.g. ["/path/to/model.pth"]

    The model_strings attribute is automatically generated from the paths attribute, and contains the actual paths to the models on our cluster.
    If a path is given, the model_string is the path itself.
    """

    paths: list[str]

    model_strings: list[str] = dataclasses.field(init=False)

    task_type: str = "multiclass"

    def __post_init__(self):
        # Initialize Model paths
        self.model_strings = []

        for path in self.paths:
            self.model_strings.append(path)

    def to_dict(self):
        return dataclasses.asdict(self)


### TASK TYPE CONFIGS ###


@dataclass
class TabPFNClassificationConfig:
    balance_probabilities: bool = False
    multiclass_decoder: Literal["shuffle", "none", "local_shuffle", "rotate"] = (
        "shuffle"
    )


@dataclass
class TabPFNRegressionConfig:
    regression_y_preprocess_transforms: Tuple[str | None, ...] = (
        None,
        "safepower",
    )
    cancel_nan_borders: bool = True


@dataclass
class TabPFNSurvivalConfig:
    pass


### MODEL TYPE CONFIGS ###


@dataclass
class StackingConfig:
    params_stacked: Tuple[Dict[str, Any], ...]
    cv: int
    append_other_model_types: bool

    final_estimator: ClassifierMixin = LogisticRegression()


@dataclass
class WeightedAverageConfig:
    params_stacked: Tuple[Dict[str, Any], ...]
    cv: int
    n_max: int = 3


@dataclass
class TabPFNRFConfig:
    min_samples_split: int = 1000
    min_samples_leaf: int = 5
    max_depth: int = 5
    splitter: Literal["best", "random"] = "best"
    n_estimators: int = 16
    max_features: Literal["sqrt", "auto"] = "sqrt"
    criterion: Literal[
        "gini", "entropy", "log_loss", "squared_error", "friedman_mse", "poisson"
    ] = "gini"
    preprocess_X: bool = False
    preprocess_X_once: bool = False
    adaptive_tree: bool = True
    fit_nodes: bool = True
    adaptive_tree_overwrite_metric: Literal["logloss", "roc"] = None
    adaptive_tree_test_size: float = 0.2
    adaptive_tree_min_train_samples: int = 100
    adaptive_tree_min_valid_samples_fraction_of_train: int = 0.2
    adaptive_tree_max_train_samples: int = 5000
    adaptive_tree_skip_class_missing: bool = True
    max_predict_time: float = -1

    bootstrap: bool = True
    rf_average_before_softmax: bool = False
    dt_average_before_softmax: bool = True


@dataclass
class BaggingConfig:
    n_estimators: int = 32
    max_samples: [float | int] = 2048
    max_features: [float | int] = 1.0
    bootstrap: bool = True
    bootstrap_features: bool = False
