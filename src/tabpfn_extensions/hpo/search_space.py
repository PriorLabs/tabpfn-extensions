# ruff: noqa: ERA001  ; Ignore commented out code.
#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Search spaces for hyperparameter optimization of TabPFN models.

This module provides predefined search spaces for TabPFN classifier and regressor
hyperparameter optimization. It also includes utilities for customizing search spaces.
"""

from __future__ import annotations

from pathlib import Path

from hyperopt import hp
from tabpfn_common_utils.telemetry import set_extension

from tabpfn.model_loading import ModelSource, ModelVersion, download_model


def enumerate_preprocess_transforms():
    transforms = []

    names_list = [
        # ["safepower"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        # ["quantile_uni"],
        ["kdi_uni"],
        ["kdi_alpha_0.3"],
        ["kdi_alpha_3.0"],
        ["none"],
        # ["robust"],  # Similar to squashing scaler.
        ["safepower", "quantile_uni"],
        # ["none", "safepower"],
        ["none", "quantile_uni_coarse"],
        ["squashing_scaler_default", "quantile_uni_coarse"],
        ["squashing_scaler_default"],
        # ["squashing_scaler_max10"],
    ]

    for names in names_list:
        for categorical_name in [
            "numeric",
            "ordinal_very_common_categories_shuffled",
            # "onehot",
            "none",
        ]:
            for append_original in [True, False]:
                for global_transformer_name in [None, "svd", "svd_quarter_components"]:
                    transforms += [
                        [
                            {
                                # Use "name" parameter as expected by TabPFN PreprocessorConfig
                                "name": name,
                                "global_transformer_name": global_transformer_name,
                                "categorical_name": categorical_name,
                                "append_original": append_original,
                            }
                            for name in names
                        ],
                    ]
    return transforms


@set_extension("hpo")
def get_param_grid_hyperopt(
    task_type: str,
    model_version: ModelVersion = ModelVersion.V2_5,
    model_dir: Path | None = None,
    download_models_if_missing: bool = True,
) -> dict:
    """Generate the full hyperopt search space for TabPFN optimization.

    Note: This will also download the required TabPFN model checkpoints if not already
    present in the specified model directory.

    Args:
        task_type: Either "multiclass" or "regression"
        model_version: Version of the TabPFN model to use.
        model_dir: Directory to store or look for TabPFN model checkpoints.
            If None, defaults to "hpo_models" directory next to this file.
        download_models_if_missing: Whether to download model checkpoints if they
            are not found in the specified model directory.

    Returns:
        Hyperopt search space dictionary
    """
    search_space = {
        # Custom HPs
        "model_type": hp.choice(
            "model_type",
            ["single", "dt_pfn"],
        ),
        "n_estimators": hp.choice("n_estimators", [4]),
        "max_depth": hp.choice("max_depth", [2, 3, 4, 5]),  # For Decision Tree TabPFN
        # -- Model HPs
        "average_before_softmax": hp.choice("average_before_softmax", [True, False]),
        "softmax_temperature": hp.choice(
            "softmax_temperature",
            [
                0.75,
                0.8,
                0.9,
                0.95,
                1.0,
                1.05,
            ],
        ),
        # Inference config
        "inference_config/FINGERPRINT_FEATURE": hp.choice(
            "FINGERPRINT_FEATURE",
            [True, False],
        ),
        "inference_config/PREPROCESS_TRANSFORMS": hp.choice(
            "PREPROCESS_TRANSFORMS",
            enumerate_preprocess_transforms(),
        ),
        "inference_config/POLYNOMIAL_FEATURES": hp.choice(
            "POLYNOMIAL_FEATURES",
            ["no"],  # Only use "no" to avoid polynomial feature computation errors
        ),
        "inference_config/OUTLIER_REMOVAL_STD": hp.choice(
            "OUTLIER_REMOVAL_STD",
            [None, 7.0, 12.0],
        ),
        "inference_config/MIN_UNIQUE_FOR_NUMERICAL_FEATURES": hp.choice(
            "MIN_UNIQUE_FOR_NUMERICAL_FEATURES", [1, 5, 10, 30]
        ),
        # Enable this for datasets with many samples.
        # "inference_config/SUBSAMPLE_SAMPLES": hp.choice(
        #     "SUBSAMPLE_SAMPLES",
        #     [0.7, None],
        # ),
    }

    if model_dir is None:
        model_dir = (Path(__file__).parent / "hpo_models").resolve()

    if task_type == "multiclass" and model_version == ModelVersion.V2:
        model_source = ModelSource.get_classifier_v2()
    elif task_type == "multiclass" and model_version == ModelVersion.V2_5:
        model_source = ModelSource.get_classifier_v2_5()
    elif task_type == "regression":
        if model_version == ModelVersion.V2:
            model_source = ModelSource.get_regressor_v2()
        elif model_version == ModelVersion.V2_5:
            model_source = ModelSource.get_regressor_v2_5()
        search_space["inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"] = hp.choice(
            "REGRESSION_Y_PREPROCESS_TRANSFORMS",
            [
                (None,),
                (None, "safepower"),
                ("safepower",),
                # ("quantile_uni",),
            ],
        )

    else:
        raise ValueError(
            f"Unknown combination of task type {task_type} and "
            "model version {model_version}!"
        )

    # Make sure models are downloaded.
    if download_models_if_missing:
        for ckpt_name in model_source.filenames:
            download_model(
                to=model_dir / ckpt_name,
                version=model_version,
                which="classifier" if task_type == "multiclass" else "regressor",
                model_name=ckpt_name,
            )

    model_paths = [str(model_dir / ckpt_name) for ckpt_name in model_source.filenames]
    search_space["model_path"] = hp.choice("model_path", model_paths)
    return search_space
