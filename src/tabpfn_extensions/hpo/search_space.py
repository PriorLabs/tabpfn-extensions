#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Search spaces for hyperparameter optimization of TabPFN models.

This module provides predefined search spaces for TabPFN classifier and regressor
hyperparameter optimization. It also includes utilities for customizing search spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperopt import hp


def enumerate_preprocess_transforms():
    transforms = []

    names_list = [
        ["safepower"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        ["quantile_uni"],
        ["none"],
        ["robust"],
        ["safepower", "quantile_uni"],
        ["none", "safepower"],
    ]

    for names in names_list:
        for categorical_name in [
            "numeric",
            "ordinal_very_common_categories_shuffled",
            "onehot",
            "none",
        ]:
            for append_original in [True, False]:
                for subsample_features in [-1, 0.99, 0.95, 0.9]:
                    for global_transformer_name in [None, "svd"]:
                        transforms += [
                            [
                                {
                                    # Use "name" parameter as expected by TabPFN PreprocessorConfig
                                    "name": name,
                                    "global_transformer_name": global_transformer_name,
                                    "subsample_features": subsample_features,
                                    "categorical_name": categorical_name,
                                    "append_original": append_original,
                                }
                                for name in names
                            ],
                        ]
    return transforms


class TabPFNSearchSpace:
    """Utility class for creating and customizing TabPFN hyperparameter search spaces.

    This class provides methods to generate default search spaces for both classification
    and regression tasks, as well as customizing parameter ranges.

    Examples:
        ```python
        # Get default classifier search space
        clf_space = TabPFNSearchSpace.get_classifier_space()

        # Get customized classifier search space
        custom_space = TabPFNSearchSpace.get_classifier_space(
            n_ensemble_range=(5, 15),
            temp_range=(0.1, 0.5)
        )

        # Use with TunedTabPFNClassifier
        from tabpfn_extensions.hpo import TunedTabPFNClassifier

        clf = TunedTabPFNClassifier(
            n_trials=50,
            search_space=custom_space
        )
        ```
    """

    @staticmethod
    def get_classifier_space(
        n_ensemble_range: tuple[int, int] = (1, 8),
        temp_range: tuple[float, float] = (0.75, 1.0),
    ) -> dict[str, Any]:
        """Get a search space for classification tasks.

        Args:
            n_ensemble_range: Range for n_estimators parameter as (min, max)
            temp_range: Range for softmax_temperature as (min, max)

        Returns:
            Dictionary with search space parameters
        """
        # Generate values within the ranges
        n_ensemble_values = list(range(n_ensemble_range[0], n_ensemble_range[1] + 1))
        temp_values = [
            round(temp_range[0] + i * 0.05, 2)
            for i in range(int((temp_range[1] - temp_range[0]) / 0.05) + 1)
        ]

        # Create simplified search space suitable for HPO
        return {
            "n_estimators": n_ensemble_values,
            "softmax_temperature": temp_values,
            "average_before_softmax": [True, False],
        }

    @staticmethod
    def get_regressor_space(
        n_ensemble_range: tuple[int, int] = (1, 8),
        temp_range: tuple[float, float] = (0.75, 1.0),
    ) -> dict[str, Any]:
        """Get a search space for regression tasks.

        Args:
            n_ensemble_range: Range for n_estimators parameter as (min, max)
            temp_range: Range for softmax_temperature as (min, max)

        Returns:
            Dictionary with search space parameters
        """
        # Basic space is the same as classifier space
        space = TabPFNSearchSpace.get_classifier_space(
            n_ensemble_range=n_ensemble_range,
            temp_range=temp_range,
        )

        # Add regression-specific parameters
        space.update(
            {
                # Add any regression-specific parameters here
            },
        )

        return space


def get_param_grid_hyperopt(task_type: str) -> dict:
    """Generate the full hyperopt search space for TabPFN optimization.

    Args:
        task_type: Either "multiclass" or "regression"

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
            [None, 7.0, 9.0, 12.0],
        ),
        "inference_config/SUBSAMPLE_SAMPLES": hp.choice(
            "SUBSAMPLE_SAMPLES",
            [0.99, None],
        ),
    }

    local_dir = (Path(__file__).parent / "hpo_models").resolve()

    if task_type == "multiclass":
        model_paths = [
            str(local_dir / "tabpfn-v2-classifier.ckpt"),
            str(local_dir / "tabpfn-v2-classifier-od3j1g5m.ckpt"),
            str(local_dir / "tabpfn-v2-classifier-gn2p4bpt.ckpt"),
            str(local_dir / "tabpfn-v2-classifier-znskzxi4.ckpt"),
            str(local_dir / "tabpfn-v2-classifier-llderlii.ckpt"),
            str(local_dir / "tabpfn-v2-classifier-vutqq28w.ckpt"),
        ]
    elif task_type == "regression":
        model_paths = [
            str(local_dir / "tabpfn-v2-regressor-09gpqh39.ckpt"),
            str(local_dir / "tabpfn-v2-regressor.ckpt"),
            str(local_dir / "tabpfn-v2-regressor-2noar4o2.ckpt"),
            str(local_dir / "tabpfn-v2-regressor-wyl4o83o.ckpt"),
        ]
        search_space["inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"] = hp.choice(
            "REGRESSION_Y_PREPROCESS_TRANSFORMS",
            [
                (None,),
                (None, "safepower"),
                ("safepower",),
                ("quantile_uni",),
            ],
        )
    else:
        raise ValueError(f"Unknown task type {task_type} for the search space!")

    search_space["model_path"] = hp.choice("model_path", model_paths)

    return search_space
