#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from hyperopt import hp
from pathlib import Path


def enumerate_preprocess_transforms():
    transforms = []

    names_list = [
        ["safepower"],
        ["quantile_uni_coarse"],
        ["quantile_norm_coarse"],
        #["norm_and_kdi"],
        ["quantile_uni"],
        ["none"],
        ["robust"],
        ["safepower", "quantile_uni"],
        ["none", "power"],
    ]

    # try:
    #     from kditransform import KDITransformer
    #
    #     names_list += [
    #         ["kdi_uni"],
    #         ["kdi_alpha_0.3"],
    #         ["kdi_alpha_3.0"],
    #         ["kdi", "quantile_uni"],
    #     ]
    # except:
    #     pass

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


def get_param_grid_hyperopt(task_type: str) -> dict:
    search_space = {
        # Custom HPs
        "model_type": hp.choice("model_type", ["single", "dt_pfn"]),
        "n_ensemble_repeats": hp.choice("n_ensemble_repeats", [4]),
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
            "POLYNOMIAL_FEATURES", ["no", 50]
        ),
        "inference_config/OUTLIER_REMOVAL_STD": hp.choice(
            "OUTLIER_REMOVAL_STD", [None, 7.0, 9.0, 12.0]
        ),
        "inference_config/SUBSAMPLE_SAMPLES": hp.choice(
            "SUBSAMPLE_SAMPLES", [0.99, None]
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
            str(local_dir / "tabpfn-v2-regressor-5wof9ojf.ckpt"),
        ]
        search_space["inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"] = hp.choice(
            "REGRESSION_Y_PREPROCESS_TRANSFORMS",
            [
                (None,),
                (None, "power"),
                ("power",),
                ("safepower",),
                # ("kdi_alpha_0.3",),
                # ("kdi_alpha_1.0",),
                # ("kdi_alpha_1.5",),
                # ("kdi_alpha_0.6",),
                # ("kdi_alpha_3.0",),
                ("quantile_uni",),
            ],
        )
    else:
        raise ValueError(f"Unknown task type {task_type} for the search space!")

    search_space["model_path"] = hp.choice("model_path", model_paths)

    return search_space
