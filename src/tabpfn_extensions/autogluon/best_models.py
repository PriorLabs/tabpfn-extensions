from __future__ import annotations

import copy
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import sklearn
from shared_utils.local_settings import model_string_config
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn_extensions.sklearn_ensembles.configs import (
    BaggingConfig,
    StackingConfig,
    WeightedAverageConfig,
)
from torch import nn

from configs import (
    TabPFNClassificationConfig,
    TabPFNConfig,
    TabPFNModelPathsConfig,
    TabPFNRegressionConfig,
    TabPFNRFConfig,
)
from utils import get_tabpfn

# Best model paths, descending order of performance


def get_model_strings(model_string_config):
    if model_string_config == "CLUSTER":
        model_strings = {
            "multiclass": [
                {
                    "path": "/home/muellesa/models_for_deployment/model_train_big_on_azujx8je_epoch:80,multi:Tru,multi:Fal,featu:2,feats:85_kl2dd853c_id_dyyjbfhj_epoch_49.cpkt",
                    "wandb_id": "dyyjbfhj",
                },
            ],
            "regression": [
                {
                    "path": "/home/muellesa/models_for_deployment/model_contin_mar25_regression_2noar4o2_id_6ckfgfxl_epoch_40.cpkt",
                    "wandb_id": "6ckfgfxl",
                },
            ],
            "survival": [
                {
                    "path": "/work/dlclarge2/hollmann-TabPFN/results/results/models_diff/model_pfn_type_survival_time_2023-11-09_16-29-04_26_id_75ca3846_epoch_-1.cpkt",
                    "wandb_id": "wq91e1q6",
                }
            ],
        }
    elif model_string_config == "CLUSTER_NEMO":
        # So far, PFN are not loaded via this file for NEMO benchmarks.
        pass
    elif model_string_config == "LOCAL":
        # Support for local debugging.
        from shared_utils.local_settings import local_model_path

        local_dir = Path(local_model_path).resolve()

        model_strings = {
            "multiclass": [
                {
                    "path": str(local_dir / "model_hans_classification.ckpt"),
                    "wandb_id": "-1",
                },
            ],
            "regression": [
                {
                    "path": str(local_dir / "model_hans_regression.ckpt"),
                    "wandb_id": "-1",
                },
            ],
            "survival": [
                {
                    "path": str(local_dir / "model_survival.ckpt"),
                    "wandb_id": "-1",
                }
            ],
        }
    else:
        raise ValueError(f"Unknown model_string_config {model_string_config=}")

    return model_strings


model_strings = get_model_strings(model_string_config)


### BEST CONFIGS ###

# Meaningful ensembling parameters for stacking and averaging
ensembling_params = (
    {
        "preprocess_transforms": (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        "use_poly_features": False,
    },
    {
        "preprocess_transforms": (
            PreprocessorConfig("none"),
            PreprocessorConfig("quantile_uni", categorical_name="numeric"),
        ),
        "use_poly_features": False,
    },
)


### DEFAULT CONFIGS ###

# If you want to change these configs, have a look at the InferenceTuning.ipynb
# you should run the full evaluation on valid_hard
# (`runner.TabPFNTrainingAgent(..., splits_to_evaluate={'valid_hard': None}).full_evaluation()`)
# of these defaults as well as of the new config
# You should generate a runs table with both using `scripts.tabular_evaluation_utils.wandb_get_runs`
# and add mixture scores to it with `scripts.tabular_evaluation_utils.add_mixed_scores_to_runs_df`.
# Now you should either have better performance or faster performance. Include the full resutls in your PR, ideally.

# You neeed to keep these in sync with the defaults in `tabpfn/scripts/transformer_prediction_interface.py`.

default_multiclass_config = TabPFNConfig(
    model_name="tabpfn_single_v_3_0",
    task_type="multiclass",
    model_type="single",
    paths_config=TabPFNModelPathsConfig(
        paths=[m["path"] for m in model_strings["multiclass"]]
    ),
    preprocess_transforms=(
        PreprocessorConfig(
            "quantile_uni_coarse",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
        ),
        PreprocessorConfig("none", categorical_name="numeric", subsample_features=-1),
    ),
    softmax_temperature=math.exp(-0.1),
    average_before_softmax=False,
    task_type_config=TabPFNClassificationConfig(),
    use_poly_features=False,
    add_fingerprint_features=True,
    remove_outliers=12.0,
    n_estimators=4,
    save_peak_memory="True",
    batch_size_inference=1,
    optimize_metric="roc",
)

default_regression_config = TabPFNConfig(
    model_name="tabpfn_single_fast_v_3_0",
    task_type="regression",
    model_type="single",
    paths_config=TabPFNModelPathsConfig(
        paths=[m["path"] for m in model_strings["regression"]],
        task_type="regression",
    ),
    preprocess_transforms=(
        PreprocessorConfig(
            "quantile_uni",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
        ),
        PreprocessorConfig("safepower", categorical_name="onehot"),
    ),
    task_type_config=TabPFNRegressionConfig(
        cancel_nan_borders=True,
        regression_y_preprocess_transforms=(
            None,
            "safepower",
        ),
    ),
    n_estimators=8,
    softmax_temperature=math.exp(-0.1),
    add_fingerprint_features=True,
    remove_outliers=None,
    use_poly_features=False,
    average_before_softmax=False,
    # save_peak_memory="auto", # TODO: Re-enable when memory tracking is adapted to the specific model
    save_peak_memory="True",
    batch_size_inference=1,
    optimize_metric="rmse",
)

###

best_tabpfn_configs = {
    "multiclass": {
        "single_fast": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            task_type_config=TabPFNClassificationConfig(),
            n_estimators=1,
            save_peak_memory="True",
            optimize_metric="roc",
        ),
        "single": default_multiclass_config,
        "single_light": TabPFNConfig(
            model_name="tabpfn_single_v_2_1",
            task_type="multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            task_type_config=TabPFNClassificationConfig(),
            n_estimators=8,
            save_peak_memory="True",
        ),
        "ensemble": TabPFNConfig(
            model_name="tabpfn_ensemble_v_2_1",
            task_type="multiclass",
            model_type="ensemble",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            task_type_config=TabPFNClassificationConfig(),
            n_estimators=8,
            save_peak_memory="True",
            optimize_metric="roc",
        ),
        "bagging": TabPFNConfig(
            model_name="tabpfn_bagging_v_2_1",
            task_type="multiclass",
            model_type="bagging",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            n_estimators=4,
            model_type_config=BaggingConfig(
                n_estimators=16,
                max_samples=2048,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
            ),
            task_type_config=TabPFNClassificationConfig(),
            # save_peak_memory="auto",
            optimize_metric="roc",
        ),
        "stacking": TabPFNConfig(
            model_name="tabpfn_stacking_v_2_1",
            task_type="multiclass",
            model_type="stacking",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            n_estimators=32,
            model_type_config=StackingConfig(
                cv=5,
                append_other_model_types=False,
                params_stacked=ensembling_params,
                final_estimator=sklearn.linear_model.LogisticRegression(),  # Better but slower: LogisticRegressionCV
            ),
            task_type_config=TabPFNClassificationConfig(),
            # save_peak_memory="auto", # TODO: Re-enable when memory tracking is adapted to the specific model
            optimize_metric="roc",
        ),
        "weighted_average": TabPFNConfig(
            model_name="tabpfn_weighted_average_v_2_1",
            task_type="multiclass",
            model_type="weighted_average",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            n_estimators=32,
            model_type_config=WeightedAverageConfig(
                cv=5,
                n_max=3,
                params_stacked=ensembling_params,
            ),
            task_type_config=TabPFNClassificationConfig(),
            # save_peak_memory="auto",
            optimize_metric="roc",
        ),
        "rf_pfn": TabPFNConfig(
            model_name="tabpfn_rf_v_2_1",
            task_type="multiclass",
            model_type="rf_pfn",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            n_estimators=2,
            model_type_config=TabPFNRFConfig(
                adaptive_tree_overwrite_metric="log_loss",
                rf_average_before_softmax=True,
                max_predict_time=60,
            ),
            batch_size_inference=1,
            preprocess_transforms=(PreprocessorConfig("quantile_uni"),),
            task_type_config=TabPFNClassificationConfig(),
            # save_peak_memory="auto", # TODO: Re-enable when memory tracking is adapted to the specific model
            optimize_metric="roc",
        ),
        "dt_pfn": TabPFNConfig(
            model_name="dt_tabpfn",
            task_type="multiclass",
            model_type="rf_pfn",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            n_estimators=1,
            model_type_config=TabPFNRFConfig(
                max_predict_time=-1,
                max_depth=None,
                n_estimators=1,
            ),
            batch_size_inference=1,
            preprocess_transforms=(PreprocessorConfig("quantile_uni"),),
            task_type_config=TabPFNClassificationConfig(),
            optimize_metric="roc",
        ),
    },
    "regression": {
        "single_fast": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="regression",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["regression"]],
                task_type="regression",
            ),
            task_type_config=TabPFNRegressionConfig(),
            n_estimators=1,
            # save_peak_memory="auto",
            optimize_metric="rmse",
        ),
        "single_light": TabPFNConfig(
            model_name="tabpfn_single_v_2_1",
            task_type="regression",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["regression"]],
                task_type="regression",
            ),
            task_type_config=TabPFNRegressionConfig(),
            n_estimators=8,
            # save_peak_memory="True",
        ),
        "ensemble": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="regression",
            model_type="ensemble",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["regression"]],
                task_type="regression",
            ),
            task_type_config=TabPFNRegressionConfig(),
            n_estimators=8,
            # save_peak_memory="True",
        ),
        "single": default_regression_config,
        "rf_pfn": TabPFNConfig(
            model_name="tabpfn_rf_v_2_1",
            task_type="regression",
            model_type="rf_pfn",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["regression"]],
                task_type="regression",
            ),
            n_estimators=16,
            model_type_config=TabPFNRFConfig(
                min_samples_split=300,
                criterion="friedman_mse",
                adaptive_tree_overwrite_metric="rmse",
            ),
            task_type_config=TabPFNRegressionConfig(),
            # save_peak_memory="auto", # TODO: Re-enable when memory tracking is adapted to the specific model
            optimize_metric="rmse",
        ),
        "dt_pfn": TabPFNConfig(
            model_name="dt_tabpfn",
            task_type="regression",
            model_type="rf_pfn",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["regression"]],
                task_type="regression",
            ),
            n_estimators=1,
            model_type_config=TabPFNRFConfig(
                max_predict_time=-1,
                max_depth=None,
                criterion="friedman_mse",
                adaptive_tree_overwrite_metric="rmse",
            ),
            task_type_config=TabPFNRegressionConfig(),
            batch_size_inference=1,
            preprocess_transforms=(PreprocessorConfig("quantile_uni"),),
            optimize_metric="rmse",
        ),
    },
    "survival": {
        "single_fast": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="survival",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["survival"]],
                task_type="survival",
            ),
            n_estimators=1,
            save_peak_memory="auto",
            optimize_metric="cindex",
        ),
        "single": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="survival",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["survival"]],
                task_type="survival",
            ),
            n_estimators=128,
            use_poly_features=False,
            save_peak_memory="auto",
            optimize_metric="cindex",
        ),
    },
}

best_tabpfn_configs["multiclass"]["best"] = best_tabpfn_configs["multiclass"]["single"]
best_tabpfn_configs["multiclass"]["outer_ensemble"] = deepcopy(
    best_tabpfn_configs["multiclass"]["single"]
)
best_tabpfn_configs["multiclass"]["outer_ensemble"].model_type = "outer_ensemble"

best_tabpfn_configs["quantile_regression"] = best_tabpfn_configs["regression"]


def get_best_tabpfn_config(
    task_type: str,
    model_type: str = "single_fast",
    debug: bool = False,
    model: Optional[nn.Module] = None,
    paths_config=None,
    model_config: Optional[Dict] = None,
    return_list_of_config_per_model_string: bool = False,
) -> TabPFNConfig | list[TabPFNConfig]:
    config = copy.deepcopy(best_tabpfn_configs[task_type][model_type])

    # In debug mode sets small parameters for the best models
    if debug:
        config.n_estimators = min(2, config.n_estimators)
        if config.model_type == "ensemble":
            config.paths_config.model_strings = config.paths_config.model_strings[:2]
        elif config.model_type == "bagging":
            config.model_type_config.n_estimators = min(
                2, config.model_type_config.n_estimators
            )
        elif config.model_type == "stacking":
            config.model_type_config.params_stacked = (
                config.model_type_config.params_stacked[:2]
            )
            config.model_type_config.cv = 2
        elif config.model_type == "rf_pfn":
            config.model_type_config.n_estimators = min(
                2, config.model_type_config.n_estimators
            )

    assert not (model and paths_config), "Only one of model and paths_config can be set"
    if model:
        config.model = model
        config.model_config = model_config
        config.paths_config = None

    if paths_config:
        config.paths_config = paths_config
        config.model = None
        config.model_config = None

    if return_list_of_config_per_model_string:
        assert config.paths_config is not None, (
            "paths_config must be set to return list of configs per model string!"
        )

        config_per_model_string = []
        for model_string in config.paths_config.model_strings:
            tmp_config = copy.deepcopy(config)
            tmp_config.paths_config = TabPFNModelPathsConfig(
                paths=[model_string], task_type=task_type
            )
            config_per_model_string.append(tmp_config)

        return config_per_model_string

    return config


### GETTING BEST MODELS ###


def _infer_config_overwrite(config, inference_config_overwrite, verbose=False):
    """
    This function looks for the keys in inference_config_overwrite and overwrites the corresponding value in either config, config.model_type_config, config.paths_config or config.task_type_config.
    """
    sub_configs = [
        config.model_type_config,
        config.paths_config,
        config.task_type_config,
    ]
    for key in inference_config_overwrite.keys():
        if key in config.__dict__:
            config.__dict__[key] = inference_config_overwrite[key]
        else:
            found = False
            for sub_config in sub_configs:
                if sub_config and key in sub_config.__dict__:
                    sub_config.__dict__[key] = inference_config_overwrite[key]
                    found = True
                    if verbose:
                        print(f"updated {key=} in subconfig")
                    break

                # workaround for overwriting rf-pfn's n_estimators
                if (
                    isinstance(sub_configs[0], TabPFNRFConfig)
                    and key == "rf_pfn_n_estimators"
                ):
                    sub_configs[0].n_estimators = inference_config_overwrite[key]
                    found = True
                    if verbose:
                        print(f"updated rf-pfn n-estimators {key=} in subconfig")
                    break

            if not found:
                raise ValueError(f"Unknown config key {key}")
    if verbose:
        print("building tabpfn model with config", config)


def get_best_tabpfn(
    task_type: str,
    model_type: str = "single_fast",
    paths_config: TabPFNModelPathsConfig = None,
    model=None,
    model_config=None,
    debug: bool = False,  # If True, uses small parameters for the best models. Useful for debugging and testing.
    inference_config_overwrite: dict = None,
    **kwargs,
) -> TabPFNRegressor | TabPFNClassifier:
    inference_config_overwrite = deepcopy(inference_config_overwrite)
    if (
        inference_config_overwrite is not None
        and "paths_config" in inference_config_overwrite
    ):
        assert paths_config is None, (
            "paths_config can't be set if it's in inference_config_overwrite"
        )
        paths_config = inference_config_overwrite.pop("paths_config")
    config = get_best_tabpfn_config(
        task_type,
        model_type,
        debug=debug,
        model=model,
        model_config=model_config,
        paths_config=paths_config,
    )
    if inference_config_overwrite is not None:
        _infer_config_overwrite(config, inference_config_overwrite)

    return get_tabpfn(config, **kwargs)


def get_all_best_tabpfns(
    task_type: str, debug: bool = False, **kwargs
) -> Dict[str, TabPFNRegressor | TabPFNClassifier]:
    configs = {
        model_type: get_best_tabpfn_config(task_type, model_type, debug=debug)
        for model_type in best_tabpfn_configs[task_type].keys()
    }

    return {k: get_tabpfn(config, **kwargs) for (k, config) in configs.items()}
