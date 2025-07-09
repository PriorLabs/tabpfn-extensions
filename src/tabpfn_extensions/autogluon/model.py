"""Code for integrating TabPFN into AutoGluon as a custom model and to make it part of the presets/hyperparameters."""

import os

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models import AbstractModel
from autogluon.core.utils import generate_train_test_split
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
from autogluon.tabular.configs.presets_configs import tabular_presets_dict

from best_models import get_best_tabpfn, get_best_tabpfn_config
from utils import get_tabpfn
from configs import TabPFNConfig


class TabPFNV2Model(AbstractModel):
    """AutoGluon model wrapper to the TabPFN model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_gpus=0, **kwargs):
        task_type_mapper = {
            BINARY: "multiclass",
            MULTICLASS: "multiclass",
            REGRESSION: "regression",
        }
        task_type = task_type_mapper[self.problem_type]

        ag_params = self._get_ag_params()
        # sample_rows = ag_params.get("sample_rows")
        # max_features = ag_params.get("max_features")
        max_classes = ag_params.get("max_classes")
        num_gpus = 1 if os.environ.get("TABPFN_AG_REFIT_WITH_GPU", False) else num_gpus
        device = "cuda" if num_gpus != 0 else "cpu"
        if (
            (max_classes is not None)
            and (self.num_classes is not None)
            and (self.num_classes > max_classes)
        ):
            raise AssertionError(
                f"Max allowed classes for the model is {max_classes}, "
                f"but found {self.num_classes} classes.",
            )

        hyp = self._get_model_params()
        self._use_preprocess = hyp.get("preprocess")

        ##  Make sample_rows generic
        # if sample_rows is not None and len(X) > sample_rows:
        #    X, y = self._subsample_train(X=X, y=y, num_rows=sample_rows)

        # num_features = X.shape[1]
        ## Make max_features generic
        # if max_features is not None and num_features > max_features:
        #    raise AssertionError(
        #        f"Max allowed features for the model is {max_features}, "
        #        f"but found {num_features} features."
        #    )

        if self._use_preprocess:
            X = self.preprocess(X)

        model_type_from_hyp = hyp.get("model_type", None)

        if model_type_from_hyp is not None:
            # For AutoGluon < v1.0.0)
            self.model = get_best_tabpfn(
                task_type=task_type,
                model_type=model_type_from_hyp,
                device=device,
            )
        else:
            # Assume a TabPFNConfig was given (for AutoGluon >= v1.0.0)
            if (tabpfn_config := hyp.get("tabpfn_config", None)) is None:
                raise ValueError(
                    "Either `model_type` or `tabpfn_config` must be specified.",
                )
            if not isinstance(tabpfn_config, TabPFNConfig):
                raise ValueError(
                    f"Expected `tabpfn_config` to be of type `TabPFNConfig`, but got {type(tabpfn_config)}",
                )

            self.model = get_tabpfn(
                config=tabpfn_config,
                device=device,
            )

        self.model.show_progress = False
        self.model.fit(X, y)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        torch.cuda.empty_cache()
        if self._use_preprocess:
            X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X)

        y_pred_proba = self.model.predict_proba(X)
        torch.cuda.empty_cache()
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Converts categorical to label encoded integers.

        Keeps missing values, as TabPFN automatically handles missing values internally.
        """
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X,
            )
        return X.to_numpy(dtype=np.float32)

    def _set_default_params(self):
        """By default, we only use 1 ensemble configurations to speed up inference times.

        Increase the value to improve model quality while linearly increasing inference time.
        Model quality improvement diminishes significantly beyond `n_estimators=8`.
        """
        default_params = {"preprocess": False}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "problem_types": [BINARY, MULTICLASS, REGRESSION],
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    def _get_default_auxiliary_params(self) -> dict:
        """TabPFN only works on datasets with at most 10 classes, so we set `max_classes=10`."""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            },
        )
        return default_auxiliary_params

    # Enabling parallel bagging TabPFN creates a lot of warnings / potential failures from Ray
    # Consider not setting `max_sets=1`, and only setting it in the preset hyperparameter definition.
    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set max_sets to 1 when bagging, otherwise inference time could become extremely slow.

        Set fold_fitting_strategy to sequential_local, as parallel folding causing many warnings / potential errors from Ray.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "max_sets": 1,  # from original implementation, avoids repeated validation OOF for TabPFNs
            "fold_fitting_strategy": "sequential_local",  # much faster and better GPU util but might leak GPU memory.
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _ag_params(self) -> set:
        return {"max_classes"}

    def _more_tags(self) -> dict:
        """Because TabPFN doesn't use validation data for early stopping, it supports refit_full natively."""
        return {"can_refit_full": True}

    # UNUSED / DEPRECATED
    # Make this generic by creating a generic `preprocess_train` and putting this logic prior to `_preprocess`.
    def _subsample_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_rows: int,
        random_state=0,
    ) -> (pd.DataFrame, pd.Series):
        num_rows_to_drop = len(X) - num_rows
        X, _, y, _ = generate_train_test_split(
            X=X,
            y=y,
            problem_type=self.problem_type,
            test_size=num_rows_to_drop,
            random_state=random_state,
            min_cls_count_train=1,
        )
        return X, y


def get_hyperparameter_preset_v1(
    *,
    num_gpus_tabpfn: int,
    custom_hps_preset: str,
    preset: str,
    task_type: str,
    paths_config: list[str],
    refit_tabpfn=False,
    paths_config_names: list[str] | None = None,
):
    tabpfn_configs: list[TabPFNConfig] = []
    prio = 110  # high prio because fast
    if custom_hps_preset == "full_and_tabpfn":
        tabpfn_configs.append(
            get_best_tabpfn_config(
                task_type=task_type,
                model_type="single_light",
                paths_config=paths_config,
            ),
        )
    elif custom_hps_preset == "full_and_rf_pfn":
        if task_type == "regression":
            raise NotImplementedError(
                "custom_hps_preset='full_and_rf_pfn' not supported for regression as RF PFN ist not supported for RF so far!"
            )
        tabpfn_configs.append(
            get_best_tabpfn_config(
                task_type=task_type,
                model_type="rf_pfn",
                paths_config=paths_config,
            ),
        )
        prio = 80  # lower prio because slower
    elif custom_hps_preset == "ensemble_of_tabpfns":
        tabpfn_configs.extend(
            get_best_tabpfn_config(
                task_type=task_type,
                model_type="single_light",  # results in len(tabpfn_configs)*8 many ensemble members.
                return_list_of_config_per_model_string=True,
                paths_config=paths_config,
            ),
        )
    else:
        raise ValueError(f"custom_hps_preset={custom_hps_preset} not supported.")

    default_ag_hps = hyperparameter_config_dict[
        tabular_presets_dict[preset]["hyperparameters"]
    ]

    fit_resources = {"num_gpus": num_gpus_tabpfn}
    if os.name != "nt":
        fit_resources["num_cpus"] = len(os.sched_getaffinity(0)) // num_gpus_tabpfn

    tabpfn_args = [
        {
            "ag_args": {"priority": prio + p_idx, "name_suffix": "Gini"},
            "ag_args_fit": fit_resources,
            "ag_args_ensemble": {"refit_folds": True} if refit_tabpfn else {},
            "tabpfn_config": conf,
        }
        for p_idx, conf in enumerate(tabpfn_configs)
    ]

    if paths_config_names is not None:
        for p_idx, p_n in enumerate(paths_config_names):
            tabpfn_args[p_idx]["ag_args"]["name_suffix"] = "_" + p_n

    return dict(
        full_and_tabpfn={
            **default_ag_hps,
            TabPFNV2Model: tabpfn_args,
        },
        full_and_rf_pfn={
            **default_ag_hps,
            TabPFNV2Model: tabpfn_args,
        },
        ensemble_of_tabpfns={
            TabPFNV2Model: tabpfn_args,
        },
    )[custom_hps_preset]


def get_hyperparameter_presets(num_gpus_tabpfn, model_type, refit_tabpfn=False):
    # if version.parse(autogluon.__version__) >= version.parse("1.0.0"):
    #    raise NotImplementedError("outdated for 1.0.0")

    tabpfn_args = [
        {
            "ag_args": {"priority": 111},
            "ag_args_fit": {"num_gpus": num_gpus_tabpfn},
            "ag_args_ensemble": {"refit_folds": True} if refit_tabpfn else {},
            "model_type": model_type,  # This is the crucial part for TabPFNV2Model to pick up
        },
    ]

    return {
        "single_tabpfn": {TabPFNV2Model: tabpfn_args},
        "stacked_tabpfn": {TabPFNV2Model: tabpfn_args},
        "stacked_tabpfn_and_rf": {
            TabPFNV2Model: tabpfn_args,
            "RF": hyperparameter_config_dict["default"]["RF"],
            "XT": hyperparameter_config_dict["default"]["XT"],
            "GBM": hyperparameter_config_dict["default"]["GBM"],
        },
        "full_and_tabpfn": {
            **hyperparameter_config_dict["default"],
            TabPFNV2Model: tabpfn_args,
        },
        "full": None,
    }
