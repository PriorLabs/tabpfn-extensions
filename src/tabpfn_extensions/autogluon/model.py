from __future__ import annotations

from typing import TYPE_CHECKING

from torch.cuda import is_available

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model import preprocessing
from tabpfn_extensions.autogluon.utils import FixedSafePowerTransformer, _check_inputs
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)

if TYPE_CHECKING:
    import pandas as pd


class TabPFNV2Model(AbstractModel):
    """AutoGluon model wrapper for TabPFN (Tabular PFN v2).

    This class integrates the Tabular Neural Network (TabPFN) model into
    AutoGluon's framework, allowing it to be used as a component within
    AutoGluon's ensemble training process. It handles data preprocessing,
    categorical feature encoding, and manages the underlying TabPFNClassifier
    or TabPFNRegressor instances.

    The implementation is based on TabArena: A Living Benchmark for Machine Learning on Tabular Data,
    Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas,
    Frank Hutter, Preprint., 2025,

    Original Code: https://github.com/autogluon/tabrepo/tree/main/tabrepo/benchmark/models/ag/tabpfnv2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._cat_features = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        """Preprocesses the input DataFrame, including label encoding for categorical features.
        This method is called by AutoGluon's internal pipeline.
        """
        X = super()._preprocess(X, **kwargs)
        self._cat_indices = []

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(
                X=X
            )

            # Detect/set cat features and indices
            if self._cat_features is None:
                self._cat_features = self._feature_generator.features_in[:]
            self._cat_indices = [X.columns.get_loc(col) for col in self._cat_features]

        return X

    # FIXME: What is the minimal model artifact?
    #  If zeroshot, maybe we don't save weights for each fold in bag and instead load from a single weights file?
    # FIXME: Crashes during model download if bagging with parallel fit.
    #  Consider adopting same download logic as TabPFNMix which doesn't crash during model download.
    # FIXME: Maybe support child_oof somehow with using only one model and being smart about inference time?
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Fits the TabPFN model using the provided data.

        This method initializes either a `TabPFNClassifier` or `TabPFNRegressor`
        based on the problem type and trains it. It handles device selection
        (CPU/GPU) and passes relevant hyperparameters to the TabPFN model.

        Parameters
        ----------
        X : pandas.DataFrame
            The training input samples.
        y : pandas.Series
            The target values.
        num_cpus : int, default=1
            Number of CPUs available for training. Passed to TabPFN's `n_jobs`.
        num_gpus : int, default=0
            Number of GPUs available for training. If > 0 and CUDA is available,
            training will be performed on GPU.
        **kwargs :
            Additional keyword arguments from AutoGluon's training process.
        """
        preprocessing.SafePowerTransformer = FixedSafePowerTransformer
        preprocessing.KDITransformerWithNaN._check_inputs = _check_inputs

        ag_params = self._get_ag_params()
        max_classes = ag_params.get("max_classes")
        is_classification = self.problem_type in ["binary", "multiclass"]

        if is_classification:
            if max_classes is not None and self.num_classes > max_classes:
                raise AssertionError(
                    f"Max allowed classes for the model is {max_classes}, but found {self.num_classes} classes.",
                )

            model_base = TabPFNClassifier
        else:
            model_base = TabPFNRegressor

        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not is_available()):
            # FIXME: warn instead and switch to CPU.
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        X = self.preprocess(X, is_train=True)

        hps = self._get_model_params()
        hps["device"] = device
        hps["n_jobs"] = num_cpus
        hps["random_state"] = 42  # TODO: get seed from AutoGluon.
        hps["categorical_features_indices"] = self._cat_indices
        hps["ignore_pretraining_limits"] = True  # to ignore warnings and size limits

        max_depth_rf_pfn = hps.pop("max_depth")

        # Resolve inference_config
        inference_config = {
            _k: v
            for k, v in hps.items()
            if k.startswith("inference_config/") and (_k := k.split("/")[-1])
        }
        if inference_config:
            hps["inference_config"] = inference_config
        for k in list(hps.keys()):
            if k.startswith("inference_config/"):
                del hps[k]

        # TODO: remove power from search space and TabPFNv2 codebase
        # Power transform can fail. To avoid this, make all power be safepower instead.
        if "PREPROCESS_TRANSFORMS" in inference_config:
            safe_config = []
            for preprocessing_dict in inference_config["PREPROCESS_TRANSFORMS"]:
                if preprocessing_dict["name"] == "power":
                    preprocessing_dict["name"] = "safepower"
                safe_config.append(preprocessing_dict)
            inference_config["PREPROCESS_TRANSFORMS"] = safe_config
        if "REGRESSION_Y_PREPROCESS_TRANSFORMS" in inference_config:
            safe_config = []
            for preprocessing_name in inference_config[
                "REGRESSION_Y_PREPROCESS_TRANSFORMS"
            ]:
                if preprocessing_name == "power":
                    preprocessing_name = "safepower"
                safe_config.append(preprocessing_name)
            inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = safe_config

        # Resolve model_type
        n_ensemble_repeats = hps.pop("n_ensemble_repeats", hps["n_estimators"])
        model_is_rf_pfn = hps.pop("model_type", "no") == "rf_pfn"
        if model_is_rf_pfn:
            hps["n_estimators"] = 1
            rf_model_base = (
                RandomForestTabPFNClassifier
                if is_classification
                else RandomForestTabPFNRegressor
            )
            self.model = rf_model_base(
                tabpfn=model_base(**hps),
                categorical_features=self._cat_indices,
                n_estimators=n_ensemble_repeats,
                max_depth=max_depth_rf_pfn,
            )
        else:
            self.model = model_base(**hps)

        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _get_default_resources(self) -> tuple[int, int]:
        """Determines the default CPU and GPU resources available for the model."""
        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = min(ResourceManager.get_gpu_count_torch(), 1)
        return num_cpus, num_gpus

    def _set_default_params(self):
        """Sets default hyperparameters for the TabPFN model."""
        default_params = {
            "random_state": 42,
            "ignore_pretraining_limits": True,  # to ignore warnings and size limits
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        """Returns the list of supported problem types for the TabPFN model."""
        return ["binary", "multiclass", "regression"]

    def _get_default_auxiliary_params(self) -> dict:
        """Returns the default auxiliary parameters for the TabPFN model."""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_classes": 10,
            },
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local,
        as parallel folding crashes if model weights aren't pre-downloaded.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            # FIXME: Find a work-around to avoid crash if parallel and weights are not downloaded
            "fold_fitting_strategy": "sequential_local",
            "refit_folds": True,  # Better to refit the model for faster inference and similar quality as the bag.
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,  # noqa: ARG003
        **kwargs,  # noqa: ARG003
    ) -> int:
        """Heuristic memory estimate based on TabPFN's memory estimate logic in:
        https://github.com/PriorLabs/TabPFN/blob/57a2efd3ebdb3886245e4d097cefa73a5261a969/src/tabpfn/model/memory.py#L147
        This is based on GPU memory usage, but hopefully with overheads it also approximates CPU memory usage.
        """
        # features_per_group = 2  # Based on TabPFNv2 default (unused) # noqa: ERA001
        n_layers = 12  # Based on TabPFNv2 default
        embedding_size = 192  # Based on TabPFNv2 default
        dtype_byte_size = 2  # Based on TabPFNv2 default

        model_mem = 14489108  # Based on TabPFNv2 default

        n_samples, n_features = X.shape[0], X.shape[1]
        n_feature_groups = n_features + 1  # TODO: Unsure how to calculate this

        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = (
            n_samples * n_feature_groups * embedding_size * n_layers * dtype_byte_size
        )

        baseline_overhead_mem_est = 1e9  # 1 GB generic overhead

        # Add some buffer to each term + 1 GB overhead to be safe
        total_mem_bytes = int(
            model_mem + 4 * X_mem + 1.5 * activation_mem + baseline_overhead_mem_est
        )

        return total_mem_bytes

    @classmethod
    def _class_tags(cls):
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        """Returns the additional tags for the TabPFN model."""
        return {"can_refit_full": True}
