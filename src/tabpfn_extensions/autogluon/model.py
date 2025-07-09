from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from tabpfn_extensions.autogluon.utils import FixedSafePowerTransformer, _check_inputs

if TYPE_CHECKING:
    import pandas as pd


class TabPFNV2Model(AbstractModel):
    """AutoGluon model wrapper for TabPFN (Tabular PFN v2).

    This class integrates the Tabular Neural Network (TabPFN) model into
    AutoGluon's framework, allowing it to be used as a component within
    AutoGluon's ensemble training process. It handles data preprocessing,
    categorical feature encoding, and manages the underlying TabPFNClassifier
    or TabPFNRegressor instances.

    The implementation is based on TabArena: A Living Benchmark for Machine
    Learning on Tabular Data (Erickson et al., 2025).
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
        from tabpfn.model import preprocessing

        preprocessing.SafePowerTransformer = FixedSafePowerTransformer
        preprocessing.KDITransformerWithNaN._check_inputs = _check_inputs

        from torch.cuda import is_available

        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.model.loading import resolve_model_path

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

        _, model_dir, _, _ = resolve_model_path(
            model_path=None,
            which="classifier" if is_classification else "regressor",
        )
        if is_classification:
            if "classification_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("classification_model_path")
            if "regression_model_path" in hps:
                del hps["regression_model_path"]
        else:
            if "regression_model_path" in hps:
                hps["model_path"] = model_dir / hps.pop("regression_model_path")
            if "classification_model_path" in hps:
                del hps["classification_model_path"]

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
        n_ensemble_repeats = hps.pop("n_ensemble_repeats", None)
        model_is_rf_pfn = hps.pop("model_type", "no") == "rf_pfn"
        if model_is_rf_pfn:
            from tabpfn_extensions.rf_pfn import (
                RandomForestTabPFNClassifier,
                RandomForestTabPFNRegressor,
            )

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
            )
        else:
            if n_ensemble_repeats is not None:
                hps["n_estimators"] = n_ensemble_repeats
            self.model = model_base(**hps)

        self.model = self.model.fit(
            X=X,
            y=y,
        )

    def _get_default_resources(self) -> tuple[int, int]:
        """Determines the default CPU and GPU resources available for the model."""
        from torch.cuda import is_available

        from autogluon.common.utils.resource_utils import ResourceManager

        num_cpus = ResourceManager.get_cpu_count_psutil()
        num_gpus = 1 if is_available() else 0
        return num_cpus, num_gpus

    def _set_default_params(self):
        """Sets default hyperparameters for the TabPFN model.
        The n_ensemble_repeats is only used in the RandomForestTabPFNClassifier/RandomForestTabPFNRegressor.
        """
        default_params = {
            "model_type": "single",
            "n_ensemble_repeats": 8,
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

    def _ag_params(self) -> set:
        """Returns the set of auxiliary parameters for the TabPFN model."""
        return {"max_classes"}

    def _more_tags(self) -> dict:
        """Returns the additional tags for the TabPFN model."""
        return {"can_refit_full": True}
