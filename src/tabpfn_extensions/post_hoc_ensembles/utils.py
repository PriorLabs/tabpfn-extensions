from __future__ import annotations

import functools
from typing import Any, Literal

import numpy as np
import torch
from hyperopt.pyll import stochastic

from tabpfn.inference_config import InferenceConfig
from tabpfn.model_loading import ModelVersion
from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt


@functools.cache
def _read_ckpt_ag_limits(model_path: str) -> dict[str, int] | None:
    """Read AutoGluon max_* limits from a TabPFN checkpoint's embedded inference_config.

    Returns a dict of {max_rows, max_features, max_classes}, or None if the
    checkpoint does not embed an inference_config (v2 and v2.5 do not).
    """
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    ic = ck.get("inference_config")
    if isinstance(ic, dict):
        return {
            "max_rows": ic["MAX_NUMBER_OF_SAMPLES"],
            "max_features": ic["MAX_NUMBER_OF_FEATURES"],
            "max_classes": ic["MAX_NUMBER_OF_CLASSES"],
        }
    return None


def _get_tabpfn_ag_limits(
    task_type: Literal["regression", "multiclass"],
    model_version: ModelVersion,
    model_path: str,
) -> dict[str, int]:
    """Resolve AutoGluon max_rows/max_features/max_classes from the checkpoint.

    Prefer the checkpoint's embedded inference_config (v2.6+); fall back to
    tabpfn's static defaults via `InferenceConfig.get_default` for v2 and v2.5,
    where the config is not stored in the checkpoint.

    These values are used as `ag_args` overrides so that AutoGluon's hardcoded
    defaults for TabPFNv2 (10000 / 500 / 10) do not clip larger-capacity
    checkpoints.
    """
    embedded = _read_ckpt_ag_limits(model_path)
    if embedded is not None:
        return embedded
    cfg = InferenceConfig.get_default(task_type=task_type, model_version=model_version)
    return {
        "max_rows": cfg.MAX_NUMBER_OF_SAMPLES,
        "max_features": cfg.MAX_NUMBER_OF_FEATURES,
        "max_classes": cfg.MAX_NUMBER_OF_CLASSES,
    }


def prepare_tabpfnv2_config(
    raw_config: dict[str, Any],
    n_estimators: int,
    balance_probabilities: bool | None,
    ignore_pretraining_limits: bool,
    *,
    refit_folds: bool = True,
    ag_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Prepare a raw TabPFN hyperparameter configuration for TabPFNv2.

    This function:
    - Converts tuple values into lists for JSON compatibility.
    - Ensures the `ag_args_ensemble` dict exists and applies `refit_folds`.
    - Sets `n_estimators` and `ignore_pretraining_limits` flags.
    - Applies or removes `balance_probabilities`.
    - Handles the special case when `model_type == 'dt_pfn'`.
    - Removes the deprecated `max_depth` key if present.
    - Optionally injects per-checkpoint `max_rows`/`max_features`/`max_classes`
      overrides into `ag_args` via `ag_limits`.

    Parameters
    ----------
    raw_config : Dict[str, Any]
        Hyperparameter dict sampled from Hyperopt.
    n_estimators : int
        Number of estimators in the ensemble (must be ≥ 1).
    balance_probabilities : Optional[bool]
        If True/False, set for classification; if None, removed (regression).
    ignore_pretraining_limits : bool
        Whether to bypass default pretraining limits.
    refit_folds : bool, optional
        Whether each fold should be refit (default is True).
    ag_limits : Optional[Dict[str, int]]
        Per-checkpoint AutoGluon limits (`max_rows`, `max_features`,
        `max_classes`) to merge into `ag_args_fit`. Overrides the hardcoded
        10000/500/10 defaults in AutoGluon's TabPFNv2 integration so that
        larger-capacity checkpoints (v2.5, v2.6) are not artificially capped.

    Returns:
    -------
    Dict[str, Any]
        A cleaned and fully-specified TabPFNv2 config.

    -------

    Note: RF-PFN is not supported at the moment and we
    disable its relevant parameters here.
    """
    # Shallow copy and tuple-to-list conversion
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in raw_config.items()}

    # Ensure ensemble args exist
    ensemble_args = config.setdefault("ag_args_ensemble", {})
    ensemble_args["refit_folds"] = refit_folds

    # Set core parameters
    config["n_estimators"] = n_estimators
    config["ignore_pretraining_limits"] = ignore_pretraining_limits

    # Classification vs. regression
    if balance_probabilities is not None:
        config["balance_probabilities"] = balance_probabilities
    else:
        config.pop("balance_probabilities", None)

    # TODO: Enable RF-PFN at some point
    config["model_type"] = "single"

    # Special case for dt_pfn
    # TODO: This code is unused until we support RF-PFN
    if config.get("model_type") == "dt_pfn":
        config["n_ensemble_repeats"] = config["n_estimators"]
        config["n_estimators"] = 1

    # Remove deprecated keys
    config.pop("max_depth", None)

    # Per-checkpoint AutoGluon limit overrides. AutoGluon's TabPFNv2 integration
    # hardcodes max_rows=10000 / max_features=500 / max_classes=10 (tuned for v2);
    # we override via ag_args_fit (AutoGluon's params_aux bucket) so v2.5/v2.6
    # checkpoints can use their full capacity.
    if ag_limits is not None:
        ag_args_fit = config.setdefault("ag_args_fit", {})
        ag_args_fit.update(ag_limits)

    return config


def search_space_func(
    task_type: Literal["regression", "multiclass"],
    n_ensemble_models: int,
    n_estimators: int,
    ignore_pretraining_limits: bool,
    balance_probabilities: bool | None = None,
    seed: int = 42,
    model_version: ModelVersion = ModelVersion.V2_5,
) -> list[dict[str, Any]]:
    """Sample and prepare multiple TabPFNv2 hyperparameter sets.

    Each dict in the returned list is ready for AutoGluon ensemble building.

    Parameters
    ----------
    task_type : Literal["regression", "multiclass"]
        Task type; regression will drop probability balancing.
    n_ensemble_models : int
        Number of configs to generate (must be > 1).
    n_estimators : int
        Estimators per model (must be > 0).
    ignore_pretraining_limits : bool
        Whether to bypass default pretraining limits.
    balance_probabilities : Optional[bool], optional
        Classification probability balancing; ignored for regression.
    seed : int, optional
        RNG seed for reproducibility (default is 42).

    Returns:
    -------
    List[Dict[str, Any]]
        A list of cleaned TabPFNv2 configurations.

    Raises:
    ------
    ValueError
        If `n_ensemble_models <= 1` or `n_estimators <= 0`.
    """
    if n_ensemble_models <= 1:
        raise ValueError(f"n_ensemble_models must be >1 (got {n_ensemble_models})")
    if n_estimators <= 0:
        raise ValueError(f"n_estimators must be >0 (got {n_estimators})")

    if task_type == "regression":
        balance_probabilities = None

    search_space = get_param_grid_hyperopt(
        task_type=task_type, model_version=model_version
    )
    rng = np.random.default_rng(seed)
    tabpfn_configs = []
    for _ in range(n_ensemble_models):
        raw_config = dict(stochastic.sample(search_space, rng=rng))
        ag_limits = _get_tabpfn_ag_limits(
            task_type=task_type,
            model_version=model_version,
            model_path=raw_config["model_path"],
        )
        tabpfn_configs.append(
            prepare_tabpfnv2_config(
                raw_config=raw_config,
                n_estimators=n_estimators,
                balance_probabilities=balance_probabilities,
                ignore_pretraining_limits=ignore_pretraining_limits,
                ag_limits=ag_limits,
            )
        )

    assert len(tabpfn_configs) > 0

    return tabpfn_configs
