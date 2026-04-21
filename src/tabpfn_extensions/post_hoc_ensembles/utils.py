from __future__ import annotations

from typing import Any, Literal

import numpy as np
from hyperopt.pyll import stochastic

from tabpfn.model_loading import ModelVersion
from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt


def prepare_tabpfnv2_config(
    raw_config: dict[str, Any],
    n_estimators: int,
    balance_probabilities: bool | None,
    ignore_pretraining_limits: bool,
    *,
    refit_folds: bool = True,
) -> dict[str, Any]:
    """Prepare a raw TabPFN hyperparameter configuration for TabPFNv2.

    This function:
    - Converts tuple values into lists for JSON compatibility.
    - Ensures the `ag_args_ensemble` dict exists and applies `refit_folds`.
    - Sets `n_estimators` and `ignore_pretraining_limits` flags.
    - Applies or removes `balance_probabilities`.
    - Handles the special case when `model_type == 'dt_pfn'`.
    - Removes the deprecated `max_depth` key if present.
    - Disables AutoGluon's v2-era static `max_rows` / `max_features` /
      `max_classes` guardrails so gating is delegated to TabPFN itself.

    Parameters
    ----------
    raw_config : Dict[str, Any]
        Hyperparameter dict sampled from Hyperopt.
    n_estimators : int
        Number of estimators in the ensemble (must be ≥ 1).
    balance_probabilities : Optional[bool]
        If True/False, set for classification; if None, removed (regression).
    ignore_pretraining_limits : bool
        Whether to bypass default pretraining limits in the wrapped TabPFN.
    refit_folds : bool, optional
        Whether each fold should be refit (default is True).

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

    # AutoGluon's TabPFNv2 integration hardcodes max_rows=10000 /
    # max_features=500 / max_classes=10 in `_get_default_auxiliary_params`.
    # These are v2-era values and are wrong for v2.5 (50K / 2000) and v2.6
    # (100K / 2000). Rather than mirroring them here, we disable the AG-side
    # checks entirely and let TabPFN's own validation (which knows the actual
    # loaded checkpoint's `inference_config.MAX_NUMBER_OF_*`) be the single
    # authority. TabPFN gating itself is still controlled by
    # `ignore_pretraining_limits` above.
    ag_args_fit = config.setdefault("ag_args_fit", {})
    ag_args_fit.update({"max_rows": None, "max_features": None, "max_classes": None})

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
    tabpfn_configs = [
        prepare_tabpfnv2_config(
            raw_config=dict(stochastic.sample(search_space, rng=rng)),
            n_estimators=n_estimators,
            balance_probabilities=balance_probabilities,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )
        for _ in range(n_ensemble_models)
    ]

    assert len(tabpfn_configs) > 0

    return tabpfn_configs
