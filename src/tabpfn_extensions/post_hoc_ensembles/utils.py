from __future__ import annotations

from typing import Literal

import numpy as np
from hyperopt.pyll import stochastic

from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt


def prepare_tabpfnv2_config(
    raw_config: dict,
    n_estimators: int,
    balance_probabilities: bool | None,
    ignore_pretraining_limits: bool,
    *,
    refit_folds: bool = True,
) -> dict:
    """Cleans and prepares a raw TabPFN hyperparameter configuration.

    This function performs several steps:
    - Converts tuple values to lists for compatibility.
    - Ensures 'refit_folds' is set to True in 'ag_args_ensemble'.
    - Sets 'n_estimators' to a specified value.
    - Applies special logic for 'dt_pfn' model type.
    - Removes the 'max_depth' key if it exists.
    """
    raw_config = {
        k: list(v) if isinstance(v, tuple) else v for k, v in raw_config.items()
    }
    if "ag_args_ensemble" not in raw_config:
        raw_config["ag_args_ensemble"] = {}
    raw_config["ag_args_ensemble"]["refit_folds"] = True

    # Set TabPFN parameters
    raw_config["n_estimators"] = n_estimators
    raw_config["ignore_pretraining_limits"] = ignore_pretraining_limits
    if balance_probabilities:
        raw_config["balance_probabilities"] = balance_probabilities

    model_type = raw_config.get("model_type")

    if model_type == "dt_pfn":
        raw_config["n_ensemble_repeats"] = raw_config["n_estimators"]
        raw_config["n_estimators"] = 1

    raw_config.pop("max_depth", None)

    return raw_config


def search_space_func(
    task_type: Literal["regression", "multiclass"],
    n_estimators: int,
    ignore_pretraining_limits: bool,
    n_ensemble_models: int,
    balance_probabilities: bool | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate a list of random configurations for TabPFNv2 from its search space.

    These configurations can be used by AutoGluon to construct an ensemble model.
    """
    assert n_ensemble_models > 0, "n_ensemble_models must be > 0"
    assert n_estimators > 0, "n_estimators must be > 0"

    if task_type == "regression":
        balance_probabilities = None

    search_space = get_param_grid_hyperopt(task_type=task_type)
    rng = np.random.default_rng(seed)
    return [
        prepare_tabpfnv2_config(
            raw_config=dict(stochastic.sample(search_space, rng=rng)),
            n_estimators=n_estimators,
            balance_probabilities=balance_probabilities,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )
        for _ in range(n_ensemble_models)
    ]
