from __future__ import annotations

from typing import Literal

import numpy as np
from hyperopt.pyll import stochastic

from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt


def prepare_tabpfnv2_config(
            raw_config: dict, *,
            refit_folds: bool = True,
            default_n_estimators: int | None = 16,
            ) -> dict:
    """Set refit folds to True and convert tuples to lists."""
    raw_config = {
        k: list(v) if isinstance(v, tuple) else v for k, v in raw_config.items()
    }
    if "ag_args_ensemble" not in raw_config:
        raw_config["ag_args_ensemble"] = {}
    raw_config["ag_args_ensemble"]["refit_folds"] = True

    raw_config["n_estimators"] = default_n_estimators
    model_type = raw_config.get("model_type")

    if model_type == "dt_pfn":
        raw_config["n_ensemble_repeats"] = raw_config["n_estimators"]
        raw_config["n_estimators"] = 1

    raw_config.pop("max_depth", None)

    return raw_config


def search_space_func(
    task_type: Literal["regression", "multiclass"],
    num_random_configs: int = 200,
    seed=1234,
) -> list[dict]:
    """Generate a list of random configurations for TabPFNv2 from Search Space,
    which can be used as by Autogluon to construct a ensemble model.

    Also includes lazy imports for hyperopt since it is not in base requirements of package
    """
    search_space = get_param_grid_hyperopt(task_type=task_type)
    rng = np.random.default_rng(seed)
    stochastic.sample(search_space, rng=rng)
    return [
        prepare_tabpfnv2_config(dict(stochastic.sample(search_space, rng=rng)))
        for _ in range(num_random_configs)
    ]
