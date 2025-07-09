"""Implementation taken from TabArena: A Living Benchmark for Machine Learning on Tabular Data,
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas,
Frank Hutter, Preprint., 2025,

Original Code: https://github.com/autogluon/tabrepo/tree/main/tabrepo/benchmark/models/ag/tabpfnv2
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import scipy

from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import FLOAT_DTYPES


def prepare_tabpfnv2_config(raw_config: dict, *, refit_folds: bool = True) -> dict:
    """Set refit folds to True and convert tuples to lists."""
    raw_config = {
        k: list(v) if isinstance(v, tuple) else v for k, v in raw_config.items()
    }
    if "ag_args_ensemble" not in raw_config:
        raw_config["ag_args_ensemble"] = {}
    raw_config["ag_args_ensemble"]["refit_folds"] = True

    return raw_config


def search_space_func(
    task_type: Literal["regression", "multiclass"],
    num_random_configs: int = 200,
    seed=1234,
) -> list[dict]:
    '''
    Generate a list of random configurations for TabPFNv2 from Search Space,
    which can be used as by Autogluon to construct a ensemble model.

    Also includes lazy imports for hyperopt since it is not in base requirements of package
    '''
    from hyperopt.pyll import stochastic
    from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt
    
    search_space = get_param_grid_hyperopt(task_type=task_type)
    rng = np.random.default_rng(seed)
    stochastic.sample(search_space, rng=rng)
    return [
        prepare_tabpfnv2_config(dict(stochastic.sample(search_space, rng=rng)))
        for _ in range(num_random_configs)
    ]


# TODO: merge into codebase or remove KDITransformer from search space
def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
    """Check inputs before fit and transform."""
    return self._validate_data(
        X,
        reset=in_fit,
        accept_sparse=False,
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )


# TODO: merge into TabPFnv2 codebase
class FixedSafePowerTransformer(PowerTransformer):
    """Fixed version of safe power THAT FOLLOWS BASIC SKLEARN STANDARD ANS THUS DOES NOT HAVE A BUG WHEN CLONING
    WHY IS THIS SO HARD?
    """

    def __init__(
        self,
        variance_threshold: float = 1e-3,
        large_value_threshold: float = 100,
        method="yeo-johnson",
        standardize=True,
        copy=True,
    ):
        super().__init__(method=method, standardize=standardize, copy=copy)
        self.variance_threshold = variance_threshold
        self.large_value_threshold = large_value_threshold

        self.revert_indices_ = None

    def _find_features_to_revert_because_of_failure(
        self,
        transformed_X: np.ndarray,
    ) -> None:
        # Calculate the variance for each feature in the transformed data
        variances = np.nanvar(transformed_X, axis=0)

        # Identify features where the variance is not close to 1
        mask = np.abs(variances - 1) > self.variance_threshold
        non_unit_variance_indices = np.where(mask)[0]

        # Identify features with values greater than the large_value_threshold
        large_value_indices = np.any(transformed_X > self.large_value_threshold, axis=0)
        large_value_indices = np.nonzero(large_value_indices)[0]

        # Identify features to revert based on either condition
        self.revert_indices_ = np.unique(
            np.concatenate([non_unit_variance_indices, large_value_indices]),
        )

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"overflow encountered",
                    category=RuntimeWarning,
                )
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except scipy.optimize._optimize.BracketError:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        if np.isnan(lmbda):
            return x

        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore

    def _revert_failed_features(
        self,
        transformed_X: np.ndarray,
        original_X: np.ndarray,
    ) -> np.ndarray:
        # Replace these features with the original features
        if self.revert_indices_ and (self.revert_indices_) > 0:
            transformed_X[:, self.revert_indices_] = original_X[:, self.revert_indices_]

        return transformed_X

    def fit(self, X: np.ndarray, y: Any | None = None) -> FixedSafePowerTransformer:
        super().fit(X, y)

        # Check and revert features as necessary
        self._find_features_to_revert_because_of_failure(super().transform(X))  # type: ignore
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        transformed_X = super().transform(X)
        return self._revert_failed_features(transformed_X, X)  # type: ignore
