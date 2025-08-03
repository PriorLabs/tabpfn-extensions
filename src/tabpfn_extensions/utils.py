#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import itertools
import logging
import os
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import numpy as np

# Type checking imports
if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class TabPFNEstimator(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


def is_tabpfn(estimator: Any) -> bool:
    """Check if an estimator is a TabPFN model."""
    try:
        return any(
            [
                "TabPFN" in str(estimator.__class__),
                "TabPFN" in str(estimator.__class__.__bases__),
                any("TabPFN" in str(b) for b in estimator.__class__.__bases__),
                "tabpfn.base_model.TabPFNBaseModel" in str(estimator.__class__.mro()),
            ],
        )
    except (AttributeError, TypeError):
        return False


try:
    from tabpfn.utils import infer_device_and_type
except ImportError:
    # Fallback for environments without tabpfn package, in these environments
    # prediction uses the API client, so we can just return "cpu"

    # torch device mock dataclass
    @dataclass
    class DeviceType:
        type: str

    def infer_device_and_type(device: Literal["cpu", "cuda", "auto"]):
        if device in ("cuda"):
            warnings.warn(
                "CUDA device requested but 'tabpfn' package not found. "
                "Falling back to CPU as the client-based API does not support GPU.",
                UserWarning,
                stacklevel=2,
            )
        return DeviceType(type="cpu")


USE_TABPFN_LOCAL = os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"

try:
    from tabpfn_client import (
        TabPFNClassifier as ClientTabPFNClassifierBase,
        TabPFNRegressor as ClientTabPFNRegressorBase,
    )

    # Debug info controlled by environment variable
    # (using logging rather than print for better debugging)
    if os.getenv("TABPFN_DEBUG", "false").lower() == "true":
        logging.info("Using TabPFN client")

    # Wrapper classes to add device parameter
    # we can't use *args because scikit-learn needs to know the parameters of the constructor
    class ClientTabPFNClassifier(ClientTabPFNClassifierBase):
        def __init__(
            self,
            device: str | None = None,
            categorical_features_indices: list[int] | None = None,
            model_path: str = "default",
            n_estimators: int = 4,
            softmax_temperature: float = 0.9,
            balance_probabilities: bool = False,
            average_before_softmax: bool = False,
            ignore_pretraining_limits: bool = False,
            inference_precision: Literal["autocast", "auto"] = "auto",
            random_state: int
            | np.random.RandomState
            | np.random.Generator
            | None = None,
            inference_config: dict | None = None,
            paper_version: bool = False,
        ) -> None:
            self.device = device
            # Categorical features need to be passed but are not used by client
            self.categorical_features_indices = categorical_features_indices
            if categorical_features_indices is not None:
                warnings.warn(
                    "categorical_features_indices is not supported in the client version of TabPFN and will be ignored",
                    UserWarning,
                    stacklevel=2,
                )
            if "/" in model_path:
                model_name = model_path.split("/")[-1].split("-")[-1].split(".")[0]
                if model_name == "classifier":
                    model_name = "default"
                self.model_path = model_name
            else:
                self.model_path = model_path

            super().__init__(
                model_path=self.model_path,
                n_estimators=n_estimators,
                softmax_temperature=softmax_temperature,
                balance_probabilities=balance_probabilities,
                average_before_softmax=average_before_softmax,
                ignore_pretraining_limits=ignore_pretraining_limits,
                inference_precision=inference_precision,
                random_state=random_state,
                inference_config=inference_config,
                paper_version=paper_version,
            )

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Return parameters for this estimator."""
            params = super().get_params(deep=deep)
            params.pop("device")
            params.pop("categorical_features_indices")
            return params

    class ClientTabPFNRegressor(ClientTabPFNRegressorBase):
        def __init__(
            self,
            device: str | None = None,
            categorical_features_indices: list[int] | None = None,
            model_path: str = "default",
            n_estimators: int = 8,
            softmax_temperature: float = 0.9,
            average_before_softmax: bool = False,
            ignore_pretraining_limits: bool = False,
            inference_precision: Literal["autocast", "auto"] = "auto",
            random_state: int
            | np.random.RandomState
            | np.random.Generator
            | None = None,
            inference_config: dict | None = None,
            paper_version: bool = False,
        ) -> None:
            self.device = device
            self.categorical_features_indices = categorical_features_indices
            if categorical_features_indices is not None:
                warnings.warn(
                    "categorical_features_indices is not supported in the client version of TabPFN and will be ignored",
                    UserWarning,
                    stacklevel=2,
                )

            if "/" in model_path:
                model_name = model_path.split("/")[-1].split("-")[-1].split(".")[0]
                if model_name == "regressor":
                    model_name = "default"
                self.model_path = model_name
            else:
                self.model_path = model_path

            super().__init__(
                model_path=self.model_path,
                n_estimators=n_estimators,
                softmax_temperature=softmax_temperature,
                average_before_softmax=average_before_softmax,
                ignore_pretraining_limits=ignore_pretraining_limits,
                inference_precision=inference_precision,
                random_state=random_state,
                inference_config=inference_config,
                paper_version=paper_version,
            )

            # This is the client wrapper - distribution output parameters are not supported

        def predict(self, X, output_type=None, **kwargs):
            """Predict target values for X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                The input samples.

            output_type : str, default=None
                Type of output to return. Options are:
                - None: Default prediction (mean)
                - "full": Return distribution dictionary with criterion object
                - Other values are passed to the parent predict

            **kwargs : Additional keyword arguments
                Passed to the parent predict method.

            Returns:
            -------
            y : array-like of shape (n_samples,) or dict
                The predicted values or the full distribution output dictionary.
            """
            # For regular prediction, just call the parent method
            if output_type != "full":
                return super().predict(X)

            # Handle output_type="full" - we need to wrap the client output
            # in a compatible format that includes a criterion object
            try:
                # Import the distribution class from TabPFN package
                import torch

                from tabpfn.model.bar_distribution import FullSupportBarDistribution

                # Get prediction output from client (already contains all the data)
                client_output = super().predict(X, output_type="full")

                # Create a proper criterion object using the data from client
                # The client already returns logits and borders we can use
                criterion = FullSupportBarDistribution(
                    borders=torch.tensor(client_output["borders"]),
                )

                # Add the criterion to the client output
                result = dict(client_output)  # Make a copy
                result["criterion"] = criterion

                return result

            except ImportError:
                # TabPFN package not available
                raise ValueError(
                    "output_type='full' requires the TabPFN package with "
                    "FullSupportBarDistribution to be installed",
                )

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Return parameters for this estimator."""
            params = super().get_params(deep=deep)
            params.pop("device")
            params.pop("categorical_features_indices")
            return params

except ImportError:
    TabPFNClassifierWrapper = None
    TabPFNRegressorWrapper = None

try:
    from tabpfn import (
        TabPFNClassifier as LocalTabPFNClassifier,
        TabPFNRegressor as LocalTabPFNRegressor,
    )
except ImportError:
    LocalTabPFNClassifier = None
    LocalTabPFNRegressor = None


def get_tabpfn_models() -> tuple[type, type]:
    """Get TabPFN models with fallback between different versions.

    Attempts to import TabPFN models in the following order:
    1. Standard TabPFN package (if USE_TABPFN_LOCAL is True)
    2. TabPFN client

    Returns:
        tuple[type, type]: A tuple containing (TabPFNClassifier, TabPFNRegressor) classes

    Raises:
        ImportError: If none of the TabPFN implementations could be imported
    """
    # First try standard TabPFN package (if local usage is enabled)
    if USE_TABPFN_LOCAL and LocalTabPFNClassifier is not None:
        # Debug info controlled by environment variable
        # (using logging rather than print for better debugging)
        logging.info("Using TabPFN package")

        return LocalTabPFNClassifier, LocalTabPFNRegressor
    elif TabPFNClassifierWrapper is not None:
        return TabPFNClassifierWrapper, TabPFNRegressorWrapper
    else:
        raise ImportError(
            "No TabPFN implementation could be imported. Install with one of the following:\n"
            "pip install tabpfn    # For standard TabPFN package\n"
            "pip install tabpfn-client  # For TabPFN client (API-based inference)",
        )


TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()


def infer_categorical_features(
    X: np.ndarray,
    categorical_features: list[int] | None = None,
) -> list[int]:
    """Infer the categorical features from the input data.

    Features are identified as categorical if any of these conditions are met:
    1. The feature index is in the provided categorical_features list AND has few unique values
    2. The feature has few unique values compared to the dataset size
    3. The feature has string/object/category data type (pandas DataFrame)
    4. The feature contains string values (numpy array)

    Parameters:
        X (np.ndarray or pandas.DataFrame): The input data.
        categorical_features (list[int], optional): Initial list of categorical feature indices.
            If None, will start with an empty list.

    Returns:
        list[int]: The indices of the categorical features.
    """
    if categorical_features is None:
        categorical_features = []

    max_unique_values_as_categorical_feature = 10
    min_unique_values_as_numerical_feature = 10

    _categorical_features: list[int] = []

    # First detect based on data type (string/object features)
    is_pandas = hasattr(X, "dtypes")

    if is_pandas:
        # Handle pandas DataFrame - use pandas' own type detection
        import pandas as pd

        for i, col_name in enumerate(X.columns):
            col = X[col_name]
            # Use pandas' built-in type checks for categorical features
            if (
                pd.api.types.is_categorical_dtype(col)
                or pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)
            ):
                _categorical_features.append(i)
    else:
        # Handle numpy array - check if any columns contain strings
        for i in range(X.shape[1]):
            if X.dtype == object:  # Check entire array dtype
                # Try to access first non-nan value to check its type
                col = X[:, i]
                for val in col:
                    if val is not None and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        if isinstance(val, str):
                            _categorical_features.append(i)
                            break

    # Then detect based on unique values
    for i in range(X.shape[-1]):
        # Skip if already identified as categorical
        if i in _categorical_features:
            continue

        # Get unique values - handle differently for pandas and numpy
        n_unique = X.iloc[:, i].nunique() if is_pandas else len(np.unique(X[:, i]))

        # Filter categorical features, with too many unique values
        if (
            i in categorical_features
            and n_unique <= max_unique_values_as_categorical_feature
        ):
            _categorical_features.append(i)

        # Filter non-categorical features, with few unique values
        elif (
            i not in categorical_features
            and n_unique < min_unique_values_as_numerical_feature
            and X.shape[0] > 100
        ):
            _categorical_features.append(i)

    return _categorical_features


def softmax(logits: NDArray) -> NDArray:
    """Apply softmax function to convert logits to probabilities.

    Args:
        logits: Input logits array of shape (n_samples, n_classes) or (n_classes,)

    Returns:
        Probabilities where values sum to 1 across the last dimension
    """
    # Handle both 2D and 1D inputs
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    # Apply exponential to each logit with numerical stability
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)  # Subtract max for numerical stability

    # Sum across classes and normalize
    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    probs = exp_logits / sum_exp_logits

    # Return in the same shape as input
    if logits.ndim == 1:
        return probs.reshape(-1)
    return probs


def product_dict(d: dict[str, list[T]]) -> Iterator[dict[str, T]]:
    """Cartesian product of a dictionary of lists.

    This function takes a dictionary where each value is a list, and returns
    an iterator over dictionaries where each key is mapped to one element
    from the corresponding list.

    Parameters:
        d: A dictionary mapping keys to lists of values.

    Returns:
        An iterator over dictionaries, each being one element of the cartesian
        product of the input dictionary.

    Example:
        >>> list(product_dict({'a': [1, 2], 'b': ['x', 'y']}))
        [{'a': 1, 'b': 'x'}, {'a': 1, 'b': 'y'}, {'a': 2, 'b': 'x'}, {'a': 2, 'b': 'y'}]
    """
    keys = d.keys()
    values = [d[key] for key in keys]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


# Get the TabPFN models with our wrappers applied
TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()
