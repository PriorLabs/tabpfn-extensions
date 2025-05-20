#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""SHAP value computation and visualization for TabPFN.

This module provides functions to calculate and visualize SHAP (SHapley Additive exPlanations)
values for TabPFN models. SHAP values help understand model predictions by attributing
the contribution of each input feature to the output prediction.

Key features:
- Efficient parallel computation of SHAP values
- Support for both TabPFN and TabPFN-client backends
- Specialized explainers for TabPFN models
- Visualization functions for feature importance and interactions
- Backend-specific optimizations for faster SHAP computation

Example usage:
    ```python
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.interpretability import get_shap_values, plot_shap

    # Train a TabPFN model
    model = TabPFNClassifier()
    model.fit(X_train, y_train)

    # Calculate SHAP values
    shap_values = get_shap_values(model, X_test)

    # Visualize feature importance
    plot_shap(shap_values)
    ```
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tabpfn_extensions.utils import is_tabpfn


def calculate_shap_subset(args: tuple) -> np.ndarray:
    """Calculate SHAP values for a specific feature in a parallel context.

    This helper function is used by parallel_permutation_shap to enable
    efficient parallel computation of SHAP values for each feature.

    Args:
        args: A tuple containing:
            - X_subset: Feature matrix for which to calculate SHAP values
            - background: Background data for the explainer
            - model: The model for which to calculate SHAP values
            - feature_idx: The index of the feature to calculate SHAP values for

    Returns:
        np.ndarray: SHAP values for the specified feature
    """
    import shap

    X_subset, background, model, feature_idx = args
    explainer = shap.PermutationExplainer(model, background)
    return explainer.shap_values(X_subset)[:, feature_idx]


def parallel_permutation_shap(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    background: np.ndarray | pd.DataFrame | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """Calculate SHAP values efficiently using parallel processing.

    This function distributes the SHAP value calculation across multiple
    processes, with each process computing values for a different feature.
    This is much faster than calculating all SHAP values at once, especially
    for large datasets or complex models.

    Args:
        model: The model for which to calculate SHAP values. Must have a prediction method.
        X: Feature matrix for which to calculate SHAP values.
        background: Background data for the explainer. If None, X is used as background data.
        n_jobs: Number of processes to use for parallel computation.
            If -1, all available CPU cores are used.

    Returns:
        np.ndarray: Matrix of SHAP values with shape (n_samples, n_features).
    """
    # If no background data provided, use X
    if background is None:
        background = X

    # Split data into chunks for parallel processing
    n_features = X.shape[1]
    feature_indices = list(range(n_features))

    # Prepare arguments for parallel processing
    args_list = [(X, background, model, idx) for idx in feature_indices]

    # Run parallel computation
    with Pool(processes=n_jobs) as pool:
        shap_values_per_feature = pool.map(calculate_shap_subset, args_list)

    # Combine results
    return np.column_stack(shap_values_per_feature)


def plot_shap(shap_values: np.ndarray) -> None:
    """Plot SHAP values for the given test data.

    This function creates several visualizations of SHAP values:
    1. Aggregated feature importances across all examples
    2. Per-sample feature importances
    3. Important feature interactions (if multiple samples provided)

    Args:
        shap_values: The SHAP values to plot, typically from get_shap_values().

    Returns:
        None: This function only produces visualizations.
    """
    import shap

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    shap.plots.bar(shap_values=shap_values, show=False)
    plt.title("Aggregate feature importances across the test examples")
    plt.show()
    shap.summary_plot(shap_values=shap_values, show=False)
    # plot the distribution of importances for each feature over all samples
    plt.title(
        "Feature importances for each feature for each test example (a dot is one feature for one example)",
    )
    plt.show()

    most_important = shap_values.abs.mean(0).values.argsort()[-1]
    if len(shap_values) > 1:
        plot_shap_feature(shap_values, most_important)


def plot_shap_feature(
    shap_values_: Any,
    feature_name: int | str,
    n_plots: int = 1,
) -> None:
    """Plot feature interactions for a specific feature based on SHAP values.

    Args:
        shap_values_: SHAP values object containing the data to plot.
        feature_name: The feature index or name to plot interactions for.
        n_plots: Number of interaction plots to create. Defaults to 1.

    Returns:
        None: This function only produces visualizations.
    """
    import shap

    # we can use shap.approximate_interactions to guess which features
    # may interact with age
    inds = shap.utils.potential_interactions(
        shap_values_[:, feature_name],
        shap_values_,
    )

    # make plots colored by each of the top three possible interacting features
    for i in range(n_plots):
        shap.plots.scatter(
            shap_values_[:, feature_name],
            color=shap_values_[:, inds[i]],
            show=False,
        )
        plt.title(
            f"Feature {feature_name} with a color coding representing the value of ({inds[i]})",
        )


def get_shap_values(
    estimator: Any,
    test_x: pd.DataFrame | np.ndarray | torch.Tensor,
    attribute_names: list[str] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute SHAP values for a model's predictions on input features.

    This function calculates SHAP (SHapley Additive exPlanations) values that
    attribute the contribution of each input feature to the model's output.
    It automatically selects the appropriate SHAP explainer based on the model.

    Args:
        estimator: The model to explain, typically a TabPFNClassifier or scikit-learn compatible model.
        test_x: The input features to compute SHAP values for.
        attribute_names: Column names for the features when test_x is a numpy array.
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        np.ndarray: The computed SHAP values with shape (n_samples, n_features).
    """
    if isinstance(test_x, torch.Tensor):
        test_x = test_x.cpu().numpy()

    if isinstance(test_x, np.ndarray):
        test_x = pd.DataFrame(test_x)
        if attribute_names is not None:
            test_x.columns = attribute_names
        else:
            test_x = test_x.rename(columns={c: str(c) for c in test_x.columns})

    if hasattr(estimator, "predict_function_for_shap"):
        predict_function_for_shap = estimator.predict_function_for_shap
    else:
        predict_function_for_shap = (
            "predict_proba" if hasattr(estimator, "predict_proba") else "predict"
        )

    if hasattr(estimator, "fit_at_predict_time"):
        if not estimator.fit_at_predict_time:
            pass

    def get_shap() -> np.ndarray:
        if is_tabpfn(estimator):
            explainer = get_tabpfn_explainer(
                estimator,
                test_x,
                predict_function_for_shap,
                **kwargs,
            )
        else:
            explainer = get_default_explainer(
                estimator,
                test_x,
                predict_function_for_shap,
                **kwargs,
            )
        return explainer(test_x)

    if hasattr(estimator, "show_progress"):
        show_progress_ = estimator.show_progress
        estimator.show_progress = False
        try:
            shap_values = get_shap()
        finally:
            estimator.show_progress = show_progress_
    else:
        shap_values = get_shap()

    return shap_values


def get_tabpfn_explainer(
    estimator: Any,
    test_x: pd.DataFrame,
    predict_function_for_shap: str | Callable = "predict",
    **kwargs: Any,
) -> Any:
    """Create a SHAP explainer specifically optimized for TabPFN models.

    Args:
        estimator: The TabPFN model to explain.
        test_x: The input features to compute SHAP values for.
        predict_function_for_shap: Function name or callable to use for prediction.
            Defaults to "predict".
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        Any: A configured SHAP explainer for the TabPFN model.
    """
    import shap

    return shap.Explainer(
        getattr(estimator, predict_function_for_shap)
        if isinstance(predict_function_for_shap, str)
        else predict_function_for_shap,
        np.ones(test_x.iloc[0:1, :].shape) * float("nan"),
        **kwargs,
    )


def get_default_explainer(
    estimator: Any,
    test_x: pd.DataFrame,
    predict_function_for_shap: str | Callable = "predict",
    **kwargs: Any,
) -> Any:
    """Create a standard SHAP explainer for non-TabPFN models.

    Args:
        estimator: The model to explain.
        test_x: The input features to compute SHAP values for.
        predict_function_for_shap: Function name or callable to use for prediction.
            Defaults to "predict".
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        Any: A configured SHAP explainer for the model.
    """
    import shap

    shap.maskers.Independent(test_x, max_samples=1000)

    return shap.Explainer(
        getattr(estimator, predict_function_for_shap)
        if isinstance(predict_function_for_shap, str)
        else predict_function_for_shap,
        test_x,
        **kwargs,
    )
