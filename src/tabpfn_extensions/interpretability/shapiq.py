#  Licensed under the Apache License, Version 2.0
"""TabPFN explainer adapters that use the ShapIQ library for model interpretability.

This module provides functions to create shapiq explainers for TabPFN models that support
both basic Shapley values and interaction indices for more detailed model explanations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from tabpfn_common_utils.telemetry import set_extension

if TYPE_CHECKING:
    import numpy as np

    import tabpfn


@set_extension("interpretability")
def get_tabpfn_explainer(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
    data: pd.DataFrame | np.ndarray,
    labels: pd.DataFrame | np.ndarray,
    index: str = "k-SII",
    max_order: int = 2,
    class_index: int | None = None,
    **kwargs,
):
    """Get a TabPFNExplainer from shapiq.

    This function returns the TabPFN explainer from the shapiq[1]_ library. The explainer uses
    a remove-and-recontextualize paradigm of model explanation[2]_[3]_ to explain the predictions
    of a TabPFN model. See ``shapiq.TabPFNExplainer`` documentation for more information regarding
    the explainer object.

    Args:
        model (tabpfn.TabPFNRegressor or tabpfn.TabPFNClassifier): The TabPFN model to explain.

        data (pd.DataFrame or np.ndarray): The background data to use for the explainer.

        labels (pd.DataFrame or np.ndarray): The labels for the background data.

        index: The index to use for the explanation. See shapiq documentation for more information
            and an up-to-date list of available indices. Defaults to "k-SII" and "SV" (Shapley
            Values like SHAP) with ``max_order=1``.

        max_order (int): The maximum order of interactions to consider. Defaults to 2.

        class_index (int, optional): The class index of the model to explain. If not provided, the
            class index will be set to 1 per default for classification models. This argument is
            ignored for regression models. Defaults to None.

        **kwargs: Additional keyword arguments to pass to the explainer.

    Returns:
        shapiq.TabPFNExplainer: The TabPFN explainer.

    References:
        .. [1] shapiq repository: https://github.com/mmschlk/shapiq
        .. [2] Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., Hüllermeier, E. (2024). shapiq: Shapley Interactions for Machine Learning. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. url: https://openreview.net/forum?id=knxGmi6SJi
        .. [3] Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., Rügamer, D. (2024). Interpretable Machine Learning for TabPFN. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_23

    """
    # Defer the import to avoid circular imports
    try:
        import shapiq  # Import the main package
        # Current version of shapiq has TabPFNExplainer in the base module
    except ImportError:
        raise ImportError(
            "Package 'shapiq' is required for model explanation. "
            "Please install it with: pip install shapiq",
        )

    # make data to array if it is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # make labels to array if it is a pandas Series
    if isinstance(labels, (pd.Series, pd.DataFrame)):
        labels = labels.values

    # TabPFNExplainer is directly available in the shapiq module
    return shapiq.TabPFNExplainer(
        model=model,
        data=data,
        labels=labels,
        index=index,
        max_order=max_order,
        class_index=class_index,
        **kwargs,
    )


@set_extension("interpretability")
def get_tabpfn_imputation_explainer(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
    data: pd.DataFrame | np.ndarray,
    index: str = "k-SII",
    max_order: int = 2,
    imputer: str = "marginal",
    class_index: int | None = None,
    **kwargs,
):
    """Gets a TabularExplainer from shapiq with using imputation.

    This function returns the TabularExplainer from the shapiq[1]_[2]_ library. The explainer uses an
    imputation-based paradigm of feature removal for the explanations similar to SHAP[3]_. See
    ``shapiq.TabularExplainer`` documentation for more information regarding the explainer object.

    Args:
        model (tabpfn.TabPFNRegressor or tabpfn.TabPFNClassifier): The TabPFN model to explain.

        data (pd.DataFrame or np.ndarray): The background data to use for the explainer.

        index: The index to use for the explanation. See shapiq documentation for more information
            and an up-to-date list of available indices. Defaults to "k-SII" and "SV" (Shapley
            Values like SHAP) with ``max_order=1``.

        max_order (int): The maximum order of interactions to consider. Defaults to 2.

        imputer: The imputation method to use. See ``shapiq.TabularExplainer`` documentation for
            more information and an up-to-date list of available imputation methods.

        class_index (int, optional): The class index of the model to explain. If not provided, the
            class index will be set to 1 per default for classification models. This argument is
            ignored for regression models. Defaults to None.

        **kwargs: Additional keyword arguments to pass to the explainer.

    Returns:
        shapiq.TabularExplainer: The TabularExplainer.

    References:
        .. [1] shapiq repository: https://github.com/mmschlk/shapiq
        .. [2] Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., Hüllermeier, E. (2024). shapiq: Shapley Interactions for Machine Learning. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. url: https://openreview.net/forum?id=knxGmi6SJi
        .. [3] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30 (pp. 4765--4774).

    """
    # Defer the import to avoid circular imports
    try:
        import shapiq  # Import the main package
        # Current version of shapiq has TabularExplainer in the base module
    except ImportError:
        raise ImportError(
            "Package 'shapiq' is required for model explanation. "
            "Please install it with: pip install shapiq",
        )

    # make data to array if it is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # TabularExplainer is directly available in the shapiq module
    return shapiq.TabularExplainer(
        model=model,
        data=data,
        index=index,
        max_order=max_order,
        imputer=imputer,
        class_index=class_index,
        **kwargs,
    )
