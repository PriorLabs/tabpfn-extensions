#  Licensed under the Apache License, Version 2.0
"""TabPFN explainer adapters that use the ShapIQ library for model interpretability.

This module provides functions to create shapiq explainers for TabPFN models that support
both basic Shapley values and interaction indices for more detailed model explanations.

Two explanation paradigms are exposed:

* :func:`get_tabpfn_explainer` — *remove-and-recontextualize* (Rundel et al. 2024).
  Re-fits TabPFN on each feature subset; cannot benefit from the KV cache
  because the training set changes per coalition.

* :func:`get_tabpfn_imputation_explainer` — *imputation-based* removal (marginal /
  conditional / baseline). The training set is fixed across coalitions, so the
  KV cache (``fit_mode="fit_with_cache"``) drastically reduces wall time. A
  runtime warning is emitted if the cache isn't enabled when this explainer
  is constructed.

Both explainers route to the approximator recommended for TabPFN in the
discussion at tabpfn-extensions#342 unless the caller selects one explicitly
(see :func:`_resolve_approximator`):
first-order Shapley values (``index="SV"``) use OddSHAP, while higher-order
interactions use ProxySHAP. Both approximators fit a tree surrogate on sampled
coalitions — OddSHAP on LightGBM, ProxySHAP on XGBoost — so those backends ship
with the ``interpretability`` extra.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.utils import warn_if_no_kv_cache

if TYPE_CHECKING:
    import numpy as np

    import tabpfn


def _resolve_approximator(shapiq, approximator, index: str, n_features: int):
    """Pick the approximator for an explainer when the caller didn't.

    Routing follows the recommendation in the discussion at
    https://github.com/PriorLabs/tabpfn-extensions/issues/342#issuecomment-4977956527:
    first-order Shapley values (``index="SV"``) use :class:`shapiq.OddSHAP`, the
    current best regression-based Shapley-value estimator; every other index
    (e.g. ``"k-SII"`` interactions) uses :class:`shapiq.ProxySHAP`, which
    supports all cardinal-probabilistic indices and is the state-of-the-art
    interaction estimator. ``OddSHAP`` computes only ``"SV"``, so it is never
    selected for interaction indices.

    shapiq's own ``"auto"`` selection does not yet pick these estimators; once
    it does (https://github.com/mmschlk/shapiq/issues/565), this routing can be
    dropped in favor of ``approximator="auto"``.

    A non-``None`` ``approximator`` (a shapiq ``Approximator`` instance or a
    literal string such as ``"auto"``) is returned unchanged, so callers keep
    full control.
    """
    if approximator is not None:
        return approximator
    if index == "SV":
        return shapiq.OddSHAP(n=n_features)
    return "proxyshap"


@set_extension("interpretability")
def get_tabpfn_explainer(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
    data: pd.DataFrame | np.ndarray,
    labels: pd.DataFrame | np.ndarray,
    index: str = "k-SII",
    max_order: int = 2,
    class_index: int | None = None,
    approximator=None,
    **kwargs,
):
    """Get a TabPFNExplainer (remove-and-recontextualize) from shapiq.

    The explainer uses the remove-and-recontextualize paradigm of model
    explanation [2]_ [3]_: for each coalition `S`, TabPFN is re-fit on the
    columns in `S` and predictions are made with that re-fitted model. This
    is expensive because every coalition triggers a fresh fit.

    NOTE: This path **does not benefit from the KV cache** even when the
    underlying model is configured with ``fit_mode="fit_with_cache"``. Each
    coalition does exactly one fit + one predict, so there are no repeated
    predicts to amortize the cache over. If you want the cache to actually
    speed things up, prefer :func:`get_tabpfn_imputation_explainer` (which
    runs ``budget`` predicts against a single fit).

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

        approximator: The shapiq approximator to estimate the coalition game with. Defaults to
            ``None``, which routes to the shapiq authors' recommendation: OddSHAP for first-order
            Shapley values (``index="SV"``) and ProxySHAP for interaction indices (see
            :func:`_resolve_approximator`). Pass a shapiq ``Approximator`` instance or a literal
            string (e.g. ``"auto"``) to override the routing.

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
    if isinstance(labels, pd.Series | pd.DataFrame):
        labels = labels.values

    approximator = _resolve_approximator(shapiq, approximator, index, data.shape[1])

    # TabPFNExplainer is directly available in the shapiq module
    return shapiq.TabPFNExplainer(
        model=model,
        data=data,
        labels=labels,
        index=index,
        max_order=max_order,
        class_index=class_index,
        approximator=approximator,
        **kwargs,
    )


@set_extension("interpretability")
def get_tabpfn_imputation_explainer(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
    data: pd.DataFrame | np.ndarray,
    index: str = "k-SII",
    max_order: int = 2,
    imputer: str = "baseline",
    class_index: int | None = None,
    approximator=None,
    **kwargs,
):
    """Gets a TabularExplainer from shapiq with imputation-based feature removal.

    The explainer uses an imputation-based paradigm of feature removal [3]_:
    for each coalition, masked features are filled by an imputer and TabPFN
    is queried for a prediction. The training set is fixed across coalitions,
    so the KV cache makes this dramatically faster than the
    remove-and-recontextualize path (cf. :func:`get_tabpfn_explainer`). A
    warning is emitted if the model is not configured for the cache.

    The default imputer is ``"baseline"`` (one fixed fill value per feature,
    so each coalition costs exactly one forward pass). Marginal/conditional
    imputers draw multiple samples per coalition and are 50-100x slower in
    practice without commensurate gains for in-context models — switch to
    them only if you have a specific reason.

    Args:
        model (tabpfn.TabPFNRegressor or tabpfn.TabPFNClassifier): The TabPFN model to explain.
            Should be constructed with ``fit_mode="fit_with_cache"`` to engage
            the KV-cache fast path.

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

        approximator: The shapiq approximator to estimate the coalition game with. Defaults to
            ``None``, which routes to the shapiq authors' recommendation: OddSHAP for first-order
            Shapley values (``index="SV"``) and ProxySHAP for interaction indices (see
            :func:`_resolve_approximator`). Pass a shapiq ``Approximator`` instance or a literal
            string (e.g. ``"auto"``) to override the routing.

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

    warn_if_no_kv_cache(model, context="Imputation-based SHAP")

    # make data to array if it is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    approximator = _resolve_approximator(shapiq, approximator, index, data.shape[1])

    # shapiq emits a UserWarning when ``TabularExplainer`` is constructed with a
    # TabPFN model, recommending ``TabPFNExplainer`` (Rundel) instead. In this
    # wrapper the user has explicitly chosen the imputation path — precisely
    # because Rundel cannot benefit from the KV cache (one predict per
    # coalition fit) while imputation-based removal can. The warning is
    # misleading here, so silence just that specific message.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*TabPFN model with the.*shapiq\.TabularExplainer.*",
            category=UserWarning,
        )
        return shapiq.TabularExplainer(
            model=model,
            data=data,
            index=index,
            max_order=max_order,
            imputer=imputer,
            class_index=class_index,
            approximator=approximator,
            **kwargs,
        )
