#  Licensed under the Apache License, Version 2.0
"""TabPFN explainer adapters that use the ShapIQ library for model interpretability.

This module provides functions to create shapiq explainers for TabPFN models that support
both basic Shapley values and interaction indices for more detailed model explanations.

Three entry points, each with a different paradigm for "missing" features:

* :func:`get_tabpfn_explainer` — remove-and-recontextualize
  (Rundel et al. 2024). Refits the model on each feature subset.

* :func:`get_tabpfn_imputation_explainer` — sample the missing features from
  a background distribution (marginal / conditional imputation).

* :func:`get_tabpfn_nan_explainer` — rely on TabPFN's native NaN-handling as
  the masking mechanism: a masked feature is literally set to ``NaN`` in
  the input and TabPFN absorbs it as "missing". This is the cheapest path
  on v3 because the training-set KV cache is reused across every coalition
  (see "Speeding up with the v3 KV cache" in the module README).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tabpfn_common_utils.telemetry import set_extension

if TYPE_CHECKING:
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


@set_extension("interpretability")
def get_tabpfn_nan_explainer(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
    data: pd.DataFrame | np.ndarray,
    index: str = "SV",
    max_order: int = 1,
    class_index: int | None = None,
    **kwargs,
):
    """Gets a TabularExplainer that masks features with NaN.

    When a coalition leaves a feature out, this explainer sets that feature to
    ``NaN`` in the input and lets TabPFN's native NaN handling absorb it as
    "missing". This matches the paradigm of the (now-removed) ``shap`` adapter
    that used ``shap.Explainer`` with an all-NaN background row, and is the
    fastest path on TabPFN v3: since the training set never changes across
    coalitions, ``fit_mode="fit_with_cache"`` + ``keep_cache_on_device=True``
    lets every coalition evaluation reuse one on-device KV cache.

    This is different from :func:`get_tabpfn_imputation_explainer` — that one
    samples the absent features from a background distribution. Here we do
    not sample: a masked feature is genuinely missing, and TabPFN is the one
    that decides how to handle that.

    Args:
        model: The TabPFN model to explain. For the v3 KV-cache fast path,
            construct it with ``fit_mode="fit_with_cache"`` before calling
            ``.fit(X, y)``, then set ``model.executor_.keep_cache_on_device
            = True`` so the per-estimator caches stay on the GPU across
            repeated coalition evaluations.

        data: Background data (used only for shapiq's required ``data``
            argument — it does **not** drive the imputation, since we
            replace with NaN rather than sampling).

        index: The Shapley-style index to compute. Defaults to ``"SV"``
            (plain Shapley values). See shapiq docs for alternatives.

        max_order: Maximum interaction order. Defaults to ``1`` (individual
            feature attributions, equivalent to classical SHAP values).

        class_index: Class to explain for classification models. Defaults
            to ``None`` (shapiq uses class 1). Ignored for regression.

        **kwargs: Passed through to ``shapiq.TabularExplainer``.

    Returns:
        shapiq.TabularExplainer: wired up with a NaN-passthrough imputer.

    Example:
        >>> from tabpfn import TabPFNClassifier
        >>> from tabpfn_extensions.interpretability import shapiq as tpe_shapiq
        >>> clf = TabPFNClassifier(fit_mode="fit_with_cache", device="cuda")
        >>> clf.fit(X_train, y_train)
        >>> clf.executor_.keep_cache_on_device = True  # optional but faster
        >>> explainer = tpe_shapiq.get_tabpfn_nan_explainer(
        ...     model=clf, data=X_train, class_index=1
        ... )
        >>> iv = explainer.explain(x=X_test[0], budget=2 ** X_train.shape[1])
    """
    try:
        import shapiq
        from shapiq.explainer.utils import get_predict_function_and_model_type
        from shapiq.imputer.marginal_imputer import MarginalImputer
    except ImportError:
        raise ImportError(
            "Package 'shapiq' is required for model explanation. "
            "Please install it with: pip install shapiq",
        )

    class _NaNImputer(MarginalImputer):
        """Replace absent features with NaN instead of sampling from background.

        Overriding ``value_function`` short-circuits the marginal sampling
        loop — no ``sample_size`` draws per coalition, just one batched
        ``predict`` call per coalition batch.
        """

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            x_masked = np.tile(self.x, (coalitions.shape[0], 1)).astype(float)
            x_masked[~coalitions] = np.nan
            return self.predict(x_masked)

    if isinstance(data, pd.DataFrame):
        data = data.values

    # Build the class-aware predict callable the same way shapiq's base
    # Explainer would — the Imputer constructor accepts any callable model,
    # so this bypasses the "_shapiq_predict_function not yet attached"
    # problem when we instantiate the imputer before the TabularExplainer.
    predict_fn, _ = get_predict_function_and_model_type(model, class_index=class_index)

    def _callable_model(X: np.ndarray) -> np.ndarray:
        return predict_fn(model, X)

    nan_imputer = _NaNImputer(
        model=_callable_model,
        data=data,
        sample_size=1,  # unused — value_function is overridden
    )

    # Shapiq warns "not recommended" when TabularExplainer wraps a TabPFN
    # model, because it assumes the default (sampling-based) imputers. For
    # our NaN imputer the warning is misleading — the NaN masking IS
    # TabPFN's native handling of missing features.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using a TabPFN model with the ``shapiq.TabularExplainer``.*",
        )
        return shapiq.TabularExplainer(
            model=model,
            data=data,
            imputer=nan_imputer,
            index=index,
            max_order=max_order,
            class_index=class_index,
            **kwargs,
        )
