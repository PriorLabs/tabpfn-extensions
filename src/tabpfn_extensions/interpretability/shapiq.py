#  Licensed under the Apache License, Version 2.0
"""TabPFN explainer adapters that use the ShapIQ library for model interpretability.

This module provides functions to create shapiq explainers for TabPFN models that support
both basic Shapley values and interaction indices for more detailed model explanations.

Three explanation paradigms are exposed:

* :func:`get_tabpfn_explainer` — *remove-and-recontextualize* (Rundel et al. 2024).
  Re-fits TabPFN on each feature subset; cannot benefit from the KV cache
  because the training set changes per coalition.

* :func:`get_tabpfn_imputation_explainer` — *imputation-based* removal (marginal /
  conditional / baseline). The training set is fixed across coalitions, so the
  KV cache (``fit_mode="fit_with_cache"``) drastically reduces wall time. A
  runtime warning is emitted if the cache isn't enabled when this explainer
  is constructed.

* :func:`get_tabpfn_inf_explainer` — *missingness-based* removal. A masked
  feature is set to ``+inf`` and TabPFN's native missing-value handling
  absorbs it as "missing" (no sampling, one forward pass per coalition).
  The training set is fixed across coalitions, so it benefits from the KV
  cache just like the imputation explainer. Requires the model to be built
  with ``inference_config={"PASSTHROUGH_INF": True}`` (tabpfn>=8.1.0) so
  ``+inf`` reaches the model instead of being rejected at validation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.utils import warn_if_no_kv_cache

if TYPE_CHECKING:
    import tabpfn


def _require_shapiq():
    """Import and return the ``shapiq`` package, or raise a helpful error."""
    try:
        import shapiq
    except ImportError:
        raise ImportError(
            "Package 'shapiq' is required for model explanation. Install it with: "
            "pip install 'tabpfn-extensions[interpretability]'",
        ) from None
    return shapiq


def _build_tabular_explainer(shapiq, **kwargs):
    """Construct a ``shapiq.TabularExplainer`` with ``kwargs``, silencing one warning.

    shapiq emits a ``UserWarning`` when ``TabularExplainer`` is built with a
    TabPFN model, recommending ``TabPFNExplainer`` (Rundel) instead. That advice
    assumes the sampling-based imputers; the wrappers here deliberately use an
    imputation/masking removal path — which, unlike Rundel, benefits from the KV
    cache (one predict per coalition, no re-fit) — so the warning is misleading
    and only that specific message is silenced.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*TabPFN model with the.*shapiq\.TabularExplainer.*",
            category=UserWarning,
        )
        return shapiq.TabularExplainer(**kwargs)


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

        **kwargs: Additional keyword arguments to pass to the explainer.

    Returns:
        shapiq.TabPFNExplainer: The TabPFN explainer.

    References:
        .. [1] shapiq repository: https://github.com/mmschlk/shapiq
        .. [2] Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., Hüllermeier, E. (2024). shapiq: Shapley Interactions for Machine Learning. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. url: https://openreview.net/forum?id=knxGmi6SJi
        .. [3] Rundel, D., Kobialka, J., von Crailsheim, C., Feurer, M., Nagler, T., Rügamer, D. (2024). Interpretable Machine Learning for TabPFN. In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2154. Springer, Cham. https://doi.org/10.1007/978-3-031-63797-1_23

    """
    shapiq = _require_shapiq()

    # make data to array if it is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # make labels to array if it is a pandas Series
    if isinstance(labels, pd.Series | pd.DataFrame):
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
    imputer: str = "baseline",
    class_index: int | None = None,
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

        **kwargs: Additional keyword arguments to pass to the explainer.

    Returns:
        shapiq.TabularExplainer: The TabularExplainer.

    References:
        .. [1] shapiq repository: https://github.com/mmschlk/shapiq
        .. [2] Muschalik, M., Baniecki, H., Fumagalli, F., Kolpaczki, P., Hammer, B., Hüllermeier, E. (2024). shapiq: Shapley Interactions for Machine Learning. In: The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. url: https://openreview.net/forum?id=knxGmi6SJi
        .. [3] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30 (pp. 4765--4774).

    """
    shapiq = _require_shapiq()

    warn_if_no_kv_cache(model, context="Imputation-based SHAP")

    # make data to array if it is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    return _build_tabular_explainer(
        shapiq,
        model=model,
        data=data,
        index=index,
        max_order=max_order,
        imputer=imputer,
        class_index=class_index,
        **kwargs,
    )


def _model_has_inf_passthrough(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
) -> bool:
    """Whether ``model`` is configured to pass ``+/-inf`` through to TabPFN.

    Returns ``True`` only if we can positively confirm the ``PASSTHROUGH_INF``
    inference-config flag is enabled. For estimators we can't introspect (e.g.
    the tabpfn-client backend, which has no ``get_inference_config``) this
    returns ``False`` — we simply can't vouch for inf support.
    """
    get_inference_config = getattr(model, "get_inference_config", None)
    if get_inference_config is None:
        return False
    return bool(getattr(get_inference_config(), "PASSTHROUGH_INF", False))


@set_extension("interpretability")
def get_tabpfn_inf_explainer(
    model: tabpfn.TabPFNRegressor | tabpfn.TabPFNClassifier,
    data: pd.DataFrame | np.ndarray,
    index: str = "SV",
    max_order: int = 1,
    class_index: int | None = None,
    **kwargs,
):
    """Gets a TabularExplainer that masks missing features with ``+inf``.

    When a coalition leaves a feature out, this explainer sets that feature to
    ``+inf`` and lets TabPFN's native missing-value handling absorb it as
    "missing" — no sampling from a background distribution, just one forward
    pass per coalition. Since the training set never changes across
    coalitions, this is the fastest path on TabPFN v3: construct the model
    with ``fit_mode="fit_with_cache"`` and set
    ``model.executor_.keep_cache_on_device = True`` after ``.fit()`` so every
    coalition evaluation reuses one on-device KV cache.

    This differs from :func:`get_tabpfn_imputation_explainer` — that one
    *samples* the absent features from a background distribution, so their
    values are drawn from the data. Here nothing is sampled: a masked feature
    is genuinely missing and TabPFN decides how to handle it. ``+inf`` (rather
    than ``NaN``) is used deliberately: ``NaN`` is transformed by TabPFN's
    preprocessing pipeline before it reaches the model, whereas ``+inf`` is
    carried through and handled natively as missingness.

    IMPORTANT: this requires the model to be constructed with
    ``inference_config={"PASSTHROUGH_INF": True}`` (available in
    ``tabpfn>=8.1.0``). Without it, TabPFN rejects non-finite inputs at
    validation and this function raises ``ValueError`` up front rather than
    letting every coalition evaluation fail later.

    Args:
        model: The TabPFN model to explain. Must be constructed with
            ``inference_config={"PASSTHROUGH_INF": True}``. For the v3
            KV-cache fast path, also pass ``fit_mode="fit_with_cache"`` before
            ``.fit(X, y)`` and set ``model.executor_.keep_cache_on_device =
            True`` afterwards.

        data: Background data. Only its shape (number of features) and a single
            row for shapiq's compatibility check are used — it does **not**
            drive the masking, since absent features are replaced with ``+inf``
            rather than sampled.

        index: The Shapley-style index to compute. Defaults to ``"SV"`` (plain
            Shapley values). See shapiq docs for alternatives.

        max_order: Maximum interaction order. Defaults to ``1`` (individual
            feature attributions, equivalent to classical SHAP values).

        class_index: Class to explain for classification models. Defaults to
            ``None`` (shapiq uses class 1). Ignored for regression.

        **kwargs: Passed through to ``shapiq.TabularExplainer``.

    Returns:
        shapiq.TabularExplainer: wired up with an ``+inf``-passthrough imputer.

    Raises:
        ValueError: If ``model`` can be introspected and does not have
            ``PASSTHROUGH_INF`` enabled.

    Example:
        >>> from tabpfn import TabPFNClassifier
        >>> from tabpfn_extensions.interpretability import shapiq as tpe_shapiq
        >>> clf = TabPFNClassifier(
        ...     inference_config={"PASSTHROUGH_INF": True},
        ...     fit_mode="fit_with_cache",
        ... )
        >>> clf.fit(X_train, y_train)
        >>> clf.executor_.keep_cache_on_device = True  # optional but faster
        >>> explainer = tpe_shapiq.get_tabpfn_inf_explainer(
        ...     model=clf, data=X_train, class_index=1
        ... )
        >>> iv = explainer.explain(x=X_test[0], budget=2 ** X_train.shape[1])
    """
    # Deferred import (kept function-local). Once the top package is confirmed
    # present, its submodules are part of the same install and import safely.
    shapiq = _require_shapiq()
    from shapiq.explainer.utils import get_predict_function_and_model_type
    from shapiq.imputer.marginal_imputer import MarginalImputer

    # Fail fast: without inf passthrough, TabPFN rejects +inf at validation.
    if not _model_has_inf_passthrough(model):
        raise ValueError(
            "get_tabpfn_inf_explainer masks missing features with +inf, which "
            "requires the TabPFN model to be constructed with "
            'inference_config={"PASSTHROUGH_INF": True} (available in '
            "tabpfn>=8.1.0). Without it, TabPFN rejects non-finite inputs at "
            "validation. Re-create the model, e.g. "
            'TabPFNClassifier(inference_config={"PASSTHROUGH_INF": True}, '
            'fit_mode="fit_with_cache"), and refit it before explaining.',
        )

    warn_if_no_kv_cache(model, context="Inf-masking SHAP")

    class _InfImputer(MarginalImputer):
        """Replace absent features with ``+inf`` instead of sampling.

        ``value_function`` masks with ``+inf`` (one predict per coalition, no
        sampling). ``calc_empty_prediction`` computes ``v(empty)`` as a single
        ``f(inf, ..., inf)`` row instead of the base class's predict over the
        whole background, which OOMs on large training sets.
        """

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            x_masked = np.tile(self.x, (coalitions.shape[0], 1))
            # int/bool arrays can't hold +inf; promote them. float and object already can.
            if np.issubdtype(x_masked.dtype, np.integer) or x_masked.dtype == bool:
                x_masked = x_masked.astype(float)
            x_masked[~coalitions] = np.inf
            return self.predict(x_masked)

        def calc_empty_prediction(self) -> float:
            empty_row = np.full((1, self.n_features), np.inf)
            empty_prediction = float(np.mean(self.predict(empty_row)))
            self.empty_prediction = empty_prediction
            if self.normalize:
                self.normalization_value = empty_prediction
            return empty_prediction

    if isinstance(data, pd.DataFrame):
        data = data.values

    # Wrap the model as a plain callable so the imputer can be built before the
    # TabularExplainer attaches its own predict function.
    predict_fn, _ = get_predict_function_and_model_type(model, class_index=class_index)

    def _callable_model(X: np.ndarray) -> np.ndarray:
        return predict_fn(model, X)

    inf_imputer = _InfImputer(
        model=_callable_model,
        data=data,
        sample_size=1,  # unused — value_function is overridden
    )

    return _build_tabular_explainer(
        shapiq,
        model=model,
        data=data,
        imputer=inf_imputer,
        index=index,
        max_order=max_order,
        class_index=class_index,
        **kwargs,
    )
