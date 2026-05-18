#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Bridge helpers for using the SHAP library's plotting ecosystem with
Shapley values computed by shapiq.

We use shapiq for the actual Shapley-value computation — it's faster and
extension-friendly for TabPFN — but the SHAP library's plotting ecosystem
(``shap.plots.waterfall``, ``beeswarm``, ``summary``, ``dependence``, etc.)
is mature and widely used. This module bridges the two.

The ``shap`` package is **not** part of the ``interpretability`` extra. Install
it separately (``pip install shap``) if you want to use these helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import shap
    from numpy.typing import ArrayLike


def shapiq_to_shap_explanation(
    explainer: Any,
    X: ArrayLike,
    *,
    budget: int,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Compute first-order Shapley values with a shapiq explainer for each
    row in ``X`` and wrap them in a ``shap.Explanation`` ready for use with
    ``shap.plots.*`` and ``shap.summary_plot``.

    Mirrors the pattern in ``examples/interpretability/shap_example.py``:
    one ``.explain(...)`` call per row, stack the first-order arrays into an
    ``(n, d)`` matrix, average baseline values, and pass everything to
    ``shap.Explanation``.

    Args:
        explainer: A shapiq explainer — e.g. one returned by
            ``get_tabpfn_imputation_explainer(..., index="SV", max_order=1)``.
        X: ``(n, d)`` array of rows to explain.
        budget: Number of model evaluations shapiq is allowed per row. For
            small ``d`` and exact Shapley values, pass ``2**d``.
        feature_names: Optional list of feature name strings (length ``d``).
            Used by ``shap.plots.*`` for axis labels.

    Returns:
        A ``shap.Explanation`` with ``values.shape == (n, d)``.

    Notes:
        Only first-order Shapley values are wrapped. ``shap.Explanation``
        doesn't represent higher-order interactions; for those, use
        shapiq's native plots on the ``InteractionValues`` object.

        Requires ``shap`` to be installed (``pip install shap``). It is
        kept out of the ``interpretability`` extra by design — shapiq is
        the runtime dependency, shap is opt-in for plotting.
    """
    import shap

    X_arr = np.asarray(X)
    n = len(X_arr)
    ivs = [explainer.explain(x=X_arr[i], budget=budget) for i in range(n)]
    values = np.stack([iv.get_n_order_values(1) for iv in ivs])
    # Pass per-row baselines through unchanged. For the imputation path the
    # background is fixed and these are all the same value; for the Rundel
    # remove-and-recontextualize path baselines genuinely vary per row.
    # shap.Explanation accepts a 1-d (n,) array natively.
    base_values = np.array([iv.baseline_value for iv in ivs])
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_arr,
        feature_names=feature_names,
    )
