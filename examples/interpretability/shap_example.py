"""Compute SHAP values for a TabPFN model with ShapIQ, then visualize them
using the SHAP library's plotting functions.

We use shapiq for the actual Shapley-value computation (it's faster and
extension-friendly for TabPFN) but the SHAP library's plotting ecosystem is
mature and widely used. This example shows how to bridge the two: wrap shapiq
output in a `shap.Explanation` and call `shap.plots.*` / `shap.summary_plot`.

The `shap` package is not part of the `interpretability` extra (we depend on
shapiq for compute). Install it separately to run this example:

    pip install shap

Dataset: California housing (regression, d=8).

The TabPFN model is constructed with `fit_mode="fit_with_cache"` to engage the
KV cache, which speeds up the computation of Shapley values by one to two
orders of magnitude; the `get_tabpfn_imputation_explainer` wrapper emits a
warning if the cache isn't enabled.
"""

from __future__ import annotations

import numpy as np
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNRegressor
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

housing = fetch_california_housing(as_frame=False)
X, y, feature_names = housing.data, housing.target, list(housing.feature_names)

X_train, X_test, y_train, _ = train_test_split(
    X, y, train_size=1000, test_size=200, random_state=0,
)
n_explain = 30
X_explain = X_test[:n_explain]

# Engage the KV cache: fit_mode='fit_with_cache' is set BEFORE fit(). The
# shapiq wrapper warns if the cache isn't enabled.
reg = TabPFNRegressor(fit_mode="fit_with_cache")
reg.fit(X_train, y_train)

explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
    model=reg,
    data=X_train,
    index="SV",
    max_order=1,
)

# Compute Shapley values for n_explain rows. Each call produces an
# `InteractionValues` object; we extract the (d,) 1st-order array per row and
# stack into the (n, d) matrix that shap.Explanation expects.
print(f"Computing Shapley values for {n_explain} rows...")
ivs = [explainer.explain(x=X_explain[i], budget=256) for i in range(n_explain)]
shap_values = np.stack([iv.get_n_order_values(1) for iv in ivs])

# baseline_value is the model's expected output when *every* feature is masked
# — i.e. the prediction on the empty coalition. We average across rows to get
# the scalar E[f(X)] that shap.Explanation wants for base_values.
base_value = float(np.mean([iv.baseline_value for iv in ivs]))

# Wrap shapiq's output in a shap.Explanation so the full shap.plots.* family
# accepts it directly.
explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(n_explain, base_value),
    data=X_explain,
    feature_names=feature_names,
)

# 1. Summary plot — beeswarm of feature attributions across all explained rows
shap.summary_plot(explanation)

# 2. Scatter plot — SHAP value of feature 0 vs. its raw value, colored by the
# feature shap picks as its strongest interaction partner. (New-API equivalent
# of the legacy shap.dependence_plot.)
shap.plots.scatter(explanation[:, 0])

# 3. Bar plot — mean(|SHAP|) ranking of features
shap.plots.bar(explanation)

# 4. Beeswarm plot — same data as summary, new-API styling
shap.plots.beeswarm(explanation)

# 5. Waterfall plot — explain a single row (E[f(X)] -> f(x) breakdown)
shap.plots.waterfall(explanation[0])
