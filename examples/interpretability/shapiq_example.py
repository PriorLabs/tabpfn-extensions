"""Compute Shapley values and pairwise Shapley interactions for a TabPFN model
using the ShapIQ library, and visualize them with shapiq's native plots.

Two paradigms for "feature removal" are illustrated:

  1. Imputation-based (`get_tabpfn_imputation_explainer`): masked features are
     filled by an imputer (default: baseline). The baseline imputer models
     missing values by the mean of the training set for numeric features 
     and the mode for categorical features. The training set is fixed
     across coalitions, so the KV-cache fast path applies — make sure
     to construct the model with `fit_mode="fit_with_cache"` and set
     `executor_.keep_cache_on_device = True` after fit().

  2. Remove-and-recontextualize (`get_tabpfn_explainer`): TabPFN is re-fit on
     each coalition's column subset. Does not benefit from the KV cache
     (one predict per fit).

Dataset: California housing (regression, d=8).
"""

from __future__ import annotations

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNRegressor
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

housing = fetch_california_housing(as_frame=False)
X, y, feature_names = housing.data, housing.target, list(housing.feature_names)

X_train, X_test, y_train, _ = train_test_split(
    X, y, train_size=1000, test_size=200, random_state=0,
)
x_explain = X_test[0]

# Construct the regressor with the KV cache fast path. fit_mode must be set
# BEFORE .fit(); keep_cache_on_device=True is set AFTER .fit().
reg = TabPFNRegressor(fit_mode="fit_with_cache")
reg.fit(X_train, y_train)
# keep_cache_on_device is usually on by default — set explicitly as a safety net.
reg.executor_.keep_cache_on_device = True

# Exact enumeration for d=8 is 2**8 = 256 coalitions.
budget = 256


# -----------------------------------------------------------------------------
# 1. Imputation-based explainer (uses the KV cache)
# -----------------------------------------------------------------------------
imputation_explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
    model=reg,
    data=X_train,
    index="SV",     # plain Shapley values
    max_order=1,
)
print("Computing imputation-based Shapley values...")
sv_imp = imputation_explainer.explain(x=x_explain, budget=budget)
sv_imp.plot_force(feature_names=feature_names)


# -----------------------------------------------------------------------------
# 2. Pairwise Shapley interactions via the same explainer (k-SII at max_order=2)
# -----------------------------------------------------------------------------
interaction_explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
    model=reg,
    data=X_train,
    index="k-SII",  # k-Shapley Interaction Index — extends SHAP to interactions
    max_order=2,
)
print("Computing pairwise Shapley interactions (k-SII)...")
iv_interactions = interaction_explainer.explain(x=x_explain, budget=budget)

# Network plot: features as nodes, sized by individual SV; edges colored by
# pairwise interaction strength. Specific to shapiq (not in the shap library).
iv_interactions.plot_network(feature_names=feature_names)

# Upset plot of top interactions
iv_interactions.plot_upset(feature_names=feature_names)

# Result: Pairwise Shapley interactions uncover a strong interaction between
# Latitude and Longitude, which are not visible at lower orders.


# -----------------------------------------------------------------------------
# 3. Remove-and-recontextualize (Rundel) — slower (no KV cache)
# -----------------------------------------------------------------------------
# Commented out because this path is much slower than the imputation explainer
# above, as it doesn't benefit from the KV cache: each coalition
# triggers a fresh TabPFN fit on a different column subset, then does exactly
# one predict against that fit.
#
#     rundel_explainer = tabpfn_shapiq.get_tabpfn_explainer(
#         model=reg,
#         data=X_train,
#         labels=y_train,
#         index="SV",
#         max_order=1,
#     )
#     sv_rundel = rundel_explainer.explain(x=x_explain, budget=budget)
#     sv_rundel.plot_force(feature_names=feature_names)
