"""Explain TabPFN predictions with shapiq.

Demonstrates three explainer flavors from
``tabpfn_extensions.interpretability.shapiq``:

  1. ``get_tabpfn_nan_explainer`` — masks absent features with ``NaN`` and
     lets TabPFN's native missing-value handling absorb them. Fastest path
     on TabPFN v3 because the training-set KV cache is built once and
     reused across every coalition.

  2. ``get_tabpfn_explainer`` — remove-and-recontextualize (Rundel et al.
     2024). Refits the model on each feature subset. Semantically
     different from (1); cannot use the KV cache.

  3. Shapley interactions (higher-order) via ``index="FSII", max_order=2``
     on either explainer.

WARNING: May run slowly on CPU. For GPU acceleration on v3, pass
``fit_mode="fit_with_cache"`` and then set
``clf.executor_.keep_cache_on_device = True`` after ``.fit()`` — this is
what makes the NaN explainer much faster than the refit explainer.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Keep budget low for demo speed; use 2**n_features (here 2**30) or a larger
# number for real explanations.
n_model_evals = 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=200, random_state=42
)
x_explain = X_test[0]


# ---------------------------------------------------------------------------
# 1) NaN-passthrough explainer — fastest on TabPFN v3
# ---------------------------------------------------------------------------
# Pass ``fit_mode="fit_with_cache"`` so the v3 KV cache is built once at fit
# time; then pin the caches on-device so every coalition evaluation during
# explanation reuses the same on-GPU representation of the training set.
clf_nan = TabPFNClassifier(fit_mode="fit_with_cache", device="auto")
clf_nan.fit(X_train, y_train)
if hasattr(clf_nan, "executor_") and hasattr(clf_nan.executor_, "keep_cache_on_device"):
    clf_nan.executor_.keep_cache_on_device = True

nan_explainer = tabpfn_shapiq.get_tabpfn_nan_explainer(
    model=clf_nan,
    data=X_train,
    index="SV",
    max_order=1,
    class_index=1,
)

print("Calculating Shapley values via NaN-passthrough explainer...")
shapley_values = nan_explainer.explain(x=x_explain, budget=n_model_evals)
shapley_values.plot_force(feature_names=feature_names)


# ---------------------------------------------------------------------------
# 2) Remove-and-recontextualize explainer — semantically different
# ---------------------------------------------------------------------------
# Here the model is refit on each feature subset, so the KV cache does NOT
# carry over between coalitions — use the default fit_mode.
clf_refit = TabPFNClassifier(device="auto")

refit_explainer = tabpfn_shapiq.get_tabpfn_explainer(
    model=clf_refit,
    data=X_train,
    labels=y_train,
    index="SV",
    verbose=True,
)

print("Calculating Shapley values via remove-and-recontextualize explainer...")
shapley_values_refit = refit_explainer.explain(x=x_explain, budget=n_model_evals)
shapley_values_refit.plot_force(feature_names=feature_names)


# ---------------------------------------------------------------------------
# 3) Pairwise Shapley interactions on the refit explainer
# ---------------------------------------------------------------------------
interaction_explainer = tabpfn_shapiq.get_tabpfn_explainer(
    model=clf_refit,
    data=X_train,
    labels=y_train,
    index="FSII",  # Faithful Shapley Interaction Index
    max_order=2,  # pairwise interactions
    verbose=True,
)

print("Calculating Shapley interaction values...")
interaction_values = interaction_explainer.explain(x=x_explain, budget=n_model_evals)
interaction_values.plot_upset(feature_names=feature_names)
