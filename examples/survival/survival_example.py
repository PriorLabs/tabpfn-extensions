#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""SurvivalTabPFN demo with baselines and a 4-cluster visualization:
- Train SurvivalTabPFN on a survival dataset
- Print C-index for TabPFN, Cox, and RSF
- Plot ALL survival curves colored by 4 clusters (no PCA)

Note: This may run slowly on CPU-only systems—GPU is recommended.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    StandardScaler as Std,
)
from sksurv.datasets import (  # change dataset here if desired
    load_whas500,
)
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import check_y_survival

from tabpfn_extensions.survival import SurvivalTabPFN  # must expose predict_survival_at
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

# Silence only the specific TabPFN power-transform overflow warnings
warnings.filterwarnings(
    "ignore",
    message=r"overflow encountered in cast",
    category=RuntimeWarning,
    module=r"tabpfn\.preprocessors\.safe_power_transformer",
)

# ============================ Load & split data ================================
X, y = load_whas500()
# X: pandas DataFrame (mixed dtypes), y: structured array with ('event','time')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
event_test, time_test = check_y_survival(y_test)

# ============================ Train SurvivalTabPFN =============================
tabpfn = SurvivalTabPFN(
    cls_model=TabPFNClassifier(n_estimators=8, random_state=42),
    reg_model=TabPFNRegressor(n_estimators=8, random_state=42),
)
tabpfn.fit(X_train, y_train)

# ============================ Risk & C-index (TabPFN) =========================
risk_tabpfn = tabpfn.predict(X_test)  # higher = riskier
cindex_tabpfn = concordance_index_censored(event_test, time_test, risk_tabpfn)[0]
print(f"TabPFN C-index: {cindex_tabpfn:.3f}")

# ============================ Cross-validation C-index =========================
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sksurv.metrics import concordance_index_censored
from sksurv.util import check_y_survival

# --- CV config ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def cindex_scorer(estimator, X, y):
    """Scorer compatible with scikit-learn; returns scalar C-index for a fold."""
    event, time = check_y_survival(y)
    risk = estimator.predict(X)  # higher = riskier
    return concordance_index_censored(event, time, risk)[0]


# --- Preprocessing for baselines (TabPFN does not need this) ---
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

# --- Define models (use your existing ones if already created) ---
from tabpfn_extensions.survival import SurvivalTabPFN
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

tabpfn_cv = SurvivalTabPFN(
    cls_model=TabPFNClassifier(n_estimators=16, random_state=42),
    reg_model=TabPFNRegressor(n_estimators=16, random_state=42),
    n_auto_horizons=20,
    auto_horizon_quantile_range=(0.05, 0.95),
)

cox_cv = Pipeline(
    [
        ("pre", pre),
        (
            "cox",
            CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=100, max_iter=100000
            ),
        ),
    ]
)

rsf_cv = Pipeline(
    [
        ("pre", pre),
        (
            "rsf",
            RandomSurvivalForest(
                n_estimators=300,
                min_samples_split=10,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)

# --- Run CV (returns array of C-index values for each fold) ---
scores_tabpfn = cross_val_score(tabpfn_cv, X, y, scoring=cindex_scorer, cv=cv, n_jobs=1)
scores_cox = cross_val_score(cox_cv, X, y, scoring=cindex_scorer, cv=cv, n_jobs=1)
scores_rsf = cross_val_score(rsf_cv, X, y, scoring=cindex_scorer, cv=cv, n_jobs=1)

print(f"CV C-index TabPFN: {scores_tabpfn.mean():.3f} ± {scores_tabpfn.std():.3f}")
print(f"CV C-index CoxPH : {scores_cox.mean():.3f} ± {scores_cox.std():.3f}")
print(f"CV C-index RSF   : {scores_rsf.mean():.3f} ± {scores_rsf.std():.3f}")

# ============================ Survival curves S(t|x) ==========================
# Use a compact time grid up to the 95th percentile of observed test times
t_grid = np.linspace(0.0, float(np.percentile(time_test, 95)), 200)
S = tabpfn.predict_survival_at(X_test, t_grid)  # shape: (n_samples, len(t_grid))

# ============================ 4-cluster visualization =========================
# Cluster on failure curves (1 - S); color all curves by cluster; show cluster means
F = 1.0 - S
Z = Std(with_mean=True, with_std=True).fit_transform(F)

kmeans = KMeans(n_clusters=4, n_init=50, random_state=42)
labels = kmeans.fit_predict(Z)

# Plot all curves colored by cluster
plt.figure(figsize=(9.5, 6.5))
cmap = plt.get_cmap("tab10")
for c in range(4):
    idx = np.where(labels == c)[0]
    for j in idx:
        plt.plot(t_grid, S[j], color=cmap(c), alpha=0.20, linewidth=1.2)
    if len(idx) > 0:
        mean_curve = S[idx].mean(axis=0)
        plt.plot(
            t_grid,
            mean_curve,
            color=cmap(c),
            linewidth=3,
            label=f"Cluster {c} (n={len(idx)})",
        )

plt.xlabel("time")
plt.ylabel("Survival probability S(t|x)")
plt.title("All survival curves colored by 4 clusters\n(thick lines = cluster means)")
plt.ylim(0.1, 1)
plt.grid(visible=True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
