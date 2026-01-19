#%%
from tabpfn_extensions.pval_crt import tabpfn_crt

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
"""
Dataset suite for validating Conditional Randomization Tests (CRT).

Each dataset generator returns:
    X : np.ndarray (n, p)
    y : np.ndarray (n,)
    relevant_idx : set[int]  # ground-truth conditionally relevant features
"""

from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_breast_cancer,
)
from sklearn.preprocessing import StandardScaler

#%%
# ---------------------------
# Utilities
# ---------------------------

def _is_categorical(arr, max_unique=10):
    arr = np.asarray(arr)
    uniq = np.unique(arr[~np.isnan(arr)])
    return len(uniq) < max_unique

def _rng(seed):
    return np.random.RandomState(seed)

def _noise(rng, n, scale=1.0):
    return scale * rng.randn(n)

# ============================================================
# A. Linear synthetic (5)
# ============================================================

def linear_sparse(n=500, p=10, seed=0):
    rng = _rng(seed)
    X = rng.randn(n, p)
    y = 3*X[:,0] - 2*X[:,1] + X[:,2] + _noise(rng, n)
    return X, y, {0,1,2}

def linear_dense(n=500, seed=0):
    rng = _rng(seed)
    X = rng.randn(n, 5)
    y = X.sum(axis=1) + _noise(rng, n)
    return X, y, {0,1,2,3,4}

def linear_weak_signal(n=500, seed=0):
    rng = _rng(seed)
    X = rng.randn(n, 5)
    y = 0.5*X[:,0] + 0.5*X[:,1] + _noise(rng, n)
    return X, y, {0,1}

def linear_noise_block(n=500, p=20, seed=0):
    rng = _rng(seed)
    X = rng.randn(n, p)
    y = X[:,0] + X[:,1] + _noise(rng, n)
    return X, y, {0,1}

def linear_correlated(n=500, seed=0):
    rng = _rng(seed)
    X0 = rng.randn(n)
    X1 = X0 + 0.1*rng.randn(n)
    X = np.column_stack([X0, X1, rng.randn(n,3)])
    y = X0 + _noise(rng, n)
    return X, y, {0}

# ============================================================
# B. Nonlinear synthetic (7)
# ============================================================

def friedman1(n=500, seed=0):
    rng = _rng(seed)
    X = rng.rand(n, 10)
    y = (
        10*np.sin(np.pi*X[:,0]*X[:,1])
        + 20*(X[:,2]-0.5)**2
        + 10*X[:,3]
        + 5*X[:,4]
        + _noise(rng, n)
    )
    return X, y, {0,1,2,3,4}

def friedman2(n=500, seed=0):
    rng = _rng(seed)
    X = rng.rand(n, 10)
    y = (
        X[:,0]**2
        + (X[:,1]*X[:,2])
        - X[:,3]
        + np.sin(X[:,4])
        + _noise(rng, n)
    )
    return X, y, {0,1,2,3,4}

def friedman3(n=500, seed=0):
    rng = _rng(seed)
    X = rng.rand(n, 10)
    y = (
        np.arctan((X[:,0]+X[:,1])/(X[:,2]+0.1))
        + X[:,3]**2
        + _noise(rng, n)
    )
    return X, y, {0,1,2,3}

def xor_interaction(n=500, seed=0):
    rng = _rng(seed)
    X = rng.randn(n, 5)
    y = ((X[:,0] > 0) ^ (X[:,1] > 0)).astype(int)
    return X, y, {0,1}

def additive_plus_interaction(n=500, seed=0):
    rng = _rng(seed)
    X = rng.randn(n, 6)
    y = X[:,0] + X[:,1]*X[:,2] + _noise(rng, n)
    return X, y, {0,1,2}

def threshold_feature(n=500, seed=0):
    rng = _rng(seed)
    X = rng.rand(n, 5)
    y = (X[:,0] > 0.5).astype(float) + _noise(rng, n, 0.1)
    return X, y, {0}

def nonlinear_conditional_null(n=500, seed=0):
    rng = _rng(seed)
    X0 = rng.randn(n)
    X1 = X0 + 0.1*rng.randn(n)
    X = np.column_stack([X0, X1])
    y = np.sin(X0) + _noise(rng, n)
    return X, y, {0}

# ============================================================
# C. Correlated / causal structures (4)
# ============================================================

def proxy_feature(n=500, seed=0):
    rng = _rng(seed)
    X0 = rng.randn(n)
    X1 = X0 + 0.05*rng.randn(n)
    X = np.column_stack([X0, X1, rng.randn(n,3)])
    y = X0 + _noise(rng, n)
    return X, y, {0}

def chain_structure(n=500, seed=0):
    rng = _rng(seed)
    X2 = rng.randn(n)
    X1 = X2 + 0.1*rng.randn(n)
    X0 = X1 + 0.1*rng.randn(n)
    X = np.column_stack([X0, X1, X2])
    y = X0 + _noise(rng, n)
    return X, y, {0}

def collider_structure(n=500, seed=0):
    rng = _rng(seed)
    X0 = rng.randn(n)
    X1 = rng.randn(n)
    Z = X0 + X1 + 0.1*rng.randn(n)
    X = np.column_stack([X0, X1])
    y = Z + _noise(rng, n)
    return X, y, set()

def grouped_correlation(n=500, seed=0):
    rng = _rng(seed)
    Z = rng.randn(n)
    X = np.column_stack([
        Z + 0.1*rng.randn(n),
        Z + 0.1*rng.randn(n),
        rng.randn(n),
        rng.randn(n),
    ])
    y = Z + _noise(rng, n)
    return X, y, {0}

# ============================================================
# D. Real-world benchmarks (4)
# ============================================================
def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return X, y, {2,3,8}  # bmi, bp, s5

def california_housing():
    data = fetch_california_housing()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    return X, y, {0,2}  # MedInc, AveRooms

def wine_quality_red():
    import pandas as pd
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        sep=";"
    )
    X = df.drop("quality", axis=1).values
    y = df["quality"].values
    return X, y, {10,1}  # alcohol, volatile acidity

def breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, {20,22}  # worst radius, worst texture

# ============================================================
# Registry
# ============================================================

DATASET_REGISTRY = {
    # Linear
    "linear_sparse": linear_sparse,
    "linear_dense": linear_dense,
    "linear_weak_signal": linear_weak_signal,
    "linear_noise_block": linear_noise_block,
    "linear_correlated": linear_correlated,

    # Nonlinear
    "friedman1": friedman1,
    "friedman2": friedman2,
    "friedman3": friedman3,
    # # "xor_interaction": xor_interaction,
    "additive_interaction": additive_plus_interaction,
    "threshold_feature": threshold_feature,
    "nonlinear_conditional_null": nonlinear_conditional_null,

    # Correlated / causal
    # "proxy_feature": proxy_feature,
    # "chain_structure": chain_structure,
    # "collider_structure": collider_structure,
    # "grouped_correlation": grouped_correlation,

    # Real
    "diabetes": diabetes,
    # "california_housing": california_housing,
    # "wine_quality_red": wine_quality_red,
    "breast_cancer": breast_cancer,
}
#%%
# assumes:
# - DATASET_REGISTRY from previous message
# - tabpfn_crt already defined

# ---------------------------
# Configuration
# ---------------------------

ALPHA = 0.05
B = 1000
N_REPEATS = 25
BASE_SEED = 0

results_rows = []
dataset_summary_rows = []


# ---------------------------
# Main evaluation loop
# ---------------------------
for rep in range(N_REPEATS):
    seed = BASE_SEED + rep
    for dataset_name, dataset_fn in DATASET_REGISTRY.items():

        print(f"\n=== Dataset: {dataset_name} ===")
        # if dataset_name != "additive_interaction":
        #     continue
        
        X, y, relevant_idx = dataset_fn()
        n, p = X.shape

        feature_results = []

        for j in tqdm(range(p), desc=f"Testing features ({p})"):

            xj = X[:, j]

            res = tabpfn_crt(
                X=X,
                y=y,
                j=j,
                B=B,
                alpha=ALPHA,
                seed=seed,
                K=1000
            )

            is_relevant = j in relevant_idx
            rejected = res["reject_null"]
            X[:,j]
            results_rows.append({
                "dataset": dataset_name,
                "feature": j,
                "is_relevant": is_relevant,
                "reject_null": rejected,
                "p_value": res["p_value"],
                "y_is_categorical": res["y_is_categorical"],
                "xj_is_categorical": res["xj_is_categorical"],
            })

            feature_results.append((is_relevant, rejected))

        # ---------------------------
        # Dataset-level metrics
        # ---------------------------

        feature_results = np.asarray(feature_results, dtype=int)

        if feature_results.size > 0:
            is_rel = feature_results[:, 0]
            rej = feature_results[:, 1]

            tp = np.sum((is_rel == 1) & (rej == 1))
            fn = np.sum((is_rel == 1) & (rej == 0))
            fp = np.sum((is_rel == 0) & (rej == 1))
            tn = np.sum((is_rel == 0) & (rej == 0))

            dataset_summary_rows.append({
                "dataset": dataset_name,
                "n_features": p,
                "n_relevant": int(is_rel.sum()),
                "power": tp / max(tp + fn, 1),
                "type1_error": fp / max(fp + tn, 1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            })

# ---------------------------
# Convert to DataFrames
# ---------------------------

df_features = pd.DataFrame(results_rows)
df_datasets = (
    pd.DataFrame(dataset_summary_rows)
    .groupby("dataset")
    .agg(
        power_mean=("power", "mean"),
        power_std=("power", "std"),
        type1_mean=("type1_error", "mean"),
        type1_std=("type1_error", "std"),
    )
    .reset_index()
)

df_plot = (
    df_features
    .groupby(["dataset", "feature", "is_relevant"])
    .p_value.mean()
    .reset_index()
)

# Global summary
df_global = pd.DataFrame([{
    "avg_power": df_datasets["power"].mean(),
    "avg_type1_error": df_datasets["type1_error"].mean(),
    "median_power": df_datasets["power"].median(),
    "median_type1_error": df_datasets["type1_error"].median(),
}])


# ---------------------------
# Save tables (Overleaf-ready)
# ---------------------------

df_features.to_csv("crt_feature_level_results.csv", index=False)
df_datasets.to_csv("crt_dataset_level_results.csv", index=False)
df_global.to_csv("crt_global_summary.csv", index=False)

print("\nSaved:")
print(" - crt_feature_level_results.csv")
print(" - crt_dataset_level_results.csv")
print(" - crt_global_summary.csv")

#%% ============================================================
# LaTeX tables: Power and Type-I error
# ============================================================

def save_latex_table(df, filename, caption, label, float_fmt="%.3f"):
    tex = df.to_latex(
        index=False,
        float_format=lambda x: float_fmt % x if isinstance(x, (float, np.floating)) else x,
        caption=caption,
        label=label,
        escape=False,
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tex)

# Dataset-level power / type-I table
save_latex_table(
    df_datasets[
        ["dataset", "n_features", "n_relevant", "power", "type1_error"]
    ],
    filename="table_power_type1.tex",
    caption=(
        "Empirical power and type-I error of the TabPFN Conditional Randomization Test "
        "across synthetic and real-world datasets."
    ),
    label="tab:power_type1",
)

# Global summary table
save_latex_table(
    df_global,
    filename="table_global_summary.tex",
    caption="Aggregate performance metrics across all datasets.",
    label="tab:global_summary",
)
#%% ============================================================
# ECDF plots
# ============================================================

def plot_ecdf(pvals, label, ax):
    x = np.sort(pvals)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label)

fig, ax = plt.subplots(figsize=(6,4))

plot_ecdf(
    df_plot.loc[df_plot.is_relevant, "p_value"],
    "CRT – relevant",
    ax,
)
plot_ecdf(
    df_plot.loc[~df_plot.is_relevant, "p_value"],
    "CRT – irrelevant",
    ax,
)

ax.plot([0,1],[0,1],"k--",alpha=0.5)
ax.set_xlabel("p-value")
ax.set_ylabel("ECDF")
ax.set_title("CRT p-value calibration")
ax.legend()
plt.tight_layout()
plt.savefig("ecdf_crt.png", dpi=200)
plt.show()
#%% ============================================================
# QQ plot: null p-values vs Uniform(0,1)
# ============================================================

null_p = np.sort(df_plot.loc[~df_plot.is_relevant, "p_value"].values)
u = (np.arange(1, len(null_p)+1) - 0.5) / len(null_p)

plt.figure(figsize=(4,4))
plt.plot(u, null_p, "o", ms=3)
plt.plot([0,1],[0,1],"k--")
plt.xlabel("Uniform(0,1)")
plt.ylabel("Empirical null p-values")
plt.title("CRT null p-value QQ plot")
plt.tight_layout()
plt.savefig("qq_null_crt.png", dpi=200)
plt.show()
