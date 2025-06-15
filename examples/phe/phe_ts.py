#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
This example trains multiple TabPFN models, which is computationally intensive.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit

from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNRegressor,
)

btc = fetch_openml(
    data_id=43563, as_frame=True
)  # ↔ name="Digital-currency---Time-series", version=1
df = btc.frame.copy()

df = (
    df.rename(columns={"Unnamed:_0": "date"})
    .assign(date=lambda x: pd.to_datetime(x["date"]))
    .set_index("date")
    .sort_index()
)

print("Head of raw data")
print(df.head(), "\n")

# ------------------------------------------------------------------
# 2️⃣  Features & target
target_col = "close_SAR"
y_raw = df[target_col].to_numpy()
y = y_raw / y_raw.max()
X = df.drop(columns=["close_SAR", "close_USD"]).to_numpy()

# ------------------------------------------------------------------
# 3️⃣  Chronological train/test split (last fold = test)
ts = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(ts.split(X))[-1]
X_tr, X_te = X[train_idx], X[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]

# ------------------------------------------------------------------
# 4️⃣  Baseline model: TabPFNRegressor
print("🔹 TabPFNRegressor (baseline)")
baseline = TabPFNRegressor()
baseline.fit(X_tr, y_tr)
pred_base = baseline.predict(X_te)

mae_base_rel = mean_absolute_error(y_te, pred_base)
mae_base_raw = mae_base_rel * y_raw.max()
print(f"   MAE (relative): {mae_base_rel:.4f}")
print(f"   MAE (SAR):      {mae_base_raw:,.2f}")


# ------------------------------------------------------------------
# no CV respect AutoTabPFNRegressor

print("\n🔹 AutoTabPFNRegressor (holdout)")
auto_holdout = AutoTabPFNRegressor(max_time=60 * 3)
auto_holdout.fit(X_tr, y_tr)
pred_auto_holdout = auto_holdout.predict(X_te)

mae_auto_holdout_rel = mean_absolute_error(y_te, pred_auto_holdout)
mae_auto_holdout_raw = mae_auto_holdout_rel * y_raw.max()
print(f"   MAE (relative): {mae_auto_holdout_rel:.4f}")
print(f"   MAE (SAR):      {mae_auto_holdout_raw:,.2f}")


# ------------------------------------------------------------------
# 5️⃣  AutoTabPFNRegressor with TimeSeriesSplit
print("\n🔹 AutoTabPFNRegressor (time-series aware CV)")
auto = AutoTabPFNRegressor(
    max_time=60 * 3,  # quick run
    random_state=42,
    phe_init_args={
        "cv_splitter": ts,
        "validation_method": "cv",
        "n_folds": 5,
        "max_models": 10,
        "n_repeats": 1,
    },
)
auto.fit(X_tr, y_tr)
pred_auto = auto.predict(X_te)

mae_auto_rel = mean_absolute_error(y_te, pred_auto)
mae_auto_raw = mae_auto_rel * y_raw.max()
print(f"   MAE (relative): {mae_auto_rel:.4f}")
print(f"   MAE (SAR):      {mae_auto_raw:,.2f}")
