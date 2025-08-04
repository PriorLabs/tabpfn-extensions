#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
This example trains multiple TabPFN models, which is computationally intensive.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNRegressor,
)


def fetch_boston_housing_manually():
    """Fetches the Boston housing dataset from its original source at CMU.
    This is a replacement for the deprecated function from scikit-learn.

    Returns:
        (data, target): A tuple of numpy arrays. `data` is the feature matrix,
                        and `target` is the regression target.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target


# Regression


X_manual, y_manual = fetch_boston_housing_manually()

# Root Mean Squared Error (RMSE): 5.279701791369698
# Mean Absolute Error (MAE): 3.7002804556292683
# R-squared (R^2): 0.6316622564943378
df_boston = fetch_openml(data_id=531, as_frame=True)
X_fetch_openml, y_fetch_openml = df_boston.data, df_boston.target

print("X_manual.shape, y_manual.shape:", X_manual.shape, y_manual.shape)
print(
    "X_fetch_openml.shape, y_fetch_openml.shape:",
    X_fetch_openml.shape,
    y_fetch_openml.shape,
)

print("X_fetch_openml.head():", X_fetch_openml.head())
print("y_fetch_openml.head():", y_fetch_openml.head())

print("Correct data types after loading:")
print(X_fetch_openml.dtypes)


X_train, X_test, y_train, y_test = train_test_split(
    X_fetch_openml,
    y_fetch_openml,
    test_size=0.30,
    random_state=42,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

tabpfn_base = TabPFNRegressor(device=device)
tabpfn_base.fit(X_train, y_train)
predictions_tabpfn_base = tabpfn_base.predict(X_test)
print(
    "Root Mean Squared Error (RMSE):",
    np.sqrt(mean_squared_error(y_test, predictions_tabpfn_base)),
)
print(
    "Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions_tabpfn_base)
)
print("R-squared (R^2):", r2_score(y_test, predictions_tabpfn_base))

reg = AutoTabPFNRegressor(max_time=60 * 6, device=device, phe_fit_args={"verbosity": 0})
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print(
    "Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, predictions))
)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))
