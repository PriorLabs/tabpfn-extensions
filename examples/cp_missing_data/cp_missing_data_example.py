"""Example of conformal prediction intervals for TabPFN with missing data.

Demonstrates how to obtain valid prediction intervals when the dataset
contains missing values, using split conformal prediction calibrated
per missing-data mask pattern.

Note: This algorithm works well when the number of unique missing patterns is small.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabpfn_extensions.cp_missing_data import CPMDATabPFNRegressor

# generate some data
np.random.seed(42)  # For reproducibility
X = np.random.rand(500, 2)
y = X @ np.array([5, 5]) + np.random.rand(500)

# add missing values in X under MCAR
X[np.random.randint(0, 500, 200), np.random.randint(0, 2, 200)] = np.nan

# Train/test split
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Check how many unique patterns there are
unique_patterns = pd.DataFrame(X_train).isna().astype(int).drop_duplicates()
print(f"Number of unique missing data patterns: {len(unique_patterns)}")
print("\nUnique patterns:")
print(unique_patterns)

# Use TabPFN+CP-MDA, the interval
# Note the interval needs to be symmetric
model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=123)
model.fit(X_train, y_train)
print("\nCalibration results:")
print(model.calibration_results_)

# Apply the model to new cases
CP_results = model.predict(X_test)

# Show first 5
print("\nConformal prediction results:")
print(f"Lower bound (corrected): {CP_results[0][:5]}")
print(f"Predictions: {CP_results[1][:5]}")
print(f"Upper bound (corrected): {CP_results[2][:5]}")
print(f"Lower bound (uncorrected): {CP_results[3][:5]}")
print(f"Upper bound (uncorrected): {CP_results[4][:5]}")

# Optionally, pass a custom TabPFNRegressor to control settings such as
# n_estimators or device. The custom estimator is cloned internally so
# it does not need to be fitted beforehand.
# from tabpfn_extensions.utils import TabPFNRegressor
# model = CPMDATabPFNRegressor(
#     quantiles=[0.05, 0.5, 0.95],
#     val_size=0.5,
#     seed=123,
#     tabpfn_estimator=TabPFNRegressor(n_estimators=16, device="cpu"),
# )
