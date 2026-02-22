"""Provides a detailed example of obtaining conformalised prediction intervals when there is missing data.

This script demonstrates the complete workflow for obtaining conformal prediction intervals
for the TabPFNRegressor when these are missing values in the dataset. The process is shown
in two steps. Using the training data to train the model and obtain correction terms for
each mask, and appying the corrcetion terms with the trained model to a new dataset.

Note: This algorithm works well when the missing pattern is small.
"""

import numpy as np
import pandas as pd
import warnings

from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

from tabpfn_extensions.cp_missing_data import CPMDATabPFNRegressor, CPMDATabPFNRegressorNewData

# generate some data
np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 2)
Y = np.random.rand(100)

# add missing values in X under MCAR
X[np.random.randint(0, 100, 40), np.random.randint(0, 2, 40)] = np.nan

# Check how many unique patterns there are 
unique_patterns = pd.DataFrame(X).isnull().astype(int).drop_duplicates()
print(f"Number of unique missing data patterns: {len(unique_patterns)}")
print("\nUnique patterns:")
print(unique_patterns)

# Use TabPFN+CP-MDA
model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed = 123)
calibration_results, model_fit = model.fit(X, Y)
print(calibration_results)

# Apply the model to new cases 
cp_apply = CPMDATabPFNRegressorNewData(model_fit, quantiles=[0.05, 0.5, 0.95], calibration_results=calibration_results)
CP_results = cp_apply.predict(X)

print("\nConformal prediction results:")
print(f"Lower bound (corrected): {CP_results[0][:5]}")  # Show first 5
print(f"Predictions: {CP_results[1][:5]}")
print(f"Upper bound (corrected): {CP_results[2][:5]}")
print(f"Lower bound (uncorrected): {CP_results[3][:5]}")
print(f"Upper bound (uncorrected): {CP_results[4][:5]}")