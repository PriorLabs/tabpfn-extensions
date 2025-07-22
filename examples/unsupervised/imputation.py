#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# --- 1. Load and Prepare Data ---
# Load the breast cancer dataset
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]

# Split the data into training and testing sets
X_train, X_test = train_test_split(
    X,
    test_size=0.33,
    random_state=42,
)

# --- 2. Introduce Missing Values ---
# Create a copy of the test set to introduce missing values (NaNs)
X_test_missing = X_test.copy()
n_samples, n_features = X_test_missing.shape

# Introduce 20% missing values in the first three columns for demonstration
missing_fraction = 0.2
n_missing = int(n_samples * missing_fraction)

for col_idx in range(3):
    # Choose random rows to set to NaN
    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
    X_test_missing[missing_indices, col_idx] = np.nan

print(f"Introduced {np.isnan(X_test_missing).sum()} missing values into the test set.")

# --- 3. Initialize the Unsupervised Model ---
# Initialize TabPFN models for regression and classification tasks.
# The unsupervised model uses these to model the data distribution.
clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)

# Initialize the main unsupervised model
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

# --- 4. Fit and Impute ---
# Fit the model on the complete training data (without missing values)
print("Fitting the unsupervised model on the training data...")
model_unsupervised.fit(X_train)

# Perform imputation on the test set
print("Imputing missing values...")
X_imputed_tensor = model_unsupervised.impute(
    X_test_missing,
    n_permutations=5,  # Fewer permutations for a quicker example
)

# --- 5. Verify Results ---
# Check that the imputed data no longer contains any NaN values
n_missing_after = torch.isnan(X_imputed_tensor).sum().item()

print(f"\nNumber of missing values after imputation: {n_missing_after}")
print(f"Imputation complete. Shape of imputed data: {X_imputed_tensor.shape}")

# Optional: Calculate the Mean Squared Error for the imputed values,
# since we know the original ground truth values.
original_nan_mask = np.isnan(X_test_missing)
imputed_values = X_imputed_tensor.numpy()[original_nan_mask]
original_values = X_test[original_nan_mask]

mse = np.mean((imputed_values - original_values) ** 2)
print(f"Mean Squared Error of imputed values vs. original values: {mse:.4f}")
