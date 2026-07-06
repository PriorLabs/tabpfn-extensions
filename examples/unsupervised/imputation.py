#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised
from tabpfn_extensions.unsupervised import simple_impute

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

# Keep the ground-truth values of the injected cells so we can score the result.
original_nan_mask = np.isnan(X_test_missing)
original_values = X_test[original_nan_mask]

# Initialize TabPFN models for the regression and classification sub-problems that
# imputation is built on.
clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)

# --- 3. Impute Missing Values ---
# `simple_impute` fills in missing values one column at a time: for each column
# that has NaNs it fits a TabPFN model on the rows where that column is observed,
# using all the other columns as features, and predicts the rows where it is
# missing. It works directly on a single table, so we stack the (complete)
# training rows with the test rows to give every column as much context as
# possible when it is predicted.
print("\nImputing missing values...")
X_all = np.vstack([X_train, X_test_missing])
X_all_imputed = simple_impute(X_all, tabpfn_reg=reg, tabpfn_clf=clf)

# Pull the imputed test rows back out and score them against the ground truth.
X_test_imputed = X_all_imputed[len(X_train) :]
imputed_values = X_test_imputed[original_nan_mask]

mse = np.mean((imputed_values - original_values) ** 2)
print(f"Imputation complete. Mean Squared Error vs. original values: {mse:.4f}")

# --- 4. Density-based imputation with TabPFNUnsupervisedModel ---
# When you need to *sample* imputations (rather than take the best estimate), or
# to condition on a causal DAG, TabPFNUnsupervisedModel models the joint
# distribution: it fits a reference set and imputes by averaging over random
# feature permutations and sampling from the resulting density.
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

print("\nFitting TabPFNUnsupervisedModel on the training data...")
model_unsupervised.fit(X_train)

print("Imputing missing values by sampling from the model...")
X_imputed_tensor = model_unsupervised.impute(
    X_test_missing,
    n_permutations=5,  # Fewer permutations for a quicker example
)

n_missing_after = torch.isnan(X_imputed_tensor).sum().item()
print(f"Number of missing values after imputation: {n_missing_after}")

imputed_values_density = X_imputed_tensor.numpy()[original_nan_mask]
mse_density = np.mean((imputed_values_density - original_values) ** 2)
print(f"Mean Squared Error vs. original values: {mse_density:.4f}")
