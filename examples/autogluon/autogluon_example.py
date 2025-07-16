#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
This example trains multiple TabPFN models, which is computationally intensive.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.autogluon import (
    AutogluonTabPFNClassifier,
    AutogluonTabPFNRegressor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpus = 1 if torch.cuda.is_available() else 0

# Binary
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

print("\n--- Using TabPFNClassifier (Normal TabPFN) ---")
normal_tabpfn_clf = TabPFNClassifier(device=device)
normal_tabpfn_clf.fit(X_train, y_train)
normal_tabpfn_proba = normal_tabpfn_clf.predict_proba(X_test)
normal_tabpfn_preds = np.argmax(normal_tabpfn_proba, axis=-1)
print("TabPFN Binary ROC AUC:", roc_auc_score(y_test, normal_tabpfn_proba[:, 1]))
print("TabPFN Binary Accuracy:", accuracy_score(y_test, normal_tabpfn_preds))

print("\n--- Using AutogluonTabPFNClassifier (TabPFN with AutoGluon) ---")
clf = AutogluonTabPFNClassifier(
    max_time=60 * 3,
    num_random_configs=50,
    presets="medium_quality",
    verbosity=0,
    num_gpus=n_gpus,
)
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=-1)

print(
    "Autogluon TabPFN Binary ROC AUC:",
    roc_auc_score(y_test, prediction_probabilities[:, 1]),
)
print("Autogluon TabPFN Binary Accuracy", accuracy_score(y_test, predictions))

# Multiclass
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)
print("\n--- Using TabPFNClassifier (Normal TabPFN) ---")
# We use device='cpu' for wider compatibility, change to 'cuda' if you have a GPU
normal_tabpfn_clf = TabPFNClassifier(device=device)
normal_tabpfn_clf.fit(X_train, y_train)
normal_tabpfn_proba = normal_tabpfn_clf.predict_proba(X_test)
normal_tabpfn_preds = np.argmax(normal_tabpfn_proba, axis=-1)

print(
    "TabPFN Multiclass ROC AUC:",
    roc_auc_score(y_test, normal_tabpfn_proba, multi_class="ovr"),
)
print("TabPFN Multiclass Accuracy:", accuracy_score(y_test, normal_tabpfn_preds))

print("\n--- Using AutogluonTabPFNClassifier (TabPFN with AutoGluon) ---")
clf = AutogluonTabPFNClassifier(
    max_time=60 * 3,
    presets="medium_quality",
    verbosity=0,
    num_gpus=n_gpus,
)
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=-1)

print(
    "Autogluon TabPFN Multiclass ROC AUC:",
    roc_auc_score(y_test, prediction_probabilities, multi_class="ovr"),
)
print("Autogluon TabPFN Multiclass Accuracy", accuracy_score(y_test, predictions))

# Regression
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)
print("\n--- Using TabPFNRegressor (Normal TabPFN) ---")
# We use device='cpu' for wider compatibility, change to 'cuda' if you have a GPU
normal_tabpfn_reg = TabPFNRegressor(device=device)
normal_tabpfn_reg.fit(X_train, y_train)
normal_tabpfn_preds = normal_tabpfn_reg.predict(X_test)

print(
    "TabPFN Regression Mean Squared Error (MSE):",
    mean_squared_error(y_test, normal_tabpfn_preds),
)
print(
    "TabPFN Regression Mean Absolute Error (MAE):",
    mean_absolute_error(y_test, normal_tabpfn_preds),
)
print("TabPFN Regression R-squared (R^2):", r2_score(y_test, normal_tabpfn_preds))

print("\n--- Using AutogluonTabPFNRegressor (TabPFN with AutoGluon) ---")
reg = AutogluonTabPFNRegressor(
    max_time=60 * 3,
    presets="medium_quality",
    verbosity=0,
    num_gpus=n_gpus,
)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print(
    "Autogluon TabPFN Regression Mean Squared Error (MSE):",
    mean_squared_error(y_test, predictions),
)
print(
    "Autogluon TabPFN Regression Mean Absolute Error (MAE):",
    mean_absolute_error(y_test, predictions),
)
print("Autogluon TabPFN Regression R-squared (R^2):", r2_score(y_test, predictions))
