#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
This example trains multiple TabPFN models, which is computationally intensive.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, fetch_openml
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
from sklearn.utils import Bunch

from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to your CSV
csv_path = "/home/klemens_priorlabs_ai/tabpfn-extensions/examples/phe/ICB1_mini.csv"  # change to your file
# Load CSV
frame = pd.read_csv(csv_path)
# Keep X as DataFrame (needed for make_column_selector) and y as Series
X_df = frame.iloc[:, :-1]            # pandas DataFrame
y_s  = frame.iloc[:, -1]             # pandas Series
feature_names = X_df.columns.tolist()
target_col = frame.columns[-1]
df = Bunch(
    data=X_df,                       # <-- DataFrame, not .to_numpy()
    target=y_s,                      # <-- Series
    frame=frame,
    feature_names=feature_names,
    DESCR=(
        f"User CSV loaded from {csv_path}. "
        f"Samples: {frame.shape[0]}, Features: {len(feature_names)}. "
        f"Target: '{target_col}'."
    )
)

phoneme = fetch_openml(data_id=1489, as_frame=True)
df = phoneme.frame
target_col = "class" if "class" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])           # keep as DataFrame (good for column selectors)
y = df[target_col]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

clf = AutoTabPFNClassifier(max_time=60 * 6)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
pred = np.argmax(proba, axis=1)

print("Accuracy:", accuracy_score(y_test, pred))
if proba.shape[1] == 2:
    print("ROC AUC: ", roc_auc_score(y_test, proba[:, 1]))
else:
    print("ROC AUC: n/a (non-binary): ", roc_auc_score(y_test, proba, multi_class="ovr"))