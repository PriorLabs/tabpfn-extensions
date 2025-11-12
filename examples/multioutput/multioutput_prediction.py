"""Multi-output prediction workflows for TabPFN."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.metrics import f1_score, r2_score

from tabpfn_extensions.multioutput import (
    TabPFNMultiOutputClassifier,
    TabPFNMultiOutputRegressor,
)

# ---------------------------------------------------------------------------
# 1. Multi-output regression with missing features
# ---------------------------------------------------------------------------

X_reg, y_reg = make_regression(
    n_samples=120,
    n_features=6,
    n_targets=2,
    n_informative=6,
    noise=0.05,
    random_state=0,
)

rng = np.random.default_rng(0)
X_reg_missing = X_reg.copy()
missing_mask = rng.random(X_reg_missing.shape) < 0.1
X_reg_missing[missing_mask] = np.nan

print(f"Regression missing values: {missing_mask.sum()} / {X_reg_missing.size}")

regressor = TabPFNMultiOutputRegressor(
    n_estimators=4,
    model_path="tabpfn-v2-regressor.ckpt",
)
regressor.fit(X_reg_missing, y_reg)

reg_predictions = regressor.predict(X_reg_missing)
print("Regression predictions shape:", reg_predictions.shape)
print("Regression predictions contain NaNs:", np.isnan(reg_predictions).any())

r2_per_target = [
    r2_score(y_reg[:, i], reg_predictions[:, i])
    for i in range(reg_predictions.shape[1])
]
print("Regression R2 per target:", r2_per_target)
print(
    "Regression average R2:",
    r2_score(y_reg, reg_predictions, multioutput="uniform_average"),
)

# ---------------------------------------------------------------------------
# 2. Multi-output classification (multi-label) with the same wrapper pattern
# ---------------------------------------------------------------------------

X_clf, y_clf = make_multilabel_classification(
    n_samples=150,
    n_features=6,
    n_classes=3,
    n_labels=2,
    allow_unlabeled=False,
    random_state=1,
)

X_clf = X_clf.astype(np.float32)
clf_missing = X_clf.copy()
clf_missing[rng.random(clf_missing.shape) < 0.1] = np.nan

classifier = TabPFNMultiOutputClassifier(n_estimators=4)
classifier.fit(clf_missing, y_clf)

clf_predictions = classifier.predict(clf_missing)
print("Classification predictions shape:", clf_predictions.shape)
print("Classification predictions contain NaNs:", np.isnan(clf_predictions).any())

micro_f1 = f1_score(y_clf, clf_predictions, average="micro")
print("Classification micro-F1:", micro_f1)
