"""Multi-output prediction workflows for TabPFN."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

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

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

regressor = TabPFNMultiOutputRegressor(n_estimators=4)
regressor.fit(X_reg_train, y_reg_train)

reg_predictions = regressor.predict(X_reg_test)
print("Regression predictions shape:", reg_predictions.shape)

r2_per_target = [
    r2_score(y_reg_test[:, i], reg_predictions[:, i])
    for i in range(reg_predictions.shape[1])
]
print("Regression R2 per target:", r2_per_target)
print(
    "Regression average R2:",
    r2_score(y_reg_test, reg_predictions, multioutput="uniform_average"),
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

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

classifier = TabPFNMultiOutputClassifier(n_estimators=4)
classifier.fit(X_clf_train, y_clf_train)

clf_predictions = classifier.predict_proba(X_clf_test)
print("Classification predictions shape:", clf_predictions.shape)

micro_roc_auc = roc_auc_score(y_clf_test, clf_predictions, average="micro")
print("Classification micro-ROC-AUC:", micro_roc_auc)
