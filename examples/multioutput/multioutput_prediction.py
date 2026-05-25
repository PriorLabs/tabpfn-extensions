"""Multi-output prediction with TabPFN.

TabPFN's classifier/regressor are single-output. For multiple regression
targets or multi-label classification, wrap TabPFN with sklearn's standard
``MultiOutputRegressor`` / ``MultiOutputClassifier``. No TabPFN-specific
wrapper class is needed — this file exists to make that pattern discoverable.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Flexible imports per examples/README.md — support both the local tabpfn
# package and the tabpfn-client backend.
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor


# ─────────────────────────── multi-output regression ─────────────────────────

X_reg, y_reg = make_regression(
    n_samples=120,
    n_features=6,
    n_targets=2,
    n_informative=6,
    noise=0.05,
    random_state=0,
)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg = MultiOutputRegressor(TabPFNRegressor(n_estimators=4))
reg.fit(X_tr, y_tr)
y_pred = reg.predict(X_te)

print("Regression predictions shape:", y_pred.shape)
print("R^2 per target:", r2_score(y_te, y_pred, multioutput="raw_values"))
print("R^2 (uniform avg):", r2_score(y_te, y_pred, multioutput="uniform_average"))


# ─────────────────────────── multi-label classification ──────────────────────

X_clf, y_clf = make_multilabel_classification(
    n_samples=150,
    n_features=6,
    n_classes=3,
    n_labels=2,
    allow_unlabeled=False,
    random_state=1,
)
X_clf = X_clf.astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

clf = MultiOutputClassifier(TabPFNClassifier(n_estimators=4))
clf.fit(X_tr, y_tr)

# predict_proba returns a list (one (n_samples, n_classes) array per target).
# Stack the positive-class probabilities for a micro-ROC-AUC over labels.
proba = np.stack([p[:, 1] for p in clf.predict_proba(X_te)], axis=1)
print("Classification proba shape:", proba.shape)
print("micro-ROC-AUC:", roc_auc_score(y_te, proba, average="micro"))
