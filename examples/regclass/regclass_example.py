import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LinearRegression
import torch

from tabpfn import TabPFNRegressor, TabPFNClassifier

from tabpfn_extensions.regclass.wrappers import DistributionalRegressorAsClassifier


# ------------------------------------------------------------------
# 1.  Synthetic data – continuous spread with sign → class
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
N, D = 1000, 40
X      = rng.normal(size=(N, D))
coef   = rng.normal(size=D)
noise  = rng.normal(scale=0.5, size=N)
spread = X @ coef + noise      # continuous     (regression target)
labels = (spread >= 0).astype(int)  # binary class (≥0 ⇒ class 1 else 0)

X_tr, X_te, y_tr_reg, y_te_reg, y_tr_cls, y_te_cls = train_test_split(
    X, spread, labels, test_size=0.3, random_state=42
)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n--- Binary Classification Example ---")
# -------------------------------------------------
# 2.  Plain TabPFNClassifier on thresholded labels
# -------------------------------------------------
clf_plain = TabPFNClassifier(n_estimators=8, random_state=0)
clf_plain.fit(X_tr, y_tr_cls)
prob_plain = clf_plain.predict_proba(X_te)[:, 1]
logloss_plain = log_loss(y_te_cls, prob_plain)
auc_plain  = roc_auc_score(y_te_cls, prob_plain)
print(f"Log Loss – TabPFNClassifier (plain labels) : {logloss_plain:.3f}")
print(f"ROC AUC – TabPFNClassifier (plain labels) : {auc_plain:.3f}")


# -------------------------------------------------
# 3.  DistributionalRegressorAsClassifier
# -------------------------------------------------
reg   = TabPFNRegressor(n_estimators=8, random_state=0)
clf_weight = DistributionalRegressorAsClassifier(reg, thresholds=[0], decision_strategy="weighted")
clf_weight.fit(X_tr, y_tr_reg)
prob_weighted = clf_weight.predict_proba(X_te)[:, 1]
logloss_weighted = log_loss(y_te_cls, prob_weighted)
dist_auc_weighted  = roc_auc_score(y_te_cls, prob_weighted)
print(f"Log Loss – DistributionalRegressorAsClassifier (weighted): {logloss_weighted:.3f}")
print(f"ROC AUC – DistributionalRegressorAsClassifier (weighted): {dist_auc_weighted:.3f}")


clf_prob = DistributionalRegressorAsClassifier(reg, thresholds=[0], decision_strategy="probabilistic")
clf_prob.fit(X_tr, y_tr_reg)
prob_weighted = clf_prob.predict_proba(X_te)[:, 1]
logloss_prob = log_loss(y_te_cls, prob_weighted)
dist_auc_prob  = roc_auc_score(y_te_cls, prob_weighted)
print(f"Log Loss – DistributionalRegressorAsClassifier (probabilistic): {logloss_prob:.3f}")
print(f"ROC AUC – DistributionalRegressorAsClassifier (probabilistic): {dist_auc_prob:.3f}")


print("\n--- Multiclass Classification Example ---")

multiclass_thresholds = [-0.5, 0.5]
labels_multiclass = np.digitize(spread, bins=multiclass_thresholds)
X_tr_mc, X_te_mc, y_tr_reg_mc, y_te_reg_mc, y_tr_cls_mc, y_te_cls_mc = train_test_split(
    X, spread, labels_multiclass, test_size=0.3, random_state=42
)

# -------------------------------------------------
# 4.  Plain TabPFNClassifier on thresholded labels (Multiclass)
# -------------------------------------------------
clf_plain_mc = TabPFNClassifier(n_estimators=8, random_state=0)
clf_plain_mc.fit(X_tr_mc, y_tr_cls_mc)
prob_plain_mc = clf_plain_mc.predict_proba(X_te_mc)
logloss_plain_mc = log_loss(y_te_cls_mc, prob_plain_mc)
auc_plain_mc = roc_auc_score(y_te_cls_mc, prob_plain_mc, multi_class='ovr', average='macro')
print(f"Log Loss – TabPFNClassifier (plain labels, Multiclass) : {logloss_plain_mc:.3f}")
print(f"ROC AUC – TabPFNClassifier (plain labels, Multiclass) : {auc_plain_mc:.3f}")


# ------------------------------------------------------------------
# 5.  DistributionalRegressorAsClassifier (Multiclass setup)
# ------------------------------------------------------------------
reg_multiclass = TabPFNRegressor(n_estimators=8, random_state=0)

# Weighted strategy for multiclass
clf_weight_mc = DistributionalRegressorAsClassifier(reg_multiclass, thresholds=multiclass_thresholds, decision_strategy="weighted")
clf_weight_mc.fit(X_tr_mc, y_tr_reg_mc)
prob_weighted_mc = clf_weight_mc.predict_proba(X_te_mc)
logloss_weighted_mc = log_loss(y_te_cls_mc, prob_weighted_mc)
auc_weighted_mc = roc_auc_score(y_te_cls_mc, prob_weighted_mc, multi_class='ovr', average='macro')
print(f"Log Loss – DistributionalRegressorAsClassifier (weighted, Multiclass): {logloss_weighted_mc:.3f}")
print(f"ROC AUC – DistributionalRegressorAsClassifier (weighted, Multiclass): {auc_weighted_mc:.3f}")

# Probabilistic strategy for multiclass
clf_prob_mc = DistributionalRegressorAsClassifier(reg_multiclass, thresholds=multiclass_thresholds, decision_strategy="probabilistic")
clf_prob_mc.fit(X_tr_mc, y_tr_reg_mc)
prob_prob_mc = clf_prob_mc.predict_proba(X_te_mc)
logloss_prob_mc = log_loss(y_te_cls_mc, prob_prob_mc)
auc_prob_mc = roc_auc_score(y_te_cls_mc, prob_prob_mc, multi_class='ovr', average='macro')
print(f"Log Loss – DistributionalRegressorAsClassifier (probabilistic, Multiclass): {logloss_prob_mc:.3f}")
print(f"ROC AUC – DistributionalRegressorAsClassifier (probabilistic, Multiclass): {auc_prob_mc:.3f}")

