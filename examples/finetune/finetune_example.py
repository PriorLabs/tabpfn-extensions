from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import torch
from tabpfn import TabPFNClassifier

from tabpfn_extensions.finetune.finetune_classifier import FinetunedTabPFNClassifier

# 1. Load and prepare the data
# We use a small subset for a quick demonstration.
print("--- 1. Loading Data ---")
X_all, y_all = fetch_covtype(return_X_y=True, shuffle=True)
X, y = X_all[:10000], y_all[:10000]

# Create a final hold-out test set. This is NOT used during fine-tuning.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.\n")

# Calculate ROC AUC
def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        return roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")

# 2. Initial model evaluation on test set

base_clf = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', n_estimators=2)
base_clf.fit(X_train, y_train)

base_pred_proba = base_clf.predict_proba(X_test)
roc_auc = calculate_roc_auc(y_test, base_pred_proba)
log_loss_score = log_loss(y_test, base_pred_proba)

print(f"📊 Initial Test ROC: {roc_auc:.4f}")
print(f"📊 Initial Test Log Loss: {log_loss_score:.4f}\n")

# 3. Initialize and run fine-tuning
print("--- 2. Initializing and Fitting Model ---\n")

# Instantiate the wrapper with your desired hyperparameters
finetuned_clf = FinetunedTabPFNClassifier(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    epochs=10,
    learning_rate=1e-5,
    meta_dataset_size=1024,
    finetune_split_ratio=0.3,
    random_state=42,
    n_estimators=2,
    patience=3
)

# 4. Call .fit() to start the fine-tuning process on the training data
finetuned_clf.fit(X_train, y_train)
print("\n")

# 5. Evaluate the fine-tuned model
print("--- 3. Evaluating Model on Held-out Test Set ---\n")
y_pred_proba = finetuned_clf.predict_proba(X_test)

roc_auc = calculate_roc_auc(y_test, y_pred_proba)
loss = log_loss(y_test, y_pred_proba)

print(f"📊 Final Test ROC: {roc_auc:.4f}")
print(f"📊 Final Test Log Loss: {loss:.4f}")