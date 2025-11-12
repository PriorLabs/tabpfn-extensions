import numpy as np
import torch
from sklearn.datasets import fetch_covtype
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
from tabpfn_extensions.finetune.finetune_classifier import FinetunedTabPFNClassifier

# 1. Load and prepare the data
# We use a small subset for a quick demonstration.
print("--- 1. Loading Data ---")
X_all, y_all = fetch_covtype(return_X_y=True, shuffle=True)
X, y = X_all[:11000], y_all[:11000]

# df = pd.read_csv("/home/anurag_priorlabs_ai/tabpfn-extensions/PrudentialLifeInsuranceAssessment.csv")

# print(df.columns)
# X = df.drop(columns=["Id", "Response"])
# y = df["Response"]
# Create a final hold-out test set. This is NOT used during fine-tuning.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.\n")

# Calculate ROC AUC
def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])   # pyright: ignore[reportReturnType]
    return roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted") # pyright: ignore[reportReturnType]

# 2. Initial model evaluation on test set

base_clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=2)
base_clf.fit(X_train, y_train)

base_pred_proba = base_clf.predict_proba(X_test)
roc_auc = calculate_roc_auc(y_test, base_pred_proba) # pyright: ignore[reportReturnType, reportArgumentType]
log_loss_score = log_loss(y_test, base_pred_proba)

print(f"📊 Initial Test ROC: {roc_auc:.4f}")
print(f"📊 Initial Test Log Loss: {log_loss_score:.4f}\n")

# 3. Initialize and run fine-tuning
print("--- 2. Initializing and Fitting Model ---\n")

# Instantiate the wrapper with your desired hyperparameters
finetuned_clf = FinetunedTabPFNClassifier(
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,
    learning_rate=1e-6,
    n_inference_context_samples=10_000,
    finetune_split_ratio=0.3,
    random_state=42,
    n_estimators=2,
    patience=3,
    ignore_pretraining_limits=True,
    grad_clip_value=1.0,
)

# 4. Call .fit() to start the fine-tuning process on the training data
finetuned_clf.fit(X_train, y_train)  # pyright: ignore[reportArgumentType]
print("\n")

# 5. Evaluate the fine-tuned model
print("--- 3. Evaluating Model on Held-out Test Set ---\n")
y_pred_proba = finetuned_clf.predict_proba(X_test)  # pyright: ignore[reportArgumentType]

roc_auc = calculate_roc_auc(y_test, y_pred_proba)  # pyright: ignore[reportArgumentType]
loss = log_loss(y_test, y_pred_proba)

print(f"📊 Final Test ROC: {roc_auc:.4f}")
print(f"📊 Final Test Log Loss: {loss:.4f}")
