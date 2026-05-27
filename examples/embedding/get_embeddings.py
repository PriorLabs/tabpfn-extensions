"""TabPFN Embedding Example

This example demonstrates how to extract embeddings from TabPFN models and use them
for classification and regression tasks via the scikit-learn style
``TabPFNEmbedding`` transformer.

NOTE: This example requires the full TabPFN implementation (pip install tabpfn).
It will not work with the TabPFN client (pip install tabpfn-client) because
the embedding functionality is not available in the client version.
"""

import warnings

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", "Ill-conditioned matrix")

# Note: You need to install the full TabPFN package for this example
# pip install tabpfn
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
print("Loading classification dataset (breast_cancer)...")
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)

# Baseline: vanilla logistic regression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
print(
    "Baseline Logistic Regression Accuracy: "
    f"{accuracy_score(y_test, model.predict(X_test)):.4f}",
)

# Vanilla TabPFN embeddings (n_fold=0).
# fit_transform returns embeddings for the training set; transform handles
# unseen data using the same full-data model.
embedding = TabPFNEmbedding(
    n_fold=0,
    model=TabPFNClassifier(n_estimators=1, random_state=42),
)
train_embeddings = embedding.fit_transform(X_train, y_train)
test_embeddings = embedding.transform(X_test)

# TabPFN embeddings are shaped (n_estimators, n_samples, embed_dim); pick the
# first ensemble member for a 2D matrix that sklearn estimators accept.
model = LogisticRegression(max_iter=5000)
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Logistic Regression with TabPFN (Vanilla) Accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}",
)

# K-fold cross-validated TabPFN embeddings (the robust variant). fit_transform
# returns OOF embeddings; transform on unseen data uses the final model
# trained on all of X_train.
embedding = TabPFNEmbedding(
    n_fold=10,
    model=TabPFNClassifier(n_estimators=1, random_state=42),
)
train_embeddings = embedding.fit_transform(X_train, y_train)  # OOF
test_embeddings = embedding.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Logistic Regression with TabPFN (K-Fold CV) Accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}",
)

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
# space_ga: 3107 samples, 6 features — a clean benchmark where TabPFN
# embeddings clearly outperform a plain Ridge regression baseline.
print("\nLoading regression dataset (space_ga from OpenML)...")
dataset = fetch_openml("space_ga", version=1, as_frame=False)
X, y = dataset["data"], dataset["target"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Baseline: vanilla Ridge regression on raw features
model = Ridge()
model.fit(X_train, y_train)
print(
    "Baseline Ridge Regression R² Score: "
    f"{r2_score(y_test, model.predict(X_test)):.4f}",
)

# Vanilla TabPFN embeddings
embedding = TabPFNEmbedding(
    n_fold=0,
    model=TabPFNRegressor(n_estimators=1, random_state=42),
)
train_embeddings = embedding.fit_transform(X_train, y_train)
test_embeddings = embedding.transform(X_test)

model = Ridge()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Ridge with TabPFN (Vanilla) R² Score: " f"{r2_score(y_test, y_pred):.4f}",
)

# K-fold cross-validated TabPFN embeddings
embedding = TabPFNEmbedding(
    n_fold=10,
    model=TabPFNRegressor(n_estimators=1, random_state=42),
)
train_embeddings = embedding.fit_transform(X_train, y_train)  # OOF
test_embeddings = embedding.transform(X_test)

model = Ridge()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Ridge with TabPFN (K-Fold CV) R² Score: " f"{r2_score(y_test, y_pred):.4f}",
)
