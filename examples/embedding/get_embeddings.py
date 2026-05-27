"""TabPFN Embedding Example

This example demonstrates how to extract embeddings from TabPFN models and use them
for classification and regression tasks via the scikit-learn style
``TabPFNEmbedding`` transformer.

NOTE: This example requires the full TabPFN implementation (pip install tabpfn).
It will not work with the TabPFN client (pip install tabpfn-client) because
the embedding functionality is not available in the client version.
"""

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

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
    X, y, test_size=0.5, random_state=42,
)

# Baseline: vanilla logistic regression
model = LogisticRegression(max_iter=1000)
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
model = LogisticRegression(max_iter=1000)
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

model = LogisticRegression(max_iter=1000)
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Logistic Regression with TabPFN (K-Fold CV) Accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}",
)

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42,
)

# Baseline: vanilla linear regression
model = LinearRegression()
model.fit(X_train, y_train)
print(
    "Baseline Linear Regression R2 Score: "
    f"{r2_score(y_test, model.predict(X_test)):.4f}",
)

# Vanilla TabPFN embeddings
embedding = TabPFNEmbedding(
    n_fold=0,
    model=TabPFNRegressor(n_estimators=1, random_state=42),
)
train_embeddings = embedding.fit_transform(X_train, y_train)
test_embeddings = embedding.transform(X_test)

model = LinearRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Linear Regression with TabPFN (Vanilla) R2 Score: "
    f"{r2_score(y_test, y_pred):.4f}",
)

# K-fold cross-validated TabPFN embeddings
embedding = TabPFNEmbedding(
    n_fold=10,
    model=TabPFNRegressor(n_estimators=1, random_state=42),
)
train_embeddings = embedding.fit_transform(X_train, y_train)  # OOF
test_embeddings = embedding.transform(X_test)

model = LinearRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    "Linear Regression with TabPFN (K-Fold CV) R2 Score: "
    f"{r2_score(y_test, y_pred):.4f}",
)
