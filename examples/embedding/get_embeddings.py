from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, TabPFNEmbedding

# Load and evaluate classification dataset
print("Loading classification dataset (kc1)...")
X, y = fetch_openml(name='kc1', version=1, as_frame=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train and evaluate vanilla logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Baseline Logistic Regression Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

# Train and evaluate TabPFN embeddings (vanilla)
clf = TabPFNClassifier(n_estimators=1,random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
train_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="train")
test_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="test")

model = LogisticRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(f"Logistic Regression with TabPFN (Vanilla) Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# Train and evaluate TabPFN embeddings (K-fold cross-validation)
clf = TabPFNClassifier(n_estimators=1,random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)
train_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="train")
test_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="test")

model = LogisticRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(f"Logistic Regression with TabPFN (K-Fold CV) Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Load and evaluate regression dataset
print("\nLoading regression dataset (kin8nm)...")
X,y = fetch_openml(name='kin8nm' , version=1, as_frame=False, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train and evaluate vanilla linear regression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Baseline Linear Regression R2 Score: {r2_score(y_test, model.predict(X_test)):.4f}")

# Train and evaluate TabPFN embeddings (vanilla)
reg = TabPFNRegressor(n_estimators=1,random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=0)
train_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="train")
test_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="test")

model = LinearRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(f"Linear Regression with TabPFN (Vanilla) R2 Score: {r2_score(y_test, y_pred):.4f}")


# Train and evaluate TabPFN embeddings (K-fold cross-validation)
reg = TabPFNRegressor(n_estimators=1,random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=10)
train_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="train")
test_embeddings = embedding_extractor.get_embeddings(X_train,y_train,X_test,data_source="test")

model = LinearRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])

print(f"Linear Regression with TabPFN (K-Fold CV) R2 Score: {r2_score(y_test, y_pred):.4f}")