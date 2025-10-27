#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
"""

from sklearn.model_selection import train_test_split
from sksurv.datasets import load_breast_cancer
from sksurv.metrics import concordance_index_censored

from tabpfn_extensions.survival_pfn import SurvivalTabPFN

X, y = load_breast_cancer()
test_size = 0.20

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
)

# Create and fit classifier with appropriate settings
model = TabPFNSurvivalAnalysis(random_state=42)
model.fit(X_train, y_train)
risk_scores = model.predict(X_test)
score = concordance_index_censored(
    [n[0] for n in y_test],
    [n[1] for n in y_test],
    risk_scores,
)[0]
print("Concordance index score:", score)

# Alternatively, it is possible to predict and score in single function call
print("Concordance index score (via score method):", model.score(X_test, y_test))

# Expected output, approximately:
# Concordance index score: 0.71907
