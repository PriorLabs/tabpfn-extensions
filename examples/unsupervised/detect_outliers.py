#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import torch
from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import unsupervised

# Try to import TabPFN from different sources
try:
    # Try the standard TabPFN package first
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    print("Using TabPFN package")
except ImportError:
    try:
        # Try the TabPFN client as fallback
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor

        print("Using TabPFN client")
    except ImportError:
        # Last resort - try old tabpfn_extensions imports
        from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor

        print("Using TabPFN from extensions")

# Load data
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]
attribute_names = df["feature_names"]

# Initialize models with compatible parameters
# Using defaults for compatibility between TabPFN and client
clf = TabPFNClassifier()
reg = TabPFNRegressor()
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf, tabpfn_reg=reg,
)

# Run outlier detection
exp_outlier = unsupervised.experiments.OutlierDetectionUnsupervisedExperiment(
    task_type="unsupervised",
)
results = exp_outlier.run(
    tabpfn=model_unsupervised,
    X=torch.tensor(X),
    y=torch.tensor(y),
    attribute_names=attribute_names,
    indices=[4, 12],  # Analyze features 4 and 12
)
