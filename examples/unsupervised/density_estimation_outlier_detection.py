#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
The unsupervised model runs multiple TabPFN models for outlier detection.
"""

import torch
from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# Load data
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]
attribute_names = df["feature_names"]

clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
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
