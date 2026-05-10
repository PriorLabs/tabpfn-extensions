#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
The unsupervised model runs multiple TabPFN models for outlier detection.
"""

import torch
from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.unsupervised.experiments import (
    OutlierDetectionUnsupervisedExperiment,
)

# Load data
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]
attribute_names = df["feature_names"]

# Enable the v3 KV cache: outlier detection runs n_permutations * n_features
# predicts against the same fitted model, so caching the encoder pass over
# the training set avoids redoing it for each (permutation, feature) pair.
# TabPFNUnsupervisedModel sets keep_cache_on_device=True automatically after
# fit when the estimator was constructed with fit_mode="fit_with_cache".
clf = TabPFNClassifier(n_estimators=3, fit_mode="fit_with_cache")
reg = TabPFNRegressor(n_estimators=3, fit_mode="fit_with_cache")
model_unsupervised = TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

# Run outlier detection
exp_outlier = OutlierDetectionUnsupervisedExperiment(
    task_type="unsupervised",
)
results = exp_outlier.run(
    tabpfn=model_unsupervised,
    X=torch.tensor(X),
    y=torch.tensor(y),
    attribute_names=attribute_names,
    indices=[4, 12],  # Analyze features 4 and 12
)
