#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
"""Example: causally-informed synthetic data via a DAG.

Shows how to pass a Directed Acyclic Graph of inter-feature dependencies to
``TabPFNUnsupervisedModel.generate_synthetic_data``. With a DAG, each feature
is generated in topological order and conditioned only on its declared
parents — useful for counterfactual generation or scenarios where the user
encodes domain knowledge about feature dependencies.

NB: the DAG used here is **illustrative only**, not a validated causal model
of wine chemistry. Replace it with your domain's actual DAG.
"""
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

df = load_wine(return_X_y=False)
X, y = df["data"], df["target"]
attribute_names = df["feature_names"]

# Illustrative DAG over wine features. Each key depends on the listed parents.
wine_dag_by_name = {
    "alcohol": [],
    "malic_acid": [],
    "ash": ["magnesium"],
    "alcalinity_of_ash": ["ash", "magnesium"],
    "magnesium": [],
    "total_phenols": ["flavanoids", "nonflavanoid_phenols", "proanthocyanins"],
    "flavanoids": [],
    "nonflavanoid_phenols": [],
    "proanthocyanins": [],
    "color_intensity": ["flavanoids", "proanthocyanins", "total_phenols"],
    "hue": ["color_intensity"],
    "od280/od315_of_diluted_wines": ["flavanoids", "total_phenols"],
    "proline": ["alcohol", "total_phenols"],
}

# Convert feature-name DAG → integer-index DAG (the public API expects ints)
name_to_idx = {n: i for i, n in enumerate(attribute_names)}
dag = {
    name_to_idx[child]: [name_to_idx[p] for p in parents]
    for child, parents in wine_dag_by_name.items()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)
model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

exp_synthetic = unsupervised.experiments.GenerateSyntheticDataExperiment(
    task_type="unsupervised",
)

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

results = exp_synthetic.run(
    tabpfn=model_unsupervised,
    X=X_tensor,
    y=y_tensor,
    attribute_names=attribute_names,
    temp=1.0,
    n_samples=X_train.shape[0] * 3,  # 3x original samples
    indices=list(range(X_train.shape[1])),
    n_permutations=3,
    dag=dag,
)
