from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_data_with_categorical():
    os.environ["FAST_TEST_MODE"] = "1"
    n = 20
    X = np.column_stack([np.random.randint(0, 3, size=n), np.random.randn(n)])
    attrs = ["cat", "cont"]

    clf = TabPFNClassifier(n_estimators=1, random_state=0)
    reg = TabPFNRegressor(n_estimators=1, random_state=0)
    model_unsup = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

    exp = unsupervised.experiments.GenerateSyntheticDataExperiment(
        task_type="unsupervised"
    )
    X_tensor = torch.tensor(X, dtype=torch.float32)
    categorical_features = torch.tensor([0], dtype=torch.long)

    exp.run(
        tabpfn=model_unsup,
        X=X_tensor,
        y=None,
        attribute_names=attrs,
        temp=1.0,
        n_samples=10,
        indices=np.arange(X_tensor.shape[1]),
        categorical_features=categorical_features,
    )

    assert isinstance(exp.synthetic_X, torch.Tensor)
    assert exp.synthetic_X.shape[1] == X_tensor.shape[1]
