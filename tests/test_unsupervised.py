from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_data_with_categorical(monkeypatch):
    monkeypatch.setenv("FAST_TEST_MODE", "1")
    n = 20
    X = np.column_stack([np.random.randint(0, 3, size=n), np.random.randn(n)])

    clf = TabPFNClassifier(n_estimators=1, random_state=0)
    reg = TabPFNRegressor(n_estimators=1, random_state=0)
    model_unsup = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=clf,
        tabpfn_reg=reg,
    )
    model_unsup.set_categorical_features([0])
    model_unsup.fit(X)

    n_samples = 10
    synthetic_X = model_unsup.generate_synthetic_data(n_samples=n_samples)

    assert isinstance(synthetic_X, torch.Tensor)
    assert synthetic_X.shape == (n_samples, X.shape[1])
