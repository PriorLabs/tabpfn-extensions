from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_data_with_categorical(monkeypatch):
    """Test generating synthetic data with categorical features."""
    monkeypatch.setenv("FAST_TEST_MODE", "1")
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    clf = TabPFNClassifier(n_estimators=1, random_state=0)
    reg = TabPFNRegressor(n_estimators=1, random_state=0)
    model_unsup = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=clf,
        tabpfn_reg=reg,
    )
    #X[:, 0] = (X[:, 0] > X[:, 0].mean()).astype(int)
    model_unsup.set_categorical_features([0])
    model_unsup.fit(X)

    n_samples = 10
    synthetic_X = model_unsup.generate_synthetic_data(n_samples=n_samples)

    assert isinstance(synthetic_X, torch.Tensor)
    assert synthetic_X.shape == (n_samples, X.shape[1])
