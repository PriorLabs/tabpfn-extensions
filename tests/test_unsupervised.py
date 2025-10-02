from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_data_mixed(monkeypatch):
    """Test generating synthetic data with categorical features."""
    monkeypatch.setenv("FAST_TEST_MODE", "1")
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    clf = TabPFNClassifier(n_estimators=1, random_state=0)
    reg = TabPFNRegressor(n_estimators=1, random_state=0)
    model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=clf,
        tabpfn_reg=reg,
    )
    X[:, 0] = (X[:, 0] > X[:, 0].mean()).astype(int)
    X = X[:, :3]  # Use only first 3 features for speed
    model.set_categorical_features([0])
    model.fit(X)

    n_samples = 10
    synthetic_X = model.generate_synthetic_data(n_samples=n_samples)

    assert isinstance(synthetic_X, torch.Tensor)
    assert synthetic_X.shape == (n_samples, X.shape[1])


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_data_categorical(monkeypatch):
    """Test generating synthetic data with categorical features."""
    monkeypatch.setenv("FAST_TEST_MODE", "1")

    X = np.random.randint(5, size=(5, 2))
    X_tensor = torch.tensor(X)

    tabpfn_clf = TabPFNClassifier(n_estimators=1)
    tabpfn_reg = TabPFNRegressor(n_estimators=1)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf, tabpfn_reg)
    model.set_categorical_features([0, 1])
    model.fit(X_tensor)
    n_samples = 10
    synthetic_X = model.generate_synthetic_data(n_samples=n_samples)

    assert isinstance(synthetic_X, torch.Tensor)
    assert synthetic_X.shape == (n_samples, X.shape[1])
