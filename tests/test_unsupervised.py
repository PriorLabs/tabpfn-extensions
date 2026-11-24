from __future__ import annotations

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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

    # Fixed seed: a constant column 0 leaves the column-1 model no usable feature.
    X = np.random.default_rng(0).integers(5, size=(5, 2))
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


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_outliers_accepts_array_logits(monkeypatch):
    """`logits` is a tensor from the local package and an array from the client."""
    from tabpfn.regressor import FullSupportBarDistribution

    n_rows, n_features, n_bars = 6, 2, 8
    # float64 throughout, as the client's own criterion and logits are.
    criterion = FullSupportBarDistribution(
        borders=torch.linspace(-3.0, 3.0, n_bars + 1, dtype=torch.float64)
    )

    class ArrayLogitsRegressor:
        def predict(self, X, output_type):
            assert output_type == "full"
            return {
                "logits": np.zeros((len(X), n_bars)),
                "criterion": criterion,
            }

    X = torch.randn(n_rows, n_features)
    model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=None, tabpfn_reg=ArrayLogitsRegressor()
    )
    model.X_ = X

    def density(X_predict, _X_fit, _conditional_idx, column_idx):
        return model.tabpfn_reg, X_predict, X_predict[:, column_idx]

    monkeypatch.setattr(model, "density_", density)

    log_p = model.outliers_single_permutation_(X, feature_permutation=[0, 1])

    assert log_p.shape == (n_rows,)
    assert torch.isfinite(log_p).all()


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_impute_single_permutation_moves_samples_to_input_device(monkeypatch):
    """Samples on another device than the imputed table must not raise."""
    # A fake CUDA tensor fails the device check like a real one, without a GPU.
    with FakeTensorMode(allow_non_fake_inputs=True):
        sample_on_cuda = torch.zeros(2, device="cuda")
    X = torch.randn(20, 2)
    model = unsupervised.TabPFNUnsupervisedModel(
        TabPFNClassifier(n_estimators=1), TabPFNRegressor(n_estimators=1)
    )
    model.X_ = X
    monkeypatch.setattr(
        unsupervised.TabPFNUnsupervisedModel,
        "density_",
        lambda _self, X_nan, *_: (None, X_nan, None),
    )
    monkeypatch.setattr(
        unsupervised.TabPFNUnsupervisedModel,
        "sample_from_model_prediction_",
        lambda *_: (None, sample_on_cuda),
    )
    X_missing = X.clone()
    # Two NaNs: a one-element value on another device does not raise.
    X_missing[:2, 0] = float("nan")

    model.impute_single_permutation_(X_missing, [0, 1])
