from __future__ import annotations

from typing import ClassVar

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


class _DeviceSpyTensor(torch.Tensor):
    """Tensor that records the keyword arguments of `.to`, to test device handling."""

    to_kwargs: ClassVar[dict] = {}

    def to(self, *args, **kwargs):
        type(self).to_kwargs = dict(kwargs)
        return torch.Tensor.to(torch.as_tensor(self), *args, **kwargs)


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_impute_single_permutation_moves_samples_to_input_device(monkeypatch):
    """Sampled values must be moved to the device of the table they fill.

    With tabpfn>=9.1 the regressor returns its samples on the model device (e.g.
    CUDA) while the table being imputed lives on the CPU. Assigning them without
    a device move raised "Expected all tensors to be on the same device".
    """
    monkeypatch.setenv("FAST_TEST_MODE", "1")
    X = torch.randn(20, 2)
    model = unsupervised.TabPFNUnsupervisedModel(
        TabPFNClassifier(n_estimators=1), TabPFNRegressor(n_estimators=1)
    )
    # Bypass fit and density_: this test needs no model weights.
    model.X_ = X

    def fake_density(_self, X_nan, *_args):
        return None, X_nan, None

    monkeypatch.setattr(unsupervised.TabPFNUnsupervisedModel, "density_", fake_density)

    def fake_sample(self, column_idx, X_fit, model_, X_predict, t):
        sampled = torch.zeros(X_predict.shape[0]).as_subclass(_DeviceSpyTensor)
        return torch.zeros(X_predict.shape[0], 1), sampled

    monkeypatch.setattr(
        unsupervised.TabPFNUnsupervisedModel,
        "sample_from_model_prediction_",
        fake_sample,
    )
    X_missing = X.clone()
    X_missing[0, 0] = float("nan")
    _DeviceSpyTensor.to_kwargs = {}
    model.impute_single_permutation_(X_missing, [0, 1])
    assert _DeviceSpyTensor.to_kwargs.get("device") == X_missing.device
