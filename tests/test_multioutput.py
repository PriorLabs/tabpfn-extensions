"""Tests for the multi-output TabPFN wrappers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

from tabpfn_extensions.multioutput import (
    TabPFNMultiOutputRegressor,
)


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_multioutput_regression_with_missing_values(backend):
    """Multi-output regression should handle missing values transparently."""
    X, y = make_regression(
        n_samples=40,
        n_features=5,
        n_targets=2,
        n_informative=5,
        noise=0.1,
        random_state=0,
    )

    model = TabPFNMultiOutputRegressor()

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()

    score = r2_score(y, predictions, multioutput="uniform_average")
    assert score > 0.3


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_multioutput_regression_default_estimator_clone(backend):
    """TabPFN kwargs should be cloneable when estimator is created internally."""
    rng = np.random.default_rng(321)
    X, y = make_regression(
        n_samples=30,
        n_features=4,
        n_targets=2,
        n_informative=4,
        noise=0.2,
        random_state=1,
    )

    X[rng.random(X.shape) < 0.15] = np.nan

    model = TabPFNMultiOutputRegressor()

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()

    cloned_model = clone(model)
    cloned_model.fit(X, y)
    cloned_predictions = cloned_model.predict(X)

    assert cloned_predictions.shape == y.shape
    assert np.isfinite(cloned_predictions).all()

    cloned_score = r2_score(y, cloned_predictions, multioutput="uniform_average")
    assert cloned_score > 0.2
