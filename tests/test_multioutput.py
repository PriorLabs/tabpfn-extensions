"""Tests for the multi-output TabPFN wrappers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.metrics import f1_score, r2_score

from tabpfn_extensions.multioutput import (
    TabPFNMultiOutputClassifier,
    TabPFNMultiOutputRegressor,
)


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_multioutput_regression_with_missing_values(backend):
    """Multi-output regression should handle missing values transparently."""

    rng = np.random.default_rng(123)
    X, y = make_regression(
        n_samples=40,
        n_features=5,
        n_targets=2,
        n_informative=5,
        noise=0.1,
        random_state=0,
    )

    missing_mask = rng.random(X.shape) < 0.1
    X[missing_mask] = np.nan

    if backend == "tabpfn":
        from tabpfn_extensions.utils import LocalTabPFNRegressor as _Regressor
    elif backend == "tabpfn_client":
        from tabpfn_extensions.utils import ClientTabPFNRegressor as _Regressor
    else:  # pragma: no cover - defensive for unexpected fixtures
        pytest.fail(f"Unknown backend: {backend}")

    regressor_kwargs = {"n_estimators": 1}
    if backend == "tabpfn":
        regressor_kwargs["model_path"] = "tabpfn-v2-regressor.ckpt"

    estimator = _Regressor(**regressor_kwargs)
    model = TabPFNMultiOutputRegressor(estimator=estimator)

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

    regressor_kwargs = {"n_estimators": 1}
    if backend == "tabpfn":
        regressor_kwargs["model_path"] = "tabpfn-v2-regressor.ckpt"
    elif backend != "tabpfn_client":  # pragma: no cover - defensive for unexpected fixtures
        pytest.fail(f"Unknown backend: {backend}")

    model = TabPFNMultiOutputRegressor(**regressor_kwargs)

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


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_multioutput_classification_with_missing_values(backend):
    """Multi-output classification should support native TabPFN wrappers."""

    rng = np.random.default_rng(321)
    X, y = make_multilabel_classification(
        n_samples=60,
        n_features=5,
        n_classes=3,
        n_labels=2,
        allow_unlabeled=False,
        random_state=0,
    )

    X = X.astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan

    if backend == "tabpfn":
        from tabpfn_extensions.utils import LocalTabPFNClassifier as _Classifier
    elif backend == "tabpfn_client":
        from tabpfn_extensions.utils import ClientTabPFNClassifier as _Classifier
    else:  # pragma: no cover - defensive for unexpected fixtures
        pytest.fail(f"Unknown backend: {backend}")

    classifier_kwargs = {"n_estimators": 1}
    if backend == "tabpfn":
        classifier_kwargs["model_path"] = "tabpfn-v2-classifier.ckpt"

    estimator = _Classifier(**classifier_kwargs)
    model = TabPFNMultiOutputClassifier(estimator=estimator)

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()

    score = f1_score(y, predictions, average="micro")
    assert score > 0.5
