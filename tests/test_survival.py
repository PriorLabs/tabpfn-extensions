from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
# --- Import dependencies ---
from sksurv.base import SurvivalAnalysisMixin

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.survival import SurvivalTabPFN


# A tiny, reusable synthetic dataset
@pytest.fixture
def tiny_data():
    """Provides a small, consistent dataset for testing."""
    X = np.array(
        [
            [0.5, 0.2],
            [0.8, 0.1],
            [0.3, 0.9],
            [0.6, 0.4],
            [0.1, 0.7],
            [0.9, 0.8],
            [0.2, 0.3],
            [0.7, 0.6],
            [0.4, 0.5],
            [0.0, 1.0],
        ]
    )

    y_data = [
        (True, 10.0),  # event
        (False, 20.0),  # censored
        (True, 5.0),  # event
        (True, 15.0),  # event
        (False, 8.0),  # censored
        (True, 1.0),  # event
        (False, 3.0),  # censored
        (True, 12.0),  # event
        (True, 6.0),  # event
        (False, 7.0),  # censored
    ]

    # Convert to sksurv's structured array format
    y = np.array(y_data, dtype=[("event", "bool"), ("time", "f8")])
    return X, y


# --- Tests ---


def test_model_initialization():
    """Tests if the model initializes correctly."""
    model = SurvivalTabPFN(random_state=42)

    # Check that it's a valid sksurv model
    assert isinstance(model, SurvivalAnalysisMixin)

    # Check that it correctly initializes the internal models
    assert isinstance(model.cls_model, TabPFNClassifier)
    assert isinstance(model.reg_model, TabPFNRegressor)

    # Check that parameters are passed down
    assert model.cls_model.random_state == 42
    assert model.reg_model.random_state == 42

    # Check sksurv API property
    assert model._predict_risk_score is True


def test_model_predict(tiny_data):
    """Tests if 'predict' returns risk scores of the correct shape and type."""
    X, y = tiny_data
    model = SurvivalTabPFN(random_state=42).fit(X, y)

    # Test prediction on training data
    preds_train = model.predict(X)
    assert preds_train.shape == (X.shape[0],)
    assert np.issubdtype(preds_train.dtype, np.floating)

    # Test prediction on new, unseen data
    X_test = np.random.rand(5, X.shape[1])  # 5 new samples
    preds_test = model.predict(X_test)
    assert preds_test.shape == (5,)
    assert np.issubdtype(preds_test.dtype, np.floating)


def test_model_score(tiny_data):
    """Tests the 'score' method for sksurv API compliance."""
    X, y = tiny_data
    model = SurvivalTabPFN(random_state=42).fit(X, y)

    # .score() will call .predict() and then run concordance_index_censored
    score = model.score(X, y)

    assert isinstance(score, float)
    # Concordance index should be between 0.0 and 1.0
    assert 0.0 <= score <= 1.0


def test_reproducibility(tiny_data):
    """Tests that random_state ensures deterministic predictions."""
    X, y = tiny_data
    X_test = np.random.rand(5, X.shape[1])

    # First model and prediction
    model_1 = SurvivalTabPFN(random_state=42)
    model_1.fit(X, y)
    preds_1 = model_1.predict(X_test)

    # Second model and prediction with the same seed
    model_2 = SurvivalTabPFN(random_state=42)
    model_2.fit(X, y)
    preds_2 = model_2.predict(X_test)

    # Assert that the outputs are identical
    assert_array_equal(preds_1, preds_2)
