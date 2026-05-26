"""Tests for the CP_missing_data extension.

This file tests the CPMDATabPFNRegressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import train_test_split

try:
    from tabpfn_extensions.cp_missing_data import (
        CPMDATabPFNRegressor,
    )
except ImportError:
    pytest.skip("Required libraries (tabpfn) not installed", allow_module_level=True)


# -------- Fixtures --------
@pytest.fixture
def X_train():
    return np.array(
        [
            [0.1, np.nan],
            [0.3, 0.4],
            [np.nan, 0.6],
            [0.7, 0.8],
            [0.2, np.nan],
            [0.2, np.nan],
            [0.9, 0.4],
            [np.nan, 0.4],
            [0.3, 0.2],
            [np.nan, 0.9],
            [0.8, np.nan],
            [0.1, 0.2],
            [np.nan, 0.5],
            [0.3, 0.7],
            [0.7, np.nan],
            [0.7, np.nan],
            [0.3, 0.4],
            [np.nan, 0.2],
            [0.9, 0.7],
            [np.nan, 0.3],
            [0.3, 0.7],
            [0.4, 0.8],
            [0.5, 0.4],
            [0.7, 0.2],
            [0.8, 0.3],
        ]
    )


@pytest.fixture
def Y_train():
    return np.array(
        [1, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 2, 3, 5, 6, 8, 4, 2, 1, 2, 3]
    )


@pytest.fixture
def X_new():
    return np.array(
        [
            [0.1, 0.1],
            [0.3, np.nan],
            [np.nan, 0.6],
        ]
    )


@pytest.fixture
def seed():
    return 123


# -- Test --
def test_fit_returns_self(X_train, Y_train, seed):
    """fit() should return the estimator itself for sklearn compatibility."""
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    result = model.fit(X_train, Y_train)
    assert result is model


def test_model_CP(X_train, Y_train, seed):
    """Tests if the calibration corrections are of the correct shape and type."""
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    model.fit(X_train, Y_train)

    # Replicate the split to get the validation set and find its unique masks.
    _, X_val, _, _ = train_test_split(
        X_train, Y_train, test_size=0.5, random_state=seed
    )
    missing_df = pd.DataFrame(X_val).isna().astype(int).drop_duplicates()

    # check type, size of the calibration results
    assert model.calibration_results_.shape[0] == missing_df.shape[0]
    assert model.calibration_results_.shape[1] == 5
    assert isinstance(model.calibration_results_, pd.DataFrame)


def test_reproducibility(X_train, Y_train, seed):
    """Tests that random_state ensures deterministic correction terms."""
    model_1 = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    model_1.fit(X_train, Y_train)

    # Second model with the same seed
    model_2 = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    model_2.fit(X_train, Y_train)

    # Assert that the outputs are identical
    assert_array_equal(model_1.calibration_results_, model_2.calibration_results_)


def test_predict(X_train, Y_train, seed, X_new):
    """Tests if the predictions have the correct shape and type."""
    # fit model
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    model.fit(X_train, Y_train)

    # Apply the model to new cases
    CP_results = model.predict(X_new)

    assert CP_results[1].size == X_new.shape[0]
    assert isinstance(CP_results[1], np.ndarray)
    assert len(CP_results) == 5


def test_validate_quantiles_wrong_length(seed):
    """_validate_quantiles should raise if quantiles does not have 3 elements."""
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.95], val_size=0.5, seed=seed)
    with pytest.raises(ValueError, match="exactly 3 elements"):
        model._validate_quantiles([0.05, 0.95])


def test_validate_quantiles_not_symmetric(seed):
    """_validate_quantiles should raise if quantiles are not symmetric."""
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    with pytest.raises(ValueError, match="symmetric"):
        model._validate_quantiles([0.05, 0.5, 0.96])


def test_validate_quantiles_not_increasing(seed):
    """_validate_quantiles should raise if quantiles are not strictly increasing."""
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size=0.5, seed=seed)
    with pytest.raises(ValueError, match="strictly increasing"):
        model._validate_quantiles([0.5, 0.3, 0.5])
