"""Tests for the CP_missing_data extension.

This file tests the CPMDATabPFNRegressor and CPMDATabPFNRegressorNewData functions,
which attempts to obtain correct uncertainity estimates in case if missing data. 
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
        CPMDATabPFNRegressorNewData,
    )
except ImportError:
    pytest.skip("Required libraries (tabpfn) not installed", allow_module_level=True)

# -------- Fixtures --------

@pytest.fixture
def X_train():
    return np.array([
        [0.1, np.nan], [0.3, 0.4], [np.nan, 0.6], [0.7, 0.8],
        [0.2, np.nan], [0.2, np.nan], [0.9, 0.4], [np.nan, 0.4],
        [0.3, 0.2], [np.nan, 0.9], [0.8, np.nan], [0.1, 0.2],
        [np.nan, 0.5], [0.3, 0.7], [0.7, np.nan], [0.7, np.nan],
        [0.3, 0.4], [np.nan, 0.2], [0.9, 0.7], [np.nan, 0.3],
        [0.3, 0.7], [0.4, 0.8], [0.5, 0.4], [0.7, 0.2], [0.8, 0.3],
    ])


@pytest.fixture
def Y_train():
    return np.array([1,3,1,2,3,4,5,6,1,2,3,4,5,6,7,2,3,5,6,8,4,2,1,2,3])


@pytest.fixture
def X_new():
    return np.array([
        [0.1, 0.1],
        [0.3, np.nan],
        [np.nan, 0.6],
    ])


@pytest.fixture
def seed():
    return 123


# -- Test --

def test_model_CP(X_train, Y_train, seed):
    """Tests if the calibration corrections are of the correct shape and type."""
    model = CPMDATabPFNRegressor(quantiles = [0.05, 0.5, 0.95], val_size = 0.5, seed = seed)
    calibration_results, model_fit = model.fit(X_train, Y_train)

    # Replicate the split to get the validation set and find its unique masks.
    _, X_val, _, _ = train_test_split(X_train, Y_train, test_size=0.5, random_state=seed)
    missing_df = pd.DataFrame(X_val).isnull().astype(int).drop_duplicates()

    # check type, size of the calibration results
    assert calibration_results.shape[0] == missing_df.shape[0]
    assert calibration_results.shape[1] == 5
    assert isinstance(calibration_results, pd.DataFrame)


def test_reproducibility(X_train, Y_train, seed):
    """Tests that random_state ensures deterministic correction terms."""

    model_1 = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size = 0.5, seed = seed)
    calibration_results_1, model_fit_1 = model_1.fit(X_train, Y_train)

    # Second model with the same seed
    model_2 = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95] , val_size = 0.5, seed = seed)
    calibration_results_2, model_fit_2 = model_2.fit(X_train, Y_train)

    # Assert that the outputs are identical
    assert_array_equal(calibration_results_1, calibration_results_2)


def test_predict(X_train, Y_train, seed, X_new):
    """Tests if the predictions have the correct shape and type."""

    # fit model 
    model = CPMDATabPFNRegressor(quantiles=[0.05, 0.5, 0.95], val_size = 0.5, seed = seed)
    calibration_results, model_fit = model.fit(X_train, Y_train)

    # Apply the model to new cases 
    cp_apply = CPMDATabPFNRegressorNewData(model_fit, quantiles=[0.05, 0.5, 0.95], calibration_results=calibration_results)
    CP_results = cp_apply.predict(X_new)

    assert CP_results[1].size== X_new.shape[0]
    assert isinstance(CP_results[1], np.ndarray)
    assert len(CP_results)== 5