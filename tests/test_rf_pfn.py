"""Tests for the TabPFN-based Random Forest implementation.

This file tests the RF-PFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import random
from typing_extensions import override

import numpy as np
import pytest
import torch
from sklearn.datasets import load_diabetes, load_digits

from tabpfn_extensions.rf_pfn.sklearn_based_random_forest_tabpfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


class TestRandomForestClassifier(BaseClassifierTests):
    """Test RandomForestTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based RandomForestClassifier as the estimator."""
        return RandomForestTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            n_estimators=1,  # Use few trees for speed
            max_depth=2,  # Shallow trees for speed
            random_state=42,
            max_predict_time=5,  # Limit prediction time
        )

    @pytest.mark.skip(reason="RandomForestTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    @pytest.mark.skip(
        reason="RandomForestTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass

    def test_random_state(self, estimator, tabpfn_classifier):
        """Test that random_state properly controls randomness in RandomForestTabPFNClassifier.

        This test verifies:
        - Same random_state produces identical decision paths
        - Different random_state produces different decision paths
        """
        X_digits, y_digits = load_digits(return_X_y=True)

        rf_clf_1 = RandomForestTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            n_estimators=2,
            max_depth=3,
            random_state=42,
        )
        rf_clf_2 = RandomForestTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            n_estimators=2,
            max_depth=3,
            random_state=42,
        )
        rf_clf_3 = RandomForestTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            n_estimators=2,
            max_depth=3,
            random_state=123,
        )

        rf_clf_1.fit(X_digits, y_digits)
        rf_clf_2.fit(X_digits, y_digits)
        rf_clf_3.fit(X_digits, y_digits)

        test_random_decision_path(X_digits, y_digits, rf_clf_1, rf_clf_2, rf_clf_3)


class TestRandomForestRegressor(BaseRegressorTests):
    """Test RandomForestTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based RandomForestRegressor as the estimator."""
        return RandomForestTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            n_estimators=1,  # Use few trees for speed
            max_depth=2,  # Shallow trees for speed
            random_state=42,
            max_predict_time=5,  # Limit prediction time
        )

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    @override
    def test_with_pandas(self, estimator, pandas_regression_data):
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        super().test_with_pandas(estimator, pandas_regression_data)

    @pytest.mark.skip(reason="RandomForestTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    @pytest.mark.skip(
        reason="RandomForestTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass

    def test_random_state(self, estimator, tabpfn_regressor):
        """Test that random_state properly controls randomness in RandomForestTabPFNRegressor.

        This test verifies:
        - Same random_state produces identical decision paths
        - Different random_state produces different decision paths
        """

        X_diabetes, y_diabetes = load_diabetes(return_X_y=True)

        rf_reg_1 = RandomForestTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            n_estimators=2,
            max_depth=3,
            random_state=42,
        )
        rf_reg_2 = RandomForestTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            n_estimators=2,
            max_depth=3,
            random_state=42,
        )
        rf_reg_3 = RandomForestTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            n_estimators=2,
            max_depth=3,
            random_state=123,
        )

        rf_reg_1.fit(X_diabetes, y_diabetes)
        rf_reg_2.fit(X_diabetes, y_diabetes)
        rf_reg_3.fit(X_diabetes, y_diabetes)

        test_random_decision_path(X_diabetes, y_diabetes, rf_reg_1, rf_reg_2, rf_reg_3)


def test_random_decision_path(
    X, y, estimator_same_seed_1, estimator_same_seed_2, estimator_diff_seed
):
    # Get decision paths for both instances with same random_state
    decision_path_1 = estimator_same_seed_1.estimators_[0].decision_path(X)
    decision_path_2 = estimator_same_seed_2.estimators_[0].decision_path(X)

    # Assert same random_state produces same decision paths
    assert (
        decision_path_1.toarray() == decision_path_2.toarray()
    ).all(), "Same random_state should produce identical decision paths"

    # Get decision path for estimator with different random_state
    decision_path_3 = estimator_diff_seed.estimators_[0].decision_path(X)

    # Assert different random_state produces different decision paths
    assert not (
        decision_path_1.toarray() == decision_path_3.toarray()
    ).all(), "Different random_state should produce different decision paths"
