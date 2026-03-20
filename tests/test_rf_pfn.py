"""Tests for the TabPFN-based Random Forest implementation.

This file tests the RF-PFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import random
from typing_extensions import override

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

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
        X, Y = make_classification(
            n_samples=5000, n_features=500, n_redundant=0, random_state=1
        )

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
            random_state=12,
        )

        rf_clf_1.fit(X, Y)
        rf_clf_2.fit(X, Y)
        rf_clf_3.fit(X, Y)

        check_random_decision_path(X, rf_clf_1, rf_clf_2, rf_clf_3)
        assert_tree_path(rf_clf_1, X)
        assert_tree_path(rf_clf_2, X)
        assert_tree_path(rf_clf_3, X)


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
        X, Y = make_regression(n_samples=5000, n_features=500, random_state=1)

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
            random_state=29,
        )

        rf_reg_1.fit(X, Y)
        rf_reg_2.fit(X, Y)
        rf_reg_3.fit(X, Y)

        check_random_decision_path(X, rf_reg_1, rf_reg_2, rf_reg_3)
        assert_tree_path(rf_reg_1, X)
        assert_tree_path(rf_reg_2, X)
        assert_tree_path(rf_reg_3, X)


def check_random_decision_path(
    X, estimator_same_seed_1, estimator_same_seed_2, estimator_diff_seed
):
    """Test that random_state properly controls decision paths across all estimators.

    This test verifies:
    1. Same random_state produces identical decision paths for all estimators
    2. Different random_state produces different decision paths for all estimators
    """
    n_estimators = len(estimator_same_seed_1.estimators_)

    # Check all estimators
    for i in range(n_estimators):
        # Get decision paths for both instances with same random_state
        decision_path_1 = estimator_same_seed_1.estimators_[i].decision_path(X)
        decision_path_2 = estimator_same_seed_2.estimators_[i].decision_path(X)

        # Assert same random_state produces same decision paths for each estimator
        assert safe_sparse_equal(
            decision_path_1, decision_path_2
        ), f"Same random_state should produce identical decision paths for estimator {i}"

        # Get decision path for estimator with different random_state
        decision_path_3 = estimator_diff_seed.estimators_[i].decision_path(X)
        # Assert different random_state produces different decision paths for each estimator
        assert not safe_sparse_equal(
            decision_path_1, decision_path_3
        ), f"Same random_state should produce different decision paths for estimator {i}"


def assert_tree_path(rf_clf, X):
    """Test that each estimator in a random forest has different decision paths.

    Since each tree now receives a deterministically different random seed,
    they should produce different tree structures and thus different decision paths.

    Parameters
    ----------
    rf_clf : RandomForestTabPFNClassifier or RandomForestTabPFNRegressor
        A fitted random forest estimator with multiple estimators.
    X : array-like
        The input samples to compute decision paths for.
    """
    # Collect all decision paths
    decision_paths = [
        rf_clf.estimators_[i].decision_path(X) for i in range(len(rf_clf.estimators_))
    ]

    # Verify all pairs of estimators have different decision paths
    for i in range(len(decision_paths)):
        for j in range(i + 1, len(decision_paths)):
            assert not safe_sparse_equal(decision_paths[i], decision_paths[j])


def safe_sparse_equal(a, b):
    # sparse array of different shape returns bool instead of sparse array.
    # This utility function handles all the cases for the safe sparse matrix comparison
    if a.shape != b.shape:
        return False
    return (a != b).nnz == 0
