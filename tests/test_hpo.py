"""Tests for the TabPFN hyperparameter optimization (HPO) implementation.

This file tests the HPO implementations in tabpfn_extensions.hpo.
"""

from __future__ import annotations

import pytest

from conftest import FAST_TEST_MODE
from utils import get_small_test_search_space

# Try to import the HPO module, but skip tests if hyperopt is not available
try:
    from tabpfn_extensions.hpo import TunedTabPFNClassifier, TunedTabPFNRegressor

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    pytest.skip(
        "hyperopt not installed. Install with 'pip install \"tabpfn-extensions[hpo]\"'",
        allow_module_level=True,
    )

from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests

# Using get_small_test_search_space from tests.utils - all HPO modules should use it


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTunedTabPFNClassifier(BaseClassifierTests):
    """Test TunedTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a HPO-based TabPFN classifier as the estimator."""
        n_trials = 3 if FAST_TEST_MODE else 10  # Very limited trials for fast testing

        # Use minimal search space in fast test mode
        search_space = get_small_test_search_space() if FAST_TEST_MODE else None

        return TunedTabPFNClassifier(
            n_trials=n_trials,
            metric="accuracy",
            random_state=42,
            search_space=search_space,
        )

    @pytest.mark.skip(reason="Tuned TabPFN models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for HPO."""
        pass

    @pytest.mark.skip(
        reason="TunedTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass


class TestTunedTabPFNClassifierROCAUC(BaseClassifierTests):
    """Test TunedTabPFNClassifier with ROC AUC metric."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a HPO-based TabPFN classifier as the estimator."""
        n_trials = 3 if FAST_TEST_MODE else 10  # Very limited trials for fast testing

        # Use minimal search space in fast test mode
        search_space = get_small_test_search_space() if FAST_TEST_MODE else None

        return TunedTabPFNClassifier(
            n_trials=n_trials,
            metric="roc_auc",
            random_state=42,
            search_space=search_space,
        )


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTunedTabPFNRegressor(BaseRegressorTests):
    """Test TunedTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a HPO-based TabPFN regressor as the estimator."""
        n_trials = 2  # Very limited trials for fast testing

        # Use minimal search space in fast test mode
        search_space = get_small_test_search_space() if FAST_TEST_MODE else None

        return TunedTabPFNRegressor(
            n_trials=n_trials,
            metric="rmse",
            random_state=42,
            search_space=search_space,
        )

    @pytest.mark.skip(reason="Tuned TabPFN models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for HPO."""
        pass

    @pytest.mark.skip(
        reason="TunedTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass
