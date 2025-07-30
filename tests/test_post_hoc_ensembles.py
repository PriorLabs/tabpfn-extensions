"""Tests for the TabPFN Post-Hoc Ensembles (PHE) implementation.

This file tests the PHE implementations in tabpfn_extensions.post_hoc_ensembles.
"""

from __future__ import annotations

import os

import pytest
from sklearn.utils.estimator_checks import check_estimator

from conftest import FAST_TEST_MODE
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


def _run_sklearn_estimator_checks(estimator_instance, non_deterministic_indices):
    """Helper to run scikit-learn's check_estimator with retries."""
    os.environ["SK_COMPATIBLE_PRECISION"] = "True"
    nan_test_index = 9

    for i, (name, check) in enumerate(
        check_estimator(estimator_instance, generate_only=True)
    ):
        if i == nan_test_index and "allow_nan" in estimator_instance._get_tags():
            continue

        n_retries = 5
        while n_retries > 0:
            try:
                check(estimator_instance)
                break  # Test passed
            except Exception as e:
                if i in non_deterministic_indices and n_retries > 1:
                    n_retries -= 1
                    continue
                # Raise the error on the last retry or for deterministic tests
                raise e


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestAutoTabPFNClassifier(BaseClassifierTests):
    """Test AutoTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a PHE-based TabPFN classifier as the estimator."""
        # For PHE, we can make tests faster by limiting time and using minimal models
        # NOTE: If max_time is set too low, AutoGluon will fail to fit any models during
        # the fit() call. This is especially true when building a TabPFN-only ensemble
        # and can be hard to debug as it may only fail on certain CI hardware.
        max_time = 10 if FAST_TEST_MODE else 20  # Very limited time for fast testing

        # Minimize the model portfolio for faster testing
        phe_init_args = {"verbosity": 1}
        phe_fit_args = {
            "num_bag_folds": 0,  # Disable bagging
            "num_bag_sets": 1,  # Minimal value for bagging sets
            "num_stack_levels": 0,  # Disable stacking
            "fit_weighted_ensemble": False,
            "ag_args_ensemble": {},
        }

        return AutoTabPFNClassifier(
            max_time=max_time,
            random_state=42,
            phe_init_args=phe_init_args,
            phe_fit_args=phe_fit_args,
            n_ensemble_models=3,
        )

    @pytest.mark.skip(reason="PHE models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for PHE."""
        pass

    @pytest.mark.skip(
        reason="Not fully compatible with sklearn estimator checks yet, TODO",
    )
    def test_passes_estimator_checks(self, estimator):
        clf_non_deterministic = [30, 31]
        _run_sklearn_estimator_checks(estimator, clf_non_deterministic)

    @pytest.mark.skip(
        reason="AutoTabPFNClassifier can't handle text features with float64 dtype requirement",
    )
    def test_with_text_features(self, estimator, dataset_generator):
        pass


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestAutoTabPFNRegressor(BaseRegressorTests):
    """Test AutoTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a PHE-based TabPFN regressor as the estimator."""
        # For PHE, we can make tests faster by limiting time and using minimal models
        # NOTE: If max_time is set too low, AutoGluon will fail to fit any models during
        # the fit() call. This is especially true when building a TabPFN-only ensemble
        # and can be hard to debug as it may only fail on certain CI hardware.
        max_time = 10 if FAST_TEST_MODE else 20  # Very limited time for fast testing

        # Minimize the model portfolio for faster testing
        phe_init_args = {"verbosity": 1}
        phe_fit_args = {
            "num_bag_folds": 0,  # Disable bagging
            "num_bag_sets": 1,  # Minimal value for bagging sets
            "num_stack_levels": 0,  # Disable stacking
            "fit_weighted_ensemble": False,
            "ag_args_ensemble": {},
        }

        return AutoTabPFNRegressor(
            max_time=max_time,
            random_state=42,
            phe_init_args=phe_init_args,
            phe_fit_args=phe_fit_args,
            n_ensemble_models=3,
        )

    @pytest.mark.skip(reason="PHE models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for PHE."""
        pass

    @pytest.mark.skip(
        reason="Not fully compatible with sklearn estimator checks yet, TODO",
    )
    def test_passes_estimator_checks(self, estimator):
        reg_non_deterministic = [27, 28]
        _run_sklearn_estimator_checks(estimator, reg_non_deterministic)

    @pytest.mark.skip(
        reason="AutoTabPFNRegressor can't handle text features with float64 dtype requirement",
    )
    def test_with_text_features(self, estimator, dataset_generator):
        pass


# Additional PHE-specific tests
class TestPHESpecificFeatures:
    """Test PHE-specific features that aren't covered by the base tests."""

    pass
