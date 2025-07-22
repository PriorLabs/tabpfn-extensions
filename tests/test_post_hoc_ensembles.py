"""Tests for the TabPFN Post-Hoc Ensembles (PHE) implementation.

This file tests the PHE implementations in tabpfn_extensions.post_hoc_ensembles.
"""

from __future__ import annotations
import os
import pytest

from conftest import FAST_TEST_MODE
from sklearn.utils.estimator_checks import check_estimator

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestAutoTabPFNClassifier(BaseClassifierTests):
    """Test AutoTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a PHE-based TabPFN classifier as the estimator."""
        # For PHE, we can make tests faster by limiting time and using minimal models
        max_time = 15 if FAST_TEST_MODE else 30  # Very limited time for fast testing

        # Minimize the model portfolio for faster testing
        phe_init_args = {}
        phe_fit_args = {
            "num_bag_folds": None,
            "num_bag_sets": None,
            "num_stack_levels": None,
            "fit_weighted_ensemble": False,
        }

        return AutoTabPFNClassifier(
            max_time=max_time,
            random_state=42,
            phe_init_args=phe_init_args,
            phe_fit_args=phe_fit_args,
            n_ensemble_models=5,
        )

    @pytest.mark.skip(reason="PHE models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for PHE."""
        pass

    @pytest.mark.skip(
        reason="AutoTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        os.environ["SK_COMPATIBLE_PRECISION"] = "True"
        raise_on_error = True
        nan_test = 9

        # Precision issues do not allow for such deterministic behavior as expected, thus retrying certain tests to show it can work.
        clf_non_deterministic_for_reasons = [
            31,
            30,
        ]

        for est, non_deterministic in [
            (AutoTabPFNClassifier(device="cuda"), clf_non_deterministic_for_reasons),
        ]:
            lst = []
            for i, x in enumerate(check_estimator(est, generate_only=True)):
                if (i == nan_test) and ("allow_nan" in x[0]._get_tags()):
                    # sklearn test does not check for the tag!
                    continue

                n_tests = 5
                while n_tests:
                    try:
                        x[1](x[0])
                    except Exception as e:
                        if i in non_deterministic:
                            n_tests -= 1
                            continue
                        if raise_on_error:
                            raise e
                        lst.append((i, x, e))
                    break


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestAutoTabPFNRegressor(BaseRegressorTests):
    """Test AutoTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a PHE-based TabPFN regressor as the estimator."""
        # For PHE, we can make tests faster by limiting time and using minimal models
        max_time = 15 if FAST_TEST_MODE else 30  # Very limited time for fast testing

        # Minimize the model portfolio for faster testing
        phe_init_args = {"verbosity": 0}
        phe_fit_args = {
            "num_bag_folds": 0,
            "num_bag_sets": 0,
            "num_stack_levels": 0,
            "fit_weighted_ensemble": False,
        }

        return AutoTabPFNRegressor(
            max_time=max_time,
            random_state=42,
            phe_init_args=phe_init_args,
            phe_fit_args=phe_fit_args,
            n_ensemble_models=5,
        )

    @pytest.mark.skip(reason="PHE models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for PHE."""
        pass

    #@pytest.mark.skip(
    #    reason="AutoTabPFN needs additional work to pass all sklearn estimator checks",
    #)
    def test_passes_estimator_checks(self, estimator):
        os.environ["SK_COMPATIBLE_PRECISION"] = "True"
        raise_on_error = True
        nan_test = 9

        # Precision issues do not allow for such deterministic behavior as expected, thus retrying certain tests to show it can work.
        reg_non_deterministic_for_reasons = [
            27,
            28,
        ]

        for est, non_deterministic in [
            (AutoTabPFNRegressor(device="cuda"), reg_non_deterministic_for_reasons),
        ]:
            lst = []
            for i, x in enumerate(check_estimator(est, generate_only=True)):
                if (i == nan_test) and ("allow_nan" in x[0]._get_tags()):
                    # sklearn test does not check for the tag!
                    continue

                n_tests = 5
                while n_tests:
                    try:
                        x[1](x[0])
                    except Exception as e:
                        if i in non_deterministic:
                            n_tests -= 1
                            continue
                        if raise_on_error:
                            raise e
                        lst.append((i, x, e))
                    break

    @pytest.mark.skip(
        reason="AutoTabPFNRegressor can't handle text features with float64 dtype requirement",
    )
    def test_with_text_features(self, estimator, dataset_generator):
        pass


# Additional PHE-specific tests
class TestPHESpecificFeatures:
    """Test PHE-specific features that aren't covered by the base tests."""

    pass
