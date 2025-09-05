"""Tests for the TabPFN Post-Hoc Ensembles (PHE) implementation.

This file tests the PHE implementations in tabpfn_extensions.post_hoc_ensembles.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
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
        check_estimator(estimator_instance, generate_only=True),
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

    def test_ignore_pretraining_limits_allows_large_dataset(self, monkeypatch):
        """Training should succeed on >10k rows when limits are ignored."""
        captured_hps: dict = {}
        captured_rows: dict = {}

        class DummyPredictor:
            def __init__(self, *args, **kwargs):
                pass

            def fit(
                self,
                *,
                train_data,
                time_limit,
                presets,
                hyperparameters,
                num_gpus,
                **kwargs,
            ):
                self.train_data = train_data
                captured_rows["n"] = len(train_data)
                captured_hps.update(hyperparameters)
                from autogluon.tabular.models import TabPFNV2Model

                if len(train_data) > 10000 and not hyperparameters[TabPFNV2Model][
                    "ag_args"
                ].get(
                    "ignore_constraints",
                    False,
                ):
                    raise AssertionError("ag.max_rows=10000 limit triggered")
                return self

            def features(self):
                return [c for c in self.train_data.columns if c != "_target_"]

            def predict(self, X):
                return pd.Series(np.zeros(len(X)))

        monkeypatch.setattr(
            "autogluon.tabular.TabularPredictor",
            DummyPredictor,
        )

        # Create dataset slightly above the 10k limit
        X = pd.DataFrame(np.random.randn(10050, 2), columns=["a", "b"])
        y = pd.Series(np.random.randn(10050))

        fit_kwargs = {
            "max_time": 1,
            "device": "cpu",
            "n_ensemble_models": 1,
            "n_estimators": 1,
            "phe_fit_args": {
                "num_bag_folds": 0,
                "num_stack_levels": 0,
                "fit_weighted_ensemble": False,
                "ag_args_ensemble": {},
            },
        }

        model = AutoTabPFNRegressor(ignore_pretraining_limits=True, **fit_kwargs)
        model.fit(X, y)

        from autogluon.tabular.models import TabPFNV2Model

        assert captured_rows["n"] > 10000
        assert captured_hps[TabPFNV2Model]["ag_args"]["ignore_constraints"] is True

        model_no_flag = AutoTabPFNRegressor(
            ignore_pretraining_limits=False,
            **fit_kwargs,
        )
        with pytest.raises(AssertionError):
            model_no_flag.fit(X, y)
