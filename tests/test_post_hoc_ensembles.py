"""Tests for the TabPFN Post-Hoc Ensembles (PHE) implementation.

This file tests the PHE implementations in tabpfn_extensions.post_hoc_ensembles.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.estimator_checks import check_estimator

from conftest import FAST_TEST_MODE
from tabpfn.model_loading import ModelVersion
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


class MockTabPFNClassifier:
    """Mock TabPFNClassifier that behaves like DummyClassifier."""

    def __init__(self, **kwargs):
        self._dummy_model = DummyClassifier(strategy="stratified")
        self._is_fitted = False
        self.ignore_pretraining_limits = kwargs["ignore_pretraining_limits"]
        # AG 1.5's TabPFN model class reads `devices_` via `get_device()`.
        self.devices_ = (torch.device("cpu"),)

    def fit(self, X, y, **kwargs):
        # Simulate the 10k row limit check
        if len(X) > 10000 and not self.ignore_pretraining_limits:
            raise AssertionError("training set size is above 10k rows")

        self._dummy_model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X, **kwargs):
        if not self._is_fitted:
            raise ValueError("This MockTabPFNClassifier instance is not fitted yet.")
        return self._dummy_model.predict(X)

    def predict_proba(self, X, **kwargs):
        if not self._is_fitted:
            raise ValueError("This MockTabPFNClassifier instance is not fitted yet.")
        return self._dummy_model.predict_proba(X)

    def to(self, device):  # pragma: no cover - invoked only by AG's _set_device
        return self


class MockTabPFNRegressor:
    """Mock TabPFNRegressor that behaves like DummyRegressor."""

    def __init__(self, **kwargs):
        self._dummy_model = DummyRegressor(strategy="mean")
        self._is_fitted = False
        self.ignore_pretraining_limits = kwargs["ignore_pretraining_limits"]
        # AG 1.5's TabPFN model class reads `devices_` via `get_device()`.
        self.devices_ = (torch.device("cpu"),)

    def fit(self, X, y, **kwargs):
        # Simulate the 10k row limit check
        if len(X) > 10000 and not self.ignore_pretraining_limits:
            raise AssertionError("training set size is above 10k rows")

        self._dummy_model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X, **kwargs):
        if not self._is_fitted:
            raise ValueError("This MockTabPFNRegressor instance is not fitted yet.")
        return self._dummy_model.predict(X)

    def to(self, device):  # pragma: no cover - invoked only by AG's _set_device
        return self


# Additional PHE-specific tests
class TestPHESpecificFeatures:
    """Test PHE-specific features that aren't covered by the base tests."""

    def test_ignore_pretraining_limits_allows_large_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Training should succeed on datasets beyond the base model's row
        limit when ``ignore_pretraining_limits=True``.

        We pin to v2 (10k row limit) so the assertion fires at a size that is
        also cheap to fit; the default v2.5 has a 100k limit which would
        require a much larger synthetic dataset to trigger.
        """
        # Patch TabPFN models
        monkeypatch.setattr(
            "tabpfn.TabPFNClassifier",
            MockTabPFNClassifier,
        )
        monkeypatch.setattr(
            "tabpfn.TabPFNRegressor",
            MockTabPFNRegressor,
        )

        # Create dataset above v2's 10k row limit
        X = pd.DataFrame(np.random.randn(20_000, 2), columns=["a", "b"])
        y = pd.Series(np.random.randn(20_000))

        # Test with ignore_pretraining_limits=True
        model = AutoTabPFNRegressor(
            ignore_pretraining_limits=True,
            max_time=5,
            model_version=ModelVersion.V2,
        )
        model.fit(X, y)

        # Test with ignore_pretraining_limits=False (should fail)
        model_no_flag = AutoTabPFNRegressor(
            ignore_pretraining_limits=False,
            max_time=5,
            model_version=ModelVersion.V2,
        )

        # TODO: would be better to check for the specific AssertionError
        # caught by AutoGluon:
        # AssertionError: ag.max_rows=10000 but...
        with pytest.raises(RuntimeError):
            model_no_flag.fit(X, y)

    @pytest.mark.parametrize(
        ("model_version", "expected_ag_name"),
        [
            # `ag_name` is the per-version identifier set by each AutoGluon
            # TabPFN model class -- AG uses it as the leaderboard prefix.
            (ModelVersion.V2, "RealTabPFN-v2"),
            (ModelVersion.V2_5, "RealTabPFN-v2.5"),
        ],
    )
    def test_routes_to_per_version_autogluon_class(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_version: ModelVersion,
        expected_ag_name: str,
    ):
        """``model_version`` should select the AutoGluon TabPFN model class
        whose per-version max_rows/max_features/max_classes limits match: V2
        -> RealTabPFNv2Model, V2_5 -> RealTabPFNv25Model. Mocks TabPFN at
        the leaf so AutoGluon runs end-to-end cheaply, then asserts on the
        leaderboard's TabPFN sub-model names (each is ``<ag_name>[_N]_BAG_L1``)
        that all of them came from the expected AG class -- catches both
        wrong-version routing and any leak of the other version's class
        into the ensemble.
        """
        monkeypatch.setattr("tabpfn.TabPFNClassifier", MockTabPFNClassifier)
        monkeypatch.setattr("tabpfn.TabPFNRegressor", MockTabPFNRegressor)

        X = pd.DataFrame(np.random.randn(40, 2), columns=["a", "b"])
        y = pd.Series([0, 1] * 20)

        clf = AutoTabPFNClassifier(
            model_version=model_version,
            n_ensemble_models=3,
            max_time=10,
            ignore_pretraining_limits=True,
        )
        clf.fit(X, y)

        # Each TabPFN sub-model is named "<ag_name>[_N]_BAG_L1"; parse the
        # ag_name back out by taking everything up to the first underscore.
        # This works because "RealTabPFN-v2" and "RealTabPFN-v2.5" differ at
        # position 13 (dot vs underscore) -- a v2.5 name does not split-prefix
        # to "RealTabPFN-v2".
        leaderboard_names = clf.predictor_.leaderboard(silent=True)["model"].tolist()
        tabpfn_ag_names = {
            name.split("_", 1)[0]
            for name in leaderboard_names
            if name.startswith("RealTabPFN-")
        }
        assert tabpfn_ag_names == {expected_ag_name}, (
            f"expected only {expected_ag_name!r} in trained TabPFN sub-models, "
            f"got {tabpfn_ag_names} (leaderboard: {leaderboard_names})"
        )
