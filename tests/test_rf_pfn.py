"""Tests for the TabPFN-based Random Forest implementation.

This file tests the RF-PFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import random
from typing_extensions import override

import numpy as np
import pytest
import torch

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


@pytest.mark.local_compatible
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
