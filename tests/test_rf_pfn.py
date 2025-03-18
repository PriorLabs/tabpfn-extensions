"""Tests for the TabPFN-based Random Forest implementation.

This file tests the RF-PFN implementations in tabpfn_extensions.rf_pfn.
"""

from __future__ import annotations

import pytest

from tabpfn_extensions.rf_pfn.sklearn_based_random_forest_tabpfn import (
    RandomForestTabPFNClassifier, 
    RandomForestTabPFNRegressor
)
from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests


@pytest.mark.requires_tabpfn
class TestRandomForestClassifier(BaseClassifierTests):
    """Test RandomForestTabPFNClassifier using the BaseClassifierTests framework."""
    
    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a TabPFN-based RandomForestClassifier as the estimator."""
        return RandomForestTabPFNClassifier(
            tabpfn=tabpfn_classifier,
            n_estimators=5,  # Use few trees for speed
            max_depth=3,     # Shallow trees for speed
            random_state=42,
            max_predict_time=30  # Limit prediction time
        )
    
    # Skip pandas and text feature tests as they're not fully supported by RandomForestTabPFN
    @pytest.mark.skip(reason="RandomForestTabPFN doesn't fully support pandas DataFrames")
    def test_with_pandas(self, estimator, pandas_classification_data):
        pass
        
    @pytest.mark.skip(reason="RandomForestTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

@pytest.mark.requires_tabpfn
class TestRandomForestRegressor(BaseRegressorTests):
    """Test RandomForestTabPFNRegressor using the BaseRegressorTests framework."""
    
    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a TabPFN-based RandomForestRegressor as the estimator."""
        return RandomForestTabPFNRegressor(
            tabpfn=tabpfn_regressor,
            n_estimators=5,  # Use few trees for speed
            max_depth=3,     # Shallow trees for speed
            random_state=42,
            max_predict_time=30  # Limit prediction time
        )
        
    # Skip pandas and text feature tests as they're not fully supported by RandomForestTabPFN
    @pytest.mark.skip(reason="RandomForestTabPFN doesn't fully support pandas DataFrames")
    def test_with_pandas(self, estimator, pandas_regression_data):
        pass
        
    @pytest.mark.skip(reason="RandomForestTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass