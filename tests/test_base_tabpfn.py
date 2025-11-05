"""Tests for base TabPFN functionality.

These tests check the core functionality of TabPFN models with
various data types, configurations, and sklearn compatibility.
The structure is designed to be easily reusable for testing other estimators.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.estimator_checks import check_estimator

from conftest import (
    DEFAULT_TEST_SIZE,
    FAST_TEST_MODE,
    SMALL_TEST_SIZE,
)


class BaseClassifierTests:
    """Base test suite for classifier models.

    This class can be inherited by specific estimator test classes.
    For example:

    class TestMyCustomClassifier(BaseClassifierTests):
        @pytest.fixture
        def estimator(self):
            return MyCustomClassifier()
    """

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_fit_predict(self, estimator, classification_data):
        """Test basic fit and predict functionality."""
        X, y = classification_data

        # Fit the model
        estimator.fit(X, y)

        # Make predictions
        y_pred = estimator.predict(X)

        # Check predictions shape and type
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

        # Check accuracy (should be high on this simple dataset)
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.6, f"Accuracy was only {accuracy:.2f}"

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_predict_proba(self, estimator, classification_data):
        """Test probability prediction functionality."""
        X, y = classification_data

        # Fit the model
        estimator.fit(X, y)

        # Make probability predictions
        y_proba = estimator.predict_proba(X)

        # Check predictions shape
        assert y_proba.shape == (X.shape[0], len(np.unique(y)))

        # Check probabilities sum to 1
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)

        # Check values are in [0, 1]
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_not_fitted(self, estimator, classification_data):
        """Test appropriate error raising when not fitted."""
        X, _ = classification_data

        # Try to predict without fitting
        with pytest.raises((NotFittedError, ValueError)):
            estimator.predict(X)

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_pandas(self, estimator, pandas_classification_data):
        """Test that estimator works with pandas DataFrames."""
        X, y = pandas_classification_data

        # Fit the model
        estimator.fit(X, y)

        # Make predictions
        y_pred = estimator.predict(X)

        # Check predictions
        assert len(y_pred) == len(y)

        # Score should be reasonable
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.6

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_multiclass(self, estimator, multiclass_data):
        """Test that estimator works with multi-class data."""
        X, y = multiclass_data

        # Fit the model
        estimator.fit(X, y)

        # Make predictions
        y_pred = estimator.predict(X)

        # Check predictions shape and type
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

        # Check accuracy
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.5, f"Multiclass accuracy was only {accuracy:.2f}"

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_is_sklearn_compatible(self, estimator):
        """Test that estimator follows scikit-learn conventions."""
        # Check that it follows the estimator API
        assert is_classifier(estimator)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")
        assert hasattr(estimator, "predict_proba")
        assert hasattr(estimator, "get_params")
        assert hasattr(estimator, "set_params")

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Test estimator with various types of datasets."""
        data_types = ["basic", "correlated", "outliers", "special_values"]

        for data_type in data_types:
            # Generate dataset - use the generic data generator
            X, y = dataset_generator.generate_data(
                task_type="classification",
                n_samples=SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE,
                n_features=5,
                n_classes=2,
                data_type=data_type,
            )

            # Fit and predict
            estimator.fit(X, y)
            y_pred = estimator.predict(X)

            # Check accuracy
            accuracy = accuracy_score(y, y_pred)
            assert (
                accuracy > 0.6
            ), f"Failed with data_type {data_type}: accuracy {accuracy:.2f}"

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_missing_values(self, estimator, dataset_generator):
        """Test with missing values (requires imputation)."""
        # Use the enhanced utility for missing data
        X, y = dataset_generator.generate_missing_values_dataset(
            n_samples=SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE,
            n_features=5,
            missing_rate=0.1,
            task_type="classification",
        )

        # Fit and predict
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        # Check accuracy
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.6

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_text_features(self, estimator, dataset_generator):
        """Test with text features (after encoding)."""
        # Use the dedicated text feature dataset generator
        X, y = dataset_generator.generate_text_dataset(
            n_samples=SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE,
            task_type="classification",
        )

        # Fit and predict
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        # Check accuracy
        accuracy = accuracy_score(y, y_pred)
        assert accuracy > 0.6

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_extreme_cases(self, estimator):
        """Test TabPFN classifier with extreme cases."""
        # Very small dataset
        X_tiny = np.random.rand(5, 3)  # Just 5 samples, 3 features
        y_tiny = np.array([0, 1, 0, 1, 0])  # Binary classification

        # Should fit and predict without errors
        estimator.fit(X_tiny, y_tiny)
        y_pred = estimator.predict(X_tiny)
        assert y_pred.shape == y_tiny.shape

        # Single class dataset
        X_single = np.random.rand(10, 3)  # 10 samples, 3 features
        y_single = np.zeros(10)  # All samples are class 0

        # Should handle single class case gracefully
        estimator.fit(X_single, y_single)
        y_pred = estimator.predict(X_single)
        assert y_pred.shape == y_single.shape
        assert np.all(y_pred == 0)

    @pytest.mark.slow
    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_passes_estimator_checks(self, estimator):
        """Run scikit-learn's estimator compatibility checks."""
        check_estimator(estimator)


class BaseRegressorTests:
    """Base test suite for regressor models.

    This class can be inherited by specific estimator test classes.
    For example:

    class TestMyCustomRegressor(BaseRegressorTests):
        @pytest.fixture
        def estimator(self):
            return MyCustomRegressor()
    """

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_fit_predict(self, estimator, regression_data):
        """Test basic fit and predict functionality."""
        X, y = regression_data

        # Fit the model
        estimator.fit(X, y)

        # Make predictions
        y_pred = estimator.predict(X)

        # Check predictions shape and type
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)

        # Check R2 score
        r2 = r2_score(y, y_pred)
        print(f"DEBUG: R2 score: {r2}")
        assert r2 > 0.5, f"R2 score was only {r2:.2f}"

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_not_fitted(self, estimator, regression_data):
        """Test appropriate error raising when not fitted."""
        X, _ = regression_data

        # Try to predict without fitting
        with pytest.raises((NotFittedError, ValueError)):
            estimator.predict(X)

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_pandas(self, estimator, pandas_regression_data):
        """Test that estimator works with pandas DataFrames."""
        X, y = pandas_regression_data

        # Fit the model
        estimator.fit(X, y)

        # Make predictions
        y_pred = estimator.predict(X)

        # Check predictions
        assert len(y_pred) == len(y)

        # Score should be reasonable
        r2 = r2_score(y, y_pred)
        assert r2 > 0.5

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_is_sklearn_compatible(self, estimator):
        """Test that estimator follows scikit-learn conventions."""
        # Check that it follows the estimator API
        assert is_regressor(estimator)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")
        assert hasattr(estimator, "get_params")
        assert hasattr(estimator, "set_params")

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Test estimator with various types of datasets."""
        data_types = ["basic", "correlated", "outliers", "special_values"]

        for data_type in data_types:
            # Generate dataset - use the generic data generator
            X, y = dataset_generator.generate_data(
                task_type="regression",
                n_samples=30 if FAST_TEST_MODE else 60,
                n_features=5,
                data_type=data_type,
            )

            # Fit and predict
            estimator.fit(X, y)
            y_pred = estimator.predict(X)

            # Check R2 score
            r2 = r2_score(y, y_pred)
            assert r2 > 0.5, f"Failed with data_type {data_type}: R2 {r2:.2f}"

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_missing_values(self, estimator, dataset_generator):
        """Test with missing values (requires imputation)."""
        # Use the enhanced utility for missing data
        X, y = dataset_generator.generate_missing_values_dataset(
            n_samples=30 if FAST_TEST_MODE else 60,
            n_features=5,
            missing_rate=0.1,
            task_type="regression",
        )

        # Fit and predict
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        # Check R2 score
        r2 = r2_score(y, y_pred)
        assert r2 > 0.5

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_with_text_features(self, estimator, dataset_generator):
        """Test with text features (after encoding)."""
        # Use the dedicated text feature dataset generator
        X, y = dataset_generator.generate_text_dataset(
            n_samples=30 if FAST_TEST_MODE else 60,
            task_type="regression",
        )

        # Fit and predict
        estimator.fit(X, y)
        y_pred = estimator.predict(X)

        # Check R2 score
        r2 = r2_score(y, y_pred)
        assert r2 > 0.5

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_extreme_cases(self, estimator):
        """Test TabPFN regressor with extreme cases."""
        # Very small dataset with constant y
        X_tiny = np.random.rand(5, 3)  # Just 5 samples, 3 features
        y_tiny = np.zeros(5)  # Random target values

        # Should fit and predict without errors
        estimator.fit(X_tiny, y_tiny)
        y_pred = estimator.predict(X_tiny)
        assert y_pred.shape == y_tiny.shape

        # Test with outliers in target
        X_outlier = np.random.rand(20, 3)
        y_outlier = np.random.rand(20)
        y_outlier[0] = 1000.0  # Extreme outlier

        estimator.fit(X_outlier, y_outlier)
        y_pred = estimator.predict(X_outlier)
        assert y_pred.shape == y_outlier.shape

    @pytest.mark.slow
    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_passes_estimator_checks(self, estimator):
        """Run scikit-learn's estimator compatibility checks."""
        # This runs all the sklearn estimator checks
        check_estimator(estimator)


# Concrete test classes for TabPFN models


class TestTabPFNClassifier(BaseClassifierTests):
    """Test suite for TabPFNClassifier."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide the TabPFN classifier as the estimator."""
        return tabpfn_classifier

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_ensemble_configurations(self, estimator, classification_data):
        """Test classifier with different ensemble configurations."""
        X, y = classification_data

        for N_ensemble_configurations in [1, 5]:
            # Set the ensemble configurations
            estimator.N_ensemble_configurations = N_ensemble_configurations

            # Fit the model
            estimator.fit(X, y)

            # Make predictions
            y_pred = estimator.predict(X)

            # Check predictions
            assert y_pred.shape == y.shape

    @pytest.mark.skip(reason="Slow test not required as this is tested in base package")
    def test_passes_estimator_checks(self, estimator):
        """Run scikit-learn's estimator compatibility checks."""
        pass


class TestTabPFNRegressor(BaseRegressorTests):
    """Test suite for TabPFNRegressor."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide the TabPFN regressor as the estimator."""
        return tabpfn_regressor

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test_ensemble_configurations(self, estimator, regression_data):
        """Test regressor with different ensemble configurations."""
        X, y = regression_data

        for N_ensemble_configurations in [1, 5]:
            # Set the ensemble configurations
            estimator.N_ensemble_configurations = N_ensemble_configurations

            # Fit the model
            estimator.fit(X, y)

            # Make predictions
            y_pred = estimator.predict(X)

            # Check predictions
            assert y_pred.shape == y.shape

    @pytest.mark.local_compatible
    @pytest.mark.client_compatible
    def test_bar_distribution_return(self, estimator, regression_data):
        """Test that the regressor can return bar distributions."""
        X, y = regression_data

        # Fit the model
        estimator.fit(X, y)

        # Print information about the estimator type for debugging
        estimator_type = str(type(estimator))
        is_client = "client" in estimator_type.lower()

        print(f"DEBUG: Estimator type is: {estimator_type}")
        print(f"DEBUG: Is client version: {is_client}")

        # Get predictions with distribution
        try:
            # Use the output_type="full" parameter supported by TabPFN package
            result = estimator.predict(X, output_type="full")

            # Debug: show keys in result
            print(f"DEBUG: Result keys: {list(result.keys())}")

        except (TypeError, ValueError, RuntimeError) as e:
            # Skip if not supported
            pytest.skip(f"Distribution output not supported: {str(e)}")

        # Check that result is a dictionary with the expected keys
        assert isinstance(result, dict), "Distribution result should be a dictionary"
        assert "logits" in result, "Distribution result should have 'logits' key"
        assert "criterion" in result, "Distribution result should have 'criterion' key"

        # Check that criterion object is present and has required methods
        criterion = result["criterion"]
        assert criterion is not None, "Criterion should not be None"
        assert hasattr(criterion, "sample"), "Criterion should have sample method"

    @pytest.mark.skip(reason="Slow test not required as this is tested in base package")
    def test_passes_estimator_checks(self, estimator):
        """Run scikit-learn's estimator compatibility checks."""
        pass
