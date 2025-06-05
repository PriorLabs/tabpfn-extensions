"""Test suite for TabPFGen Data Synthesizer Extension."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

# Python version check
PYTHON_VERSION_OK = sys.version_info >= (3, 10)
SKIP_REASON_PYTHON = "Python >=3.10 required for tabpfgen_datasynthesizer extension"

# Import modules to test (with proper error handling)
try:
    from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
    from tabpfn_extensions.tabpfgen_datasynthesizer.tabpfgen_wrapper import (
        TABPFGEN_AVAILABLE,
    )
    from tabpfn_extensions.tabpfgen_datasynthesizer.utils import (
        analyze_class_distribution,
        calculate_synthetic_quality_metrics,
        combine_datasets,
        validate_tabpfn_data,
    )

    EXTENSION_IMPORTABLE = True
except ImportError:
    EXTENSION_IMPORTABLE = False
    # Create dummy objects to prevent test collection errors
    TabPFNDataSynthesizer = None
    TABPFGEN_AVAILABLE = False


# Combined skip condition - skip if Python version is too old OR if extension can't be imported
SKIP_EXTENSION_TESTS = not PYTHON_VERSION_OK or not EXTENSION_IMPORTABLE
SKIP_TABPFGEN_TESTS = SKIP_EXTENSION_TESTS or not TABPFGEN_AVAILABLE

# Skip reasons
SKIP_REASON_EXTENSION = (
    SKIP_REASON_PYTHON
    if not PYTHON_VERSION_OK
    else "tabpfgen_datasynthesizer extension not available"
)
SKIP_REASON_TABPFGEN = (
    SKIP_REASON_EXTENSION if SKIP_EXTENSION_TESTS else "TabPFGen not available"
)


# Test data fixtures
@pytest.fixture
def classification_data():
    """Generate small classification dataset for testing."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, n_informative=3, random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate small regression dataset for testing."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],
        n_informative=4,
        random_state=42,
    )
    return X, y


@pytest.mark.skipif(SKIP_EXTENSION_TESTS, reason=SKIP_REASON_EXTENSION)
class TestTabPFNDataSynthesizer:
    """Test the TabPFNDataSynthesizer class."""

    def test_init(self):
        """Test synthesizer initialization."""
        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=100)
        assert synthesizer.n_sgld_steps == 100
        assert synthesizer.device == "auto"

    @pytest.mark.skipif(SKIP_TABPFGEN_TESTS, reason=SKIP_REASON_TABPFGEN)
    def test_generate_classification(self, classification_data):
        """Test classification data generation."""
        X, y = classification_data

        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=50)  # Small for testing
        X_synth, y_synth = synthesizer.generate_classification(
            X, y, n_samples=20, balance_classes=True
        )

        assert X_synth.shape == (20, X.shape[1])
        assert y_synth.shape == (20,)
        assert set(np.unique(y_synth)).issubset(set(np.unique(y)))

    @pytest.mark.skipif(SKIP_TABPFGEN_TESTS, reason=SKIP_REASON_TABPFGEN)
    def test_generate_regression(self, regression_data):
        """Test regression data generation."""
        X, y = regression_data

        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=50)  # Small for testing
        X_synth, y_synth = synthesizer.generate_regression(
            X, y, n_samples=20, use_quantiles=True
        )

        assert X_synth.shape == (20, X.shape[1])
        assert y_synth.shape == (20,)
        assert np.isfinite(X_synth).all()
        assert np.isfinite(y_synth).all()

    @pytest.mark.skipif(SKIP_TABPFGEN_TESTS, reason=SKIP_REASON_TABPFGEN)
    def test_balance_dataset(self, imbalanced_data):
        """Test the balance_dataset method."""
        X, y = imbalanced_data

        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=30)  # Very small for testing
        X_synth, y_synth, X_combined, y_combined = synthesizer.balance_dataset(
            X, y, visualize=False
        )

        # Check return types and shapes
        assert isinstance(X_synth, np.ndarray)
        assert isinstance(y_synth, np.ndarray)
        assert isinstance(X_combined, np.ndarray)
        assert isinstance(y_combined, np.ndarray)

        # Check that combined data includes original data
        assert len(X_combined) >= len(X)
        assert len(y_combined) >= len(y)

        # Check that synthetic data was generated
        assert len(X_synth) > 0
        assert len(y_synth) > 0

        # Check feature dimensions match
        assert X_synth.shape[1] == X.shape[1]
        assert X_combined.shape[1] == X.shape[1]

    @pytest.mark.skipif(SKIP_EXTENSION_TESTS, reason=SKIP_REASON_EXTENSION)
    def test_pandas_input(self, classification_data):
        """Test that pandas DataFrames are handled correctly."""
        X, y = classification_data
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=50)
        try:
            X_synth, y_synth = synthesizer.generate_classification(
                X_df, y_series, n_samples=10
            )
            assert isinstance(X_synth, np.ndarray)
            assert isinstance(y_synth, np.ndarray)
        except ImportError:
            pytest.skip("TabPFGen not available")


@pytest.mark.skipif(SKIP_EXTENSION_TESTS, reason=SKIP_REASON_EXTENSION)
class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_tabpfn_data_valid(self, classification_data):
        """Test data validation with valid data."""
        X, y = classification_data
        is_valid, message = validate_tabpfn_data(X, y)
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)

    def test_validate_tabpfn_data_large(self):
        """Test data validation with large dataset."""
        X = np.random.randn(15000, 10)  # Too large
        y = np.random.randint(0, 2, 15000)

        is_valid, message = validate_tabpfn_data(X, y, max_samples=10000)
        assert not is_valid
        assert "samples" in message.lower()

    def test_validate_tabpfn_data_many_features(self):
        """Test data validation with too many features."""
        X = np.random.randn(100, 150)  # Too many features
        y = np.random.randint(0, 2, 100)

        is_valid, message = validate_tabpfn_data(X, y, max_features=100)
        assert not is_valid
        assert "features" in message.lower()

    def test_validate_tabpfn_data_imbalanced(self, imbalanced_data):
        """Test data validation with imbalanced data."""
        X, y = imbalanced_data

        is_valid, message = validate_tabpfn_data(X, y)
        # Should detect imbalance
        assert "imbalanced" in message.lower() or "balance_dataset" in message.lower()

    def test_combine_datasets_append(self, classification_data):
        """Test combine_datasets with append strategy."""
        X, y = classification_data
        X_synth = np.random.randn(20, X.shape[1])
        y_synth = np.random.randint(0, 2, 20)

        X_combined, y_combined = combine_datasets(
            X, y, X_synth, y_synth, strategy="append"
        )

        assert X_combined.shape[0] == len(X) + len(X_synth)
        assert y_combined.shape[0] == len(y) + len(y_synth)

    def test_combine_datasets_replace(self, classification_data):
        """Test combine_datasets with replace strategy."""
        X, y = classification_data
        X_synth = np.random.randn(20, X.shape[1])
        y_synth = np.random.randint(0, 2, 20)

        X_combined, y_combined = combine_datasets(
            X, y, X_synth, y_synth, strategy="replace"
        )

        assert X_combined.shape[0] == len(X_synth)
        assert y_combined.shape[0] == len(y_synth)
        np.testing.assert_array_equal(X_combined, X_synth)
        np.testing.assert_array_equal(y_combined, y_synth)

    def test_combine_datasets_balanced(self, classification_data):
        """Test combine_datasets with balanced strategy."""
        X, y = classification_data
        X_synth = np.random.randn(50, X.shape[1])  # Different size
        y_synth = np.random.randint(0, 2, 50)

        X_combined, y_combined = combine_datasets(
            X, y, X_synth, y_synth, strategy="balanced"
        )

        # Should have equal amounts of original and synthetic data
        expected_size = 2 * min(len(X), len(X_synth))
        assert X_combined.shape[0] == expected_size
        assert y_combined.shape[0] == expected_size

    def test_combine_datasets_invalid_strategy(self, classification_data):
        """Test combine_datasets with invalid strategy."""
        X, y = classification_data
        X_synth = np.random.randn(20, X.shape[1])
        y_synth = np.random.randint(0, 2, 20)

        with pytest.raises(ValueError):
            combine_datasets(X, y, X_synth, y_synth, strategy="invalid")

    def test_analyze_class_distribution(self, classification_data):
        """Test class distribution analysis."""
        X, y = classification_data

        analysis = analyze_class_distribution(y, "Test Dataset")

        assert isinstance(analysis, dict)
        assert "classes" in analysis
        assert "counts" in analysis
        assert "percentages" in analysis
        assert "total_samples" in analysis
        assert "imbalance_ratio" in analysis
        assert "is_balanced" in analysis

        assert analysis["total_samples"] == len(y)
        assert len(analysis["classes"]) == len(np.unique(y))
        assert sum(analysis["counts"]) == len(y)
        assert abs(sum(analysis["percentages"]) - 100.0) < 1e-10

    def test_calculate_synthetic_quality_metrics(self, classification_data):
        """Test synthetic data quality metrics calculation."""
        X, y = classification_data

        # Create synthetic data (just random for testing)
        X_synth = np.random.randn(50, X.shape[1])
        y_synth = np.random.choice(np.unique(y), 50)

        metrics = calculate_synthetic_quality_metrics(X, X_synth, y, y_synth)

        assert isinstance(metrics, dict)

        # Check that we get some expected metrics
        expected_metrics = ["mean_absolute_error", "std_absolute_error"]
        for metric in expected_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], (int, float))
                assert np.isfinite(metrics[metric])

    def test_calculate_synthetic_quality_metrics_no_labels(self, classification_data):
        """Test quality metrics calculation without labels."""
        X, y = classification_data
        X_synth = np.random.randn(50, X.shape[1])

        metrics = calculate_synthetic_quality_metrics(X, X_synth)

        assert isinstance(metrics, dict)
        # Should still get feature-based metrics
        if "mean_absolute_error" in metrics:
            assert isinstance(metrics["mean_absolute_error"], (int, float))


@pytest.mark.skipif(SKIP_TABPFGEN_TESTS, reason=SKIP_REASON_TABPFGEN)
class TestIntegration:
    """Integration tests requiring TabPFGen."""

    def test_end_to_end_classification_with_balancing(self, imbalanced_data):
        """Test complete classification workflow with balancing."""
        X, y = imbalanced_data

        # Validate original data
        is_valid, message = validate_tabpfn_data(X, y)
        print(f"Validation: {message}")

        # Analyze original distribution
        original_analysis = analyze_class_distribution(y, "Original")
        assert not original_analysis["is_balanced"]  # Should be imbalanced

        # Balance dataset
        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=20)  # Small for testing
        X_synth, y_synth, X_balanced, y_balanced = synthesizer.balance_dataset(
            X, y, visualize=False
        )

        # Analyze balanced distribution
        balanced_analysis = analyze_class_distribution(y_balanced, "Balanced")

        # Should be more balanced than original
        assert (
            balanced_analysis["imbalance_ratio"] < original_analysis["imbalance_ratio"]
        )

        # Calculate quality metrics
        quality_metrics = calculate_synthetic_quality_metrics(X, X_synth, y, y_synth)
        assert isinstance(quality_metrics, dict)

        # Test combination
        X_combined_append, y_combined_append = combine_datasets(
            X, y, X_synth, y_synth, strategy="append"
        )

        assert len(X_combined_append) == len(X_balanced)
        assert len(y_combined_append) == len(y_balanced)

    def test_end_to_end_regression(self, regression_data):
        """Test complete regression workflow."""
        X, y = regression_data

        # Generate synthetic regression data
        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=20)
        X_synth, y_synth = synthesizer.generate_regression(
            X, y, n_samples=30, visualize=False
        )

        # Combine data
        X_combined, y_combined = combine_datasets(
            X, y, X_synth, y_synth, strategy="append"
        )

        # Calculate quality
        quality_metrics = calculate_synthetic_quality_metrics(X, X_synth)

        assert X_combined.shape[0] == len(X) + len(X_synth)
        assert isinstance(quality_metrics, dict)

        # Basic sanity checks
        assert np.isfinite(X_synth).all()
        assert np.isfinite(y_synth).all()
        assert X_synth.shape[1] == X.shape[1]


@pytest.mark.skipif(SKIP_EXTENSION_TESTS, reason=SKIP_REASON_EXTENSION)
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_tabpfgen_not_available(self, monkeypatch):
        """Test behavior when TabPFGen is not available."""
        # Mock TabPFGen as unavailable
        import tabpfn_extensions.tabpfgen_datasynthesizer.tabpfgen_wrapper as wrapper

        monkeypatch.setattr(wrapper, "TABPFGEN_AVAILABLE", False)

        with pytest.raises(ImportError, match="TabPFGen is required"):
            TabPFNDataSynthesizer()

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        is_valid, message = validate_tabpfn_data(X, y)
        # Should handle gracefully and detect empty dataset
        assert not is_valid
        assert "empty" in message.lower()

    def test_single_class_dataset(self):
        """Test handling of single-class datasets."""
        X = np.random.randn(100, 5)
        y = np.zeros(100)  # All same class

        is_valid, message = validate_tabpfn_data(X, y)
        analysis = analyze_class_distribution(y, "Single Class")

        assert analysis["num_classes"] == 1
        assert analysis["imbalance_ratio"] == 1.0  # Perfectly balanced (trivially)

    def test_mismatched_dimensions(self):
        """Test handling of mismatched X and y dimensions."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 90)  # Wrong size

        is_valid, message = validate_tabpfn_data(X, y)
        # Should detect dimension mismatch
        assert not is_valid
        assert "mismatch" in message.lower()


# Performance and benchmarking tests
@pytest.mark.skipif(SKIP_TABPFGEN_TESTS, reason=SKIP_REASON_TABPFGEN)
class TestPerformance:
    """Test performance characteristics."""

    def test_generation_performance(self, classification_data):
        """Test that generation completes in reasonable time."""
        import time

        X, y = classification_data
        synthesizer = TabPFNDataSynthesizer(n_sgld_steps=10)  # Very small for speed

        start_time = time.time()
        X_synth, y_synth = synthesizer.generate_classification(
            X, y, n_samples=20, visualize=False
        )
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 30.0  # 30 seconds max for small dataset
        assert len(X_synth) == 20

    @pytest.mark.skipif(SKIP_EXTENSION_TESTS, reason=SKIP_REASON_EXTENSION)
    def test_utility_function_performance(self):
        """Test that utility functions perform well on larger datasets."""
        import time

        # Create larger dataset
        X = np.random.randn(5000, 20)
        y = np.random.randint(0, 5, 5000)

        start_time = time.time()
        is_valid, message = validate_tabpfn_data(X, y)
        analysis = analyze_class_distribution(y, "Large Dataset")
        elapsed_time = time.time() - start_time

        # Should be fast for utility functions
        assert elapsed_time < 5.0  # 5 seconds max
        assert isinstance(analysis, dict)


# Test to verify version checking works correctly
class TestVersionCompatibility:
    """Test version compatibility checks."""

    def test_python_version_detection(self):
        """Test that Python version detection works correctly."""
        assert isinstance(PYTHON_VERSION_OK, bool)
        assert isinstance(SKIP_EXTENSION_TESTS, bool)
        assert isinstance(SKIP_TABPFGEN_TESTS, bool)

        # On Python <3.10, extension tests should be skipped
        if sys.version_info < (3, 10):
            assert SKIP_EXTENSION_TESTS
            assert "Python >=3.10 required" in SKIP_REASON_EXTENSION

    def test_skip_conditions_logic(self):
        """Test that skip conditions are logically correct."""
        # SKIP_TABPFGEN_TESTS should always be True if SKIP_EXTENSION_TESTS is True
        if SKIP_EXTENSION_TESTS:
            assert SKIP_TABPFGEN_TESTS

        # If Python version is OK and extension is importable,
        # then SKIP_EXTENSION_TESTS should be False
        if PYTHON_VERSION_OK and EXTENSION_IMPORTABLE:
            assert not SKIP_EXTENSION_TESTS


if __name__ == "__main__":
    # Print version information for debugging
    print(f"Python version: {sys.version}")
    print(f"Python >=3.10: {PYTHON_VERSION_OK}")
    print(f"Extension importable: {EXTENSION_IMPORTABLE}")
    print(f"TabPFGen available: {TABPFGEN_AVAILABLE}")
    print(f"Skip extension tests: {SKIP_EXTENSION_TESTS}")
    print(f"Skip TabPFGen tests: {SKIP_TABPFGEN_TESTS}")

    # Run tests with pytest
    pytest.main([__file__, "-v"])
