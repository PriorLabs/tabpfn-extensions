"""Tests for TabEBM (Tabular Energy-Based Model) functionality.

This file tests the TabEBM implementation in tabpfn_extensions.tabebm.
TabEBM uses TabPFN as an energy function to generate synthetic tabular data
through Stochastic Gradient Langevin Dynamics (SGLD) sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification

from conftest import DEFAULT_TEST_SIZE, FAST_TEST_MODE, SMALL_TEST_SIZE

# Try to import TabEBM, but skip tests if dependencies are not available
try:
    from tabpfn_extensions.tabebm.tabebm import TabEBM, to_numpy
except ImportError:
    pytest.skip(
        "TabEBM dependencies not available. Install with 'pip install \"tabpfn-extensions\"'",
        allow_module_level=True,
    )


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTabEBM:
    """Test suite for TabEBM class."""

    @pytest.fixture
    def classification_data(self):
        """Generate synthetic classification data for testing."""
        n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Generate synthetic multiclass data for testing."""
        n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def tabebm_model(self):
        """Create a TabEBM model instance for testing."""
        return TabEBM(max_data_size=1000)

    def test_tabebm_initialization(self, tabebm_model):
        """Test TabEBM model initialization."""
        assert tabebm_model.max_data_size == 1000
        assert hasattr(tabebm_model, "model")
        assert hasattr(tabebm_model, "device")
        assert tabebm_model.device in ["cuda", "cpu"]
        assert isinstance(tabebm_model._fitted_models_cache, dict)
        assert len(tabebm_model._fitted_models_cache) == 0

    def test_basic_generation(self, tabebm_model, classification_data):
        """Test basic synthetic data generation."""
        X, y = classification_data

        # Generate a small number of samples for testing
        num_samples = 3 if FAST_TEST_MODE else 5
        sgld_steps = 5 if FAST_TEST_MODE else 10

        synthetic_data = tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Check output structure
        assert isinstance(synthetic_data, dict)
        unique_classes = np.unique(y)
        assert len(synthetic_data) == len(unique_classes)

        # Check each class has correct shape
        for class_idx in range(len(unique_classes)):
            class_key = f"class_{class_idx}"
            assert class_key in synthetic_data
            assert isinstance(synthetic_data[class_key], np.ndarray)
            assert synthetic_data[class_key].shape == (num_samples, X.shape[1])

    def test_generation_with_multiclass(self, tabebm_model, multiclass_data):
        """Test synthetic data generation with multiclass data."""
        X, y = multiclass_data

        num_samples = 2 if FAST_TEST_MODE else 3
        sgld_steps = 5 if FAST_TEST_MODE else 10

        synthetic_data = tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Check that we have data for all classes
        unique_classes = np.unique(y)
        assert len(synthetic_data) == len(unique_classes)

        for class_idx in range(len(unique_classes)):
            class_key = f"class_{class_idx}"
            assert class_key in synthetic_data
            assert synthetic_data[class_key].shape == (num_samples, X.shape[1])

    def test_generation_with_pandas(self, tabebm_model, classification_data):
        """Test generation with pandas DataFrame input."""
        X, y = classification_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.DataFrame(y, columns=["target"])

        num_samples = 2 if FAST_TEST_MODE else 3
        sgld_steps = 3 if FAST_TEST_MODE else 5

        synthetic_data = tabebm_model.generate(
            X=X_df,
            y=y_series,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Check output is still numpy arrays
        for class_key in synthetic_data:
            assert isinstance(synthetic_data[class_key], np.ndarray)
            assert synthetic_data[class_key].shape == (num_samples, X.shape[1])

    def test_generation_with_torch_tensors(self, tabebm_model, classification_data):
        """Test generation with PyTorch tensor input."""
        X, y = classification_data
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        num_samples = 2 if FAST_TEST_MODE else 3
        sgld_steps = 3 if FAST_TEST_MODE else 5

        synthetic_data = tabebm_model.generate(
            X=X_tensor,
            y=y_tensor,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Check output is numpy arrays
        for class_key in synthetic_data:
            assert isinstance(synthetic_data[class_key], np.ndarray)
            assert synthetic_data[class_key].shape == (num_samples, X.shape[1])

    def test_sgld_parameters(self, tabebm_model, classification_data):
        """Test SGLD parameter effects on generation."""
        X, y = classification_data

        num_samples = 2 if FAST_TEST_MODE else 3
        sgld_steps = 3 if FAST_TEST_MODE else 5

        # Test with different step sizes
        synthetic_data_1 = tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_step_size=0.05,
            sgld_steps=sgld_steps,
            seed=42,
        )

        synthetic_data_2 = tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_step_size=0.2,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Results should be different with different step sizes
        for class_key in synthetic_data_1:
            assert not np.allclose(
                synthetic_data_1[class_key], synthetic_data_2[class_key], atol=1e-6
            )

    def test_model_caching(self, tabebm_model, classification_data):
        """Test that model fitting is cached properly."""
        X, y = classification_data

        num_samples = 2 if FAST_TEST_MODE else 3
        sgld_steps = 3 if FAST_TEST_MODE else 5

        # First generation should populate cache
        assert len(tabebm_model._fitted_models_cache) == 0

        tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Cache should now contain fitted models for each class
        unique_classes = np.unique(y)
        assert len(tabebm_model._fitted_models_cache) == len(unique_classes)


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTabEBMStaticMethods:
    """Test static methods of TabEBM class."""

    def test_to_numpy_with_numpy_array(self):
        """Test to_numpy with numpy array input."""
        X = np.random.randn(10, 5)
        result = to_numpy(X)
        assert result is not None
        assert np.array_equal(result, X)
        assert isinstance(result, np.ndarray)

    def test_to_numpy_with_torch_tensor(self):
        """Test to_numpy with PyTorch tensor input."""
        X = torch.randn(10, 5)
        result = to_numpy(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)

    def test_to_numpy_with_pandas_dataframe(self):
        """Test to_numpy with pandas DataFrame input."""
        X = pd.DataFrame(np.random.randn(10, 5))
        result = to_numpy(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)

    def test_compute_energy_with_torch_tensor(self):
        """Test compute_energy with PyTorch tensor logits."""
        # Create unnormalized logits
        logits = torch.tensor([[2.0, 1.0], [0.5, 3.0], [1.5, 0.8]])

        energy = TabEBM.compute_energy(logits)
        assert isinstance(energy, torch.Tensor)
        assert energy.shape == (3,)
        assert torch.all(energy < 0)  # Energy should be negative

    def test_compute_energy_with_numpy_array(self):
        """Test compute_energy with numpy array logits."""
        # Create unnormalized logits
        logits = np.array([[2.0, 1.0], [0.5, 3.0], [1.5, 0.8]])

        energy = TabEBM.compute_energy(logits)
        assert isinstance(energy, np.ndarray)
        assert energy.shape == (3,)
        assert np.all(energy < 0)  # Energy should be negative

    def test_compute_energy_with_probabilities_raises_error(self):
        """Test that compute_energy raises error with normalized probabilities."""
        # Create normalized probabilities (sum to 1)
        probs = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])

        with pytest.raises(ValueError, match="Logits must be unnormalized"):
            TabEBM.compute_energy(probs)

    def test_compute_energy_return_unnormalized_prob(self):
        """Test compute_energy with return_unnormalized_prob=True."""
        logits = torch.tensor([[2.0, 1.0], [0.5, 3.0]])

        unnorm_probs = TabEBM.compute_energy(logits, return_unnormalized_prob=True)
        assert isinstance(unnorm_probs, torch.Tensor)
        assert torch.all(unnorm_probs > 0)  # Should be positive

    def test_add_surrogate_negative_samples_numpy(self):
        """Test add_surrogate_negative_samples with numpy arrays."""
        X = np.random.randn(10, 3)

        X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
            X, distance_negative_class=2.0
        )

        assert isinstance(X_ebm, np.ndarray)
        assert isinstance(y_ebm, np.ndarray)
        assert X_ebm.shape[0] > X.shape[0]  # Should have added surrogate samples
        assert X_ebm.shape[1] == X.shape[1]  # Same number of features
        assert len(np.unique(y_ebm)) == 2  # Binary labels: 0 for real, 1 for surrogates
        assert np.sum(y_ebm == 0) == X.shape[0]  # Real samples labeled as 0

    def test_add_surrogate_negative_samples_torch(self):
        """Test add_surrogate_negative_samples with PyTorch tensors."""
        X = torch.randn(10, 3)

        X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
            X, distance_negative_class=2.0
        )

        assert isinstance(X_ebm, torch.Tensor)
        assert isinstance(y_ebm, torch.Tensor)
        assert X_ebm.shape[0] > X.shape[0]  # Should have added surrogate samples
        assert X_ebm.shape[1] == X.shape[1]  # Same number of features
        assert len(torch.unique(y_ebm)) == 2  # Binary labels
        assert torch.sum(y_ebm == 0) == X.shape[0]  # Real samples labeled as 0

    def test_add_surrogate_negative_samples_2d(self):
        """Test add_surrogate_negative_samples with 2D data (special case)."""
        X = np.random.randn(5, 2)  # 2D data

        X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
            X, distance_negative_class=3.0
        )

        # For 2D, should add exactly 4 corner points
        expected_surrogates = 4
        assert X_ebm.shape[0] == X.shape[0] + expected_surrogates
        assert np.sum(y_ebm == 1) == expected_surrogates

    def test_train_test_split_allow_full_train(self):
        """Test train_test_split_allow_full_train with test_size=0."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        X_train, X_val, y_train, y_val = TabEBM.train_test_split_allow_full_train(
            X, y, test_size=0, random_state=42
        )

        # In full train mode, training data should be the original data
        assert np.array_equal(X_train, X)
        assert np.array_equal(y_train, y)
        # Validation sets should still exist (from sklearn's split)
        assert X_val.shape[0] > 0
        assert y_val.shape[0] > 0

    def test_train_test_split_normal_mode(self):
        """Test train_test_split_allow_full_train with normal test_size."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        X_train, X_val, y_train, y_val = TabEBM.train_test_split_allow_full_train(
            X, y, test_size=0.3, random_state=42
        )

        # Should behave like normal train_test_split
        assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
        assert y_train.shape[0] + y_val.shape[0] == y.shape[0]
        assert X_val.shape[0] == int(0.3 * X.shape[0])  # Approximately


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTabEBMErrorHandling:
    """Test error handling and edge cases for TabEBM."""

    def test_invalid_input_types(self):
        """Test TabEBM with invalid input types."""
        tabebm_model = TabEBM()

        # Test with invalid X type
        with pytest.raises(ValueError):
            # Use type ignore to test error handling with invalid input
            tabebm_model.generate(
                X="invalid",  # type: ignore
                y=np.array([0, 1, 0, 1]),
                num_samples=2,
                sgld_steps=3,
            )

    def test_mismatched_input_shapes(self):
        """Test TabEBM with mismatched X and y shapes."""
        tabebm_model = TabEBM()

        X = np.random.randn(10, 5)
        y = np.array([0, 1, 0])  # Wrong length

        # This should be handled gracefully by the preprocessing
        with pytest.raises((ValueError, IndexError)):
            tabebm_model.generate(
                X=X,
                y=y,
                num_samples=2,
                sgld_steps=3,
            )

    def test_single_class_data(self):
        """Test TabEBM with single class data."""
        tabebm_model = TabEBM()

        X = np.random.randn(10, 3)
        y = np.zeros(10)  # All same class

        num_samples = 2
        sgld_steps = 3

        synthetic_data = tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            seed=42,
        )

        # Should work with single class
        assert len(synthetic_data) == 1
        assert "class_0" in synthetic_data
        assert synthetic_data["class_0"].shape == (num_samples, X.shape[1])

    @pytest.mark.slow
    def test_generation_with_debug_mode(self, classification_data):
        """Test generation with debug mode enabled."""
        X, y = classification_data
        tabebm_model = TabEBM()

        num_samples = 2
        sgld_steps = 10  # More steps to see debug output

        # This should work without errors and print debug info
        synthetic_data = tabebm_model.generate(
            X=X,
            y=y,
            num_samples=num_samples,
            sgld_steps=sgld_steps,
            debug=True,
            seed=42,
        )

        assert len(synthetic_data) == len(np.unique(y))
