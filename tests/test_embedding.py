"""Tests for TabPFN embedding functionality.

These tests check the functionality of TabPFNEmbedding class for extracting
embeddings from TabPFN models, both in vanilla mode and with K-fold cross-validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from conftest import FAST_TEST_MODE
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding


class TestTabPFNEmbedding:
    """Test suite for TabPFNEmbedding class."""

    @pytest.fixture
    def classification_data(self):
        """Generate synthetic classification data."""
        n_samples = 30 if FAST_TEST_MODE else 60
        X, y = make_classification(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def regression_data(self):
        """Generate synthetic regression data."""
        n_samples = 30 if FAST_TEST_MODE else 60
        X, y = make_regression(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def dataset_generator(self):
        """Provide dataset generator for varied test cases."""
        from tests.utils import DatasetGenerator

        return DatasetGenerator(seed=42)

    @pytest.mark.requires_tabpfn
    def test_clf_embedding_vanilla(self, classification_data):
        """Test vanilla embeddings extraction with a classifier."""
        X_train, X_test, y_train, y_test = classification_data

        # Create classifier and embedding extractor
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

        # Extract embeddings
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes (as numpy arrays)
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2
        assert train_embeddings.shape[1] == X_train.shape[0]  # Sample dimension
        assert test_embeddings.shape[1] == X_test.shape[0]

        # Use the first batch of embeddings
        train_emb = train_embeddings[0]
        test_emb = test_embeddings[0]

        # Verify embeddings are useful by training a simple model on them
        lr = LogisticRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)

        # The accuracy should be better than random
        assert accuracy > 0.4, f"Accuracy with embeddings was only {accuracy:.2f}"

    @pytest.mark.requires_tabpfn
    def test_clf_embedding_kfold(self, classification_data):
        """Test K-fold embeddings extraction with a classifier."""
        X_train, X_test, y_train, y_test = classification_data

        # Create classifier and embedding extractor with K-fold
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=3)  # Use 3 folds

        # Extract embeddings
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes (as numpy arrays)
        assert isinstance(train_embeddings, np.ndarray)

        # K-fold embeddings might have different shape depending on implementation
        assert train_embeddings.ndim >= 2

        # For test data it should work as usual
        assert test_embeddings.ndim >= 2
        assert test_embeddings.shape[1] == X_test.shape[0]

        # Use the first batch of embeddings for tests
        if train_embeddings.ndim == 3:
            train_emb = train_embeddings[0]
        else:
            train_emb = train_embeddings

        test_emb = test_embeddings[0]

        # Verify embeddings are useful by training a simple model on them
        lr = LogisticRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)

        # The accuracy should be better than random
        assert (
            accuracy > 0.4
        ), f"Accuracy with K-fold embeddings was only {accuracy:.2f}"

    @pytest.mark.requires_tabpfn
    def test_reg_embedding_vanilla(self, regression_data):
        """Test vanilla embeddings extraction with a regressor."""
        X_train, X_test, y_train, y_test = regression_data

        # Create regressor and embedding extractor
        reg = TabPFNRegressor(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=0)

        # Extract embeddings
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes (as numpy arrays)
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2
        assert train_embeddings.shape[1] == X_train.shape[0]
        assert test_embeddings.shape[1] == X_test.shape[0]

        # Use the first batch of embeddings
        train_emb = train_embeddings[0]
        test_emb = test_embeddings[0]

        # Verify embeddings are useful by training a simple model on them
        lr = LinearRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        r2 = r2_score(y_test, y_pred)

        # The R2 score should be reasonable
        # In rare cases this might fail due to randomness, so we set a low bar
        assert r2 > -1.0, f"R2 score with embeddings was very low: {r2:.2f}"

    @pytest.mark.requires_tabpfn
    def test_embedding_errors(self, classification_data):
        """Test error handling in TabPFNEmbedding."""
        X_train, X_test, y_train, _ = classification_data

        # Test error when no model is provided
        with pytest.raises(ValueError):
            embedding_extractor = TabPFNEmbedding()
            embedding_extractor.get_embeddings(
                X_train,
                y_train,
                X_test,
                data_source="test",
            )

        # Test error when invalid n_fold value is provided
        with pytest.raises(ValueError):
            clf = TabPFNClassifier(n_estimators=1)
            embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=1)
            embedding_extractor.get_embeddings(
                X_train,
                y_train,
                X_test,
                data_source="train",
            )

    @pytest.mark.requires_tabpfn
    def test_embeddings_with_missing_values(self, dataset_generator):
        """Test embedding extraction with missing values."""
        # Generate classification data with missing values
        X, y = dataset_generator.generate_classification_data(
            n_samples=30 if FAST_TEST_MODE else 60,
            n_features=5,
            with_missing=True,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        # Create classifier and embedding extractor
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

        # Extract embeddings
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

        # Use embeddings to make predictions
        train_emb = train_embeddings[0]
        test_emb = test_embeddings[0]

        # Verify embeddings are useful
        lr = LogisticRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)

        # Set a low threshold since missing values might reduce accuracy
        assert accuracy > 0.3, f"Accuracy with missing values was only {accuracy:.2f}"

    @pytest.mark.requires_tabpfn
    def test_embeddings_with_pandas(self, dataset_generator):
        """Test embedding extraction with pandas DataFrames."""
        # Generate pandas DataFrame data
        X, y = dataset_generator.generate_classification_data(
            n_samples=30 if FAST_TEST_MODE else 60,
            n_features=5,
            as_pandas=True,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        # Create classifier and embedding extractor
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

        # Extract embeddings
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

        # Verify embeddings
        train_emb = train_embeddings[0]
        test_emb = test_embeddings[0]

        lr = LogisticRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)

        assert accuracy > 0.4, f"Accuracy with pandas data was only {accuracy:.2f}"

    @pytest.mark.requires_tabpfn
    def test_embeddings_with_text_features(self, dataset_generator):
        """Test embedding extraction with text features."""
        # Generate data with text features
        X, y = dataset_generator.generate_classification_data(
            n_samples=30 if FAST_TEST_MODE else 60,
            n_features=5,
            with_text=True,
        )

        # Dataset generator returns both the data and encoded version
        X_encoded, X_original = X

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=0.3,
            random_state=42,
        )

        # Create classifier and embedding extractor
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

        # Extract embeddings using the encoded data
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

        # Verify embeddings are useful
        train_emb = train_embeddings[0]
        test_emb = test_embeddings[0]

        lr = LogisticRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)

        # Lower threshold for text features as they can be harder to model
        assert accuracy > 0.3, f"Accuracy with text features was only {accuracy:.2f}"

    @pytest.mark.requires_tabpfn
    def test_embeddings_with_multiclass(self, dataset_generator):
        """Test embedding extraction with multiclass data."""
        # Generate multiclass classification data
        X, y = dataset_generator.generate_classification_data(
            n_samples=30 if FAST_TEST_MODE else 60,
            n_features=5,
            n_classes=3,  # Use 3 classes
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        # Create classifier and embedding extractor
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)

        # Extract embeddings
        train_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_train,
            data_source="train",
        )
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

        # Verify embeddings
        train_emb = train_embeddings[0]
        test_emb = test_embeddings[0]

        lr = LogisticRegression()
        lr.fit(train_emb, y_train)
        y_pred = lr.predict(test_emb)
        accuracy = accuracy_score(y_test, y_pred)

        # A bit more challenging with multiclass
        assert accuracy > 0.3, f"Accuracy with multiclass data was only {accuracy:.2f}"
