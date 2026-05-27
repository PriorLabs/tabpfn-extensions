"""Tests for TabPFN embedding functionality.

These tests check the functionality of TabPFNEmbedding class for extracting
embeddings from TabPFN models, both in vanilla mode and with K-fold cross-validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from conftest import (
    DEFAULT_TEST_SIZE,
    FAST_TEST_MODE,
    SMALL_TEST_SIZE,
)
from tabpfn_extensions.embedding import TabPFNEmbedding
from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor


@pytest.mark.local_compatible
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestTabPFNEmbeddingLegacy:
    """Test suite for the deprecated TabPFNEmbedding.get_embeddings interface."""

    @pytest.fixture
    def classification_data(self):
        """Generate synthetic classification data."""
        n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
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
        n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
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
        from utils import DatasetGenerator

        return DatasetGenerator(seed=42)

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

    def test_embeddings_with_missing_values(self, dataset_generator):
        """Test embedding extraction with missing values."""
        # Use the missing values dataset generator
        X_missing, y = dataset_generator.generate_missing_values_dataset(
            n_samples=30 if FAST_TEST_MODE else 60,
            n_features=5,
            missing_rate=0.1,
            task_type="classification",
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_missing,
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

        assert isinstance(test_embeddings, np.ndarray)
        assert test_embeddings.ndim >= 2

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
        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

    def test_embeddings_with_text_features(self, dataset_generator):
        """Test embedding extraction with text features."""
        # Generate data with text features
        X_original, y = dataset_generator.generate_text_dataset(
            n_samples=30 if FAST_TEST_MODE else 60,
            task_type="classification",
        )

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_original,  # Use the DataFrame with text features
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
        embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_test,
            data_source="test",
        )

        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

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

        # Check embedding shapes
        assert isinstance(train_embeddings, np.ndarray)
        assert train_embeddings.ndim >= 2

    def test_get_embeddings_emits_deprecation(self, classification_data):
        """Calling the legacy get_embeddings() method emits a DeprecationWarning."""
        X_train, X_test, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        extractor = TabPFNEmbedding(n_fold=0, model=clf)
        with pytest.warns(DeprecationWarning, match="get_embeddings"):
            extractor.get_embeddings(X_train, y_train, X_test, data_source="test")

    def test_legacy_kwargs_emit_deprecation(self, classification_data):
        """tabpfn_clf / tabpfn_reg kwargs emit a DeprecationWarning at fit-time."""
        X_train, _X_test, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
        with pytest.warns(DeprecationWarning, match="tabpfn_clf"):
            extractor.fit(X_train, y_train)


@pytest.mark.local_compatible
class TestTabPFNEmbedding:
    """Test suite for the sklearn-style TabPFNEmbedding."""

    @pytest.fixture
    def classification_data(self):
        n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
        X, y = make_classification(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.fixture
    def regression_data(self):
        n_samples = SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE
        X, y = make_regression(
            n_samples=n_samples,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_vanilla_classifier(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """n_fold=0 path with an explicit classifier."""
        X_train, X_test, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        transformer = TabPFNEmbedding(n_fold=0, model=clf)
        transformer.fit(X_train, y_train)

        train_embeds = transformer.train_embeddings_
        test_embeds = transformer.transform(X_test)

        assert isinstance(train_embeds, np.ndarray)
        assert train_embeds.ndim >= 2
        assert train_embeds.shape[1] == X_train.shape[0]
        assert test_embeds.shape[1] == X_test.shape[0]
        assert train_embeds.shape[-1] == test_embeds.shape[-1]

    def test_kfold_classifier_oof_shape(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """K-fold path: train_embeddings_ covers every training sample once."""
        X_train, X_test, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        transformer = TabPFNEmbedding(n_fold=3, model=clf)
        transformer.fit(X_train, y_train)

        train_embeds = transformer.train_embeddings_
        test_embeds = transformer.transform(X_test)

        assert train_embeds.shape[1] == X_train.shape[0]
        assert test_embeds.shape[1] == X_test.shape[0]
        assert train_embeds.shape[-1] == test_embeds.shape[-1]

    def test_vanilla_regressor(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X_train, X_test, y_train, _ = regression_data
        reg = TabPFNRegressor(n_estimators=1, random_state=42)
        transformer = TabPFNEmbedding(n_fold=0, model=reg)
        transformer.fit(X_train, y_train)

        assert transformer.train_embeddings_.shape[1] == X_train.shape[0]
        assert transformer.transform(X_test).shape[1] == X_test.shape[0]

    def test_kfold_regressor(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X_train, X_test, y_train, _ = regression_data
        reg = TabPFNRegressor(n_estimators=1, random_state=42)
        transformer = TabPFNEmbedding(n_fold=3, model=reg)
        transformer.fit(X_train, y_train)

        assert transformer.train_embeddings_.shape[1] == X_train.shape[0]
        assert transformer.transform(X_test).shape[1] == X_test.shape[0]

    def test_transform_always_uses_final_model(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """transform(X_train) goes through the final full-data model — it does
        NOT return cached OOF embeddings, and should therefore differ from
        train_embeddings_ when CV is on.
        """
        X_train, _X_test, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        transformer = TabPFNEmbedding(n_fold=3, model=clf)
        transformer.fit(X_train, y_train)

        oof = transformer.train_embeddings_
        via_transform = transformer.transform(X_train)
        assert oof.shape == via_transform.shape
        assert not np.allclose(oof, via_transform)

    def test_fit_transform_returns_oof(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """fit_transform returns OOF (train_embeddings_), not transform output."""
        X_train, _X_test, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        transformer = TabPFNEmbedding(n_fold=3, model=clf)
        ft = transformer.fit_transform(X_train, y_train)
        np.testing.assert_array_equal(ft, transformer.train_embeddings_)
        assert not np.allclose(ft, transformer.transform(X_train))

    def test_invalid_n_fold(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X_train, _, y_train, _ = classification_data
        clf = TabPFNClassifier(n_estimators=1, random_state=42)
        with pytest.raises(ValueError, match="n_fold"):
            TabPFNEmbedding(n_fold=1, model=clf).fit(X_train, y_train)
        with pytest.raises(ValueError, match="n_fold"):
            TabPFNEmbedding(n_fold=-1, model=clf).fit(X_train, y_train)

    def test_auto_task_classification_warns(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Without an explicit model, classification is inferred and warns."""
        X_train, _, y_train, _ = classification_data
        transformer = TabPFNEmbedding(n_fold=0)
        with pytest.warns(UserWarning, match="No `model=` provided"):
            transformer.fit(X_train, y_train)
        assert isinstance(transformer.model_, TabPFNClassifier)

    def test_auto_task_regression_warns(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X_train, _, y_train, _ = regression_data
        transformer = TabPFNEmbedding(n_fold=0)
        with pytest.warns(UserWarning, match="No `model=` provided"):
            transformer.fit(X_train, y_train)
        assert isinstance(transformer.model_, TabPFNRegressor)

    def test_transform_before_fit_raises(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        from sklearn.exceptions import NotFittedError

        X_train, _, _, _ = classification_data
        transformer = TabPFNEmbedding(n_fold=0)
        with pytest.raises(NotFittedError):
            transformer.transform(X_train)

    def test_shuffle_independent_of_random_state(
        self,
        classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """shuffle=True with two different random_states should give different
        OOF orderings; shuffle=False should ignore random_state entirely.
        """
        X_train, _X_test, y_train, _ = classification_data

        def clf() -> TabPFNClassifier:
            return TabPFNClassifier(n_estimators=1, random_state=0)

        t1 = TabPFNEmbedding(
            n_fold=3,
            model=clf(),
            shuffle=True,
            random_state=0,
        )
        t2 = TabPFNEmbedding(
            n_fold=3,
            model=clf(),
            shuffle=True,
            random_state=1,
        )
        oof1 = t1.fit_transform(X_train, y_train)
        oof2 = t2.fit_transform(X_train, y_train)
        assert not np.allclose(oof1, oof2)

        t3 = TabPFNEmbedding(
            n_fold=3,
            model=clf(),
            shuffle=False,
            random_state=0,
        )
        t4 = TabPFNEmbedding(
            n_fold=3,
            model=clf(),
            shuffle=False,
            random_state=1,
        )
        np.testing.assert_array_equal(
            t3.fit_transform(X_train, y_train),
            t4.fit_transform(X_train, y_train),
        )
