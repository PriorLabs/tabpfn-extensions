"""Tests for the ManyClassClassifier extension.

This file tests the ManyClassClassifier, which extends TabPFN's capabilities
to handle classification problems with a large number of classes, inheriting
from a common base test suite.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Assuming tabpfn_extensions.many_class is in the python path
from tabpfn_extensions.many_class import (
    AggregationConfig,
    CodebookConfig,
    ManyClassClassifier,
)
from test_base_tabpfn import BaseClassifierTests


# Helper function (as provided in the initial problem description)
def get_classification_data(num_classes: int, num_features: int, num_samples: int):
    assert (
        num_samples >= num_classes
    ), "Number of samples must be at least the number of classes."
    X = np.random.randn(num_samples, num_features)
    y = np.concatenate(
        [
            np.arange(num_classes),
            np.random.randint(0, num_classes, size=num_samples - num_classes),
        ]
    )
    y = np.random.permutation(y)
    assert np.unique(y).size == num_classes
    return X, y


class TestManyClassClassifier(BaseClassifierTests):  # Inherit from BaseClassifierTests
    """Test suite for the ManyClassClassifier, including specific tests for its
    many-class handling capabilities and inheriting general classifier tests.
    """

    @pytest.fixture
    def estimator(
        self, tabpfn_classifier
    ):  # This fixture is required by BaseClassifierTests
        """Provides a ManyClassClassifier instance with a TabPFN base."""
        return ManyClassClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            alphabet_size=10,
            n_estimators_redundancy=2,
            random_state=42,
            codebook_config=CodebookConfig(strategy="legacy_rest"),
        )

    def test_internal_fit_predict_many_classes(self, estimator):
        """Test fit/predict when the wrapper must build a codebook."""
        n_classes = 15  # More than default alphabet_size of 10
        n_features = 4
        n_samples = n_classes * 20
        X, y = get_classification_data(
            num_classes=n_classes, num_features=n_features, num_samples=n_samples
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        estimator.fit(X_train, y_train)  # estimator is ManyClassClassifier
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)

        assert (
            not estimator.no_mapping_needed_
        ), "Mapping should have been used for 15 classes."
        assert estimator.code_book_ is not None
        assert estimator.code_book_.shape[1] == n_classes
        assert (
            estimator.estimators_ is None
        )  # Fit happens during predict_proba when mapping
        stats = estimator.codebook_statistics_
        assert stats.get("coverage_min", 0) > 0
        assert stats.get("strategy") in {"balanced_cluster", "legacy_rest"}
        assert stats.get("regeneration_attempts", 0) >= 1
        assert "best_min_pairwise_hamming_dist" in stats

        assert predictions.shape == (X_test.shape[0],)
        assert probabilities.shape == (X_test.shape[0], n_classes)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)
        assert accuracy_score(y_test, predictions) >= 0.0  # Basic check
        assert estimator.row_weights_ is not None
        assert estimator.row_train_support_ is not None
        assert estimator.row_weights_.shape[0] == estimator.code_book_.shape[0]
        assert estimator.row_train_support_.shape[0] == estimator.code_book_.shape[0]

    def test_failing_scenario_many_classes_replication(self, estimator):
        """Exercise multiple class counts to ensure coverage and stats."""
        logging.info("Testing ManyClassClassifier with a large number of classes:")
        for num_classes in [2, 10, 24, 81]:  # Reduced range for test speed
            logging.info(f"  Testing with num_classes = {num_classes}")
            X, y = get_classification_data(
                num_classes=num_classes,
                num_features=10,
                num_samples=2 * num_classes,
            )

            estimator.fit(X, y)

            if not estimator.no_mapping_needed_:
                assert estimator._get_alphabet_size() < num_classes
                assert estimator.code_book_ is not None
                assert estimator.code_book_.shape[1] == num_classes
                stats = estimator.codebook_statistics_
                assert (
                    stats.get("coverage_min", 0) > 0
                ), f"Coverage min is 0 for {num_classes} classes!"
                assert stats.get("regeneration_attempts", 0) >= 1
            else:
                assert estimator._get_alphabet_size() >= num_classes

            _ = estimator.predict(X)  # Triggers predict_proba
            _ = estimator.predict_proba(X)

            assert hasattr(estimator, "n_features_in_")
            assert estimator.n_features_in_ == X.shape[1]
        print("Large number of classes test completed.")

    def test_wrapper_retains_base_performance_on_sparse_multi_class_problem(self):
        """Check that the wrapper matches a strong base estimator on a sparse dataset.

        We mimic a 2-class-limited base model by using an alphabet of size 2 on a
        10-class problem with very few samples per class. The wrapper should come
        close to the base estimator's accuracy despite the heavy output coding.
        """
        X, y = make_blobs(
            n_samples=60,
            centers=10,
            cluster_std=0.3,
            random_state=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=0,
            stratify=y,
        )

        base_estimator = LogisticRegression(max_iter=200)
        base_estimator.fit(X_train, y_train)
        base_accuracy = accuracy_score(y_test, base_estimator.predict(X_test))

        wrapped = ManyClassClassifier(
            estimator=LogisticRegression(max_iter=200),
            alphabet_size=2,
            n_estimators=20,
            random_state=0,
            codebook_config="legacy_rest",
            aggregation_config=AggregationConfig(log_likelihood=True),
        )
        wrapped.fit(X_train, y_train)
        wrapped_accuracy = accuracy_score(y_test, wrapped.predict(X_test))

        assert wrapped_accuracy >= 0.8
        assert wrapped_accuracy >= base_accuracy - 0.05

    def test_sample_weight_is_forwarded_to_sub_estimators(self):
        """Ensure fit_params (like sample_weight) reach every cloned estimator."""
        fit_records: list[np.ndarray | None] = []

        class RecordingEstimator(BaseEstimator, ClassifierMixin):
            def fit(self, X, y, sample_weight=None):
                if sample_weight is not None:
                    fit_records.append(np.asarray(sample_weight).copy())
                else:
                    fit_records.append(None)
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.classes_[0], dtype=self.classes_.dtype)

            def predict_proba(self, X):
                n_samples = X.shape[0]
                proba = np.full(
                    (n_samples, len(self.classes_)), 1.0 / len(self.classes_)
                )
                return proba

        base_estimator = RecordingEstimator()
        wrapper = ManyClassClassifier(
            estimator=base_estimator,
            alphabet_size=3,
            n_estimators=6,
            random_state=0,
        )

        rng = np.random.RandomState(0)
        X = rng.randn(30, 2)
        y = rng.randint(0, 5, size=30)
        sample_weight = rng.rand(30)

        wrapper.fit(X, y, sample_weight=sample_weight)
        wrapper.predict_proba(X[:5])

        n_estimators_used = wrapper.code_book_.shape[0]
        assert len(fit_records) == n_estimators_used
        for recorded in fit_records:
            assert recorded is not None
            np.testing.assert_allclose(recorded, sample_weight)

    def test_legacy_rest_filtering_drops_rest_samples(self):
        """Legacy rest strategy can drop rest-labeled samples before fitting."""
        fit_y_records: list[np.ndarray] = []
        fit_weight_records: list[np.ndarray | None] = []

        class RecordingEstimator(BaseEstimator, ClassifierMixin):
            def fit(self, X, y, sample_weight=None):
                fit_y_records.append(np.asarray(y).copy())
                if sample_weight is not None:
                    fit_weight_records.append(np.asarray(sample_weight).copy())
                else:
                    fit_weight_records.append(None)
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.classes_[0], dtype=self.classes_.dtype)

            def predict_proba(self, X):
                n_samples = X.shape[0]
                return np.full(
                    (n_samples, len(self.classes_)), 1.0 / len(self.classes_)
                )

        rng = np.random.RandomState(1)
        X = rng.randn(40, 3)
        y = rng.randint(0, 6, size=40)
        sample_weight = rng.rand(40)

        wrapper = ManyClassClassifier(
            estimator=RecordingEstimator(),
            alphabet_size=3,
            n_estimators=5,
            random_state=0,
            codebook_config=CodebookConfig(
                strategy="legacy_rest", legacy_filter_rest_train=True
            ),
            aggregation_config=AggregationConfig(
                log_likelihood=True, legacy_mask_rest_log_agg=True
            ),
        )
        wrapper.fit(X, y, sample_weight=sample_weight)

        proba = wrapper.predict_proba(X)

        assert proba.shape == (X.shape[0], len(wrapper.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)

        rest_code = wrapper.codebook_statistics_.get("rest_class_code")
        assert rest_code is not None

        rows_with_rest = [
            idx
            for idx in range(wrapper.code_book_.shape[0])
            if np.any(wrapper.Y_train_per_estimator[idx] == rest_code)
        ]
        assert rows_with_rest, "Expected at least one row to include the rest symbol."

        assert wrapper.row_train_support_ is not None
        expected_support = np.sum(wrapper.Y_train_per_estimator != rest_code, axis=1)
        np.testing.assert_array_equal(wrapper.row_train_support_, expected_support)

        assert len(fit_y_records) == int(np.count_nonzero(expected_support))
        assert len(fit_weight_records) == len(fit_y_records)

        for labels in fit_y_records:
            assert rest_code not in labels

        for weights, labels in zip(fit_weight_records, fit_y_records):
            if weights is not None:
                assert weights.shape[0] == labels.shape[0]

    def test_predict_proba_handles_sub_estimator_missing_codes(self):
        """predict_proba should expand sub-estimator outputs to the full alphabet."""
        rng = np.random.RandomState(1)
        X = rng.randn(60, 3)
        y = rng.randint(0, 5, size=60)

        wrapper = ManyClassClassifier(
            estimator=LogisticRegression(max_iter=200),
            alphabet_size=3,
            n_estimators=4,
            random_state=0,
        )

        wrapper.fit(X, y)

        # Force the first sub-problem to only observe a strict subset of the alphabet.
        rest_code = wrapper.alphabet_size_ - 1
        class_mask = y == wrapper.classes_[0]
        assert class_mask.any()
        assert (~class_mask).any()
        wrapper.Y_train_per_estimator[0, class_mask] = 0
        wrapper.Y_train_per_estimator[0, ~class_mask] = rest_code

        probas = wrapper.predict_proba(X[:7])

        assert probas.shape == (7, len(wrapper.classes_))
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-9)

    @pytest.mark.skip(reason="DecisionTreeTabPFN doesn't fully support text features")
    def test_with_text_features(self, estimator, dataset_generator):
        pass

    @pytest.mark.skip(
        reason="DecisionTreeTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass

    @pytest.mark.skip(reason="Disabled due to backend=tabpfn_client failures.")
    def test_with_pandas(self, estimator, pandas_classification_data):
        pass

    @pytest.mark.skip(
        reason="Disabled due to DecisionTreeTabPFN not supporting missing values."
    )
    def test_with_missing_values(self, estimator, dataset_generator):
        pass
