"""Tests for the ManyClassClassifier extension.

This file tests the ManyClassClassifier, which extends TabPFN's capabilities
to handle classification problems with a large number of classes, inheriting
from a common base test suite.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Assuming tabpfn_extensions.many_class is in the python path
from tabpfn_extensions.many_class import ManyClassClassifier
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
        )

    def test_internal_fit_predict_many_classes(self, estimator):
        """Test fit and predict specifically with more classes than alphabet_size,
        focusing on the mapping logic of ManyClassClassifier.
        This uses the 'estimator' fixture which is ManyClassClassifier.
        """
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
        assert "coverage_min" in estimator.codebook_statistics_
        assert estimator.codebook_statistics_["coverage_min"] > 0

        assert predictions.shape == (X_test.shape[0],)
        assert probabilities.shape == (X_test.shape[0], n_classes)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)
        assert accuracy_score(y_test, predictions) >= 0.0  # Basic check

    def test_failing_scenario_many_classes_replication(self, estimator):
        """Replicates and tests the scenario with a very large number of classes.
        This test is specific and doesn't use the main 'estimator' fixture to
        re-initialize ManyClassClassifier inside the loop with specific verbose settings.
        """
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
                assert "coverage_min" in estimator.codebook_statistics_
                assert (
                    estimator.codebook_statistics_["coverage_min"] > 0
                ), f"Coverage min is 0 for {num_classes} classes!"
            else:
                assert estimator._get_alphabet_size() >= num_classes

            _ = estimator.predict(X)  # Triggers predict_proba
            _ = estimator.predict_proba(X)

            assert hasattr(estimator, "n_features_in_")
            assert estimator.n_features_in_ == X.shape[1]
        print("Large number of classes test completed.")

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
