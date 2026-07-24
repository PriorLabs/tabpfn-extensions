"""Tests for the decoder-head readout (``get_decoder_readout`` / ``class_vote``).

Local-only: the readout reads TabPFN's ``ManyClassDecoder`` internals via the
``model_`` handle, which the client backend does not expose. The key check is an
end-to-end identity: collapsing the recovered attention weights by training label
and averaging over the ensemble reproduces ``predict_proba`` up to the head's
log-clamping. If row alignment or the attention math were wrong, this would break.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_extensions.interpretability import class_vote, get_decoder_readout
from tabpfn_extensions.utils import TabPFNClassifier


@pytest.fixture
def fitted_clf_split(classification_data):
    """A fitted local classifier plus the held-out test split it was not fit on."""
    X, y = classification_data
    n_train = 2 * len(X) // 3
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    clf = TabPFNClassifier(device="cpu", n_estimators=2, random_state=0)
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


@pytest.mark.local_compatible
def test_shapes_and_normalization(fitted_clf_split):
    clf, X_train, X_test, _, _ = fitted_clf_split
    weights, train_indices = get_decoder_readout(clf, X_test)

    assert weights.shape == (len(X_test), len(X_train))
    assert train_indices.shape == (len(X_train),)
    np.testing.assert_array_equal(train_indices, np.arange(len(X_train)))
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.local_compatible
def test_per_estimator_axis(fitted_clf_split):
    clf, X_train, X_test, _, _ = fitted_clf_split
    per_est, _ = get_decoder_readout(clf, X_test, average_over_estimators=False)
    avg, _ = get_decoder_readout(clf, X_test)

    assert per_est.shape[1:] == (len(X_test), len(X_train))
    assert per_est.shape[0] >= 1
    np.testing.assert_allclose(per_est.mean(axis=0), avg, atol=1e-5)


@pytest.mark.local_compatible
def test_class_vote_matches_predict_proba(fitted_clf_split):
    """The label-collapsed readout reproduces predict_proba (bar log-clamping)."""
    clf, _, X_test, y_train, _ = fitted_clf_split
    weights, _ = get_decoder_readout(clf, X_test)
    votes, classes = class_vote(weights, y_train)

    np.testing.assert_array_equal(classes, clf.classes_)
    np.testing.assert_allclose(votes.sum(axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(votes, clf.predict_proba(X_test), atol=0.05)
