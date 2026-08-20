"""Tests for the decoder-head readout (``get_decoder_readout`` / ``class_vote``).

Local-only: the readout reads TabPFN's ``ManyClassDecoder`` internals via the
``model_`` handle, which the client backend does not expose. The weights come from
the head's own ``attention_weights`` method, so these tests are mostly an
upstream-contract guard: the key check is an end-to-end identity where collapsing
the recovered attention weights by training label and averaging over the ensemble
reproduces ``predict_proba`` up to the head's log-clamping. That identity only holds
at ``softmax_temperature=1.0`` (the library default 0.9 sharpens the vote by
``** (1 / T)`` downstream of the readout) and in full precision, so the fixtures fit
at 1.0 with ``inference_precision=torch.float32`` and the check runs tight. If our row
alignment were wrong, or if upstream's ``attention_weights`` drifted from what its
fused ``forward`` computes, this would break.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn_extensions.interpretability import class_vote, get_decoder_readout
from tabpfn_extensions.utils import TabPFNClassifier


@pytest.fixture(params=["classification_data", "multiclass_data"])
def fitted_clf_split(request):
    """A fitted local classifier plus the held-out test split it was not fit on.

    Parametrized over a binary and a 5-class dataset. Fit at
    ``softmax_temperature=1.0`` so the label-collapsed readout matches
    ``predict_proba`` up to log-clamping alone; at the library default 0.9 the
    temperature sharpens the vote (``∝ vote ** (1 / T)``) and the gap grows with
    the class count.

    ``inference_precision`` is pinned to float32 because ``inference_precision="auto"``
    (the default) turns on bf16 autocast on CPUs with native bf16 support (Intel
    AMX / AVX512-BF16, AMD Zen 4+; see ``tabpfn.utils.infer_autocast_inference_mode``).
    In bf16 the head's ``log(clamp(vote) + 3e-5)`` and the softmax that undoes it no
    longer round-trip, so the identity below holds only to ~1e-2 relative rather than
    to float32 noise. Leaving the precision to the machine made the tight check pass
    or fail depending on which CPU a CI runner happened to land on.
    """
    X, y = request.getfixturevalue(request.param)
    n_train = 2 * len(X) // 3
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    clf = TabPFNClassifier(
        device="cpu",
        n_estimators=2,
        random_state=0,
        softmax_temperature=1.0,
        inference_precision=torch.float32,
    )
    clf.fit(X_train, y_train)
    # Guards the pin: were a forced dtype ever to stop disabling autocast upstream,
    # fail here rather than let the tight identity check flake by hardware again.
    assert getattr(clf, "use_autocast_", False) is False
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
def test_test_row_chunking_raises(classification_data, monkeypatch):
    """Test-row chunking is rejected rather than silently mislabeling axes.

    With ``fit_mode="fit_with_cache"`` and more than ``max_batched_test_rows`` test
    rows, TabPFN runs the decoder once per chunk, so the captured weights no longer
    map to a single ``(n_test, n_train)`` matrix.
    """
    from tabpfn.settings import settings

    X, y = classification_data
    n_train = 2 * len(X) // 3
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]
    clf = TabPFNClassifier(
        device="cpu", n_estimators=2, random_state=0, fit_mode="fit_with_cache"
    )
    clf.fit(X_train, y_train)

    monkeypatch.setattr(settings.tabpfn, "max_batched_test_rows", 1)
    with pytest.raises(NotImplementedError, match="test-row chunking"):
        get_decoder_readout(clf, X_test)


@pytest.mark.local_compatible
def test_class_vote_matches_predict_proba(fitted_clf_split):
    """The label-collapsed readout reproduces predict_proba exactly.

    The decoder turns its vote into logits as ``log(clamp(vote, min=1e-5) + 3e-5)``,
    and at ``softmax_temperature=1.0`` with ``average_before_softmax=False`` the
    softmax simply undoes the log: per estimator, ``predict_proba`` is the clamped,
    shifted vote renormalized, then averaged over the ensemble. Runs for both the binary and the 5-class
    dataset.
    """
    clf, _, X_test, y_train, _ = fitted_clf_split
    per_est, _ = get_decoder_readout(clf, X_test, average_over_estimators=False)
    collapsed = [class_vote(w, y_train) for w in per_est]
    votes = np.stack([v for v, _ in collapsed])
    classes = collapsed[0][1]

    np.testing.assert_array_equal(classes, clf.classes_)
    np.testing.assert_allclose(votes.sum(axis=-1), 1.0, atol=1e-4)

    shifted = np.clip(votes, 1e-5, None) + 3e-5
    expected = (shifted / shifted.sum(axis=-1, keepdims=True)).mean(axis=0)
    np.testing.assert_allclose(expected, clf.predict_proba(X_test), atol=1e-6)
