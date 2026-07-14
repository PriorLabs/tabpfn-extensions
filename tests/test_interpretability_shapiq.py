"""Tests for the shapiq interpretability adapters.

Focused on ``get_tabpfn_inf_explainer`` (PRI-290), which masks absent features
with ``+inf`` and relies on TabPFN's opt-in ``PASSTHROUGH_INF`` inference config.

These are local-only: ``inference_config``/``PASSTHROUGH_INF`` is a local-tabpfn
feature not exposed by the client backend. They run on CPU (fp32), where the
value function is deterministic and the Shapley efficiency identity is tight —
so we can compare the imputer's masking against hand-built masked predictions
exactly, rather than merely asserting the pipeline "runs".
"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_extensions.interpretability import shapiq as tpe_shapiq
from tabpfn_extensions.utils import TabPFNClassifier

pytest.importorskip("shapiq")

CLASS_INDEX = 1


@pytest.fixture
def passthrough_clf(classification_data):
    """A fitted classifier with +inf passthrough and the KV cache enabled."""
    X, y = classification_data
    clf = TabPFNClassifier(
        device="cpu",
        n_estimators=1,
        fit_mode="fit_with_cache",
        inference_config={"PASSTHROUGH_INF": True},
    )
    clf.fit(X, y)
    clf.executor_.keep_cache_on_device = True
    return clf


def _class_predict(clf):
    """The class-aware predict callable shapiq uses internally (logit space)."""
    from shapiq.explainer.utils import get_predict_function_and_model_type

    predict_fn, _ = get_predict_function_and_model_type(clf, class_index=CLASS_INDEX)
    return lambda X: predict_fn(clf, np.asarray(X, dtype=float))


@pytest.mark.local_compatible
def test_raises_without_passthrough(classification_data):
    """Without PASSTHROUGH_INF the wrapper fails fast with an actionable error,
    instead of letting a cryptic validation error surface at predict time.
    """
    X, y = classification_data
    clf = TabPFNClassifier(device="cpu", n_estimators=1)  # PASSTHROUGH_INF off
    clf.fit(X, y)
    with pytest.raises(ValueError, match="PASSTHROUGH_INF"):
        tpe_shapiq.get_tabpfn_inf_explainer(model=clf, data=X)


@pytest.mark.local_compatible
def test_value_function_masks_absent_features_with_inf(
    classification_data, passthrough_clf
):
    """The core behaviour: for each coalition the imputer sets exactly the
    *absent* features to +inf, keeps the present ones at x, and feeds that to the
    model — matching a hand-built masked prediction, batched over coalitions.
    """
    X, _ = classification_data
    d = X.shape[1]
    predict = _class_predict(passthrough_clf)

    imputer = tpe_shapiq.get_tabpfn_inf_explainer(
        model=passthrough_clf, data=X, class_index=CLASS_INDEX
    ).imputer
    x = X[0].astype(float)
    imputer.fit(x)

    # A partial coalition (keep features 0 and 2), the full and empty coalitions,
    # evaluated together to also exercise batching.
    partial = np.zeros(d, dtype=bool)
    partial[[0, 2]] = True
    coalitions = np.stack([partial, np.ones(d, bool), np.zeros(d, bool)])

    got = imputer.value_function(coalitions)

    # The imputer's masking (absent -> +inf, present -> x) must reproduce an
    # independently hand-masked prediction, across the partial, full and empty
    # coalitions. A wrong fill value (e.g. NaN) or masking the wrong features
    # would make `got` diverge from `expected`.
    expected = predict(np.stack([np.where(coal, x, np.inf) for coal in coalitions]))
    assert got.shape == (3,)
    assert np.allclose(got, expected, atol=1e-5)


@pytest.mark.local_compatible
def test_calc_empty_prediction_uses_single_all_inf_row(
    classification_data, passthrough_clf
):
    """OOM fix (PRI-290): v(empty) is one all-+inf forward pass, not a predict
    over the whole background (the base MarginalImputer behaviour that OOMs).
    """
    X, _ = classification_data
    d = X.shape[1]
    predict = _class_predict(passthrough_clf)
    explainer = tpe_shapiq.get_tabpfn_inf_explainer(
        model=passthrough_clf, data=X, class_index=CLASS_INDEX
    )

    calls = []
    orig_predict = explainer.imputer.predict

    def spy(arr):
        calls.append(np.asarray(arr))
        return orig_predict(arr)

    explainer.imputer.predict = spy
    value = explainer.imputer.calc_empty_prediction()

    assert len(calls) == 1
    row = calls[0]
    assert row.shape == (1, d)  # a single row, not len(X)
    assert np.isinf(row).all()  # all +inf
    assert np.isclose(value, predict(np.full((1, d), np.inf))[0], atol=1e-5)
