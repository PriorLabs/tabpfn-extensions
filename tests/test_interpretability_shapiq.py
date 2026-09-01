"""Tests for the shapiq interpretability adapters.

Focused on ``get_tabpfn_inf_explainer``, which masks absent features with
``+inf`` and relies on TabPFN's opt-in ``PASSTHROUGH_INF`` inference config.

The end-to-end cases are local-only, not because the feature is: the remote
backends forward ``inference_config``/``PASSTHROUGH_INF`` to the TabPFN they
run, and ``TestInfPassthroughDetection`` covers them with stubs. They run on
CPU (fp32), where the
value function is deterministic and the Shapley efficiency identity is tight —
so we can compare the imputer's masking against hand-built masked predictions
exactly, rather than merely asserting the pipeline "runs".
"""

from __future__ import annotations

from typing import ClassVar

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
    """v(empty) is one all-+inf forward pass, not a predict over the whole
    background (the base MarginalImputer behaviour that OOMs on large data).
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


@pytest.mark.local_compatible
def test_inf_explainer_handles_string_column(classification_data_with_text):
    """+inf masking works on a dataset with a string/categorical column.

    Such a row is an object array, so the masked features must stay object dtype
    for +inf to be assignable, and TabPFN absorbs the +inf as missing. Checks the
    explainer runs end to end and that masking the string feature with +inf
    reproduces a hand-built object-array prediction.
    """
    from shapiq.explainer.utils import get_predict_function_and_model_type

    df, y = classification_data_with_text
    d = df.shape[1]
    text_col = df.columns.get_loc("text")
    clf = TabPFNClassifier(
        device="cpu", n_estimators=1, inference_config={"PASSTHROUGH_INF": True}
    )
    clf.fit(df, y)
    predict_fn, _ = get_predict_function_and_model_type(clf, class_index=CLASS_INDEX)

    explainer = tpe_shapiq.get_tabpfn_inf_explainer(
        model=clf, data=df, class_index=CLASS_INDEX
    )

    # End-to-end run over the object-dtype array (numeric columns + a string).
    x0 = df.iloc[0].to_numpy()
    iv = explainer.explain(x=x0, budget=2**d)
    sv = iv.get_n_order_values(1)
    assert sv.shape == (d,)
    assert np.isfinite(sv).all()

    # Dropping only the string feature keeps an object array and matches a
    # hand-built prediction with that feature replaced by +inf.
    imputer = explainer.imputer
    imputer.fit(x0)
    coalition = np.ones((1, d), dtype=bool)
    coalition[0, text_col] = False
    got = imputer.value_function(coalition)
    expected_row = x0.astype(object).copy()
    expected_row[text_col] = np.inf
    assert np.allclose(got, predict_fn(clf, expected_row.reshape(1, -1)), atol=1e-5)


class TestInfPassthroughDetection:
    """Which backends may use the +inf masking path.

    The flag lives in one of two places: a local model resolves it into
    ``get_inference_config()``, while a remote backend only forwards the
    ``inference_config`` constructor argument. Stubs stand in for the client
    estimators, which are not a test dependency here.
    """

    class _Remote:
        """Shaped like any tabpfn-client estimator: no resolvable config."""

        def __init__(self, inference_config=None):
            self.inference_config = inference_config

    @pytest.mark.local_compatible
    @pytest.mark.client_compatible
    def test_remote_backend_with_the_flag_is_accepted(self):
        """Both the managed API and a self-hosted endpoint honour the flag."""
        model = self._Remote(inference_config={"PASSTHROUGH_INF": True})
        assert tpe_shapiq._model_has_inf_passthrough(model)

    @pytest.mark.local_compatible
    @pytest.mark.client_compatible
    def test_remote_backend_without_the_flag_is_rejected(self):
        assert not tpe_shapiq._model_has_inf_passthrough(self._Remote())
        assert not tpe_shapiq._model_has_inf_passthrough(
            self._Remote(inference_config={"PASSTHROUGH_INF": False}),
        )

    @pytest.mark.local_compatible
    @pytest.mark.client_compatible
    def test_remote_backend_without_the_flag_raises(self, classification_data):
        X, _ = classification_data
        with pytest.raises(ValueError, match="PASSTHROUGH_INF"):
            tpe_shapiq.get_tabpfn_inf_explainer(model=self._Remote(), data=X)

    @pytest.mark.local_compatible
    @pytest.mark.client_compatible
    def test_local_model_still_uses_its_resolved_config(self):
        """A local model's resolved config wins; its raw argument is not read."""

        class _Local:
            inference_config: ClassVar[dict] = {"PASSTHROUGH_INF": True}

            def __init__(self, *, resolved: bool):
                self._resolved = resolved

            def get_inference_config(self):
                return type("Cfg", (), {"PASSTHROUGH_INF": self._resolved})()

        assert tpe_shapiq._model_has_inf_passthrough(_Local(resolved=True))
        assert not tpe_shapiq._model_has_inf_passthrough(_Local(resolved=False))
