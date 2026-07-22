"""Tests for the client-backend ``output_type`` handling (GH #350).

``ClientTabPFNRegressor`` used to override ``predict`` to import
``FullSupportBarDistribution`` from ``tabpfn.model.bar_distribution`` — a
module removed in tabpfn 8.1 — and raised before ever looking at the client
output. It also silently dropped any other ``output_type`` (e.g.
``"quantiles"``) instead of passing it to the parent predict.

The override is gone: tabpfn-client >= 0.2.7 handles every output type itself
and attaches a compatible ``criterion`` to ``output_type="full"``. These tests
pin that the wrapper defers to the client untouched, with the (network-bound)
parent predict stubbed out so no API credentials are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_extensions import utils


@pytest.fixture
def client_regressor():
    """A ClientTabPFNRegressor instance (construction needs no credentials)."""
    pytest.importorskip("tabpfn_client")
    if utils.ClientTabPFNRegressor is None:
        pytest.skip("tabpfn-client wrapper not available")
    return utils.ClientTabPFNRegressor()


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_wrapper_full_output_keeps_client_criterion(monkeypatch, client_regressor):
    """predict(output_type='full') returns the client's output untouched."""
    sentinel = object()
    full_output = {
        "logits": np.zeros((4, 5), dtype=np.float32),
        "borders": np.linspace(-2.0, 2.0, 6, dtype=np.float32),
        "mean": np.zeros(4, dtype=np.float32),
        "criterion": sentinel,
    }

    def fake_predict(self, X, output_type="mean", quantiles=None):
        assert output_type == "full"
        return full_output

    monkeypatch.setattr(utils.ClientTabPFNRegressorBase, "predict", fake_predict)

    result = client_regressor.predict(np.zeros((3, 2)), output_type="full")

    assert result is full_output
    assert result["criterion"] is sentinel


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_wrapper_passes_other_output_types_through(monkeypatch, client_regressor):
    """Predict forwards output_type and extra kwargs to the parent predict."""
    received = {}

    def fake_predict(self, X, output_type="mean", quantiles=None):
        received["output_type"] = output_type
        received["quantiles"] = quantiles
        return [np.zeros(3), np.zeros(3)]

    monkeypatch.setattr(utils.ClientTabPFNRegressorBase, "predict", fake_predict)

    client_regressor.predict(
        np.zeros((3, 2)), output_type="quantiles", quantiles=[0.1, 0.9]
    )

    assert received == {"output_type": "quantiles", "quantiles": [0.1, 0.9]}
