"""Tests for the client-backend ``output_type="full"`` handling (GH #350).

``ClientTabPFNRegressor.predict(..., output_type="full")`` used to import
``FullSupportBarDistribution`` from ``tabpfn.model.bar_distribution`` — a
module removed in tabpfn 8.1 — and raised before ever looking at the client
output, even though current tabpfn-client already returns a compatible
``criterion``. It also silently dropped any other ``output_type`` (e.g.
``"quantiles"``) instead of passing it to the parent predict.

These tests pin the fixed behaviour without needing API credentials: the
criterion handling is tested through the module-level helper, and the wrapper
itself is tested with the parent's (network-bound) predict stubbed out.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tabpfn_extensions import utils
from tabpfn_extensions.utils import _client_full_output_with_criterion


def _full_output_without_criterion() -> dict:
    """A minimal client full-output payload, as returned by older clients."""
    rng = np.random.default_rng(0)
    return {
        "logits": rng.standard_normal((4, 5)).astype(np.float32),
        "borders": np.linspace(-2.0, 2.0, 6, dtype=np.float32),
        "mean": rng.standard_normal(4).astype(np.float32),
    }


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_client_criterion_is_passed_through():
    """Current tabpfn-client already attaches a criterion; keep it untouched."""
    sentinel = object()
    output = {**_full_output_without_criterion(), "criterion": sentinel}

    result = _client_full_output_with_criterion(output)

    assert result is output
    assert result["criterion"] is sentinel


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_missing_criterion_is_reconstructed_from_borders():
    """Older clients return raw arrays only; the criterion is rebuilt from borders."""
    pytest.importorskip("tabpfn")
    torch = pytest.importorskip("torch")

    output = _full_output_without_criterion()
    result = _client_full_output_with_criterion(output)

    assert "criterion" not in output, "the client output must not be mutated"
    criterion = result["criterion"]
    assert hasattr(criterion, "sample")
    torch.testing.assert_close(
        criterion.borders,
        torch.as_tensor(output["borders"]),
        rtol=0,
        atol=0,
    )
    # The reconstructed criterion must be usable, e.g. to compute the mean.
    mean = criterion.mean(torch.as_tensor(output["logits"]))
    assert mean.shape == (4,)


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_missing_criterion_without_tabpfn_raises_helpful_error(monkeypatch):
    """Without the local tabpfn package, reconstruction fails with a clear error."""
    for name in list(sys.modules):
        if name == "tabpfn" or name.startswith("tabpfn."):
            monkeypatch.delitem(sys.modules, name)
    # A None entry in sys.modules makes any "import tabpfn..." raise ImportError.
    monkeypatch.setitem(sys.modules, "tabpfn", None)

    with pytest.raises(ValueError, match="pip install tabpfn"):
        _client_full_output_with_criterion(_full_output_without_criterion())


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
    """predict(output_type='full') returns the client's own criterion untouched."""
    sentinel = object()

    def fake_predict(self, X, output_type="mean", quantiles=None):
        assert output_type == "full"
        return {**_full_output_without_criterion(), "criterion": sentinel}

    monkeypatch.setattr(utils.ClientTabPFNRegressorBase, "predict", fake_predict)

    result = client_regressor.predict(np.zeros((3, 2)), output_type="full")

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
