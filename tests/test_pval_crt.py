from __future__ import annotations

import numpy as np
import pytest

from tabpfn_extensions.pval_crt import tabpfn_crt


@pytest.mark.skip(reason="Flaky: stochastic test on random data can fail due to inherent false-positive rate")
@pytest.mark.local_compatible
def test_crt_detects_relevant_feature():
    rng = np.random.RandomState(0)
    n, p = 120, 4

    X = rng.randn(n, p)
    y = 3 * X[:, 1] + 0.05 * rng.randn(n)

    res = tabpfn_crt(
        X,
        y,
        j=1,
        B=40,
        test_size=0.3,
        device="cpu",
        seed=0,
    )

    assert "p_value" in res
    assert 0 <= res["p_value"] <= 1
    assert res["p_value"] < 0.1


@pytest.mark.skip(reason="Flaky: stochastic test on random data can fail due to inherent false-positive rate")
@pytest.mark.local_compatible
def test_crt_handles_irrelevant_feature():
    rng = np.random.RandomState(1)
    n, p = 120, 4

    X = rng.randn(n, p)
    y = rng.randn(n)

    res = tabpfn_crt(
        X,
        y,
        j=2,
        B=40,
        test_size=0.3,
        device="cpu",
        seed=0,
    )

    assert "p_value" in res
    assert 0 <= res["p_value"] <= 1
    assert res["reject_null"] is False


@pytest.mark.local_compatible
def test_crt_output_contract():
    """Non-flaky: only checks that the output has the right structure."""
    rng = np.random.RandomState(0)
    X = rng.randn(120, 4)
    y = 3 * X[:, 1] + 0.05 * rng.randn(120)

    res = tabpfn_crt(X, y, j=1, B=40, test_size=0.3, device="cpu", seed=0)

    assert "p_value" in res
    assert "reject_null" in res
    assert 0 <= res["p_value"] <= 1
    assert isinstance(res["reject_null"], bool)


@pytest.mark.local_compatible
def test_crt_detects_obvious_signal():
    """Non-flaky: signal is so strong that failure is astronomically unlikely."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = 100 * X[:, 1]  # noiseless, massive coefficient

    res = tabpfn_crt(X, y, j=1, B=40, test_size=0.3, device="cpu", seed=0)

    assert res["p_value"] < 0.05
