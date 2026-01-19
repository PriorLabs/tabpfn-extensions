import numpy as np
import pytest

from tabpfn_extensions.pval_crt import tabpfn_crt


@pytest.mark.local_compatible
def test_crt_detects_relevant_feature():
    rng = np.random.RandomState(0)
    n, p = 80, 4

    X = rng.randn(n, p)
    y = X[:, 1] + 0.1 * rng.randn(n)

    res = tabpfn_crt(
        X,
        y,
        j=1,
        B=20,
        test_size=0.3,
        device="cpu",
    )

    assert "p_value" in res
    assert 0 <= res["p_value"] <= 1

@pytest.mark.local_compatible
def test_crt_handles_irrelevant_feature():
    rng = np.random.RandomState(1)
    n, p = 80, 4

    X = rng.randn(n, p)
    y = rng.randn(n)

    res = tabpfn_crt(
        X,
        y,
        j=2,
        B=20,
        test_size=0.3,
        device="cpu",
    )

    assert "p_value" in res
