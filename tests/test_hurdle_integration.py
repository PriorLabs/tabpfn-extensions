from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tabpfn_extensions import AutoHurdleRegressor


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_tabpfn_integration(tabpfn_classifier: Any, tabpfn_regressor: Any) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 3))
    y = np.where(X[:, 0] > 0.5, 1 + np.abs(X[:, 1]), 0)
    model = AutoHurdleRegressor(tabpfn_classifier, tabpfn_regressor).fit(X, y)
    prediction = model.predict(X[:4])
    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()
    model.set_params(hurdle=False).fit(X, y)
    np.testing.assert_allclose(
        model.predict(X[:4]), model.regressor_.predict(X[:4], output_type="median")
    )
