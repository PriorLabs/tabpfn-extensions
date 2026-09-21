from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError

from tabpfn_extensions.hurdle import AutoHurdleRegressor


@pytest.fixture
def model() -> AutoHurdleRegressor:
    return AutoHurdleRegressor(DummyClassifier(), DummyRegressor())


def test_fit_clones_and_preserves_frame(
    model: AutoHurdleRegressor, mocker: MockerFixture
) -> None:
    fit = mocker.spy(DummyRegressor, "fit")
    X = pd.DataFrame({"value": [1, 2, 3, 4], "kind": ["a", "b", "a", "b"]})
    X.index = [7, 4, 9, 2]
    y = pd.Series([0, 0, 0, 4], index=X.index)
    model.fit(X, y)
    assert model.hurdle_
    assert model.zero_rate_ == 0.75
    np.testing.assert_array_equal(model.classifier_.class_prior_, [0.75, 0.25])
    assert model.regressor_.constant_.item() == 4
    pd.testing.assert_frame_equal(fit.call_args.args[1], X.iloc[[3]])
    assert not hasattr(model.regressor, "constant_")
    assert not hasattr(model.classifier, "classes_")
    assert model.n_features_in_ == 2
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    assert clone(model).get_params()["zero_threshold"] == 0.25
    with pytest.raises(ValueError, match="feature names"):
        model.predict(X[["kind", "value"]])


@pytest.mark.parametrize(
    ("y", "hurdle", "threshold", "expected"),
    [
        ([0, 0, 1, 2], "auto", 0.5, False),
        ([0, 0, 0, 2], "auto", 0.5, True),
        ([0, 0, 0, -2], "auto", 0.5, False),
        ([1, 2, 3, 4], "auto", 0.5, False),
        ([0, 1, 2, 3], "auto", 0.1, True),
        ([0, 0, 0, 2], False, 0.5, False),
        ([0, 1, 2, 3], True, 0.5, True),
    ],
)
def test_auto_gate(
    model: AutoHurdleRegressor,
    y: list[float],
    hurdle: Any,
    threshold: float,
    expected: bool,
) -> None:
    model.set_params(hurdle=hurdle, zero_threshold=threshold).fit(np.ones((4, 2)), y)
    assert model.hurdle_ == expected
    if not expected:
        assert model.regressor_.constant_.item() == np.mean(y)


@pytest.mark.parametrize(("n_zeros", "expected"), [(4, False), (5, False), (6, True)])
def test_default_zero_threshold(
    model: AutoHurdleRegressor, n_zeros: int, expected: bool
) -> None:
    y = np.ones(20)
    y[:n_zeros] = 0
    model.fit(np.ones((20, 2)), y)
    assert model.hurdle_ == expected


def test_mixture_predictions(
    model: AutoHurdleRegressor, monkeypatch: pytest.MonkeyPatch
) -> None:
    X = np.ones((8, 2))
    model.fit(X, [0, 0, 0, 0, 0, 1, 2, 3])
    p = np.array([0, 0.1, 0.5, 0.5001, 0.8, 1, -0.1, 1.1])
    # Reverse class order to check that probabilities follow the positive label.
    model.classifier_.classes_ = np.array([1, 0])
    monkeypatch.setattr(
        model.classifier_, "predict_proba", lambda _X: np.column_stack([p, 1 - p])
    )

    def uniform_predict(
        X: Any, *, output_type: str, quantiles: list[float] | None = None
    ) -> Any:
        if output_type == "mean":
            return np.full(len(X), 15.0)
        return [np.full(len(X), 10 + 10 * q) for q in quantiles]

    monkeypatch.setattr(model.regressor_, "predict", uniform_predict)
    np.testing.assert_allclose(model.predict(X), [0, 0, 0, 10.1, 13.75, 15, 0, 15])
    np.testing.assert_allclose(
        model.predict(X, output_type="mean"), np.clip(p, 0, 1) * 15
    )
    lower, median, upper = model.predict(
        X, output_type="quantiles", quantiles=[0.1, 0.5, 0.9]
    )
    np.testing.assert_allclose(median, model.predict(X))
    assert np.all(lower <= median)
    assert np.all(median <= upper)
    assert upper[4] == pytest.approx(18.75)

    monkeypatch.setattr(
        model.regressor_, "predict", lambda X, **_kw: -np.ones((21, len(X)))
    )
    np.testing.assert_array_equal(model.predict(X), 0)


def test_prediction_validation(model: AutoHurdleRegressor) -> None:
    X = np.ones((4, 2))
    with pytest.raises(NotFittedError):
        model.predict(X)
    model.fit(X, np.zeros(4))
    with pytest.raises(ValueError, match="output_type"):
        model.predict(X, output_type="mode")
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((4, 3)))


def test_refit_and_all_zero(model: AutoHurdleRegressor) -> None:
    X = np.ones((4, 2))
    model.fit(X, [0, 0, 0, 1])
    model.fit(X, [0, 0, 0, 0])
    assert model.classifier_ is None
    assert model.regressor_ is None
    np.testing.assert_array_equal(model.predict(X), np.zeros(4))
    np.testing.assert_array_equal(model.predict(X, output_type="mean"), np.zeros(4))
    np.testing.assert_array_equal(
        model.predict(X, output_type="quantiles", quantiles=[0.1, 0.9]),
        np.zeros((2, 4)),
    )
    model.fit(X, [1, 2, 3, 4])
    assert not model.hurdle_
    assert model.classifier_ is None
    with pytest.raises(ValueError):
        model.fit(X, [0, 0, np.nan, 1])
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("y", [[0, -1, 2], [1, 2, 3], [0, np.inf, 1]])
def test_invalid_forced_targets(model: AutoHurdleRegressor, y: list[float]) -> None:
    with pytest.raises(ValueError):
        model.set_params(hurdle=True).fit(np.ones((3, 2)), y)


@pytest.mark.parametrize("params", [{"hurdle": "yes"}, {"zero_threshold": -1}])
def test_invalid_parameters(model: AutoHurdleRegressor, params: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        model.set_params(**params).fit(np.ones((4, 2)), [0, 0, 0, 1])


@pytest.mark.parametrize("quantiles", [[], [0], [1], [np.nan], [-0.1]])
def test_invalid_quantiles(model: AutoHurdleRegressor, quantiles: list[float]) -> None:
    X = np.ones((4, 2))
    model.fit(X, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="quantiles"):
        model.predict(X, output_type="quantiles", quantiles=quantiles)


def test_custom_grid(model: AutoHurdleRegressor, mocker: MockerFixture) -> None:
    grid = [0.2, 0.4, 0.6]
    model.set_params(quantile_grid=grid).fit(np.ones((4, 2)), [0, 0, 0, 1])
    assert clone(model).quantile_grid == grid
    grid[0] = 0.1
    mocker.patch.object(
        model.classifier_, "predict_proba", return_value=np.array([[0.2, 0.8]])
    )
    predict = mocker.patch.object(
        model.regressor_,
        "predict",
        return_value=[np.array([2]), np.array([4]), np.array([6])],
    )
    X = np.ones((1, 2))
    np.testing.assert_allclose(model.predict(X), [3.75])
    predict.assert_called_once_with(
        X, output_type="quantiles", quantiles=[0.2, 0.4, 0.6]
    )
    np.testing.assert_allclose(
        model.predict(X, output_type="quantiles", quantiles=[0.25, 0.9]), [[2], [6]]
    )


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [0.5],
        [[0.1, 0.9]],
        [0, 0.5],
        [0.5, 1],
        [0.5, np.nan],
        [0.5, np.inf],
        [0.5, 0.5],
        [0.9, 0.1],
    ],
)
def test_invalid_grid(model: AutoHurdleRegressor, grid: Any) -> None:
    with pytest.raises(ValueError, match="quantile_grid"):
        model.set_params(quantile_grid=grid).fit(np.ones((4, 2)), [0, 0, 0, 1])
