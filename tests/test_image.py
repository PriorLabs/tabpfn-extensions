"""Tests for `ImageTransformer` and `TabPFNWithImages`.

A stub replaces the encoder so no test downloads weights; one slow test runs the
real one when its dependencies and the Hub license are in place. The decoding and
the encoder have their own test files.
"""

from __future__ import annotations

import base64
import hashlib
import io
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError

from conftest import DEFAULT_TEST_SIZE, FAST_TEST_MODE, SMALL_TEST_SIZE
from tabpfn_extensions.image import (
    GatedEncoderError,
    ImageTransformer,
    TabPFNWithImages,
    image_transformer as images_module,
)
from tabpfn_extensions.image.image_transformer import DEFAULT_N_COMPONENTS

#: The default width an image column is expanded to.
N_COMPONENTS = DEFAULT_N_COMPONENTS

#: The width of the default encoder's embedding, which the stub reproduces.
EMBEDDING_DIM = 384


def _stub_encoder(payloads: list[bytes], **kwargs: Any) -> np.ndarray:
    """A deterministic vector per payload: its sha256 digest, repeated to width.

    Stands in for the real encoder so no test downloads weights, and so the
    bytes in a cell can be anything: the stub never opens them as an image.
    """
    del kwargs
    rows = [
        np.frombuffer(hashlib.sha256(_raw(payload)).digest() * 12, dtype=np.uint8)
        for payload in payloads
    ]
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return np.stack(rows)[:, :EMBEDDING_DIM].astype(np.float32)


def _raw(source: Any) -> bytes:
    """The bytes the stub hashes: a PIL image's pixels, or the bytes themselves."""
    return source if isinstance(source, bytes) else source.tobytes()


@pytest.fixture
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_module, "encode_images", _stub_encoder)


def _never_called(*_args: object, **_kwargs: object) -> np.ndarray:
    raise AssertionError("the encoder must not run here")


def _b64(i: int) -> str:
    """A base64 cell; the stub encoder never opens the bytes, so any will do."""
    return base64.b64encode(f"image-{i}".encode()).decode("ascii")


def _png_bytes(
    colour: tuple[int, ...] | int = (255, 0, 0),
    size: tuple[int, int] = (8, 8),
    mode: str = "RGB",
) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _frame(n: int = 40, cells: list | None = None) -> pd.DataFrame:
    """A numeric column beside `photo`: base64 cells unless `cells` says otherwise."""
    photo = [_b64(i) for i in range(n)] if cells is None else cells
    return pd.DataFrame({"num": np.arange(len(photo), dtype=float), "photo": photo})


def _expander(**kwargs: Any) -> ImageTransformer:
    return ImageTransformer([1], **kwargs)


def _estimator_data(*, regression: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """A numeric column beside `photo`, plus a `y` matching the task."""
    # At least the default component count, which every image column needs.
    n = max(N_COMPONENTS + 2, SMALL_TEST_SIZE if FAST_TEST_MODE else DEFAULT_TEST_SIZE)
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {"num": rng.normal(size=n), "photo": [_b64(i % 11) for i in range(n)]}
    )
    y = rng.normal(size=n) if regression else np.arange(n) % 2
    return X, y


class TestDeclaration:
    """Which columns are expanded: the declared ones, and only when valid."""

    def test__no_declared_columns__is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(images_module, "encode_images", _never_called)

        with pytest.raises(ValueError, match="must name columns"):
            ImageTransformer([]).fit(_frame())
        with pytest.raises(ValueError, match="is required"):
            ImageTransformer(None).fit(_frame())

    @pytest.mark.usefixtures("stub_encoder")
    def test__declared_column__is_expanded_to_n_components(self) -> None:
        transformer = _expander()
        out = transformer.fit_transform(_frame())

        assert out.shape == (40, 1 + N_COMPONENTS)
        assert list(out.columns) == [
            "num",
            *[f"photo_img_{i}" for i in range(N_COMPONENTS)],
        ]
        assert list(transformer.get_feature_names_out()) == list(out.columns)
        assert out.notna().all().all()
        assert all(pd.api.types.is_float_dtype(dtype) for dtype in out.dtypes)

    def test__numpy_index_arrays__are_accepted(self) -> None:
        assert ImageTransformer(np.array([1]))._declared_positions(2) == [1]
        assert ImageTransformer(np.array([0]))._declared_positions(2) == [0]
        assert ImageTransformer(np.array([1, 0]))._declared_positions(2) == [0, 1]

    def test__index_out_of_range__is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"which has 2; got \[5\]"):
            ImageTransformer([5]).fit(_frame())

    def test__negative_index__is_refused(self) -> None:
        with pytest.raises(ValueError, match="must name columns"):
            ImageTransformer([-1]).fit(_frame())

    def test__array_input__is_refused(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            _expander().fit(np.zeros((3, 2)))
        with pytest.raises(TypeError, match="DataFrame"):
            ImageTransformer([0]).fit(np.zeros((3, 2)))


@pytest.mark.usefixtures("stub_encoder")
class TestExpansion:
    """How a declared column is read and what it turns into."""

    def test__data_uri_prefix__is_stripped(self) -> None:
        plain = _expander().fit_transform(_frame())
        prefixed = _expander().fit_transform(
            _frame(cells=[f"data:image/png;base64,{_b64(i)}" for i in range(40)])
        )

        pd.testing.assert_frame_equal(prefixed, plain)

    def test__bytes_cells__are_accepted(self) -> None:
        plain = _expander().fit_transform(_frame())
        raw = _expander().fit_transform(
            _frame(cells=[f"image-{i}".encode() for i in range(40)])
        )

        pd.testing.assert_frame_equal(raw, plain)

    def test__whitespace_in_base64__is_tolerated(self) -> None:
        plain = _expander().fit_transform(_frame())
        wrapped = _expander().fit_transform(
            _frame(cells=[f" {_b64(i)[:4]}\n{_b64(i)[4:]} " for i in range(40)])
        )

        pd.testing.assert_frame_equal(wrapped, plain)

    def test__missing_cells__are_refused_naming_rows(self) -> None:
        cells: list[str | None] = [_b64(i) for i in range(40)]
        cells[3] = None
        cells[5] = ""

        with pytest.raises(ValueError, match="row 3: no image"):
            _expander().fit(_frame(cells=cells))

    def test__non_base64_strings__are_refused_naming_rows(self) -> None:
        cells = [_b64(i) for i in range(40)]
        cells[2] = "not base64!!"

        with pytest.raises(
            ValueError, match=r"row 2: neither an existing file nor base64"
        ):
            _expander().fit(_frame(cells=cells))

    def test__fewer_rows_than_components__is_refused(self) -> None:
        with pytest.raises(ValueError, match="10 rows, fewer than the 30"):
            _expander().fit(_frame(n=10))

    def test__duplicate_column_labels__are_handled_positionally(self) -> None:
        X = _frame().set_axis(["a", "a"], axis=1)

        out = _expander().fit_transform(X)

        assert out.shape == (40, 1 + N_COMPONENTS)
        assert out.columns[0] == "a"

    def test__kept_column_named_like_an_image_feature__is_refused_before_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(images_module, "encode_images", _never_called)
        X = _frame().assign(photo_img_0=1.0)

        with pytest.raises(ValueError, match=r"repeat the labels \['photo_img_0'\]"):
            _expander().fit(X)

    def test__two_declared_columns_with_one_label__are_refused_before_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(images_module, "encode_images", _never_called)
        X = pd.DataFrame([[_b64(0), _b64(1)]] * 5, columns=["pic", "pic"])

        with pytest.raises(ValueError, match=r"\['pic_img_0', 'pic_img_1'\]"):
            ImageTransformer([0, 1], n_components=2).fit(X)

    def test__two_declared_columns__are_both_expanded_in_order(self) -> None:
        X = pd.DataFrame(
            {
                "front": [_b64(i) for i in range(12)],
                "num": np.arange(12, dtype=float),
                "back": [_b64(100 + i) for i in range(12)],
            }
        )

        transformer = ImageTransformer([2, 0], n_components=3)
        out = transformer.fit_transform(X)

        assert list(transformer.reducers_) == [0, 2]
        assert list(out.columns) == [
            "num",
            *[f"front_img_{i}" for i in range(3)],
            *[f"back_img_{i}" for i in range(3)],
        ]

    def test__pil_image_cells__are_accepted_and_left_untouched(self) -> None:
        Image = pytest.importorskip("PIL.Image")
        cells = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(40)]

        out = _expander().fit_transform(_frame(cells=cells))

        assert out.shape == (40, 1 + N_COMPONENTS)
        assert all(image.size == (8, 8) for image in cells)

    def test__caller_s_frame__is_left_untouched(self) -> None:
        X = _frame()
        before = X.copy()

        _expander().fit_transform(X)

        pd.testing.assert_frame_equal(X, before)

    def test__refit__replaces_the_last_fit(self) -> None:
        transformer = _expander(n_components=5)
        transformer.fit(_frame(n=40))

        out = transformer.fit_transform(_frame(n=10).set_axis(["num", "pic"], axis=1))

        assert list(out.columns) == ["num", *[f"pic_img_{i}" for i in range(5)]]
        assert out.shape == (10, 6)


class TestOutputIndices:
    """Translating caller positions to the expanded frame, no fit needed."""

    def test__positions_after_an_image_column__shift_down(self) -> None:
        transformer = ImageTransformer([1, 3])

        assert transformer.output_indices([0, 2, 4], n_columns=5) == [0, 1, 2]
        assert transformer.output_indices([], n_columns=5) == []
        assert transformer.output_indices(None, n_columns=5) is None

    def test__a_declared_image_position__is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"Positions \[1\] are declared image"):
            ImageTransformer([1]).output_indices([0, 1], n_columns=2)

    def test__a_declared_position_outside_the_frame__is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"which has 3; got \[1, 3\]"):
            ImageTransformer([1, 3]).output_indices([0], n_columns=3)


@pytest.mark.usefixtures("stub_encoder")
class TestTransform:
    """`transform` reapplies the fit and refuses what it cannot embed."""

    def test__row_index__is_kept_even_with_duplicate_labels(self) -> None:
        X = _frame(n=6).set_axis([5, 5, 7, 8, 9, 9], axis=0)
        transformer = _expander(n_components=3)

        fitted = transformer.fit_transform(X)
        again = transformer.transform(X)

        assert fitted.index.tolist() == [5, 5, 7, 8, 9, 9]
        assert again.index.tolist() == [5, 5, 7, 8, 9, 9]

    def test__integer_column_labels__come_back_as_the_feature_names_out(self) -> None:
        X = _frame().set_axis([0, 1], axis=1)
        transformer = _expander()

        out = transformer.fit_transform(X)

        assert list(out.columns) == list(transformer.get_feature_names_out())
        assert list(transformer.transform(X).columns) == list(out.columns)
        assert out.columns[0] == "0"

    def test__same_data__reproduces_the_fitted_columns(self) -> None:
        X = _frame()
        transformer = _expander()
        fitted = transformer.fit_transform(X)

        pd.testing.assert_frame_equal(transformer.transform(X), fitted)

    def test__unseen_images__keep_the_fitted_width(self) -> None:
        transformer = _expander()
        fitted = transformer.fit_transform(_frame())

        out = transformer.transform(_frame(cells=[_b64(1000 + i) for i in range(7)]))

        assert list(out.columns) == list(fitted.columns)
        assert out.shape[0] == 7
        assert out.notna().all().all()

    def test__missing_cell__is_refused_naming_the_row(self) -> None:
        transformer = _expander().fit(_frame())
        cells: list[str | None] = [_b64(i) for i in range(5)]
        cells[4] = None

        with pytest.raises(ValueError, match="row 4: no image"):
            transformer.transform(_frame(cells=cells))

    def test__undecodable_cell__is_refused(self) -> None:
        transformer = _expander().fit(_frame())

        with pytest.raises(ValueError, match="neither an existing file nor base64"):
            transformer.transform(_frame(cells=["???"]))

    def test__other_column_count__is_refused(self) -> None:
        transformer = _expander().fit(_frame())
        X = _frame(n=3)
        X["extra"] = 1.0

        with pytest.raises(ValueError, match="3 columns.*fitted on 2"):
            transformer.transform(X)

    def test__array_input__is_refused(self) -> None:
        transformer = _expander().fit(_frame())

        with pytest.raises(TypeError, match="DataFrame"):
            transformer.transform(np.zeros((3, 2)))

    def test__device_and_batch_size__are_handed_to_the_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[torch.device, int]] = []

        def recording(payloads: list[bytes], **kwargs: Any) -> np.ndarray:
            seen.append((kwargs["device"], kwargs["batch_size"]))
            return _stub_encoder(payloads)

        monkeypatch.setattr(images_module, "encode_images", recording)
        transformer = _expander(device="cpu", batch_size=7).fit(_frame())
        transformer.transform(_frame(n=3))

        assert seen == [("cpu", 7)] * 2


class TestSklearnInterface:
    """What scikit-learn expects of a transformer."""

    @pytest.mark.usefixtures("stub_encoder")
    def test__fit__returns_itself_and_transform_matches_fit_transform(self) -> None:
        X = _frame()

        transformer = _expander().fit(X)

        assert isinstance(transformer, ImageTransformer)
        pd.testing.assert_frame_equal(
            transformer.transform(X), _expander().fit_transform(X)
        )

    @pytest.mark.usefixtures("stub_encoder")
    def test__failed_fit__leaves_no_half_fitted_state(self) -> None:
        cells = [_b64(i) for i in range(40)]
        cells[3] = "not base64!!"
        transformer = _expander()

        with pytest.raises(
            ValueError, match="row 3: neither an existing file nor base64"
        ):
            transformer.fit(_frame(cells=cells))
        with pytest.raises(NotFittedError):
            transformer.transform(_frame())

        transformer.fit(_frame())
        with pytest.raises(
            ValueError, match="row 3: neither an existing file nor base64"
        ):
            transformer.fit(_frame(cells=cells))
        assert transformer.transform(_frame()).shape == (40, 1 + N_COMPONENTS)

    def test__transform_before_fit__raises(self) -> None:
        with pytest.raises(NotFittedError):
            _expander().transform(_frame())
        with pytest.raises(NotFittedError):
            _expander().get_feature_names_out()

    def test__clone__keeps_the_parameters(self) -> None:
        transformer = _expander(n_components=5, batch_size=3)

        cloned = clone(transformer)

        assert cloned.get_params() == transformer.get_params()
        assert cloned.image_features_indices == [1]

    @pytest.mark.usefixtures("stub_encoder")
    def test__fitted_transformer__pickles_and_transforms_alike(self) -> None:
        X = _frame()
        transformer = _expander().fit(X)

        restored = pickle.loads(pickle.dumps(transformer))  # noqa: S301

        pd.testing.assert_frame_equal(restored.transform(X), transformer.transform(X))


@pytest.mark.usefixtures("stub_encoder")
class TestTabPFNWithImages:
    """The wrapper: expand, remap categoricals, delegate to a fitted clone."""

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test__classifier__fits_on_the_expanded_frame_and_predicts(
        self, tabpfn_classifier: Any
    ) -> None:
        X, y = _estimator_data(regression=False)

        model = TabPFNWithImages(tabpfn_classifier, image_features_indices=[1])
        model.fit(X, y)

        assert len(model.image_transformer_.get_feature_names_out()) == 1 + N_COMPONENTS
        assert list(model.classes_) == [0, 1]
        assert model.predict(X).shape == (len(X),)
        assert model.predict_proba(X).shape == (len(X), 2)
        assert 0.0 <= model.score(X, y) <= 1.0
        assert is_classifier(model)
        assert not is_regressor(model)

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test__regressor__fits_and_predicts_without_predict_proba(
        self, tabpfn_regressor: Any
    ) -> None:
        X, y = _estimator_data(regression=True)

        model = TabPFNWithImages(tabpfn_regressor, image_features_indices=[1])
        model.fit(X, y)

        assert model.predict(X).shape == (len(X),)
        assert not hasattr(model, "predict_proba")
        assert not hasattr(model, "classes_")
        assert is_regressor(model)
        assert not is_classifier(model)

    @pytest.mark.local_compatible
    def test__categorical_positions__move_to_the_expanded_frame(
        self, tabpfn_classifier: Any
    ) -> None:
        X, y = _estimator_data(regression=False)
        X = X[["photo", "num"]]
        X["cat"] = np.where(np.arange(len(X)) % 3 == 0, "a", "b")
        tabpfn_classifier.categorical_features_indices = [2]

        model = TabPFNWithImages(tabpfn_classifier, image_features_indices=[0])
        model.fit(X, y)

        assert model.estimator_.categorical_features_indices == [1]
        assert tabpfn_classifier.categorical_features_indices == [2]
        assert model.predict(X).shape == (len(X),)

    @pytest.mark.local_compatible
    def test__image_position_also_categorical__is_refused_before_encoding(
        self, tabpfn_classifier: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(images_module, "encode_images", _never_called)
        X, y = _estimator_data(regression=False)
        tabpfn_classifier.categorical_features_indices = [1]

        model = TabPFNWithImages(tabpfn_classifier, image_features_indices=[1])

        with pytest.raises(ValueError, match="declared image columns"):
            model.fit(X, y)

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test__no_declared_columns__is_refused_at_fit(
        self, tabpfn_classifier: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(images_module, "encode_images", _never_called)
        X, y = _estimator_data(regression=False)

        with pytest.raises(ValueError, match="must name columns"):
            TabPFNWithImages(tabpfn_classifier, image_features_indices=[]).fit(X, y)

    @pytest.mark.local_compatible
    def test__encoder__runs_on_the_estimator_s_device(
        self, tabpfn_classifier: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []

        def recording(payloads: list[bytes], **kwargs: Any) -> np.ndarray:
            seen.append(kwargs["device"])
            return _stub_encoder(payloads)

        monkeypatch.setattr(images_module, "encode_images", recording)
        X, y = _estimator_data(regression=False)
        tabpfn_classifier.device = "cpu"

        TabPFNWithImages(tabpfn_classifier, image_features_indices=[1]).fit(X, y)

        assert seen == ["cpu"]

    @pytest.mark.local_compatible
    def test__tabpfn__refuses_columns_in_another_order(
        self, tabpfn_classifier: Any
    ) -> None:
        """The premise of the next test: TabPFN itself checks the feature names."""
        X, y = _estimator_data(regression=False)
        X = pd.DataFrame({"num": X["num"], "more": X["num"] * 2})

        tabpfn_classifier.fit(X, y)

        with pytest.raises(ValueError, match="feature names should match"):
            tabpfn_classifier.predict(X[["more", "num"]])

    @pytest.mark.local_compatible
    def test__columns_in_another_order__are_refused_by_tabpfn(
        self, tabpfn_classifier: Any
    ) -> None:
        """The kept columns keep their labels, so the swap reaches TabPFN's check."""
        from tabpfn.errors import TabPFNValidationError

        X, y = _estimator_data(regression=False)
        X["more"] = X["num"] * 2
        model = TabPFNWithImages(tabpfn_classifier, image_features_indices=[1])
        model.fit(X, y)

        with pytest.raises(TabPFNValidationError, match="feature names should match"):
            model.predict(X[["more", "photo", "num"]])

    def test__predict_before_fit__raises(self, tabpfn_classifier: Any) -> None:
        X, _ = _estimator_data(regression=False)

        with pytest.raises(NotFittedError):
            TabPFNWithImages(tabpfn_classifier, image_features_indices=[1]).predict(X)

    @pytest.mark.client_compatible
    @pytest.mark.local_compatible
    def test__clone__keeps_the_parameters_and_the_unfitted_estimator(
        self, tabpfn_classifier: Any
    ) -> None:
        model = TabPFNWithImages(
            tabpfn_classifier, image_features_indices=[1], n_components=4
        )

        cloned = clone(model)

        assert cloned.image_features_indices == [1]
        assert cloned.n_components == 4
        assert type(cloned.estimator) is type(tabpfn_classifier)
        assert cloned.estimator is not tabpfn_classifier


@pytest.mark.slow
def test__real_encoder__separates_two_colours() -> None:
    """The default DINOv3 encoder, on the CPU, when its dependencies and license are
    in place: same-colour squares embed alike, different colours apart.
    """
    pytest.importorskip("transformers")
    red = base64.b64encode(_png_bytes((255, 0, 0), size=(32, 32))).decode()
    blue = base64.b64encode(_png_bytes((0, 0, 255), size=(32, 32))).decode()
    X = pd.DataFrame({"num": range(6), "photo": [red, blue] * 3})

    try:
        out = ImageTransformer([1], n_components=2, device="cpu").fit_transform(X)
    except GatedEncoderError as e:
        pytest.skip(str(e))

    assert out.shape == (6, 3)
    features = out.iloc[:, 1:].to_numpy()
    np.testing.assert_allclose(features[0], features[2], atol=1e-4)
    assert np.abs(features[0] - features[1]).max() > 1e-2
