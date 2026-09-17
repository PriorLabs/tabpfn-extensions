"""Tests for the DINOv3 encoder: loading it and the embeddings."""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from tabpfn_extensions.image import GatedEncoderError, _embeddings
from tabpfn_extensions.image._embeddings import IMAGE_ENCODER_MODEL

#: The width of the encoder's CLS embedding.
EMBEDDING_DIM = 384


def _png_bytes(colour: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _fresh_encoder_cache() -> Iterator[None]:
    """Each test loads its own encoder and leaves none behind for the next."""
    _embeddings._load_encoder.cache_clear()
    yield
    _embeddings._load_encoder.cache_clear()


def _stub_encoder_dependencies(monkeypatch: pytest.MonkeyPatch, auto: type) -> None:
    """Stand in for transformers and torchvision, so no test needs the image extra."""
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=auto, AutoImageProcessor=auto),
    )
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace())


class TestEncoderLoading:
    """Importing the optional dependencies and fetching the weights."""

    @pytest.mark.parametrize("module", ["transformers", "torchvision"])
    def test__missing_optional_dependency__names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, module: str
    ) -> None:
        monkeypatch.setitem(sys.modules, module, None)

        with pytest.raises(ImportError, match=r"tabpfn-extensions\[image\]"):
            _embeddings.get_dino_encoder(torch.device("cpu"))

    def test__gated_repo__is_reported_with_license_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class GatedAuto:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> None:
                del kwargs
                raise OSError(f"You are trying to access a gated repo. {name}")

        _stub_encoder_dependencies(monkeypatch, GatedAuto)

        with pytest.raises(GatedEncoderError, match="huggingface.co/facebook/dinov3"):
            _embeddings.get_dino_encoder(torch.device("cpu"))

    def test__encoder__is_loaded_once_per_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []

        class FakeModel:
            def eval(self) -> FakeModel:
                return self

            def to(self, device: torch.device) -> FakeModel:
                del device
                return self

        class FakeAuto:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> FakeModel:
                del kwargs
                loaded.append(name)
                return FakeModel()

        _stub_encoder_dependencies(monkeypatch, FakeAuto)

        first_model, first_processor = _embeddings.get_dino_encoder(torch.device("cpu"))
        model, processor = _embeddings.get_dino_encoder(torch.device("cpu"))

        assert model is first_model
        assert processor is first_processor
        assert loaded == [IMAGE_ENCODER_MODEL, IMAGE_ENCODER_MODEL]


@pytest.mark.parametrize("batch_size", [0, -1])
def test__encode_images__rejects_a_batch_size_that_is_not_positive(
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        _embeddings.encode_images(
            [b"never decoded"], device="cpu", batch_size=batch_size
        )


class _Batch(dict):
    """What the processor hands the model: a mapping that moves to a device."""

    def to(self, device: torch.device) -> _Batch:
        del device
        return self


def _fake_encoder(device: torch.device) -> _embeddings.DinoEncoder:
    """A processor that numbers the images of its batch, and a model whose CLS
    token is that number, so the output tells which batch a row came from.
    """
    del device

    class Processor:
        def __call__(self, images: list[Any], return_tensors: str) -> _Batch:
            del return_tensors
            return _Batch(pixel_values=torch.arange(len(images), dtype=torch.float32))

    class Model:
        config = types.SimpleNamespace(hidden_size=3)

        def __call__(self, pixel_values: torch.Tensor) -> Any:
            hidden = pixel_values[:, None, None].expand(-1, 2, 3)
            return types.SimpleNamespace(last_hidden_state=hidden)

    return _embeddings.DinoEncoder(Model(), Processor())  # type: ignore[arg-type]


def test__encode_images__decodes_one_batch_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each batch is decoded right before its forward pass, its rows numbered
    within the column, and the batches' embeddings come back in row order.
    """
    decoded: list[tuple[int, int]] = []

    def open_images(sources: list[Any], *, first_row: int = 0) -> list[Any]:
        decoded.append((first_row, len(sources)))
        return list(sources)

    monkeypatch.setattr(_embeddings, "open_images", open_images)
    monkeypatch.setattr(_embeddings, "get_dino_encoder", _fake_encoder)

    out = _embeddings.encode_images([b"image"] * 5, device="cpu", batch_size=3)

    assert decoded == [(0, 3), (3, 2)]
    assert (out.shape, out.dtype) == ((5, 3), np.float32)
    np.testing.assert_array_equal(out[:, 0], [0, 1, 2, 0, 1])
    assert _embeddings.encode_images([], device="cpu").shape == (0, 3)


def test__encode_images__names_the_row_of_a_bad_image_within_the_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_embeddings, "get_dino_encoder", _fake_encoder)
    sources = [_png_bytes((0, 0, 0)), _png_bytes((0, 0, 0)), b"not an image"]

    with pytest.raises(ValueError, match="row 2"):
        _embeddings.encode_images(sources, device="cpu", batch_size=2)


@pytest.mark.slow
def test__encode_images__separates_two_colours() -> None:
    """The real encoder on the CPU, when its dependencies and license are in place:
    same-colour squares embed alike, as bytes or as a PIL image, different colours
    apart.
    """
    pytest.importorskip("transformers")
    Image = pytest.importorskip("PIL.Image")
    red, blue = _png_bytes((255, 0, 0)), _png_bytes((0, 0, 255))
    red_image = Image.open(io.BytesIO(red))

    try:
        out = _embeddings.encode_images([red, blue, red_image], device="cpu")
    except GatedEncoderError as e:
        pytest.skip(str(e))

    assert out.shape == (3, EMBEDDING_DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], out[2], atol=1e-4)
    assert np.abs(out[0] - out[1]).max() > 1e-2
