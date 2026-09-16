"""Tests for the DINOv3 encoder: loading it, its device, and the embeddings."""

from __future__ import annotations

import io
import sys
import types
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


class TestTorchDevice:
    def test__specs__resolve_to_torch_devices(self) -> None:
        assert _embeddings._torch_device("cpu") == torch.device("cpu")
        assert _embeddings._torch_device(None) == _embeddings._torch_device("auto")
        assert isinstance(_embeddings._torch_device("auto"), torch.device)


class TestEncoderLoading:
    """Importing the optional dependencies and fetching the weights."""

    @pytest.mark.parametrize("module", ["transformers", "torchvision"])
    def test__missing_optional_dependency__names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, module: str
    ) -> None:
        monkeypatch.setitem(sys.modules, module, None)
        monkeypatch.setattr(_embeddings, "_ENCODER", None)

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

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoModel=GatedAuto, AutoImageProcessor=GatedAuto),
        )
        monkeypatch.setattr(_embeddings, "_ENCODER", None)

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

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoModel=FakeAuto, AutoImageProcessor=FakeAuto),
        )
        monkeypatch.setattr(_embeddings, "_ENCODER", None)

        first_model, first_processor = _embeddings.get_dino_encoder(torch.device("cpu"))
        model, processor = _embeddings.get_dino_encoder(torch.device("cpu"))

        assert model is first_model
        assert processor is first_processor
        assert loaded == [IMAGE_ENCODER_MODEL, IMAGE_ENCODER_MODEL]


@pytest.mark.slow
def test__encode_image_bytes__separates_two_colours() -> None:
    """The real encoder on the CPU, when its dependencies and license are in place:
    same-colour squares embed alike, different colours apart.
    """
    pytest.importorskip("transformers")
    red, blue = _png_bytes((255, 0, 0)), _png_bytes((0, 0, 255))

    try:
        out = _embeddings.encode_image_bytes([red, blue, red], device="cpu")
    except GatedEncoderError as e:
        pytest.skip(str(e))

    assert out.shape == (3, EMBEDDING_DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], out[2], atol=1e-4)
    assert np.abs(out[0] - out[1]).max() > 1e-2
