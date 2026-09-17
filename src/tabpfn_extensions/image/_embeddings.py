#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""The frozen DINOv3 encoder and its image processor: loaded once, run in batches."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from tabpfn_extensions.image._preprocessing import open_images
from tabpfn_extensions.utils import infer_torch_device

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers import DINOv3ViTImageProcessor, DINOv3ViTModel

IMAGE_ENCODER_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
"""The one encoder: a DINOv3 ViT-S/16; its CLS token is an image's embedding."""
DEFAULT_BATCH_SIZE = 64


class DinoEncoder(NamedTuple):
    """The frozen encoder and the image processor that prepares its input."""

    model: DINOv3ViTModel
    processor: DINOv3ViTImageProcessor


class GatedEncoderError(OSError):
    """The encoder's weights are gated on the Hub and this process may not read them."""

    def __init__(self) -> None:
        super().__init__(
            f"The weights of `{IMAGE_ENCODER_MODEL}` are gated on the Hugging Face "
            f"Hub. Accept the license once at https://huggingface.co/"
            f"{IMAGE_ENCODER_MODEL}, then log in with `hf auth login` or set "
            "`HF_TOKEN`."
        )


def _raise_if_no_encoder_dependencies() -> None:
    """Name the extra that brings transformers and torchvision when one is missing."""
    try:
        import torchvision  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Image columns need the optional image dependencies (transformers, "
            'torchvision, pillow): pip install "tabpfn-extensions[image]".'
        ) from e


@functools.cache
def _load_encoder() -> DinoEncoder:
    """The encoder and its image processor, fetched once per process.

    Cached here, not on a transformer, so fitted transformers pickle without the
    weights. A failed load is not cached, so a retry once the extra is installed or
    the license accepted goes through.

    Raises:
        GatedEncoderError: When the license has not been accepted, or no token is
            available.
    """
    _raise_if_no_encoder_dependencies()
    from transformers import AutoImageProcessor, AutoModel

    try:
        model = AutoModel.from_pretrained(IMAGE_ENCODER_MODEL).eval()
        processor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER_MODEL)
    except OSError as e:
        # transformers folds the Hub's `GatedRepoError` into an `OSError`.
        if "gated" in str(e).lower():
            raise GatedEncoderError from e
        raise
    return DinoEncoder(model, processor)


def get_dino_encoder(device: torch.device) -> DinoEncoder:
    """The encoder moved to `device`, and its image processor."""
    encoder = _load_encoder()
    return DinoEncoder(encoder.model.to(device), encoder.processor)


def encode_images(
    sources: Sequence[bytes | Image],
    *,
    device: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """The CLS embedding of every image, as `(len(sources), hidden_size)` float32.

    Args:
        sources: One image per row, as an image file's bytes or path, or a PIL image.
        device: Where the encoder runs, as TabPFN's `device` argument; `None`
            means `"auto"`. The encoder runs in this process even when TabPFN
            itself runs through the client.
        batch_size: Images per forward pass.

    Raises:
        ValueError: When `batch_size` is not positive, or naming the row whose
            bytes are not an image.
    """
    if batch_size < 1:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
    torch_device = infer_torch_device("auto" if device is None else device)
    model, processor = get_dino_encoder(torch_device)
    chunks = []
    for start in range(0, len(sources), batch_size):
        # Decoded one batch at a time, so a long column never holds every image.
        images = open_images(sources[start : start + batch_size], first_row=start)
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**inputs.to(torch_device)).last_hidden_state
        chunks.append(hidden[:, 0, :].float().cpu().numpy())
    if not chunks:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
