#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""From a DataFrame cell to a PIL image the encoder's processor can take, and back."""

from __future__ import annotations

import base64
import binascii
import io
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from PIL.Image import Image

MAX_IMAGE_SIDE = 512
"""Longest side an image is shrunk to before the encoder's processor resizes it to
its own input size: a cap on memory for high-resolution inputs, with no effect on
the embedding."""


def cell_to_bytes(cell: object) -> bytes | None:
    """The image bytes one cell holds, or `None` for a missing cell.

    A `str` is base64, optionally behind a `data:...;base64,` prefix and with
    whitespace removed; `bytes` are the image file itself.

    Raises:
        ValueError: On a value that is neither, or a string that is not base64.
    """
    if isinstance(cell, bytes | bytearray | memoryview):
        return bytes(cell)
    if isinstance(cell, str):
        text = cell.strip()
        if not text:
            return None
        if text.startswith("data:"):
            text = text.split(",", 1)[-1]
        try:
            return base64.b64decode("".join(text.split()), validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"not base64: {e}") from e
    if cell is None or (pd.api.types.is_scalar(cell) and pd.isna(cell)):
        return None
    raise ValueError(f"unsupported cell type {type(cell).__name__}")


def image_to_bytes(image: Image, format: str = "PNG") -> bytes:
    """A PIL image encoded as an image file, the form a DataFrame cell takes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def _raise_if_no_pil() -> None:
    """Name the extra that brings pillow when it is missing."""
    try:
        import PIL  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Image columns need the optional image dependencies: "
            'pip install "tabpfn-extensions[image]".'
        ) from e


def open_images(payloads: Sequence[bytes]) -> list[Image]:
    """Decode each payload to an RGB PIL image no larger than `MAX_IMAGE_SIDE`.

    A palette image goes through RGBA so its transparency survives the
    conversion; any other mode, grayscale included, converts to RGB directly.

    Raises:
        ValueError: Naming the row whose bytes PIL cannot read as an image.
    """
    _raise_if_no_pil()
    import PIL.Image

    images = []
    for row, payload in enumerate(payloads):
        try:
            image = PIL.Image.open(io.BytesIO(payload))
            image.load()
        except (
            OSError,
            ValueError,
            SyntaxError,
            PIL.Image.DecompressionBombError,
        ) as e:
            raise ValueError(f"row {row}: {e}") from e
        if image.mode == "P":
            image = image.convert("RGBA")
        image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), PIL.Image.LANCZOS)
        images.append(image)
    return images
