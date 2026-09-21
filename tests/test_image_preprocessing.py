"""Tests for turning an image column's cells into PIL images, and back."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd
import pytest

from tabpfn_extensions.image import _preprocessing


def _png_bytes(
    colour: tuple[int, ...] | int = (255, 0, 0),
    size: tuple[int, int] = (8, 8),
    mode: str = "RGB",
) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test__cell_to_image_source__reads_each_kind_of_cell() -> None:
    Image = pytest.importorskip("PIL.Image")
    payload = b"image-0"
    encoded = base64.b64encode(payload).decode("ascii")

    assert _preprocessing.cell_to_image_source(encoded) == payload
    assert (
        _preprocessing.cell_to_image_source(f"data:image/png;base64,{encoded}")
        == payload
    )
    assert (
        _preprocessing.cell_to_image_source(f" {encoded[:3]}\n{encoded[3:]}") == payload
    )
    assert _preprocessing.cell_to_image_source(payload) == payload
    assert _preprocessing.cell_to_image_source(bytearray(payload)) == payload
    for missing in (None, "", "  ", float("nan"), pd.NA):
        assert _preprocessing.cell_to_image_source(missing) is None
    with pytest.raises(ValueError, match="neither an existing file nor base64"):
        _preprocessing.cell_to_image_source("not base64!!")
    image = Image.new("RGB", (2, 2))
    assert _preprocessing.cell_to_image_source(image) is image
    with pytest.raises(ValueError, match="unsupported cell type int"):
        _preprocessing.cell_to_image_source(5)


def test__cell_to_image_source__reads_a_path_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A string naming a file, absolute or relative to the working directory, and a
    `Path` are the file's path; the file is not read here.
    """
    file = tmp_path / "cat.png"
    file.write_bytes(b"not read yet")
    monkeypatch.chdir(tmp_path)

    assert _preprocessing.cell_to_image_source(str(file)) == file
    assert _preprocessing.cell_to_image_source(f" {file} ") == file
    assert _preprocessing.cell_to_image_source("cat.png") == Path("cat.png")
    assert _preprocessing.cell_to_image_source(file) == file
    with pytest.raises(ValueError, match="no such file"):
        _preprocessing.cell_to_image_source(tmp_path / "dog.png")
    with pytest.raises(ValueError, match="neither an existing file nor base64"):
        _preprocessing.cell_to_image_source(str(tmp_path / "dog.png"))


def test__open_images__converts_every_mode_to_rgb_and_caps_the_size() -> None:
    payloads = [
        _png_bytes(),
        _png_bytes(colour=7, mode="L"),
        _png_bytes(colour=3, mode="P"),
        _png_bytes(size=(1000, 600)),
    ]

    images = _preprocessing.open_images(payloads)

    assert [image.mode for image in images] == ["RGB"] * 4
    assert [image.size for image in images][:3] == [(8, 8)] * 3
    assert max(images[3].size) == _preprocessing.MAX_IMAGE_SIDE


def test__open_images__reads_a_file_from_its_path(tmp_path: Path) -> None:
    file = tmp_path / "cat.png"
    file.write_bytes(_png_bytes(colour=(10, 20, 30)))
    (tmp_path / "notes.txt").write_text("not an image")

    image = _preprocessing.open_images([file])[0]

    assert image.size == (8, 8)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    with pytest.raises(ValueError, match="row 1"):
        _preprocessing.open_images([file, tmp_path / "notes.txt"])
    with pytest.raises(ValueError, match="row 0"):
        _preprocessing.open_images([tmp_path / "dog.png"])


def test__open_images__takes_pil_images_and_leaves_them_untouched() -> None:
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("L", (600, 8), 7)

    out = _preprocessing.open_images([image])[0]

    assert out.mode == "RGB"
    assert max(out.size) == _preprocessing.MAX_IMAGE_SIDE
    assert (image.mode, image.size) == ("L", (600, 8))


def test__open_images__refuses_bytes_that_are_not_an_image() -> None:
    with pytest.raises(ValueError, match="row 1"):
        _preprocessing.open_images([_png_bytes(), b"not an image"])


def test__image_to_bytes__round_trips_through_open_images() -> None:
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (5, 7), (10, 20, 30))

    payload = _preprocessing.image_to_bytes(image)
    restored = _preprocessing.open_images([payload])[0]

    assert isinstance(payload, bytes)
    assert restored.size == (5, 7)
    assert restored.getpixel((0, 0)) == (10, 20, 30)
