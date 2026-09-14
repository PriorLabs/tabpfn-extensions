#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Turn the declared image columns of a DataFrame into numeric features.

An image column is one the caller names in `image_features_indices`: nothing is
detected, so an undeclared column is never touched. Each cell of a declared column
holds one image, as a base64 string (a `data:image/...;base64,` prefix is tolerated)
or as the image file's `bytes`. `fit` replaces the column by at most `n_components`
numeric features: the CLS embedding of a frozen DINOv3 ViT-S/16, standardised and
reduced by a PCA fit on the training rows. `transform` embeds the same columns with
the same encoder and projects them with the PCA fit at training time, so an unseen
image lands where its training neighbours are. A cell that is missing, or that
cannot be decoded as an image, is refused rather than guessed at, at fit and at
transform alike.

The encoder is downloaded from the Hugging Face Hub on first use and kept in a
module-level cache for the process. It is never stored on the transformer, so a
fitted one pickles without it. Its weights are gated: accept their license on the
Hub once and log in (`hf auth login` or `HF_TOKEN`). The optional dependencies come
with `pip install "tabpfn-extensions[image]"`.

Column handling is positional throughout: labels are the caller's and can repeat.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tabpfn_common_utils.telemetry import set_extension

__all__ = [
    "IMAGE_ENCODER_MODEL",
    "GatedEncoderError",
    "ImageTransformer",
    "encode_image_bytes",
    "resolve_device",
]

IMAGE_ENCODER_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
"""The one encoder: a DINOv3 ViT-S/16; its CLS token is an image's embedding."""
IMAGE_SIZE = 224
"""Side of the square every image is resized to, as the encoder expects."""
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
"""ImageNet channel statistics, the encoder's own normalisation."""
DEFAULT_N_COMPONENTS = 30
DEFAULT_BATCH_SIZE = 64

_ENCODER: Any = None
"""The loaded model, for the life of the process. Kept out of the transformer so
a fitted one pickles without the weights."""


class GatedEncoderError(OSError):
    """The encoder's weights are gated on the Hub and this process may not read them."""

    def __init__(self) -> None:
        super().__init__(
            f"The weights of `{IMAGE_ENCODER_MODEL}` are gated on the Hugging Face "
            f"Hub. Accept the license once at https://huggingface.co/"
            f"{IMAGE_ENCODER_MODEL}, then log in with `hf auth login` or set "
            "`HF_TOKEN`."
        )


@dataclasses.dataclass
class _FittedImageColumn:
    """One input column's fitted reduction and its features' names."""

    scaler: StandardScaler
    pca: PCA
    output_names: list[str]


@set_extension("image")
class ImageTransformer(TransformerMixin, BaseEstimator):
    """Replaces each declared image column of a DataFrame by numeric features.

    A scikit-learn transformer: `fit` on the training frame, `transform` any frame
    with the same columns. The declared columns are dropped and their features
    appended after the kept columns, so every kept column after an image column
    moves down by one; `output_indices` translates positions accordingly.

    Args:
        image_features_indices: Positions in `X` whose cells hold images. `None`
            or empty: nothing is expanded and nothing optional is imported.
        n_components: Features an image column is expanded into, at most: fewer
            when the column has fewer rows.
        device: Where the encoder runs: `"auto"`, a device string, a `torch.device`,
            or TabPFN's `device` argument as is. `"auto"` picks the device TabPFN
            would.
        batch_size: Images per encoder forward pass.

    Attributes:
        fitted_columns_: Input position -> the scaler and PCA fit on that column's
            embeddings and the names of the features they make. Empty when
            nothing was expanded.
        n_features_in_: Columns of the frame `fit` saw.
        feature_names_in_: Labels of the frame `fit` saw, as strings.
    """

    fitted_columns_: dict[int, _FittedImageColumn]
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        image_features_indices: Sequence[int] | None = None,
        *,
        n_components: int = DEFAULT_N_COMPONENTS,
        device: Any = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.image_features_indices = image_features_indices
        self.n_components = n_components
        self.device = device
        self.batch_size = batch_size

    @property
    def expanded_indices(self) -> list[int]:
        """Input positions that were expanded into image features, ascending."""
        check_is_fitted(self)
        return sorted(self.fitted_columns_)

    def fit(self, X: pd.DataFrame, y: Any = None) -> ImageTransformer:
        """Fit one reduction per declared image column in `X`.

        Args:
            X: The training frame.
            y: Ignored; present for the scikit-learn interface.

        Returns:
            Itself, fitted.
        """
        del y
        self._fit(X)
        return self

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        """`fit(X).transform(X)`, embedding each image column only once.

        Args:
            X: The training frame.
            y: Ignored; present for the scikit-learn interface.
            **fit_params: Ignored; present for the scikit-learn interface.

        Returns:
            `X` with each declared column replaced by its features.
        """
        del y, fit_params
        blocks = self._fit(X)
        if not blocks:
            return X
        return _drop_and_append(X.reset_index(drop=True), self.expanded_indices, blocks)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reapply the expansion `fit` decided on, so the width holds.

        Args:
            X: A frame with the same columns as the one `fit` saw.

        Returns:
            `X` with each declared column replaced by its features.

        Raises:
            NotFittedError: If `fit` has not run yet.
            TypeError: If `X` is not a `DataFrame`.
            ValueError: If `X` has another number of columns than at fit, or if a
                cell of an expanded column is missing or not an image.
        """
        check_is_fitted(self)
        _refuse_non_dataframe(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} columns, but this {type(self).__name__} was "
                f"fitted on {self.n_features_in_}."
            )
        if not self.expanded_indices:
            return X
        blocks = [
            self._apply_one(X, i, self.fitted_columns_[i])
            for i in self.expanded_indices
        ]
        return _drop_and_append(X.reset_index(drop=True), self.expanded_indices, blocks)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """The transformed frame's column labels: kept columns, then features.

        Args:
            input_features: Ignored; the labels come from the frame `fit` saw.

        Returns:
            The labels as an object array of strings.
        """
        del input_features
        check_is_fitted(self)
        kept = [
            name
            for i, name in enumerate(self.feature_names_in_)
            if i not in self.fitted_columns_
        ]
        expanded = [
            name
            for i in self.expanded_indices
            for name in self.fitted_columns_[i].output_names
        ]
        return np.asarray([*kept, *expanded], dtype=object)

    def output_indices(self, indices: Sequence[int] | None) -> list[int] | None:
        """Where each of `indices`, input positions, sits in the transformed frame.

        Needs no fit: only the declared positions matter. A kept column shifts down
        by however many image columns sat ahead of it.

        Args:
            indices: Input positions, or `None` for none declared.

        Returns:
            The same positions in the transformed frame, or `None` for `None`.

        Raises:
            ValueError: If one of `indices` is itself a declared image column: it
                becomes many features, so it has no single output position.
        """
        if indices is None:
            return None
        declared = self._declared_positions()
        clashing = sorted(set(indices) & set(declared))
        if clashing:
            raise ValueError(
                f"Positions {clashing} are declared image columns and cannot be "
                "translated to a single output position: an image column is "
                "expanded into numeric features, so it cannot be a categorical "
                "feature as well. Drop it from `categorical_features_indices`."
            )
        return [i - sum(1 for j in declared if j < i) for i in indices]

    def _declared_positions(self) -> list[int]:
        """The declared positions, validated, deduplicated and ascending."""
        if self.image_features_indices is None:
            return []
        positions = list(self.image_features_indices)
        if any(
            not isinstance(i, int | np.integer) or isinstance(i, bool)
            for i in positions
        ):
            raise ValueError(
                f"`image_features_indices` must hold integers, got {positions!r}."
            )
        if any(i < 0 for i in positions):
            raise ValueError(
                f"`image_features_indices` must be non-negative, got {positions!r}."
            )
        return sorted({int(i) for i in positions})

    def _fit(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        """Fit the reductions; return each expanded column's features, in order."""
        _refuse_non_dataframe(X)
        if self.n_components < 1:
            raise ValueError(
                f"`n_components` must be at least 1, got {self.n_components}."
            )
        positions = self._declared_positions()
        out_of_range = [i for i in positions if i >= X.shape[1]]
        if out_of_range:
            raise ValueError(
                f"`image_features_indices` names positions {out_of_range}, but `X` "
                f"has {X.shape[1]} columns."
            )
        # Cleared first, so refitting on an input with nothing to expand still
        # forgets the last fit.
        self.fitted_columns_ = {}
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        if not positions:
            return []

        kept_names = [
            str(column) for i, column in enumerate(X.columns) if i not in set(positions)
        ]
        taken = set(kept_names)
        blocks: list[pd.DataFrame] = []
        for position in positions:
            block, fitted = self._fit_one(
                self._embed(X, position), str(X.columns[position]), taken
            )
            self.fitted_columns_[position] = fitted
            taken.update(fitted.output_names)
            blocks.append(block)
        return blocks

    def _embed(self, X: pd.DataFrame, position: int) -> np.ndarray:
        """Embed every cell of column `position`, refusing missing or broken ones."""
        payloads = _payloads(X, position)
        try:
            return encode_image_bytes(
                payloads, device=resolve_device(self.device), batch_size=self.batch_size
            )
        except ValueError as e:
            raise ValueError(
                f"Column {_name_columns(X, [position])} holds a cell that is not an "
                f"image ({e}). Pass each image as a base64-encoded string or as the "
                "image file's bytes."
            ) from e

    def _fit_one(
        self, embeddings: np.ndarray, label: str, taken: set[str]
    ) -> tuple[pd.DataFrame, _FittedImageColumn]:
        """Fit a scaler and a PCA on one column's embeddings, named after the column.

        A PCA keeps at most as many components as it has rows or dimensions, so
        how many features there are is settled here, which is why `transform`
        reuses this fit rather than making a fresh one.
        """
        n_kept = min(self.n_components, *embeddings.shape)
        scaler = StandardScaler().fit(embeddings)
        scaled = scaler.transform(embeddings)
        # Seeded on its own: the features a column turns into are a property of
        # the data, and should not move with any estimator's seed.
        pca = PCA(n_components=n_kept, random_state=0).fit(scaled)
        reduced = pca.transform(scaled).astype(np.float32)
        output_names = _unique_names(
            [f"{label}_img_{i}" for i in range(n_kept)], taken=taken
        )
        return (
            pd.DataFrame(reduced, columns=output_names),
            _FittedImageColumn(scaler=scaler, pca=pca, output_names=output_names),
        )

    def _apply_one(
        self, X: pd.DataFrame, position: int, fitted: _FittedImageColumn
    ) -> pd.DataFrame:
        """Reapply one fitted reduction, naming its features as at fit."""
        embeddings = self._embed(X, position)
        reduced = fitted.pca.transform(fitted.scaler.transform(embeddings))
        return pd.DataFrame(reduced.astype(np.float32), columns=fitted.output_names)


def resolve_device(device: Any) -> torch.device:
    """The `torch.device` the encoder should run on, from any TabPFN-style spec.

    `"auto"` or `None` picks what TabPFN's own device resolution would; a list
    of devices means its first entry. Without the `tabpfn` package installed,
    `"auto"` means CUDA when available and the CPU otherwise.

    Args:
        device: `"auto"`, `None`, a device string, a `torch.device`, or a
            sequence of those.
    """
    if isinstance(device, list | tuple):
        device = device[0] if device else "auto"
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        return device
    try:
        from tabpfn.utils import infer_devices
    except ImportError:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    return infer_devices(device)[0]


def _cell_to_bytes(cell: object) -> bytes | None:
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


def _payloads(X: pd.DataFrame, position: int) -> list[bytes]:
    """Every cell of column `position` as image bytes, in row order.

    Raises:
        ValueError: Naming the rows that are missing or not base64.
    """
    payloads: list[bytes] = []
    missing: list[int] = []
    bad: dict[int, str] = {}
    for row, cell in enumerate(X.iloc[:, position].tolist()):
        try:
            payload = _cell_to_bytes(cell)
        except ValueError as e:
            bad[row] = str(e)
            continue
        if payload is None:
            missing.append(row)
        else:
            payloads.append(payload)
    if missing:
        raise ValueError(
            f"Column {_name_columns(X, [position])} has no image in rows "
            f"{_name_rows(missing)}. Every cell of a declared image column has to "
            "hold a base64 string or bytes: drop those rows or fill the image in first."
        )
    if bad:
        raise ValueError(
            f"Column {_name_columns(X, [position])} holds cells that are not base64 "
            f"images in rows {_name_rows(list(bad))}: {next(iter(bad.values()))}. "
            "Pass each image as a base64-encoded string (a `data:image/...;base64,` "
            "prefix is fine) or as the image file's bytes."
        )
    return payloads


def _import_pil() -> Any:
    """PIL's `Image` module, or an `ImportError` naming the extra that brings it."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Image columns need the optional image dependencies: "
            'pip install "tabpfn-extensions[image]".'
        ) from e
    return Image


def _import_transformers() -> Any:
    """Transformers' `AutoModel`, or an `ImportError` naming the extra."""
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "Image columns need the optional image dependencies (transformers, "
            'pillow): pip install "tabpfn-extensions[image]".'
        ) from e
    return AutoModel


def _get_image_encoder() -> Any:
    """The encoder, loaded once per process and set to eval.

    Raises:
        GatedEncoderError: When the license has not been accepted, or no token is
            available.
    """
    global _ENCODER
    if _ENCODER is None:
        AutoModel = _import_transformers()
        try:
            _ENCODER = AutoModel.from_pretrained(IMAGE_ENCODER_MODEL).eval()
        except OSError as e:
            # transformers folds the Hub's `GatedRepoError` into an `OSError`.
            if "gated" in str(e).lower():
                raise GatedEncoderError from e
            raise
    return _ENCODER


def _open_images(payloads: Sequence[bytes]) -> list[Any]:
    """Decode each payload to an RGB PIL image of `IMAGE_SIZE` by `IMAGE_SIZE`.

    A palette image goes through RGBA so its transparency survives the
    conversion; any other mode, grayscale included, converts to RGB directly.
    The resize is the encoder's own: straight to a square, bilinear, no crop.

    Raises:
        ValueError: Naming the row whose bytes PIL cannot read as an image.
    """
    Image = _import_pil()
    images = []
    for row, payload in enumerate(payloads):
        try:
            image = Image.open(io.BytesIO(payload))
            image.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            raise ValueError(f"row {row}: {e}") from e
        if image.mode == "P":
            image = image.convert("RGBA")
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        images.append(image)
    return images


def _pixel_values(images: Sequence[Any]) -> torch.Tensor:
    """The encoder's input for `images`: `(n, 3, IMAGE_SIZE, IMAGE_SIZE)` float32.

    Scaled to [0, 1] and normalised per channel, which is all the encoder's own
    image processor does after the resize in `_open_images`.
    """
    arrays = [np.asarray(image, dtype=np.float32) / 255.0 for image in images]
    normalised = (np.stack(arrays) - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(normalised).permute(0, 3, 1, 2).contiguous()


def encode_image_bytes(
    payloads: Sequence[bytes],
    *,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """The CLS embedding of every image, as `(len(payloads), hidden_size)` float32.

    Batches of `batch_size`, no gradients. The encoder is moved to `device` in
    place, a no-op once it is there.

    Args:
        payloads: The image files' bytes, one per row.
        device: Where the encoder runs.
        batch_size: Images per forward pass.

    Raises:
        ValueError: Naming the row whose bytes are not an image.
    """
    images = _open_images(payloads)
    model = _get_image_encoder()
    model.to(device)
    chunks = []
    for start in range(0, len(images), batch_size):
        pixels = _pixel_values(images[start : start + batch_size]).to(device)
        with torch.no_grad():
            hidden = model(pixel_values=pixels).last_hidden_state
        chunks.append(hidden[:, 0, :].float().cpu().numpy())
    if not chunks:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _drop_and_append(
    frame: pd.DataFrame, expanded: Sequence[int], blocks: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    """Drop the `expanded` positions and append `blocks` after the kept columns.

    The blocks are default-indexed, so the kept columns must be too, or `concat`
    aligns the two by label instead of position.
    """
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)


def _unique_names(candidates: Sequence[str], *, taken: set[str]) -> list[str]:
    """Each candidate, suffixed `_1`, `_2`, ... until it clashes with nothing."""
    names: list[str] = []
    seen = set(taken)
    for candidate in candidates:
        name = candidate
        suffix = 1
        while name in seen:
            name = f"{candidate}_{suffix}"
            suffix += 1
        seen.add(name)
        names.append(name)
    return names


def _refuse_non_dataframe(X: Any) -> None:
    """Raise unless `X` is a `DataFrame`, the only input that can carry images."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"Image columns need a pandas DataFrame, got {type(X).__name__}. Pass `X` "
            "as a DataFrame whose declared columns hold each image as a base64 string "
            "or as bytes."
        )


def _name_columns(X: pd.DataFrame, positions: Sequence[int]) -> str:
    """Name each of `positions` by index and label, e.g. `1 ('photo')`."""
    return ", ".join(f"{i} ({X.columns[i]!r})" for i in positions)


def _name_rows(rows: Sequence[int], limit: int = 10) -> str:
    """List the first `limit` of `rows`, then say how many more there are."""
    shown = ", ".join(str(row) for row in rows[:limit])
    if len(rows) > limit:
        shown += f" and {len(rows) - limit} more"
    return shown
