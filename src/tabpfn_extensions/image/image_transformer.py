#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Replace the declared image columns of a DataFrame by DINOv3 features.

Each cell of a declared column holds one image, as a base64 string, as the image
file's path or bytes, or as a PIL image. `fit` embeds the column with a frozen DINOv3
ViT-S/16 and fits a StandardScaler and a PCA on the training rows; `transform`
applies them, so an unseen image lands where its training neighbours are. The
column is dropped and its `n_components` features are appended after the kept
columns. A missing or undecodable cell is refused. Columns are handled by position.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.image._embeddings import DEFAULT_BATCH_SIZE, encode_images
from tabpfn_extensions.image._preprocessing import cell_to_image_source

__all__ = ["DEFAULT_N_COMPONENTS", "ImageTransformer"]

DEFAULT_N_COMPONENTS = 30


@set_extension("image")
class ImageTransformer(TransformerMixin, BaseEstimator):
    """Replaces each declared image column of a DataFrame by numeric features.

    Args:
        image_features_indices: Positions of the image columns in `X`. Such a
            column holds one image per cell, as a base64 string, the image file's
            path or bytes, or a PIL image, so its dtype is object or string.
        n_components: Features an image column is expanded into.
        device: Where the encoder runs, as TabPFN's `device` argument.
        batch_size: Images per encoder forward pass.

    Attributes:
        reducers_: Input position -> the column's fitted scaler and PCA.
        feature_names_in_: Labels of the frame `fit` saw, as strings.
    """

    reducers_: dict[int, Pipeline]
    feature_names_in_: np.ndarray

    def __init__(
        self,
        image_features_indices: Sequence[int],
        *,
        n_components: int = DEFAULT_N_COMPONENTS,
        device: Any = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.image_features_indices = image_features_indices
        self.n_components = n_components
        self.device = device
        self.batch_size = batch_size

    def fit(self, X: pd.DataFrame, y: Any = None) -> ImageTransformer:
        """Fit one scaler and PCA per declared column; `y` is ignored."""
        self.fit_transform(X)
        return self

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit every declared column and return the expanded frame."""
        del y, fit_params
        _check_frame(X)
        positions = self._declared_positions(X.shape[1])
        reducers: dict[int, Pipeline] = {}
        blocks: dict[int, np.ndarray] = {}
        for position in positions:
            embeddings = self._embed(X, position)
            if len(embeddings) < self.n_components:
                raise ValueError(
                    f"Column {position} has {len(embeddings)} rows, fewer than the "
                    f"{self.n_components} components it is expanded into."
                )
            # Seeded on its own: the features depend on the data, not on any seed.
            pca = PCA(self.n_components, random_state=0)
            reducers[position] = make_pipeline(StandardScaler(), pca).fit(embeddings)
            blocks[position] = reducers[position].transform(embeddings)
        # Assigned together at the end, so a failed fit leaves no half-fitted state.
        self.reducers_ = reducers
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        return self._assemble(X, blocks)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace the declared columns of `X` by the features fit at training."""
        check_is_fitted(self)
        _check_frame(X)
        if X.shape[1] != len(self.feature_names_in_):
            raise ValueError(
                f"X has {X.shape[1]} columns, fitted on {len(self.feature_names_in_)}."
            )
        blocks = {i: r.transform(self._embed(X, i)) for i, r in self.reducers_.items()}
        return self._assemble(X, blocks)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """The kept columns' labels, then the image features', as an object array."""
        del input_features
        check_is_fitted(self)
        names = self.feature_names_in_
        kept = [name for i, name in enumerate(names) if i not in self.reducers_]
        image = self._image_feature_names(list(names), self.reducers_)
        return np.asarray([*kept, *image], dtype=object)

    def output_indices(
        self, indices: Sequence[int] | None, *, n_columns: int
    ) -> list[int] | None:
        """Where the input positions `indices` sit once the image columns moved.

        Args:
            indices: Positions in a frame of `n_columns` columns; `None` stays `None`.
            n_columns: The width of that frame.

        Raises:
            ValueError: If one of them is a declared image column.
        """
        if indices is None:
            return None
        declared = self._declared_positions(n_columns)
        if clash := sorted(set(indices) & set(declared)):
            raise ValueError(
                f"Positions {clash} are declared image columns; they cannot also be "
                "categorical features."
            )
        return [i - sum(j < i for j in declared) for i in indices]

    def _declared_positions(self, n_columns: int) -> list[int]:
        """The declared positions, sorted and unique, all within `n_columns`."""
        if self.image_features_indices is None:
            raise ValueError("`image_features_indices` is required.")
        positions = sorted({int(i) for i in self.image_features_indices})
        if not positions or positions[0] < 0 or positions[-1] >= n_columns:
            raise ValueError(
                f"`image_features_indices` must name columns of X, which has "
                f"{n_columns}; got {list(self.image_features_indices)}."
            )
        return positions

    def _embed(self, X: pd.DataFrame, position: int) -> np.ndarray:
        """The encoder's embedding of every cell of the column."""
        sources = []
        for row, cell in enumerate(X.iloc[:, position]):
            try:
                source = cell_to_image_source(cell)
            except ValueError as e:
                raise ValueError(f"Column {position}, row {row}: {e}") from e
            if source is None:
                raise ValueError(f"Column {position}, row {row}: no image.")
            sources.append(source)
        return encode_images(sources, device=self.device, batch_size=self.batch_size)

    def _image_feature_names(
        self, labels: Sequence[str], positions: Iterable[int]
    ) -> list[str]:
        """`<label>_img_<k>` per declared column and component, in position order.

        A name a kept column or an earlier feature already bears, because a column
        is called `photo_img_0` or two declared columns share a label, gets `_0`,
        `_1`, ... appended until it is free.
        """
        positions = list(positions)
        taken = {label for i, label in enumerate(labels) if i not in positions}
        names = []
        for i in positions:
            for k in range(self.n_components):
                name = base = f"{labels[i]}_img_{k}"
                n = 0
                while name in taken:
                    name = f"{base}_{n}"
                    n += 1
                taken.add(name)
                names.append(name)
        return names

    def _assemble(self, X: pd.DataFrame, blocks: dict[int, np.ndarray]) -> pd.DataFrame:
        """Drop the image columns, append their features, keep the caller's index.

        The kept columns keep their own labels, as strings, so a frame whose columns
        differ from the fitted ones reaches the estimator as it is and TabPFN's own
        feature-name check sees it.
        """
        kept = X.iloc[:, [i for i in range(X.shape[1]) if i not in blocks]]
        features = [pd.DataFrame(block, dtype=np.float32) for block in blocks.values()]
        out = pd.concat([kept.reset_index(drop=True), *features], axis=1)
        image = self._image_feature_names(list(self.feature_names_in_), blocks)
        names = [*map(str, kept.columns), *image]
        return out.set_axis(names, axis=1).set_axis(X.index, axis=0)


def _check_frame(X: object) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Image columns need a DataFrame, got {type(X).__name__}.")
