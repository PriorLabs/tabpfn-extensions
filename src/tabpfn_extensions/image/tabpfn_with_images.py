#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""TabPFN on a frame that mixes tabular columns with pictures.

`TabPFNWithImages` wraps an estimator and an `ImageTransformer`. `fit` expands the
declared image columns into DINOv3 features, moves the estimator's
`categorical_features_indices` to where those columns now sit, and fits a clone of
the estimator on the expanded frame; `predict` expands the same columns and
delegates. The estimator only ever sees numbers, so the TabPFN client backend
works as well as the local package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.image._embeddings import DEFAULT_BATCH_SIZE
from tabpfn_extensions.image.image_transformer import (
    DEFAULT_N_COMPONENTS,
    ImageTransformer,
)
from tabpfn_extensions.misc.sklearn_compat import get_tags

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["TabPFNWithImages"]


def _estimator_has(attr: str) -> Callable[[TabPFNWithImages], bool]:
    """A check for `available_if`: the wrapped estimator has `attr`."""

    def check(self: TabPFNWithImages) -> bool:
        # Raises the original AttributeError when the estimator lacks it.
        getattr(self.estimator, attr)
        return True

    return check


@set_extension("image")
class TabPFNWithImages(BaseEstimator):
    """Fits an estimator on a frame whose declared image columns become features.

    Args:
        estimator: Typically a `TabPFNClassifier` or `TabPFNRegressor`; cloned at
            `fit`. Its `categorical_features_indices`, when set, are positions in
            the caller's frame and are moved to the expanded frame. Its `device`,
            when it has one, is where the image encoder runs.
        image_features_indices: Positions in `X` whose cells hold images, each as
            a base64 string, as the image file's path or bytes, or as a PIL image.
        n_components: Features an image column is expanded into.
        batch_size: Images per encoder forward pass.

    Attributes:
        estimator_: The fitted clone of `estimator`.
        image_transformer_: The fitted `ImageTransformer`.
    """

    estimator_: Any
    image_transformer_: ImageTransformer

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        image_features_indices: Sequence[int],
        n_components: int = DEFAULT_N_COMPONENTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.estimator = estimator
        self.image_features_indices = image_features_indices
        self.n_components = n_components
        self.batch_size = batch_size

    def fit(self, X: pd.DataFrame, y: Any, **fit_params: Any) -> TabPFNWithImages:
        """Expand the image columns of `X`, then fit a clone of the estimator."""
        images = ImageTransformer(
            self.image_features_indices,
            n_components=self.n_components,
            device=getattr(self.estimator, "device", "auto"),
            batch_size=self.batch_size,
        )
        estimator = clone(self.estimator)
        categorical = getattr(estimator, "categorical_features_indices", None)
        if categorical is not None:
            estimator.categorical_features_indices = images.output_indices(categorical)
        X = images.fit_transform(X)
        estimator.fit(X, y, **fit_params)
        self.image_transformer_, self.estimator_ = images, estimator
        return self

    def predict(self, X: pd.DataFrame, **predict_params: Any) -> Any:
        """The fitted estimator's `predict` on the expanded `X`."""
        X = self._transform(X)
        return self.estimator_.predict(X, **predict_params)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X: pd.DataFrame) -> Any:
        """The fitted estimator's `predict_proba` on the expanded `X`."""
        X = self._transform(X)
        return self.estimator_.predict_proba(X)

    def score(self, X: pd.DataFrame, y: Any) -> Any:
        """The fitted estimator's `score` on the expanded `X`."""
        X = self._transform(X)
        return self.estimator_.score(X, y)

    @property
    def classes_(self) -> Any:
        """The fitted estimator's classes; a classifier's attribute only."""
        return self.estimator_.classes_

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        return self.image_transformer_.transform(X)

    def __sklearn_tags__(self) -> Any:
        # So `is_classifier`, and with it stratified cross-validation, follow the
        # wrapped estimator.
        tags = super().__sklearn_tags__()
        tags.estimator_type = get_tags(self.estimator).estimator_type
        return tags
