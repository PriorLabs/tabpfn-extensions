#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""TabPFN on a frame that mixes tabular columns with pictures.

`TabPFNWithImages` wraps a TabPFN estimator, or any scikit-learn estimator, and an
`ImageTransformer`. At `fit` it expands the declared image columns into numeric
features, the PCA-reduced embeddings of a frozen DINOv3 ViT-S/16, moves the wrapped estimator's `categorical_features_indices` to where
those columns now sit, and fits a clone of the estimator on the expanded frame; at
`predict` it expands the same columns with the fit-time encoder and PCA and
delegates. The wrapped estimator only ever sees numbers, so the TabPFN client
backend works as well as the local package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.image.image_transformer import (
    DEFAULT_BATCH_SIZE,
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
        estimator: The estimator to fit on the expanded frame, typically a
            `TabPFNClassifier` or `TabPFNRegressor`. Cloned at `fit`; the instance
            handed in is never fitted. Its `categorical_features_indices`, when it
            has any, are read as positions in the caller's frame and moved to the
            expanded frame. Its `device`, when it has one, is where the image
            encoder runs.
        image_features_indices: Positions in `X` whose cells hold images, each as
            a base64 string or as the image file's bytes. A position may not also
            be in the estimator's `categorical_features_indices`. `None` or empty
            fits the estimator on `X` as is.
        n_components: Features an image column is expanded into, at most.
        batch_size: Images per encoder forward pass.

    Attributes:
        estimator_: The fitted clone of `estimator`.
        image_transformer_: The fitted `ImageTransformer`.
        n_features_in_: Columns of the frame `fit` saw.
        feature_names_in_: Labels of the frame `fit` saw, as strings.
    """

    estimator_: BaseEstimator
    image_transformer_: ImageTransformer
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        image_features_indices: Sequence[int] | None,
        n_components: int = DEFAULT_N_COMPONENTS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.estimator = estimator
        self.image_features_indices = image_features_indices
        self.n_components = n_components
        self.batch_size = batch_size

    def fit(self, X: pd.DataFrame, y: Any, **fit_params: Any) -> TabPFNWithImages:
        """Expand the image columns of `X`, then fit a clone of the estimator.

        Args:
            X: The training frame.
            y: The targets.
            **fit_params: Handed to the estimator's `fit`.

        Returns:
            Itself, fitted.
        """
        transformer = ImageTransformer(
            image_features_indices=self.image_features_indices,
            n_components=self.n_components,
            device=getattr(self.estimator, "device", "auto"),
            batch_size=self.batch_size,
        )
        estimator = clone(self.estimator)
        # Before any image is embedded, so a clash fails fast.
        categorical = getattr(estimator, "categorical_features_indices", None)
        if categorical is not None:
            estimator.categorical_features_indices = transformer.output_indices(
                categorical
            )
        estimator.fit(transformer.fit_transform(X), y, **fit_params)
        self.image_transformer_ = transformer
        self.estimator_ = estimator
        self.n_features_in_ = transformer.n_features_in_
        self.feature_names_in_ = transformer.feature_names_in_
        return self

    def predict(self, X: pd.DataFrame, **predict_params: Any) -> Any:
        """Expand the image columns of `X`, then predict with the fitted estimator.

        Args:
            X: A frame with the same columns as the one `fit` saw.
            **predict_params: Handed to the estimator's `predict`, for example a
                regressor's `output_type`.
        """
        check_is_fitted(self)
        return self.estimator_.predict(
            self.image_transformer_.transform(X), **predict_params
        )

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Expand the image columns of `X`, then call the estimator's `predict_proba`."""
        check_is_fitted(self)
        return np.asarray(
            self.estimator_.predict_proba(self.image_transformer_.transform(X))
        )

    def score(self, X: pd.DataFrame, y: Any) -> float:
        """The fitted estimator's own `score` on the expanded `X`."""
        check_is_fitted(self)
        return float(self.estimator_.score(self.image_transformer_.transform(X), y))

    @property
    def classes_(self) -> np.ndarray:
        """The fitted estimator's classes; a classifier's attribute only."""
        check_is_fitted(self)
        return np.asarray(self.estimator_.classes_)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        try:
            wrapped = get_tags(self.estimator)
        except (ValueError, AttributeError, TypeError):
            return tags
        tags.estimator_type = wrapped.estimator_type
        tags.classifier_tags = deepcopy(wrapped.classifier_tags)
        tags.regressor_tags = deepcopy(wrapped.regressor_tags)
        tags.target_tags.multi_output = wrapped.target_tags.multi_output
        tags.input_tags.allow_nan = wrapped.input_tags.allow_nan
        return tags
