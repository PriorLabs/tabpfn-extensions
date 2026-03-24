#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

if TYPE_CHECKING:
    import numpy as np
    from sklearn.base import BaseEstimator


def feature_selection(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 3,
    feature_names: list[str] | None = None,
    **kwargs,
) -> SequentialFeatureSelector:
    """Perform feature selection to find the most important features.

    Uses forward sequential feature selection to identify the most important
    features for the given estimator and data.

    Args:
        estimator: The model to use for feature selection
        X: Input features, shape (n_samples, n_features)
        y: Target values, shape (n_samples,)
        n_features_to_select: Number of features to select
        feature_names: Names of the features (optional)
        **kwargs: Additional parameters to pass to SequentialFeatureSelector

    Returns:
        SequentialFeatureSelector: Fitted feature selector that can be used
            to transform data to use only the selected features
    """
    if hasattr(estimator, "show_progress"):
        show_progress_ = estimator.show_progress
        estimator.show_progress = False
        try:
            return _feature_selection(
                estimator,
                X,
                y,
                n_features_to_select,
                feature_names,
                **kwargs,
            )
        finally:
            estimator.show_progress = show_progress_
    else:
        return _feature_selection(
            estimator,
            X,
            y,
            n_features_to_select,
            feature_names,
            **kwargs,
        )


def _feature_selection(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 3,
    feature_names: list[str] | None = None,
    **kwargs,
) -> SequentialFeatureSelector:
    """Internal implementation of feature selection.

    Args:
        estimator: The model to use for feature selection
        X: Input features
        y: Target values
        n_features_to_select: Number of features to select
        feature_names: Names of the features
        **kwargs: Additional parameters for SequentialFeatureSelector

    Returns:
        SequentialFeatureSelector: Fitted feature selector
    """
    # TODO: Try https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
    # TODO: Could use more feature in training, but only keep fewer in test
    # TODO: the fit function is somehow still called; We need to to change the feature selection
    #  method so it sets feature to none in test; Could be done with a wrapper that sets the feature subsets in fit
    #  and predict
    CV_FOLDS = 5

    cross_val_score(estimator, X, y, cv=CV_FOLDS)

    # TODO: Feature selection is done without CV, i.e. final CV scores might be biased (too good)
    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        direction="forward",
        **kwargs,
    )
    sfs.fit(X, y)
    sfs.get_support()
    X_transformed = sfs.transform(X)

    cross_val_score(estimator, X_transformed, y, cv=CV_FOLDS)

    if feature_names is not None:
        pass

    return sfs
