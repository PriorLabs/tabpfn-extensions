# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from sklearn.inspection import PartialDependenceDisplay

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sklearn.base import BaseEstimator


def partial_dependence_plots(
    estimator: BaseEstimator,
    X: np.ndarray,
    features: Sequence[int | tuple[int, int]],
    *,
    grid_resolution: int = 20,
    kind: str = "average",            # "average" or "individual" (ICE)
    target_class: int | None = None,  # for classification: which class's proba
    ax: Axes | None = None,
    **kwargs,
) -> PartialDependenceDisplay:
    """
    Plot partial dependence (and ICE) for 1D/2D feature(s).

    Args:
        estimator: fitted estimator or TabPFN-like estimator (fit-at-predict-time is fine)
        X: array of shape (n_samples, n_features)
        features: list of feature indices (e.g., [0, 3]) or pairs for interactions (e.g., [(0, 3)])
        grid_resolution: number of grid points per feature
        kind: "average" for PD, "individual" for ICE, "both" in newer sklearn
        target_class: for classifiers, the class index for which to plot probabilities
        ax: optional matplotlib Axes
        **kwargs: forwarded to PartialDependenceDisplay.from_estimator

    Returns:
        PartialDependenceDisplay
    """
    # Decide response method
    # - If classifier & predict_proba exists, use that (optionally select a class)
    # - Else fall back to decision_function/predict
    response_method = None
    if hasattr(estimator, "predict_proba"):
        response_method = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        response_method = "decision_function"
    else:
        response_method = "auto"

    # Some TabPFN-derivatives expose `show_progress`; silence for speed
    restore_progress = None
    if hasattr(estimator, "show_progress"):
        restore_progress = estimator.show_progress
        try:
            estimator.show_progress = False
        except Exception:
            restore_progress = None  # be tolerant

    try:
        disp = PartialDependenceDisplay.from_estimator(
            estimator,
            X,
            features=features,
            kind=kind,
            grid_resolution=grid_resolution,
            response_method=response_method,
            target=target_class,       # ignored unless needed (e.g., predict_proba multiclass)
            ax=ax,
            **kwargs,
        )
    finally:
        if restore_progress is not None:
            estimator.show_progress = restore_progress

    return disp
