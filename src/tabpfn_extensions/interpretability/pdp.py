# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from sklearn.inspection import PartialDependenceDisplay
from tabpfn_common_utils.telemetry import set_extension

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from sklearn.base import BaseEstimator


@set_extension("interpretability")
def partial_dependence_plots(
    estimator: BaseEstimator,
    X: np.ndarray,
    features: Sequence[int | tuple[int, int]],
    *,
    grid_resolution: int = 20,
    kind: str = "average",  # "average" or "individual" (ICE)
    target_class: int | None = None,  # for classification: which class's proba
    ax: Axes | None = None,
    **kwargs,
) -> PartialDependenceDisplay:
    """Plot partial dependence (and ICE) for 1D/2D feature(s).

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
    response_method = "predict_proba" if hasattr(estimator, "predict_proba") else "auto"

    restore_progress = None
    if hasattr(estimator, "show_progress"):
        restore_progress = getattr(estimator, "show_progress", None)
        try:
            estimator.show_progress = False
        except (AttributeError, TypeError):
            restore_progress = None

    try:
        disp = PartialDependenceDisplay.from_estimator(
            estimator,
            X,
            features=features,
            kind=kind,
            grid_resolution=grid_resolution,
            response_method=response_method,
            target=target_class,  # ignored unless needed (e.g., predict_proba multiclass)
            ax=ax,
            **kwargs,
        )
    finally:
        if restore_progress is not None:
            estimator.show_progress = restore_progress

    return disp
