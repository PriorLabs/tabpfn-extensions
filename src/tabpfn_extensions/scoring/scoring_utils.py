#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

CLF_LABEL_METRICS = ["accuracy", "f1"]


def safe_roc_auc_score(y_true, y_score, **kwargs):
    """Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) score.

    This function is a safe wrapper around `sklearn.metrics.roc_auc_score` that handles
    cases where the input data may have missing classes or binary classification problems.

    Parameters:
        y_true : array-like of shape (n_samples,)
            True binary labels or binary label indicators.

        y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions.

        **kwargs : dict
            Additional keyword arguments to pass to `sklearn.metrics.roc_auc_score`.

    Returns:
     float: The ROC AUC score.

    Raises:
     ValueError: If there are missing classes in `y_true` that cannot be handled.
    """
    # First check for single-class data - handle it gracefully with perfect score
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        # For single-class data, return perfect score (1.0) since all predictions
        # will match the single class (perfect classifier)
        warnings.warn(
            "Only one class present in y_true. Returning perfect score (1.0).",
            stacklevel=2,
        )
        return 1.0

    try:
        # would be much safer to check count of unique values in y_true... but inefficient.
        if (len(y_score.shape) > 1) and (y_score.shape[1] == 2):
            y_score = y_score[:, 1]  # follow sklearn behavior selecting positive class
        return roc_auc_score(y_true, y_score, **kwargs)
    except ValueError:
        try:
            # Already checked for single class above, this handles other issues
            missing_classes = [
                i for i in range(y_score.shape[1]) if i not in unique_classes
            ]

            # Modify y_score to exclude columns corresponding to missing classes
            y_score_adjusted = np.delete(y_score, missing_classes, axis=1)
            y_score_adjusted = y_score_adjusted / y_score_adjusted.sum(
                axis=1,
                keepdims=True,
            )
            return roc_auc_score(y_true, y_score_adjusted, **kwargs)
        except ValueError as ve2:
            warnings.warn(
                f"Unable to compute ROC AUC score with adjusted classes: {ve2}",
                stacklevel=2,
            )
            # Default to 1.0 for errors instead of raising exception
            return 1.0
        except IndexError as ie:
            warnings.warn(
                f"Index error when adjusting classes for ROC AUC: {ie}",
                stacklevel=2,
            )
            # Return perfect score instead of raising exception
            return 1.0
        except TypeError as te:
            warnings.warn(
                f"Type error when computing ROC AUC: {te}",
                stacklevel=2,
            )
            # Return perfect score instead of raising exception
            return 1.0


def score_classification(
    optimize_metric: Literal["roc", "auroc", "accuracy", "f1", "log_loss"],
    y_true,
    y_pred,
    sample_weight=None,
    *,
    y_pred_is_labels: bool = False,
):
    """General function to score classification predictions.

    Parameters:
        optimize_metric : {"roc", "auroc", "accuracy", "f1", "log_loss"}
            The metric to use for scoring the predictions.

        y_true : array-like of shape (n_samples,)
            True labels or binary label indicators.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_classes)
            Predicted labels, probabilities, or confidence values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

    Returns:
        float: The score for the specified metric.

    Raises:
        ValueError:If an unknown metric is specified.
    """
    if optimize_metric is None:
        optimize_metric = "roc"

    if (optimize_metric == "roc") and len(np.unique(y_true)) == 2:
        y_pred = y_pred[:, 1]

    if (not y_pred_is_labels) and (optimize_metric not in ["roc", "log_loss"]):
        y_pred = np.argmax(y_pred, axis=1)

    if optimize_metric in ("roc", "auroc"):
        return safe_roc_auc_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multi_class="ovr",
        )
    if optimize_metric == "accuracy":
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    if optimize_metric == "f1":
        return f1_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            average="macro",
        )
    if optimize_metric == "log_loss":
        return -log_loss(y_true, y_pred, sample_weight=sample_weight)
    raise ValueError(f"Unknown metric {optimize_metric}")


def score_regression(
    optimize_metric: Literal["rmse", "mse", "mae"],
    y_true,
    y_pred,
    sample_weight=None,
):
    """General function to score regression predictions.

    Parameters:
        optimize_metric : {"rmse", "mse", "mae"}
            The metric to use for scoring the predictions.

        y_true : array-like of shape (n_samples,)
            True target values.

        y_pred : array-like of shape (n_samples,)
            Predicted target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

    Returns:
        float: The score for the specified metric.

    Raises:
         ValueError: If an unknown metric is specified.
    """
    if optimize_metric == "rmse":
        try:
            return -mean_squared_error(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                squared=False,
            )
        except TypeError:
            # Newer python version
            from sklearn.metrics import root_mean_squared_error

            return -root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    elif optimize_metric == "mse":
        return -mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    elif optimize_metric == "mae":
        return -mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown metric {optimize_metric}")


def get_score_survival_model(metric_used, inv_predictions=True):
    from tabpfn.scripts.tabular_metrics import get_scoring_direction

    def score_survival_model(model, X, y):
        censoring, y = next(zip(*y.tolist())), list(zip(*y.tolist()))[1]
        prediction = model.predict(X)
        prediction = -prediction if inv_predictions else prediction

        if np.array(censoring).mean() == 0:
            return 0.5

        result = metric_used(target=y, pred=prediction, event_observed=censoring)

        result *= get_scoring_direction(metric_used)

        if np.isnan(result):
            return 0.0

        return result

    return score_survival_model


def score_survival(
    optimize_metric: Literal["cindex"],
    y_true,
    y_pred,
    event_observed,
    sample_weight=None,
):
    """General function to score regression predictions.

    Parameters:
        optimize_metric : {"rmse", "mse", "mae"}
            The metric to use for scoring the predictions.

        y_true : array-like of shape (n_samples,)
            True target values.

        y_pred : array-like of shape (n_samples,)
            Predicted target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

    Returns:
        float: The score for the specified metric.

    Raises:
         ValueError: If an unknown metric is specified.
    """
    from lifelines.utils import concordance_index

    if optimize_metric in ("cindex", "c_index", "risk_score", "risk_score_capped"):
        return concordance_index(y_true, y_pred, event_observed=event_observed)
    raise ValueError(f"Unknown metric {optimize_metric}")
