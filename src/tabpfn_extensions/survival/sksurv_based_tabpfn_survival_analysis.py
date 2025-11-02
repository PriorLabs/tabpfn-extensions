#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sksurv.base import SurvivalAnalysisMixin
from tabpfn_common_utils.telemetry import set_extension

try:
    from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor
except ImportError:
    raise ImportError(
        "TabPFN extensions utils module not found. Please make sure tabpfn_extensions is installed correctly.",
    )


@set_extension("survival")
class SurvivalTabPFN(SurvivalAnalysisMixin):
    r"""TabPFN classifier and regressor blended together to predict risk scores
    to better optimize concordance index scores for survival analysis.

    This ensemble model combines a TabPFNClassifier with a TabPFNRegressor to
    improve performance on predicting risk scores to optimize concordance_index
    scores. It leverages scikit-survival's API to add TabPFN as a new, powerful
    model option for Survival Analysis.

    For more details on TabPFN see [1]_.
    For more details on Survival Analysis, see scikit-survival documentation:
    https://scikit-survival.readthedocs.io/en/stable/index.html

    Parameters
    ----------
    cls_model : TabPFNClassifier | None, default: None
        A pre-initialised :class:`~tabpfn_extensions.utils.TabPFNClassifier`.
        When ``None`` (the default) a new classifier instance will be created
        internally using ``ignore_pretraining_limits`` and ``random_state``.
        Supplying a classifier instance allows advanced users to customise
        loading behaviour (e.g. from a specific checkpoint path) before
        passing it to :class:`SurvivalTabPFN`.

    reg_model : TabPFNRegressor | None, default: None
        A pre-initialised :class:`~tabpfn_extensions.utils.TabPFNRegressor`.
        When ``None`` (the default) a new regressor instance will be created
        internally using ``ignore_pretraining_limits`` and ``random_state``.
        Provide a custom regressor when you need to control how the
        underlying TabPFN model is configured or loaded.

    ignore_pretraining_limits : bool, default: False
        Whether to ignore the pre-training limits of the TabPFN models. These
        limits cover the number of samples, features, and classes the models
        were trained on. Setting this to ``True`` suppresses warnings or errors
        when operating outside these limits (e.g. more than 1_000 samples on
        CPU), but results may degrade.

    random_state : int | numpy.random.RandomState | None, default: None
        Controls the random seed passed to both internal TabPFN models.
        Provide an ``int`` for deterministic behaviour across runs. ``None``
        uses the library defaults.

    References:
    ----------
    .. [1] Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer,
           M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F.,
           "Accurate predictions on small data with a tabular foundation model."
           Nature, 637(8045), 319-326, 2025
    """

    def __init__(
        self,
        *,
        cls_model=None,
        reg_model=None,
        ignore_pretraining_limits=False,
        random_state=None,
    ):
        if cls_model is None:
            self.cls_model = TabPFNClassifier(
                ignore_pretraining_limits=ignore_pretraining_limits,
                random_state=random_state,
            )
        else:
            self.cls_model = cls_model

        if reg_model is None:
            self.reg_model = TabPFNRegressor(
                ignore_pretraining_limits=ignore_pretraining_limits,
                random_state=random_state,
            )
        else:
            self.reg_model = reg_model

    def fit(self, X: np.ndarray, y: np.ndarray | list | dict) -> SurvivalTabPFN:
        """Fits the survival analysis model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray | list | dict
            Training target data. Accepts either a structured numpy array with
            ``("event", "time")`` fields, a mapping with corresponding keys,
            or an iterable of ``(event_indicator, time)`` tuples.
        """
        if isinstance(y, np.ndarray) and getattr(y.dtype, "names", None):
            y_event = y["event"].astype(bool)
            y_time = y["time"]
        elif isinstance(y, dict):
            y_event = np.asarray(y["event"], dtype=bool)
            y_time = np.asarray(y["time"])
        else:
            y_event = np.array([n[0] for n in y], dtype=bool)
            y_time = np.array([n[1] for n in y])

        assert y_event.sum() >= 2, "You need atleast two events in your data."

        self.cls_model.fit(X, y_event)

        # Rank longest time to shortest time from 0.0 to 1.0, only for event times where event==True
        X_with_event = X[y_event]
        y_time_with_event = y_time[y_event]
        reversed_y_time_with_event = -y_time_with_event
        y_ranked_risk = (rankdata(reversed_y_time_with_event) - 1) / (
            reversed_y_time_with_event.shape[0] - 1
        )
        self.reg_model.fit(X_with_event, y_ranked_risk)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts risk scores for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples to predict.

        Returns:
        -------
        np.ndarray of shape (n_samples,)
            The predicted risk scores.
        """
        p_a = self.cls_model.predict_proba(X)[:, 1]
        p_b = self.reg_model.predict(X)
        preds = p_a * p_b

        return preds

    @property
    def _predict_risk_score(self):
        return True
