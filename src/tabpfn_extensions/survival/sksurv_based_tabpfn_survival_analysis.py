#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sksurv.base import SurvivalAnalysisMixin
from tabpfn_common_utils.telemetry import set_extension

# Import TabPFN models from extensions (which handles backend compatibility)
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
    cls_model_path : default: auto
        The path to the TabPFN model file for the TabPFNClassifier,
        i.e., the pre-trained weights.

        - If `"auto"`, the model will be downloaded upon first use. This
          defaults to your system cache directory, but can be overwritten
          with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
        - If a path or a string of a path, the model will be loaded from
          the user-specified location if available, otherwise it will be
          downloaded to this location.

    reg_model_path : default: auto
        The path to the TabPFN model file for the TabPFNRegressor,
        i.e., the pre-trained weights.

        - If `"auto"`, the model will be downloaded upon first use. This
          defaults to your system cache directory, but can be overwritten
          with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
        - If a path or a string of a path, the model will be loaded from
          the user-specified location if available, otherwise it will be
          downloaded to this location.

    ignore_pretraining_limits : bool, default: False
        Whether to ignore the pre-training limits of the model. The TabPFN
        models have been pre-trained on a specific range of input data. If the
        input data is outside of this range, the model may not perform well.
        You may ignore our limits to use the model on data outside the
        pre-training range.

        - If `True`, the model will not raise an error if the input data is
          outside the pre-training range. Also suppresses error when using
          the model with more than 1000 samples on CPU.
        - If `False`, you can use the model outside the pre-training range, but
          the model could perform worse.

        !!! note

            The current pre-training limits are:

            - 10_000 samples/rows
            - 500 features/columns
            - 10 classes, this is not ignorable and will raise an error
              if the model is used with more classes.

    random_state : int, RandomState instance, or None, optional, default: None
        Controls the random seed given to each TabPFN model.
        Pass an int for reproducible output across multiple function calls.

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
        cls_model_path="auto",
        reg_model_path="auto",
        ignore_pretraining_limits=False,
        random_state=None,
    ):
        self.cls_model = TabPFNClassifier(
            model_path=cls_model_path,
            ignore_pretraining_limits=ignore_pretraining_limits,
            random_state=random_state,
        )
        self.reg_model = TabPFNRegressor(
            model_path=reg_model_path,
            ignore_pretraining_limits=ignore_pretraining_limits,
            random_state=random_state,
        )

    def fit(
            self, X: np.ndarray, y: np.ndarray | list
    ) -> SurvivalTabPFN:
        """Fits the survival analysis model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray | list
            Training target data. Assumed to be an iterable of
            (event_indicator, time_to_event_or_censoring) elements.
        """
        y_event = np.array([n[0] for n in y]).astype(bool)
        y_time = np.array([n[1] for n in y])
        self.cls_model.fit(X, y_event)

        ## Rank longest time to shortest time from 0.0 to 1.0, only for event times where event==True
        X_with_event = X[y_event]
        y_time_with_event = y_time[y_event]
        reversed_y_time_with_event = -y_time_with_event
        y_ranked_risk = (rankdata(reversed_y_time_with_event) - 1) / (
            reversed_y_time_with_event.shape[0] - 1
        )
        self.reg_model.fit(X_with_event, y_ranked_risk)

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
