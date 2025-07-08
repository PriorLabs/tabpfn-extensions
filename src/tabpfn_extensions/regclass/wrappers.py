from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class DistributionalRegressorAsClassifier(ClassifierMixin, BaseEstimator):
    """
    Wrap a probabilistic regressor to behave like a multi-class classifier.

    This wrapper is designed for regressors that output a full predictive
    distribution (e.g., `TabPFNRegressor` when `output_type="full"`). It allows
    for classifying continuous regression targets into discrete buckets defined
    by specified thresholds.

    It can operate in two primary modes for converting the regressor's
    distribution into class probabilities:

    1.  **"probabilistic" (Default):**
        This strategy leverages the regressor's cumulative distribution function (CDF)
        to directly calculate the probability mass within each defined classification
        bucket. This is a principled approach that fully utilizes the uncertainty
        information from the regressor's predictive distribution. It is
        recommended for most use cases as it provides true probability estimates
        for each class based on the regressor's model.

    2.  **"weighted":**
        This strategy calculates a "weighted score" for each bucket. It
        multiplies the probability of each fine-grained interval (or "bar" in
        the context of `TabPFNRegressor`'s bar distribution) by the absolute
        value of its expected mean. These weighted values are then aggregated
        into the corresponding class buckets. This approach gives more emphasis
        to regions of the distribution that are further from the thresholds
        (i.e., "stronger" predictions), which can be useful in specific
        scenarios where the magnitude of the regression target is highly
        indicative of class certainty. Note that the output of this strategy,
        while normalized to sum to one, represents weighted scores rather than
        strict probabilities in the same sense as the "probabilistic" strategy.

    Parameters
    ----------
    estimator : object
        A scikit-learn-style probabilistic regressor that can return a
        distribution object. This estimator must implement a `predict` method
        that, when called with `output_type="full"`, returns a dictionary
        containing `'criterion'` (a distribution object with a `cdf` method,
        like `FullSupportBarDistribution` from TabPFN) and `'logits'` (the
        unnormalized log-probabilities over the distribution's support).
        `TabPFNRegressor` is a compatible estimator.

    thresholds : Sequence[float]
        A sorted sequence of numeric cut-points. These thresholds define the
        boundaries between the classification buckets (classes). If `k`
        thresholds are provided, they will define `k+1` classification classes.
        For example, `thresholds=[0]` defines two classes: `(-inf, 0]` and `(0, inf)`.
        `thresholds=[-1, 1]` defines three classes: `(-inf, -1]`, `(-1, 1]`, and `(1, inf)`.

    decision_strategy : {'probabilistic', 'weighted'}, default='probabilistic'
        The strategy to use for converting the regression distribution into
        class probabilities.
        - "probabilistic": Uses the cumulative distribution function (CDF) to
          find the probability mass in each bucket.
        - "weighted": Weights the probability of each interval by its expected
          absolute value before aggregation.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels known to the classifier. These are integers from 0 to
        `n_classes_ - 1`.

    n_classes_ : int
        The number of classes inferred from the `thresholds`.

    thresholds_ : tuple of float
        The sorted tuple of thresholds used to define the class boundaries.

    Notes
    -----
    This wrapper assumes the underlying regressor's `predict(..., output_type="full")`
    method returns a structure consistent with `TabPFNRegressor`'s
    `FullSupportBarDistribution` for the `criterion` and associated `logits`.
    Compatibility with other probabilistic regressors is not guaranteed without
    their adherence to this specific output format.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        thresholds: Sequence[float],
        decision_strategy: Literal["probabilistic", "weighted"] = "probabilistic",
    ) -> None:        
        if not hasattr(estimator, "predict"):
            raise TypeError("estimator must implement a predict() method.")
        if not all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds) - 1)):
            raise ValueError("Thresholds must be sorted in ascending order.")
        if decision_strategy not in {"probabilistic", "weighted"}:
            raise ValueError(f"decision_strategy must be 'probabilistic' or 'weighted', got '{decision_strategy}'.")


        self.estimator = estimator
        self.thresholds = thresholds
        self.decision_strategy = decision_strategy

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistributionalRegressorAsClassifier":
        """
        Fit the underlying regressor on the continuous target `y`.

        The provided `thresholds` are sorted and stored, and the number of
        classes is inferred from them. The `y` values are passed directly to the
        underlying regressor as continuous labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The continuous regression target values corresponding to `X`. These
            values will be used by the underlying regressor to learn the
            distribution.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # For compatibility with scikit-learn's validation utilities
        self.thresholds_ = tuple(sorted(self.thresholds))
        self.n_classes_ = len(self.thresholds_) + 1
        self.classes_ = np.arange(self.n_classes_)

        # Fit the wrapped regressor on the continuous labels
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the most likely class label for each sample in X.
        """
        check_is_fitted(self)
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate class probabilities for samples in X.

        The probabilities are derived from the regressor's predictive
        distribution based on the `decision_strategy` chosen during
        initialization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict class probabilities.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes_)
            The class probabilities for each sample. The columns correspond to
            the classes in `self.classes_`.
        """
        check_is_fitted(self)

        # 1. Get the full distributional output from the TabPFNRegressor
        pred_dict = self.estimator.predict(X, output_type="full")
        criterion = pred_dict["criterion"]
        logits = pred_dict["logits"]
        device = logits.device

        # --- Strategy 1: Probabilistic (CDF-based) ---
        if self.decision_strategy == "probabilistic":
            # 2. Calculate the CDF at each threshold point.
            threshold_tensor = torch.tensor(self.thresholds_, device=device, dtype=torch.float32)
            cdf_values = criterion.cdf(logits, threshold_tensor)
            cdf_values = cdf_values.cpu().detach().numpy()

            # 3. Calculate the probability for each bucket by taking differences.
            n_samples = X.shape[0]
            probs = np.zeros((n_samples, self.n_classes_))
            probs[:, 0] = cdf_values[:, 0]
            for i in range(1, self.n_classes_ - 1):
                probs[:, i] = cdf_values[:, i] - cdf_values[:, i - 1]
            probs[:, -1] = 1.0 - cdf_values[:, -1]

        # --- Strategy 2: Weighted by Magnitude ---
        elif self.decision_strategy == "weighted":
            # 2. Get per-bar probabilities and expected values (means)
            probs_per_bar = logits.softmax(-1)
            bar_dist_borders = criterion.borders.to(device, dtype=torch.float32)

            # Replicate logic from FullSupportBarDistribution.mean to get per-bar means
            bucket_widths = (bar_dist_borders[1:] - bar_dist_borders[:-1]).to(device, dtype=torch.float32)
            bucket_means = bar_dist_borders[:-1] + bucket_widths / 2
            side_normals = (
                criterion.halfnormal_with_p_weight_before(criterion.bucket_widths[0]),
                criterion.halfnormal_with_p_weight_before(criterion.bucket_widths[-1]),
            )
            bucket_means[0] = -side_normals[0].mean.to(device, dtype=torch.float32) + bar_dist_borders[1]
            bucket_means[-1] = side_normals[1].mean.to(device, dtype=torch.float32) + bar_dist_borders[-2]

            # 3. Weight probabilities by the *magnitude* of the expected value
            weighted_bar_values = probs_per_bar * torch.abs(bucket_means)

            # 4. Map the regressor's bars to the classifier's buckets
            bar_midpoints = (bar_dist_borders[:-1] + bar_dist_borders[1:]) / 2
            threshold_tensor = torch.tensor(self.thresholds_, device=device, dtype=torch.float32)
            class_indices_per_bar = torch.bucketize(bar_midpoints, threshold_tensor)

            # 5. Sum the weighted values into the corresponding class buckets
            n_samples = X.shape[0]
            weighted_scores = torch.zeros((n_samples, self.n_classes_), device=device, dtype=torch.float32)
            idx_tensor = class_indices_per_bar.unsqueeze(0).expand_as(weighted_bar_values)
            weighted_scores.scatter_add_(1, idx_tensor, weighted_bar_values)

            probs = weighted_scores.cpu().detach().numpy()
        else:
            raise ValueError(f"Unknown decision_strategy: '{self.decision_strategy}'")

        # Normalize probabilities
        row_sums = np.sum(probs, axis=1, keepdims=True)
        return probs / (row_sums + 1e-9)

    def _more_tags(self) -> dict[str, Any]:
        """Expose the underlying estimator's tags."""
        return self.estimator._more_tags()