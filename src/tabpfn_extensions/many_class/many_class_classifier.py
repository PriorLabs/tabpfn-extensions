# Copyright (c) Prior Labs GmbH 2025.
# Licensed under the Apache License, Version 2.0

"""ManyClassClassifier: TabPFN extension for handling classification with many classes.

Development Notebook: https://colab.research.google.com/drive/1HWF5IF0IN21G8FZdLVwBbLBkCMu94yBA?usp=sharing

This module provides a classifier that overcomes TabPFN's limitation on the number of
classes (typically 10) by using a meta-classifier approach based on output coding.
It works by breaking down multi-class problems into multiple sub-problems, each
within TabPFN's class limit.

This version aims to be very close to an original structural design, with key
improvements in codebook generation and using a custom `validate_data` function
for scikit-learn compatibility.

Key features (compared to a very basic output coder):
- Improved codebook generation: Uses a strategy that attempts to balance the
  number of times each class is explicitly represented and guarantees coverage.
- Codebook statistics: Optionally prints statistics about the generated codebook.
- Uses a custom `validate_data` for potentially better cross-sklearn-version
  compatibility for data validation.
- Robustness: Minor changes for better scikit-learn compatibility (e.g.,
  ensuring the wrapper is properly "fitted", setting n_features_in_).

Original structural aspects retained:
- Fitting of base estimators for sub-problems largely occurs during predict_proba calls.

Example usage:
    ```python
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tabpfn import TabPFNClassifier # Assuming TabPFN is installed
    from sklearn.datasets import make_classification

    # Create synthetic data with many classes
    n_classes_total = 15 # TabPFN might struggle with >10 if not configured
    X, y = make_classification(n_samples=300, n_features=20, n_informative=15,
                               n_redundant=0, n_classes=n_classes_total,
                               n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    # Create a TabPFN base classifier
    # Adjust N_ensemble_configurations and device as needed/available
    # TabPFN's default class limit is often 10 for the public model.
    base_clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)

    # Wrap it with ManyClassClassifier
    many_class_clf = ManyClassClassifier(
        estimator=base_clf,
        alphabet_size=10, # Max classes the base_clf sub-problems will handle
                          # This should align with TabPFN's actual capability.
        n_estimators_redundancy=3,
        random_state=42,
        log_proba_aggregation=True,
        verbose=1 # Print codebook stats
    )

    # Use like any scikit-learn classifier
    many_class_clf.fit(X_train, y_train)
    y_pred = many_class_clf.predict(X_test)
    y_proba = many_class_clf.predict_proba(X_test)

    print(f"Prediction shape: {y_pred.shape}")
    print(f"Probability shape: {y_proba.shape}")
    if hasattr(many_class_clf, 'codebook_stats_'):
        print(f"Codebook Stats: {many_class_clf.codebook_stats_}")
    ```
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, ClassVar

import numpy as np
import tqdm  # For pairwise combinations
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state

# Imports as specified by the user
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    # _check_sample_weight, # Import if sample_weight functionality is added
    check_is_fitted,
    # _check_feature_names_in is used if X is validated by wrapper directly
    # but we aim to use the custom validate_data
)
from tabpfn_common_utils.telemetry import set_extension

# Custom validate_data import
from tabpfn_extensions.misc.sklearn_compat import validate_data


# Helper function: Fits a clone of the estimator on a specific sub-problem's
# training data. This follows the original design where fitting happens during
# prediction calls.
def _fit_and_predict_proba(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    Y_train_subproblem: np.ndarray,  # Encoded labels for one sub-problem
    X_pred: np.ndarray,  # Data to predict on
) -> np.ndarray:
    """Fit a cloned base estimator on sub-problem data and predict probabilities."""
    cloned_estimator = clone(estimator)
    # Base estimator's fit method will handle X_train validation
    cloned_estimator.fit(X_train, Y_train_subproblem)

    if hasattr(cloned_estimator, "predict_proba"):
        # Base estimator's predict_proba will handle X_pred validation
        return cloned_estimator.predict_proba(X_pred)
    raise AttributeError("Base estimator must implement the predict_proba method.")


@set_extension("many_class")
class ManyClassClassifier(BaseEstimator, ClassifierMixin):
    """Output-Code multiclass strategy to extend classifiers beyond their class limit.

    This version adheres closely to an original structural design, with key
    improvements in codebook generation and using a custom `validate_data` function
    for scikit-learn compatibility. Fitting for sub-problems primarily occurs
    during prediction.

    Args:
        estimator: A classifier implementing fit() and predict_proba() methods.
        alphabet_size (int, optional): Maximum number of classes the base
            estimator can handle. If None, attempts to infer from
            `estimator.max_num_classes_`.
        n_estimators (int, optional): Number of base estimators (sub-problems).
            If None, calculated based on other parameters.
        n_estimators_redundancy (int): Redundancy factor for auto-calculated
            `n_estimators`. Defaults to 4.
        random_state (int, RandomState instance or None): Controls randomization
            for codebook generation.
        verbose (int): Controls verbosity. If > 0, prints codebook stats.
            Defaults to 0.

    Attributes:
        classes_ (np.ndarray): Unique target labels.
        code_book_ (np.ndarray | None): Generated codebook if mapping is needed.
        codebook_stats_ (dict): Statistics about the generated codebook.
        estimators_ (list | None): Stores the single fitted base estimator *only*
            if `no_mapping_needed_` is True.
        no_mapping_needed_ (bool): True if n_classes <= alphabet_size.
        classes_index_ (dict | None): Maps class labels to indices.
        X_train (np.ndarray | None): Stored training features if mapping needed.
        Y_train_per_estimator (np.ndarray | None): Encoded training labels for each sub-problem.
                                        Shape (n_estimators, n_samples).
        n_features_in_ (int): Number of features seen during `fit`.
        feature_names_in_ (np.ndarray | None): Names of features seen during `fit`.

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from tabpfn import TabPFNClassifier
        >>> from tabpfn_extensions.many_class import ManyClassClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        >>> base_clf = TabPFNClassifier()
        >>> many_clf = ManyClassClassifier(base_clf, alphabet_size=base_clf.max_num_classes_)
        >>> many_clf.fit(X_train, y_train)
        >>> y_pred = many_clf.predict(X_test)
    """

    _required_parameters: ClassVar[list[str]] = ["estimator"]

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        alphabet_size: int | None = None,
        n_estimators: int | None = None,
        n_estimators_redundancy: int = 4,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.alphabet_size = alphabet_size
        self.n_estimators = n_estimators
        self.n_estimators_redundancy = n_estimators_redundancy
        self.verbose = verbose

    def _get_alphabet_size(self) -> int:
        """Helper to get alphabet_size, inferring if necessary."""
        if self.alphabet_size is not None:
            return self.alphabet_size
        try:
            # TabPFN specific attribute, or common one for models with class limits
            return self.estimator.max_num_classes_
        except AttributeError:
            # Fallback for estimators not exposing this directly
            # Might need to be explicitly set for such estimators.
            if self.verbose > 0:
                warnings.warn(
                    "Could not infer alphabet_size from estimator.max_num_classes_. "
                    "Ensure alphabet_size is correctly set if this is not TabPFN.",
                    UserWarning,
                    stacklevel=2,
                )
            # Default to a common small number if not TabPFN and not set,
            # though this might not be optimal.
            return 10

    def _get_n_estimators(self, n_classes: int, alphabet_size: int) -> int:
        """Helper to calculate the number of estimators."""
        if self.n_estimators is not None:
            return self.n_estimators
        if n_classes <= alphabet_size:
            return 1  # Only one base estimator needed
        # Using max(2,...) ensures alphabet_size for log is at least 2
        min_estimators_theory = math.ceil(math.log(n_classes, max(2, alphabet_size)))
        # Ensure enough estimators for potential coverage based on n_classes
        min_needed_for_potential_coverage = math.ceil(
            n_classes / max(1, alphabet_size - 1)
        )
        return (
            max(min_estimators_theory, min_needed_for_potential_coverage)
            * self.n_estimators_redundancy
            * 2
        )

    def _generate_codebook(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        random_state_instance: np.random.RandomState,
    ) -> tuple[np.ndarray, dict]:
        """(Improved) Generate codebook with balanced coverage and stats."""
        if n_classes <= alphabet_size:
            raise ValueError(
                "_generate_codebook called when n_classes <= alphabet_size"
            )

        codes_to_assign = list(range(alphabet_size - 1))
        n_codes_available = len(codes_to_assign)
        rest_class_code = alphabet_size - 1

        if n_codes_available == 0:
            raise ValueError(
                "alphabet_size must be at least 2 for codebook generation."
            )

        codebook = np.full((n_estimators, n_classes), rest_class_code, dtype=int)
        coverage_count = np.zeros(n_classes, dtype=int)

        for i in range(n_estimators):
            n_assignable_this_row = min(n_codes_available, n_classes)
            noisy_counts = coverage_count + random_state_instance.uniform(
                0, 0.1, n_classes
            )
            sorted_indices = np.argsort(noisy_counts)
            selected_classes_for_row = sorted_indices[:n_assignable_this_row]
            permuted_codes = random_state_instance.permutation(codes_to_assign)
            codes_to_use = permuted_codes[:n_assignable_this_row]
            codebook[i, selected_classes_for_row] = codes_to_use
            coverage_count[selected_classes_for_row] += 1

        if np.any(coverage_count == 0):
            uncovered_indices = np.where(coverage_count == 0)[0]
            raise RuntimeError(
                f"Failed to cover classes within {n_estimators} estimators. "
                f"{len(uncovered_indices)} uncovered (e.g., {uncovered_indices[:5]}). "
                f"Increase `n_estimators` or `n_estimators_redundancy`."
            )

        stats = {
            "coverage_min": int(np.min(coverage_count)),
            "coverage_max": int(np.max(coverage_count)),
            "coverage_mean": float(np.mean(coverage_count)),
            "coverage_std": float(np.std(coverage_count)),
            "n_estimators": n_estimators,
            "n_classes": n_classes,
            "alphabet_size": alphabet_size,
        }
        if n_classes > 1:
            min_dist = n_estimators
            for j1, j2 in itertools.combinations(range(n_classes), 2):
                dist = np.sum(codebook[:, j1] != codebook[:, j2])
                min_dist = min(min_dist, dist)
            stats["min_pairwise_hamming_dist"] = int(min_dist)

        if self.verbose > 0:
            for key, value in stats.items():
                pass
        return codebook, stats

    def fit(self, X, y, **fit_params) -> ManyClassClassifier:
        """Prepare classifier using custom validate_data.
        Actual fitting of sub-estimators happens in predict_proba if mapping is needed.
        """
        # Use the custom validate_data for y
        # Assuming it handles conversion to 1D and basic checks.
        # y_numeric=True is common for classification targets.
        X, y = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )
        # After validate_data, set feature_names_in_ if X is a DataFrame
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)

        random_state_instance = check_random_state(self.random_state)
        self.classes_ = unique_labels(y)  # Use unique_labels as imported
        n_classes = len(self.classes_)

        alphabet_size = self._get_alphabet_size()
        self.no_mapping_needed_ = n_classes <= alphabet_size
        self.codebook_stats_ = {}
        self.estimators_ = None
        self.code_book_ = None
        self.classes_index_ = None
        self.X_train = None
        self.Y_train_per_estimator = None

        if n_classes == 0:
            raise ValueError("Cannot fit with no classes present.")
        if n_classes == 1:
            # Gracefully handle single-class case: fit estimator, set trivial codebook
            if self.verbose > 0:
                pass
            cloned_estimator = clone(self.estimator)
            cloned_estimator.fit(X, y, **fit_params)
            self.estimators_ = [cloned_estimator]
            self.code_book_ = np.zeros((1, 1), dtype=int)
            self.codebook_stats_ = {
                "n_classes": 1,
                "n_estimators": 1,
                "alphabet_size": 1,
            }
            return self

        if self.no_mapping_needed_:
            cloned_estimator = clone(self.estimator)
            # Base estimator fits on X_validated (already processed by custom validate_data)
            cloned_estimator.fit(X, y, **fit_params)
            self.estimators_ = [cloned_estimator]
            # Ensure n_features_in_ matches the fitted estimator if it has the attribute
            if hasattr(cloned_estimator, "n_features_in_"):
                self.n_features_in_ = cloned_estimator.n_features_in_

        else:  # Mapping is needed
            if self.verbose > 0:
                pass
            n_est = self._get_n_estimators(n_classes, alphabet_size)
            self.code_book_, self.codebook_stats_ = self._generate_codebook(
                n_classes, n_est, alphabet_size, random_state_instance
            )
            self.classes_index_ = {c: i for i, c in enumerate(self.classes_)}
            self.X_train = X  # Store validated X
            y_indices = np.array([self.classes_index_[val] for val in y])
            self.Y_train_per_estimator = self.code_book_[:, y_indices]

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X. Sub-estimators are fitted here if mapping is used."""
        # Attributes to check if fitted, adapt from user's ["_tree", "X", "y"]
        # Key attributes for this classifier: classes_ must be set, n_features_in_ for X dim check.
        check_is_fitted(self, ["classes_", "n_features_in_"])

        # Use the custom validate_data for X in predict methods as well
        # reset=False as n_features_in_ should already be set from fit
        # Align DataFrame columns if needed
        X = validate_data(
            self,
            X,
            ensure_all_finite=False,  # As requested
        )

        if self.no_mapping_needed_:
            if not self.estimators_:
                raise RuntimeError("Estimator not fitted. Call fit first.")
            return self.estimators_[0].predict_proba(X)

        if (
            self.X_train is None
            or self.Y_train_per_estimator is None
            or self.code_book_ is None
        ):
            raise RuntimeError(
                "Fit method did not properly initialize for mapping. Call fit first."
            )

        Y_pred_probas_list = [
            _fit_and_predict_proba(
                self.estimator,
                self.X_train,  # This is X_validated from fit
                self.Y_train_per_estimator[i, :],
                X,  # Pass validated X to predict on
            )
            for i in tqdm.tqdm(range(self.code_book_.shape[0]))
        ]
        Y_pred_probas_arr = np.array(Y_pred_probas_list, dtype=np.float64)

        _n_estimators, n_samples, current_alphabet_size = Y_pred_probas_arr.shape
        if n_samples == 0:
            return np.zeros((0, len(self.classes_)))

        n_orig_classes = len(self.classes_)
        rest_class_code = self._get_alphabet_size() - 1

        raw_probabilities = np.zeros((n_samples, n_orig_classes))
        counts = np.zeros(n_orig_classes, dtype=float)
        for i in range(_n_estimators):
            for j_orig_class_idx in range(n_orig_classes):
                code_assigned = self.code_book_[i, j_orig_class_idx]
                if code_assigned != rest_class_code:
                    raw_probabilities[:, j_orig_class_idx] += Y_pred_probas_arr[
                        i, :, code_assigned
                    ]
                    counts[j_orig_class_idx] += 1

        valid_counts_mask = counts > 0
        final_probabilities = np.zeros_like(raw_probabilities)
        if np.any(valid_counts_mask):
            final_probabilities[:, valid_counts_mask] = (
                raw_probabilities[:, valid_counts_mask] / counts[valid_counts_mask]
            )
        if not np.all(valid_counts_mask) and self.verbose > 0:
            warnings.warn(
                "Some classes had zero specific code assignments during aggregation.",
                RuntimeWarning,
                stacklevel=2,
            )

        prob_sum = np.sum(final_probabilities, axis=1, keepdims=True)
        safe_sum = np.where(prob_sum == 0, 1.0, prob_sum)
        final_probabilities /= safe_sum
        final_probabilities[prob_sum.squeeze() == 0] = 1.0 / n_orig_classes

        return final_probabilities

    def predict(self, X) -> np.ndarray:
        """Predict multi-class targets for X."""
        # Attributes to check if fitted, adapt from user's ["_tree", "X", "y"]
        check_is_fitted(self, ["classes_", "n_features_in_"])
        # X will be validated by predict_proba or base_estimator.predict

        if self.no_mapping_needed_ or (
            hasattr(self, "estimators_")
            and self.estimators_ is not None
            and len(self.estimators_) == 1
        ):
            if not self.estimators_:
                raise RuntimeError("Estimator not fitted. Call fit first.")
            # Base estimator's predict validates X
            return self.estimators_[0].predict(X)

        probas = self.predict_proba(X)
        if probas.shape[0] == 0:
            return np.array([], dtype=self.classes_.dtype)
        return self.classes_[np.argmax(probas, axis=1)]

    def set_categorical_features(self, categorical_features: list[int]) -> None:
        """Attempts to set categorical features on the base estimator."""
        self.categorical_features = categorical_features
        if hasattr(self.estimator, "set_categorical_features"):
            self.estimator.set_categorical_features(categorical_features)
        elif hasattr(self.estimator, "categorical_features"):
            self.estimator.categorical_features = categorical_features
        elif self.verbose > 0:
            warnings.warn(
                "Base estimator has no known categorical feature support.",
                UserWarning,
                stacklevel=2,
            )

    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags

    @property
    def codebook_statistics_(self):
        """Returns statistics about the generated codebook."""
        check_is_fitted(self, ["classes_"])  # Minimal check
        if self.no_mapping_needed_:
            return {"message": "No codebook mapping was needed."}
        return getattr(self, "codebook_stats_", {})
