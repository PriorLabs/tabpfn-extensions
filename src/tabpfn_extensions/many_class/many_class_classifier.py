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

import logging
import math
import warnings
from typing import Any, ClassVar, Literal

import numpy as np
import tqdm  # Progress bar for sub-estimator fits
from scipy.spatial.distance import pdist
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

logger = logging.getLogger(__name__)


CODEBOOK_DEFAULT_RETRIES = 3
CODEBOOK_DEFAULT_MIN_HAMMING_FRAC = 0.30
CODEBOOK_DEFAULT_SELECTION = "max_min_hamming"
CODEBOOK_HAMMING_MAX_CLASSES = 200


# Helper function: Fits a clone of the estimator on a specific sub-problem's
# training data. This follows the original design where fitting happens during
# prediction calls.
def _apply_categorical_features_to_estimator(
    estimator: BaseEstimator, categorical_features: list[int] | None
) -> None:
    """Apply stored categorical feature metadata to an estimator clone."""
    if categorical_features is None:
        return
    if hasattr(estimator, "set_categorical_features"):
        estimator.set_categorical_features(categorical_features)
    elif hasattr(estimator, "categorical_features"):
        estimator.categorical_features = categorical_features


def _fit_and_predict_proba(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    Y_train_subproblem: np.ndarray,  # Encoded labels for one sub-problem
    X_pred: np.ndarray,  # Data to predict on
    *,
    categorical_features: list[int] | None = None,
    fit_params: dict[str, Any] | None = None,
    alphabet_size: int,
) -> np.ndarray:
    """Fit a cloned base estimator on sub-problem data and predict probabilities."""
    cloned_estimator = clone(estimator)
    _apply_categorical_features_to_estimator(cloned_estimator, categorical_features)

    fit_kwargs: dict[str, Any] = fit_params.copy() if fit_params else {}
    cloned_estimator.fit(X_train, Y_train_subproblem, **fit_kwargs)

    if not hasattr(cloned_estimator, "predict_proba"):
        raise AttributeError("Base estimator must implement the predict_proba method.")

    proba = cloned_estimator.predict_proba(X_pred)
    classes_seen = getattr(cloned_estimator, "classes_", None)
    if classes_seen is None:
        raise AttributeError(
            "Base estimator must expose `classes_` after fitting to align probabilities."
        )

    full_proba = np.zeros((proba.shape[0], alphabet_size), dtype=np.float64)
    indices = np.asarray(classes_seen, dtype=int)
    full_proba[:, indices] = proba
    return full_proba


@set_extension("many_class")
class ManyClassClassifier(ClassifierMixin, BaseEstimator):
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
        codebook_strategy (str): Strategy used to create the ECOC codebook.
            ``"balanced_cluster"`` partitions classes into balanced groups for each
            estimator and is the default. ``"legacy_rest"`` uses the previous
            one-vs-rest-style codebook that reserves one symbol as a catch-all.
        n_estimators (int, optional): Number of base estimators (sub-problems).
            If None, calculated based on other parameters.
        n_estimators_redundancy (int): Redundancy factor for auto-calculated
            `n_estimators`. Defaults to 4.
        random_state (int, RandomState instance or None): Controls randomization
            for codebook generation.
        verbose (int): Controls verbosity. If > 0, prints codebook stats.
            Defaults to 0.
        log_proba_aggregation (bool): If True (default), aggregates sub-problem
            predictions using log-likelihood decoding across the full codebook.
            When False, falls back to the legacy averaging strategy that ignores
            the "rest" bucket.

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
        log_proba_aggregation (bool): Strategy flag controlling probability aggregation.

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
        codebook_strategy: str = "balanced_cluster",
        n_estimators: int | None = None,
        n_estimators_redundancy: int = 4,
        random_state: int | None = None,
        verbose: int = 0,
        log_proba_aggregation: bool = True,
        codebook_retries: int = CODEBOOK_DEFAULT_RETRIES,
        codebook_min_hamming_frac: float = CODEBOOK_DEFAULT_MIN_HAMMING_FRAC,
        codebook_selection: Literal["max_min_hamming"] = CODEBOOK_DEFAULT_SELECTION,
        codebook_hamming_max_classes: int = CODEBOOK_HAMMING_MAX_CLASSES,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.alphabet_size = alphabet_size
        self.n_estimators = n_estimators
        self.codebook_strategy = codebook_strategy
        self.n_estimators_redundancy = n_estimators_redundancy
        self.verbose = verbose
        self.log_proba_aggregation = log_proba_aggregation
        self.fit_params_: dict[str, Any] | None = None
        if codebook_retries < 1:
            raise ValueError("codebook_retries must be at least 1.")
        if codebook_min_hamming_frac < 0.0:
            raise ValueError("codebook_min_hamming_frac must be non-negative.")
        if codebook_hamming_max_classes < 0:
            raise ValueError("codebook_hamming_max_classes must be non-negative.")
        if codebook_selection != "max_min_hamming":
            raise ValueError("Unsupported codebook_selection criterion.")
        self.codebook_retries = int(codebook_retries)
        self.codebook_min_hamming_frac = float(codebook_min_hamming_frac)
        self.codebook_selection = codebook_selection
        self.codebook_hamming_max_classes = int(codebook_hamming_max_classes)

    def _set_verbosity(self) -> None:
        """Configure the module-level logger according to the estimator verbosity."""
        level = (
            logging.WARNING
            if self.verbose <= 0
            else logging.INFO
            if self.verbose == 1
            else logging.DEBUG
        )
        logger.setLevel(level)

    def _log_codebook_stats(self, stats: dict[str, Any], *, tag: str) -> None:
        if not stats:
            return
        logger.info(
            "[ManyClassClassifier] %s: %s",
            tag,
            {
                key: stats.get(key)
                for key in (
                    "strategy",
                    "codebook_selection",
                    "n_classes",
                    "alphabet_size",
                    "n_estimators",
                    "coverage_min",
                    "coverage_max",
                    "coverage_mean",
                    "coverage_std",
                    "min_pairwise_hamming_dist",
                    "mean_pairwise_hamming_dist",
                    "best_min_pairwise_hamming_dist",
                    "regeneration_attempts",
                )
                if key in stats
            },
        )

    def _log_shapes(
        self,
        *,
        X_train: np.ndarray | None,
        X: np.ndarray,
        Y_train_per_estimator: np.ndarray | None,
        proba_arr: np.ndarray | None,
    ) -> None:
        logger.debug(
            "Shapes | X_train=%s X=%s Y_train_per_estimator=%s proba_arr=%s",
            getattr(X_train, "shape", None),
            getattr(X, "shape", None),
            getattr(Y_train_per_estimator, "shape", None),
            getattr(proba_arr, "shape", None),
        )

    def _codebook_quality_key(
        self,
        *,
        min_dist: int | None,
        mean_dist: float | None,
        stats: dict[str, Any],
        attempt_seed: int,
    ) -> tuple[float, float, float, int]:
        """Produce a comparable quality key for codebook selection."""
        min_metric = float("-inf") if min_dist is None else float(min_dist)
        mean_metric = float("-inf") if mean_dist is None else float(mean_dist)
        coverage_std = stats.get("coverage_std")
        coverage_metric = (
            float("-inf") if coverage_std is None else -float(coverage_std)
        )
        return (min_metric, mean_metric, coverage_metric, attempt_seed)

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

        log_base = max(2, alphabet_size)
        min_estimators_theory = max(1, math.ceil(math.log(n_classes, log_base)))
        min_needed_for_potential_coverage = math.ceil(
            n_classes / max(1, alphabet_size - 1)
        )
        max_needed = max(min_estimators_theory, min_needed_for_potential_coverage)
        expanded = max_needed * max(1, self.n_estimators_redundancy)
        log_cap = 4 * min_estimators_theory
        capped = min(expanded, log_cap) if log_cap > 0 else expanded
        return max(max_needed, capped)

    def _summarize_codebook(
        self,
        *,
        codebook: np.ndarray,
        coverage_count: np.ndarray,
        n_estimators: int,
        n_classes: int,
        alphabet_size: int,
        strategy: str,
        has_rest_symbol: bool,
        rest_class_code: int | None,
    ) -> tuple[dict[str, Any], int | None, float | None]:
        """Compute statistics for a generated codebook and return its quality."""
        stats = {
            "coverage_min": int(np.min(coverage_count)),
            "coverage_max": int(np.max(coverage_count)),
            "coverage_mean": float(np.mean(coverage_count)),
            "coverage_std": float(np.std(coverage_count)),
            "n_estimators": n_estimators,
            "n_classes": n_classes,
            "alphabet_size": alphabet_size,
            "strategy": strategy,
            "has_rest_symbol": has_rest_symbol,
            "rest_class_code": rest_class_code,
            "codebook_selection": self.codebook_selection,
        }

        min_dist_count: int | None = None
        mean_dist_count: float | None = None
        if 1 < n_classes < self.codebook_hamming_max_classes:
            distances = pdist(codebook.T, metric="hamming") * n_estimators
            if distances.size > 0:
                min_dist_count = int(np.rint(np.min(distances)))
                mean_dist_count = float(np.mean(distances))
        stats["min_pairwise_hamming_dist"] = min_dist_count
        if mean_dist_count is not None:
            stats["mean_pairwise_hamming_dist"] = mean_dist_count
        else:
            stats["mean_pairwise_hamming_dist"] = None
        return stats, min_dist_count, mean_dist_count

    def _generate_codebook_balanced_cluster(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        random_state_instance: np.random.RandomState,
    ) -> tuple[np.ndarray, dict]:
        """Generate a dense, balanced codebook using random partitioning."""
        if n_classes <= alphabet_size:
            raise ValueError(
                "Balanced codebook generation requires n_classes > alphabet_size."
            )

        coverage_count = np.full(n_classes, n_estimators, dtype=int)
        max_attempts = max(1, self.codebook_retries)
        target = (
            math.ceil(self.codebook_min_hamming_frac * n_estimators)
            if self.codebook_min_hamming_frac > 0
            else None
        )
        best_codebook: np.ndarray | None = None
        best_stats: dict[str, Any] | None = None
        best_min_dist: int | None = None
        best_mean_dist: float | None = None
        best_quality: tuple[float, float, float, int] | None = None
        attempts_used = 0

        for attempt in range(max_attempts):
            attempt_seed = int(
                random_state_instance.randint(0, np.iinfo(np.uint32).max)
            )
            attempt_rng = np.random.RandomState(attempt_seed)
            codebook = np.zeros((n_estimators, n_classes), dtype=int)
            class_indices = np.arange(n_classes)

            highlight_rows = min(n_classes, n_estimators)
            for row_idx in range(highlight_rows):
                focus_class = row_idx % n_classes
                other_classes = np.delete(class_indices, focus_class)
                codebook[row_idx, focus_class] = 0
                if alphabet_size > 1 and other_classes.size > 0:
                    attempt_rng.shuffle(other_classes)
                    other_groups = np.array_split(other_classes, alphabet_size - 1)
                    for offset, group in enumerate(other_groups, start=1):
                        codebook[row_idx, group] = offset

            for row_idx in range(highlight_rows, n_estimators):
                row_classes = class_indices.copy()
                attempt_rng.shuffle(row_classes)
                class_groups = np.array_split(row_classes, alphabet_size)
                for code, group in enumerate(class_groups):
                    codebook[row_idx, group] = code

            stats, min_dist, mean_dist = self._summarize_codebook(
                codebook=codebook,
                coverage_count=coverage_count,
                n_estimators=n_estimators,
                n_classes=n_classes,
                alphabet_size=alphabet_size,
                strategy="balanced_cluster",
                has_rest_symbol=False,
                rest_class_code=None,
            )
            attempts_used = attempt + 1
            quality_key = self._codebook_quality_key(
                min_dist=min_dist,
                mean_dist=mean_dist,
                stats=stats,
                attempt_seed=attempt_seed,
            )
            logger.info(
                "Codebook attempt %d/%d (balanced_cluster): min_hamming=%s mean_hamming=%s",
                attempts_used,
                max_attempts,
                min_dist,
                mean_dist,
            )

            if best_quality is None or quality_key > best_quality:
                best_quality = quality_key
                best_codebook = codebook
                best_stats = stats
                best_min_dist = min_dist
                best_mean_dist = mean_dist

            if target is not None and min_dist is not None and min_dist >= target:
                logger.info(
                    "Early exit after %d attempts: achieved min Hamming %s (target %s)",
                    attempts_used,
                    min_dist,
                    target,
                )
                break

        if best_codebook is None or best_stats is None:
            raise RuntimeError("Failed to generate a valid balanced codebook.")

        best_stats["regeneration_attempts"] = attempts_used
        best_stats["best_min_pairwise_hamming_dist"] = best_min_dist
        best_stats["mean_pairwise_hamming_dist"] = best_mean_dist
        logger.info(
            "Selected balanced_cluster codebook after %d attempts with min_hamming=%s",
            attempts_used,
            best_min_dist,
        )
        return best_codebook, best_stats

    def _generate_codebook_legacy_rest(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        random_state_instance: np.random.RandomState,
    ) -> tuple[np.ndarray, dict]:
        """Generate a legacy codebook with an explicit rest symbol."""
        if n_classes <= alphabet_size:
            raise ValueError(
                "Legacy codebook generation requires n_classes > alphabet_size."
            )

        codes_to_assign = list(range(alphabet_size - 1))
        n_codes_available = len(codes_to_assign)
        rest_class_code = alphabet_size - 1

        if n_codes_available == 0:
            raise ValueError(
                "alphabet_size must be at least 2 for codebook generation."
            )
        max_attempts = max(1, self.codebook_retries)
        target = (
            math.ceil(self.codebook_min_hamming_frac * n_estimators)
            if self.codebook_min_hamming_frac > 0
            else None
        )
        best_codebook: np.ndarray | None = None
        best_stats: dict[str, Any] | None = None
        best_min_dist: int | None = None
        best_mean_dist: float | None = None
        best_quality: tuple[float, float, float, int] | None = None
        attempts_used = 0

        for attempt in range(max_attempts):
            attempt_seed = int(
                random_state_instance.randint(0, np.iinfo(np.uint32).max)
            )
            attempt_rng = np.random.RandomState(attempt_seed)
            codebook = np.full((n_estimators, n_classes), rest_class_code, dtype=int)
            coverage_count = np.zeros(n_classes, dtype=int)

            for row_idx in range(n_estimators):
                n_assignable_this_row = min(n_codes_available, n_classes)
                noisy_counts = coverage_count + attempt_rng.uniform(0, 0.1, n_classes)
                sorted_indices = np.argsort(noisy_counts)
                selected_classes_for_row = sorted_indices[:n_assignable_this_row]
                permuted_codes = attempt_rng.permutation(codes_to_assign)
                codes_to_use = permuted_codes[:n_assignable_this_row]
                codebook[row_idx, selected_classes_for_row] = codes_to_use
                coverage_count[selected_classes_for_row] += 1

            if np.any(coverage_count == 0):
                uncovered_indices = np.where(coverage_count == 0)[0]
                raise RuntimeError(
                    "Failed to cover classes within "
                    f"{n_estimators} estimators. {len(uncovered_indices)} uncovered "
                    f"(e.g., {uncovered_indices[:5]})."
                )

            stats, min_dist, mean_dist = self._summarize_codebook(
                codebook=codebook,
                coverage_count=coverage_count,
                n_estimators=n_estimators,
                n_classes=n_classes,
                alphabet_size=alphabet_size,
                strategy="legacy_rest",
                has_rest_symbol=True,
                rest_class_code=rest_class_code,
            )
            attempts_used = attempt + 1
            quality_key = self._codebook_quality_key(
                min_dist=min_dist,
                mean_dist=mean_dist,
                stats=stats,
                attempt_seed=attempt_seed,
            )
            logger.info(
                "Codebook attempt %d/%d (legacy_rest): min_hamming=%s mean_hamming=%s",
                attempts_used,
                max_attempts,
                min_dist,
                mean_dist,
            )

            if best_quality is None or quality_key > best_quality:
                best_quality = quality_key
                best_codebook = codebook
                best_stats = stats
                best_min_dist = min_dist
                best_mean_dist = mean_dist

            if target is not None and min_dist is not None and min_dist >= target:
                logger.info(
                    "Early exit after %d attempts: achieved min Hamming %s (target %s)",
                    attempts_used,
                    min_dist,
                    target,
                )
                break

        if best_codebook is None or best_stats is None:
            raise RuntimeError("Failed to generate a valid legacy codebook.")

        best_stats["regeneration_attempts"] = attempts_used
        best_stats["best_min_pairwise_hamming_dist"] = best_min_dist
        best_stats["mean_pairwise_hamming_dist"] = best_mean_dist
        logger.info(
            "Selected legacy_rest codebook after %d attempts with min_hamming=%s",
            attempts_used,
            best_min_dist,
        )
        return best_codebook, best_stats

    def fit(self, X, y, **fit_params) -> ManyClassClassifier:
        """Prepare classifier using custom validate_data.
        Actual fitting of sub-estimators happens in predict_proba if mapping is needed.
        """
        self._set_verbosity()
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
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_features_in_ = X.shape[1]

        random_state_instance = check_random_state(self.random_state)
        self.classes_ = unique_labels(y)  # Use unique_labels as imported
        n_classes = len(self.classes_)

        alphabet_size = self._get_alphabet_size()
        self.alphabet_size_ = alphabet_size
        self.no_mapping_needed_ = n_classes <= alphabet_size
        self.codebook_stats_ = {}
        self.estimators_ = None
        self.code_book_ = None
        self.classes_index_ = None
        self.X_train = None
        self.Y_train_per_estimator = None
        self.fit_params_ = fit_params.copy() if fit_params else {}

        if n_classes == 0:
            raise ValueError("Cannot fit with no classes present.")
        if n_classes == 1:
            # Gracefully handle single-class case: fit estimator, set trivial codebook
            if self.verbose > 0:
                pass
            cloned_estimator = clone(self.estimator)
            _apply_categorical_features_to_estimator(
                cloned_estimator, getattr(self, "categorical_features", None)
            )
            cloned_estimator.fit(X, y, **self.fit_params_)
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
            _apply_categorical_features_to_estimator(
                cloned_estimator, getattr(self, "categorical_features", None)
            )
            # Base estimator fits on X_validated (already processed by custom validate_data)
            cloned_estimator.fit(X, y, **self.fit_params_)
            self.estimators_ = [cloned_estimator]
            # Ensure n_features_in_ matches the fitted estimator if it has the attribute
            if hasattr(cloned_estimator, "n_features_in_"):
                self.n_features_in_ = cloned_estimator.n_features_in_

        else:  # Mapping is needed
            if self.verbose > 0:
                pass
            n_est = self._get_n_estimators(n_classes, alphabet_size)
            if self.codebook_strategy == "balanced_cluster":
                generator = self._generate_codebook_balanced_cluster
            elif self.codebook_strategy == "legacy_rest":
                generator = self._generate_codebook_legacy_rest
            else:
                raise ValueError(
                    "Unsupported codebook_strategy. Expected 'balanced_cluster' or 'legacy_rest'."
                )
            self.code_book_, self.codebook_stats_ = generator(
                n_classes, n_est, alphabet_size, random_state_instance
            )
            self._log_codebook_stats(self.codebook_stats_, tag="Codebook stats")
            self.classes_index_ = {c: i for i, c in enumerate(self.classes_)}
            self.X_train = X  # Store validated X
            y_indices = np.array([self.classes_index_[val] for val in y])
            self.Y_train_per_estimator = self.code_book_[:, y_indices]

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X. Sub-estimators are fitted here if mapping is used."""
        # Attributes to check if fitted, adapt from user's ["_tree", "X", "y"]
        # Key attributes for this classifier: classes_ must be set, n_features_in_ for X dim check.
        self._set_verbosity()
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

        iterator = range(self.code_book_.shape[0])
        iterable = tqdm.tqdm(iterator, disable=(self.verbose <= 1))
        Y_pred_probas_list = []
        categorical_features = getattr(self, "categorical_features", None)
        for i in iterable:
            Y_pred_probas_list.append(
                _fit_and_predict_proba(
                    self.estimator,
                    self.X_train,  # This is X_validated from fit
                    self.Y_train_per_estimator[i, :],
                    X,  # Pass validated X to predict on
                    categorical_features=categorical_features,
                    fit_params=self.fit_params_,
                    alphabet_size=self.alphabet_size_,
                )
            )
        Y_pred_probas_arr = np.stack(Y_pred_probas_list, axis=0)
        self._log_shapes(
            X_train=self.X_train,
            X=X,
            Y_train_per_estimator=self.Y_train_per_estimator,
            proba_arr=Y_pred_probas_arr,
        )

        _n_estimators, n_samples, current_alphabet_size = Y_pred_probas_arr.shape
        if n_samples == 0:
            return np.zeros((0, len(self.classes_)))

        n_orig_classes = len(self.classes_)
        codebook_stats = getattr(self, "codebook_stats_", {}) or {}
        has_rest_symbol = bool(codebook_stats.get("has_rest_symbol", False))
        rest_class_code = codebook_stats.get("rest_class_code")
        if has_rest_symbol and rest_class_code is None:
            rest_class_code = current_alphabet_size - 1
        use_log_agg = bool(self.log_proba_aggregation)
        if self.codebook_strategy == "balanced_cluster":
            if not use_log_agg and self.verbose > 0:
                warnings.warn(
                    "Using log-likelihood decoding for 'balanced_cluster' (more accurate).",
                    UserWarning,
                    stacklevel=2,
                )
            use_log_agg = True
        elif not has_rest_symbol:
            if not use_log_agg and self.verbose > 0:
                warnings.warn(
                    "Using log-likelihood decoding for codebooks without a rest symbol (more accurate).",
                    UserWarning,
                    stacklevel=2,
                )
            use_log_agg = True

        gather_idx = self.code_book_[:, None, :]
        gathered = np.take_along_axis(Y_pred_probas_arr, gather_idx, axis=2)

        if use_log_agg:
            gathered = np.log(np.clip(gathered, 1e-12, 1.0))
            aggregated = gathered.sum(axis=0)
            aggregated -= aggregated.max(axis=1, keepdims=True)
            exp_scores = np.exp(aggregated)
            denom = np.clip(exp_scores.sum(axis=1, keepdims=True), 1.0, None)
            probas = exp_scores / denom
            zero_mask = denom.squeeze() == 0
            if np.any(zero_mask):
                probas[zero_mask] = 1.0 / n_orig_classes
            return probas

        mask = (
            self.code_book_ != rest_class_code
            if has_rest_symbol
            else np.ones_like(self.code_book_, dtype=bool)
        )
        weighted = gathered * mask[:, None, :]
        aggregated = weighted.sum(axis=0)
        counts = mask.sum(axis=0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            averages = aggregated / np.where(counts == 0, 1.0, counts)[None, :]
        averages[:, counts == 0] = 0.0
        if has_rest_symbol and not np.all(counts > 0) and self.verbose > 0:
            warnings.warn(
                "Some classes had zero specific code assignments during aggregation.",
                RuntimeWarning,
                stacklevel=2,
            )
        row_sum = averages.sum(axis=1, keepdims=True)
        denom = np.clip(row_sum, 1.0, None)
        averages /= denom
        zero_mask = row_sum.squeeze() == 0
        if np.any(zero_mask):
            averages[zero_mask] = 1.0 / n_orig_classes

        return averages

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
        _apply_categorical_features_to_estimator(self.estimator, categorical_features)
        if (
            not hasattr(self.estimator, "set_categorical_features")
            and not hasattr(self.estimator, "categorical_features")
            and self.verbose > 0
        ):
            warnings.warn(
                "Base estimator has no known categorical feature support.",
                UserWarning,
                stacklevel=2,
            )

    def _more_tags(self) -> dict[str, Any]:
        return {"allow_nan": True}

    @property
    def codebook_statistics_(self):
        """Returns statistics about the generated codebook."""
        check_is_fitted(self, ["classes_"])  # Minimal check
        if self.no_mapping_needed_:
            return {"message": "No codebook mapping was needed."}
        return dict(getattr(self, "codebook_stats_", {}))
