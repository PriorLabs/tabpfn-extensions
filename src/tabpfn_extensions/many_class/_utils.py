from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

EPS_LOG = 1e-12
EPS_WEIGHT = 1e-6


def apply_categorical_features_to_estimator(
    estimator: BaseEstimator, categorical_features: list[int] | None
) -> None:
    """Apply stored categorical feature metadata to a cloned estimator."""
    if categorical_features is None:
        return
    if hasattr(estimator, "set_categorical_features"):
        estimator.set_categorical_features(categorical_features)
    elif hasattr(estimator, "categorical_features"):
        estimator.categorical_features = categorical_features


def as_numpy(X: Any) -> np.ndarray:
    """Convert array-like input to a NumPy array without unnecessary copies."""
    if isinstance(X, np.ndarray):
        return X
    try:
        return X.to_numpy()
    except AttributeError:
        if hasattr(X, "toarray"):
            return np.asarray(X.toarray())
        return np.asarray(X)


def align_probabilities(
    probabilities: np.ndarray, classes_seen: Iterable[int], alphabet_size: int
) -> np.ndarray:
    """Expand sub-estimator probabilities to the full alphabet size."""
    aligned = np.zeros((probabilities.shape[0], alphabet_size), dtype=np.float64)
    indices = np.asarray(list(classes_seen), dtype=int)
    aligned[:, indices] = probabilities
    return aligned


def filter_fit_params_for_mask(
    fit_params: dict[str, Any] | None,
    mask: np.ndarray | None,
    *,
    n_samples: int,
) -> dict[str, Any]:
    """Filter fit parameters according to a boolean mask when provided."""
    if not fit_params or mask is None:
        return {} if not fit_params else fit_params.copy()

    filtered: dict[str, Any] = {}
    for key, value in fit_params.items():
        if hasattr(value, "iloc"):
            try:
                filtered[key] = value.iloc[mask]
                continue
            except (TypeError, ValueError, IndexError):
                pass
        candidate = None
        if isinstance(value, (list, tuple)):
            candidate = np.asarray(value)
        else:
            try:
                candidate = np.asarray(value)
            except (TypeError, ValueError):
                candidate = None
        if candidate is not None and candidate.ndim > 0 and candidate.shape[0] == n_samples:
            masked = candidate[mask]
            if isinstance(value, list):
                filtered[key] = masked.tolist()
            elif isinstance(value, tuple):
                filtered[key] = tuple(masked.tolist())
            else:
                filtered[key] = masked
            continue
        filtered[key] = value
    return filtered


def normalize_weights(weights: np.ndarray, eps: float = EPS_WEIGHT) -> np.ndarray:
    """Normalize weights so their sum equals the number of rows."""
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    weights = np.clip(weights, eps, None)
    total = weights.sum()
    if total <= eps:
        weights = np.ones_like(weights)
        total = weights.sum()
    return weights * (len(weights) / max(total, eps))


def compute_pairwise_hamming(
    codebook: np.ndarray, *, max_classes: int
) -> tuple[int | None, float | None]:
    """Compute min/mean pairwise Hamming distance across class codewords."""
    n_estimators, n_classes = codebook.shape
    if not (1 < n_classes < max_classes):
        return None, None
    distances = pdist(codebook.T, metric="hamming") * n_estimators
    if distances.size == 0:
        return None, None
    return int(np.rint(np.min(distances))), float(np.mean(distances))


@dataclass(frozen=True)
class CodebookQuality:
    min_distance: int | None
    mean_distance: float | None
    coverage_std: float | None
    attempt_seed: int

    def as_key(self) -> tuple[float, float, float, int]:
        min_metric = float("-inf") if self.min_distance is None else float(self.min_distance)
        mean_metric = float("-inf") if self.mean_distance is None else float(self.mean_distance)
        coverage_metric = (
            float("-inf") if self.coverage_std is None else -float(self.coverage_std)
        )
        return (min_metric, mean_metric, coverage_metric, self.attempt_seed)


def summarize_codebook(
    *,
    codebook: np.ndarray,
    coverage_count: np.ndarray,
    alphabet_size: int,
    strategy: str,
    has_rest_symbol: bool,
    rest_class_code: int | None,
    attempt_count: int,
    max_classes: int,
    selection: str,
    attempt_seed: int,
) -> tuple[dict[str, Any], CodebookQuality]:
    """Summarise codebook statistics and produce a comparable quality tuple."""
    n_estimators, n_classes = codebook.shape
    coverage_min = int(np.min(coverage_count)) if coverage_count.size else 0
    coverage_max = int(np.max(coverage_count)) if coverage_count.size else 0
    coverage_mean = float(np.mean(coverage_count)) if coverage_count.size else 0.0
    coverage_std = float(np.std(coverage_count)) if coverage_count.size else 0.0

    min_dist, mean_dist = compute_pairwise_hamming(codebook, max_classes=max_classes)

    stats: dict[str, Any] = {
        "coverage_min": coverage_min,
        "coverage_max": coverage_max,
        "coverage_mean": coverage_mean,
        "coverage_std": coverage_std,
        "n_estimators": n_estimators,
        "n_classes": n_classes,
        "alphabet_size": alphabet_size,
        "strategy": strategy,
        "has_rest_symbol": has_rest_symbol,
        "rest_class_code": rest_class_code,
        "min_pairwise_hamming_dist": min_dist,
        "mean_pairwise_hamming_dist": mean_dist,
        "codebook_selection": selection,
        "regeneration_attempts": attempt_count,
    }
    quality = CodebookQuality(
        min_distance=min_dist,
        mean_distance=mean_dist,
        coverage_std=coverage_std,
        attempt_seed=attempt_seed,
    )
    return stats, quality


def make_row_usage_mask(
    codebook: np.ndarray, *, rest_class_code: int | None
) -> np.ndarray:
    """Return a boolean mask indicating which rows cover each class explicitly."""
    if rest_class_code is None:
        return np.ones_like(codebook, dtype=bool)
    return codebook != rest_class_code
