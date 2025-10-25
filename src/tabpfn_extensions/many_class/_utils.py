from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, clone

EPS_PROBA = 1e-12
EPS_WEIGHT = 1e-6


def _dataclass_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if sys.version_info >= (3, 10):  # pragma: no cover - environment dependent
        kwargs["slots"] = True
    return kwargs


DATACLASS_KWARGS = _dataclass_kwargs()

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from ._strategies import RowWeighter


def apply_categorical_features(
    estimator: BaseEstimator, categorical_features: list[int] | None
) -> None:
    """Apply stored categorical feature metadata to an estimator clone."""
    if categorical_features is None:
        return
    if hasattr(estimator, "set_categorical_features"):
        estimator.set_categorical_features(categorical_features)
    elif hasattr(estimator, "categorical_features"):
        estimator.categorical_features = categorical_features


def as_numpy(X: Any) -> np.ndarray:
    """Best-effort conversion of array-like data to a NumPy ndarray."""
    if isinstance(X, np.ndarray):
        return X
    try:
        return X.to_numpy()
    except AttributeError:
        if hasattr(X, "toarray"):
            return np.asarray(X.toarray())
        return np.asarray(X)


def align_probabilities(
    proba: np.ndarray, classes_seen: Iterable[int], alphabet_size: int
) -> np.ndarray:
    """Expand predicted probabilities to a dense alphabet-sized array."""
    full = np.zeros((proba.shape[0], alphabet_size), dtype=np.float64)
    indices = np.asarray(list(classes_seen), dtype=int)
    full[:, indices] = proba
    return full


def filter_fit_params_for_mask(
    fit_params: Mapping[str, Any] | None,
    mask: np.ndarray | None,
    *,
    n_samples: int,
) -> dict[str, Any]:
    """Return fit parameters filtered by a boolean mask when available."""
    if not fit_params:
        return {}

    if mask is None:
        return dict(fit_params)

    filtered: dict[str, Any] = {}
    for key, value in fit_params.items():
        if hasattr(value, "iloc"):
            try:
                filtered[key] = value.iloc[mask]
                continue
            except (TypeError, ValueError, IndexError):
                pass
        try:
            array_value = np.asarray(value)
        except (TypeError, ValueError):
            filtered[key] = value
            continue
        if array_value.ndim == 0 or array_value.shape[0] != n_samples:
            filtered[key] = value
            continue
        filtered_slice = array_value[mask]
        if isinstance(value, list):
            filtered[key] = filtered_slice.tolist()
        elif isinstance(value, tuple):
            filtered[key] = tuple(filtered_slice.tolist())
        else:
            filtered[key] = filtered_slice
    return filtered


def pairwise_hamming(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances (counts) between rows."""
    if matrix.shape[0] <= 1:
        return np.array([], dtype=float)
    return pdist(matrix, metric="hamming") * matrix.shape[1]


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to the number of rows with numeric safeguards."""
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), np.maximum(weights, EPS_WEIGHT), 1.0)
    total = weights.sum()
    if total <= EPS_WEIGHT:
        return np.full_like(weights, 1.0)
    return weights * (len(weights) / total)


def summarize_codebook(
    codebook: np.ndarray,
    *,
    strategy: str,
    alphabet_size: int,
    has_rest_symbol: bool,
    rest_class_code: int | None,
    hamming_max_classes: int,
) -> dict[str, Any]:
    """Summarize coverage and separability statistics for a codebook."""
    n_estimators, n_classes = codebook.shape
    if has_rest_symbol and rest_class_code is not None:
        non_rest_mask = codebook != rest_class_code
    else:
        non_rest_mask = np.ones_like(codebook, dtype=bool)

    coverage_counts = non_rest_mask.sum(axis=0)
    stats: dict[str, Any] = {
        "strategy": strategy,
        "n_estimators": int(n_estimators),
        "n_classes": int(n_classes),
        "alphabet_size": int(alphabet_size),
        "has_rest_symbol": bool(has_rest_symbol),
        "rest_class_code": int(rest_class_code)
        if rest_class_code is not None
        else None,
        "coverage_min": int(np.min(coverage_counts)) if coverage_counts.size else 0,
        "coverage_max": int(np.max(coverage_counts)) if coverage_counts.size else 0,
        "coverage_mean": float(np.mean(coverage_counts))
        if coverage_counts.size
        else 0.0,
        "coverage_std": float(np.std(coverage_counts)) if coverage_counts.size else 0.0,
    }

    if n_classes <= hamming_max_classes:
        distances = pairwise_hamming(codebook.T)
        if distances.size:
            stats["min_pairwise_hamming_dist"] = int(np.rint(np.min(distances)))
            stats["avg_pairwise_hamming_dist"] = float(np.mean(distances))
        else:
            stats["min_pairwise_hamming_dist"] = None
            stats["avg_pairwise_hamming_dist"] = None
    else:
        stats["min_pairwise_hamming_dist"] = None
        stats["avg_pairwise_hamming_dist"] = None

    return stats


# Python 3.9 compatibility: dataclass(slots=...) is supported from 3.10 onwards.


@dataclass(**DATACLASS_KWARGS)
class RowRunResult:
    proba_test: np.ndarray
    proba_train: np.ndarray
    weight: float
    support: int
    entropy: float | None
    accuracy: float | None


def run_row(
    estimator: BaseEstimator,
    X_train: Any,
    y_train_codes: np.ndarray,
    X_test: Any,
    *,
    alphabet_size: int,
    categorical_features: list[int] | None,
    mask: np.ndarray | None,
    fit_params: Mapping[str, Any] | None,
    row_weighter: RowWeighter,
) -> RowRunResult:
    """Run a single ECOC row fit/predict cycle and return diagnostics."""
    if mask is not None:
        X_train_row = X_train[mask]
        y_train_row = y_train_codes[mask]
    else:
        X_train_row = X_train
        y_train_row = y_train_codes

    support = len(y_train_row)
    if support == 0:
        uniform = np.full(
            (as_numpy(X_test).shape[0], alphabet_size), 1.0 / alphabet_size
        )
        return RowRunResult(
            proba_test=uniform,
            proba_train=np.empty((0, alphabet_size)),
            weight=EPS_WEIGHT,
            support=0,
            entropy=None,
            accuracy=None,
        )

    filtered_params = filter_fit_params_for_mask(
        fit_params, mask, n_samples=len(X_train)
    )
    cloned_estimator = clone(estimator)
    apply_categorical_features(cloned_estimator, categorical_features)
    cloned_estimator.fit(X_train_row, y_train_row, **filtered_params)

    if not hasattr(cloned_estimator, "predict_proba"):
        raise AttributeError("Base estimator must implement the predict_proba method.")

    X_train_np = as_numpy(X_train_row)
    X_test_np = as_numpy(X_test)
    proba_both = cloned_estimator.predict_proba(
        np.concatenate([X_train_np, X_test_np], axis=0)
    )
    classes_seen = getattr(cloned_estimator, "classes_", None)
    if classes_seen is None:
        raise AttributeError(
            "Base estimator must expose `classes_` after fitting to align probabilities."
        )

    aligned = align_probabilities(proba_both, classes_seen, alphabet_size)
    n_train = X_train_np.shape[0]
    proba_train = aligned[:n_train]
    proba_test = aligned[n_train:]

    weight, diagnostics = row_weighter.weight(proba_train, y_train_row, alphabet_size)
    entropy = diagnostics.get("entropy") if diagnostics else None
    accuracy = diagnostics.get("accuracy") if diagnostics else None

    return RowRunResult(
        proba_test=proba_test,
        proba_train=proba_train,
        weight=float(weight),
        support=support,
        entropy=entropy,
        accuracy=accuracy,
    )
