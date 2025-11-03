from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

import numpy as np

from ._utils import DATACLASS_KWARGS, EPS_PROBA, EPS_WEIGHT, summarize_codebook


class CodebookStrategy(Protocol):
    def generate(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a codebook and associated statistics."""


class RowWeighter(Protocol):
    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> tuple[float, dict[str, float]]:
        """Return a weight for the row and optional diagnostics."""


class Aggregator(Protocol):
    def aggregate(
        self,
        P_rows: np.ndarray,
        codebook: np.ndarray,
        row_weights: np.ndarray,
        *,
        rest_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Aggregate row probabilities into final class probabilities."""


class WeightMode(str, Enum):
    NONE = "none"
    TRAIN_ENTROPY = "train_entropy"
    TRAIN_ACC = "train_acc"


@dataclass(**DATACLASS_KWARGS)
class CodebookConfig:
    strategy: Literal["legacy_rest", "balanced_cluster"] = (
        "legacy_rest"  # legacy_rest is recommended
    )
    retries: int = 50
    min_hamming_frac: float | None = None
    hamming_max_classes: int = 200
    legacy_filter_rest_train: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in {"legacy_rest", "balanced_cluster"}:
            raise ValueError(f"Unsupported codebook strategy: {self.strategy}")
        if self.retries < 1:
            raise ValueError("codebook retries must be >= 1")
        if (
            self.min_hamming_frac is not None
            and not 0.0 <= self.min_hamming_frac <= 1.0
        ):
            raise ValueError("codebook_min_hamming_frac must be in [0, 1].")
        if self.hamming_max_classes < 1:
            raise ValueError("codebook_hamming_max_classes must be >= 1")


@dataclass(**DATACLASS_KWARGS)
class RowWeightingConfig:
    mode: WeightMode | str = WeightMode.NONE
    gamma: float = 1.0

    def resolved_mode(self) -> WeightMode:
        if isinstance(self.mode, WeightMode):
            return self.mode
        try:
            return WeightMode(self.mode)
        except ValueError as exc:
            raise ValueError("Unsupported row weighting mode") from exc


@dataclass(**DATACLASS_KWARGS)
class AggregationConfig:
    log_likelihood: bool = True
    legacy_mask_rest_log_agg: bool = True


class LegacyRestCodebookStrategy:
    def __init__(self, config: CodebookConfig) -> None:
        self.config = config

    def generate(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rest_code = alphabet_size - 1
        best_codebook: np.ndarray | None = None
        best_stats: dict[str, Any] | None = None
        best_min = -np.inf
        best_mean = -np.inf
        attempts = max(1, int(self.config.retries))
        threshold = (
            math.ceil(self.config.min_hamming_frac * n_estimators)
            if self.config.min_hamming_frac is not None
            else None
        )

        for attempt in range(attempts):
            attempt_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            try:
                codebook = self._single_attempt(
                    n_classes, n_estimators, alphabet_size, attempt_rng
                )
            except RuntimeError:
                continue
            stats = summarize_codebook(
                codebook,
                strategy="legacy_rest",
                alphabet_size=alphabet_size,
                has_rest_symbol=True,
                rest_class_code=rest_code,
                hamming_max_classes=self.config.hamming_max_classes,
            )
            min_dist = stats.get("min_pairwise_hamming_dist")
            mean_dist = stats.get("avg_pairwise_hamming_dist")
            score_min = -np.inf if min_dist is None else float(min_dist)
            score_mean = -np.inf if mean_dist is None else float(mean_dist)
            if (
                best_codebook is None
                or score_min > best_min
                or (score_min == best_min and score_mean > best_mean)
            ):
                best_codebook = codebook
                best_stats = stats
                best_min = score_min
                best_mean = score_mean
            if threshold is not None and score_min >= threshold:
                break

        if best_codebook is None or best_stats is None:
            raise RuntimeError("Failed to generate a valid legacy codebook")

        best_stats = dict(best_stats)
        best_stats["regeneration_attempts"] = attempt + 1
        best_stats["best_min_pairwise_hamming_dist"] = (
            None if best_min == -np.inf else int(best_min)
        )
        best_stats.setdefault(
            "min_pairwise_hamming_dist", best_stats["best_min_pairwise_hamming_dist"]
        )
        return best_codebook, best_stats

    def _single_attempt(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2 for legacy strategy")
        codes = list(range(alphabet_size - 1))
        rest_code = alphabet_size - 1
        codebook = np.full((n_estimators, n_classes), rest_code, dtype=int)
        coverage = np.zeros(n_classes, dtype=int)

        for row in range(n_estimators):
            assignable = min(len(codes), n_classes)
            noisy = coverage + rng.uniform(0.0, 0.1, size=n_classes)
            chosen_classes = np.argsort(noisy)[:assignable]
            shuffled_codes = rng.permutation(codes)[:assignable]
            codebook[row, chosen_classes] = shuffled_codes
            coverage[chosen_classes] += 1

        if np.any(coverage == 0):
            raise RuntimeError("Failed to cover all classes in legacy codebook attempt")
        return codebook


class BalancedClusterCodebookStrategy:
    def __init__(self, config: CodebookConfig) -> None:
        self.config = config

    def generate(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        best_codebook: np.ndarray | None = None
        best_stats: dict[str, Any] | None = None
        best_min = -np.inf
        best_mean = -np.inf
        attempts = max(1, int(self.config.retries))
        threshold = (
            math.ceil(self.config.min_hamming_frac * n_estimators)
            if self.config.min_hamming_frac is not None
            else None
        )

        for attempt in range(attempts):
            attempt_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
            codebook = self._single_attempt(
                n_classes, n_estimators, alphabet_size, attempt_rng
            )
            stats = summarize_codebook(
                codebook,
                strategy="balanced_cluster",
                alphabet_size=alphabet_size,
                has_rest_symbol=False,
                rest_class_code=None,
                hamming_max_classes=self.config.hamming_max_classes,
            )
            min_dist = stats.get("min_pairwise_hamming_dist")
            mean_dist = stats.get("avg_pairwise_hamming_dist")
            score_min = -np.inf if min_dist is None else float(min_dist)
            score_mean = -np.inf if mean_dist is None else float(mean_dist)
            if (
                best_codebook is None
                or score_min > best_min
                or (score_min == best_min and score_mean > best_mean)
            ):
                best_codebook = codebook
                best_stats = stats
                best_min = score_min
                best_mean = score_mean
            if threshold is not None and score_min >= threshold:
                break

        if best_codebook is None or best_stats is None:
            raise RuntimeError("Failed to generate a valid balanced codebook")

        best_stats = dict(best_stats)
        best_stats["regeneration_attempts"] = attempt + 1
        best_stats["best_min_pairwise_hamming_dist"] = (
            None if best_min == -np.inf else int(best_min)
        )
        best_stats.setdefault(
            "min_pairwise_hamming_dist", best_stats["best_min_pairwise_hamming_dist"]
        )
        return best_codebook, best_stats

    def _single_attempt(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if n_classes <= alphabet_size:
            raise ValueError("Balanced clustering requires n_classes > alphabet_size")

        codebook = np.zeros((n_estimators, n_classes), dtype=int)
        class_indices = np.arange(n_classes)
        highlight_rows = min(n_classes, n_estimators)

        for row in range(highlight_rows):
            focus = row % n_classes
            others = np.delete(class_indices, focus)
            codebook[row, focus] = 0
            if alphabet_size > 1:
                rng.shuffle(others)
                groups = np.array_split(others, alphabet_size - 1)
                for offset, group in enumerate(groups, start=1):
                    codebook[row, group] = offset

        for row in range(highlight_rows, n_estimators):
            rng.shuffle(class_indices)
            groups = np.array_split(class_indices, alphabet_size)
            for code, group in enumerate(groups):
                codebook[row, group] = code

        return codebook


class NoopRowWeighter:
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> tuple[float, dict[str, float]]:
        return 1.0, {}


class TrainEntropyRowWeighter:
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> tuple[float, dict[str, float]]:
        if P_train.size == 0:
            return EPS_WEIGHT, {"entropy": math.nan}
        P = np.clip(P_train, EPS_PROBA, 1.0)
        entropy = float(-np.mean(np.sum(P * np.log(P), axis=1)))
        norm = max(math.log(max(alphabet_size, 2)), EPS_PROBA)
        ratio = entropy / norm
        weight = max(EPS_WEIGHT, 1.0 - ratio) ** self.gamma
        return weight, {"entropy": entropy}


class TrainAccuracyRowWeighter:
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> tuple[float, dict[str, float]]:
        if P_train.size == 0:
            return EPS_WEIGHT, {"accuracy": math.nan}
        P = np.clip(P_train, EPS_PROBA, 1.0)
        preds = np.argmax(P, axis=1)
        accuracy = float(np.mean(preds == y_train)) if y_train.size else 0.0
        counts = np.bincount(y_train, minlength=alphabet_size).astype(float)
        if counts.sum() == 0:
            chance = 1.0 / max(alphabet_size, 1)
        else:
            chance = float(counts.max() / counts.sum())
        weight = max(EPS_WEIGHT, accuracy - chance) ** self.gamma
        return weight, {"accuracy": accuracy}


class LogLikelihoodAggregator:
    def __init__(self, mask_rest: bool) -> None:
        self.mask_rest = mask_rest

    def aggregate(
        self,
        P_rows: np.ndarray,
        codebook: np.ndarray,
        row_weights: np.ndarray,
        *,
        rest_mask: np.ndarray | None,
    ) -> np.ndarray:
        gather_idx = codebook[:, None, :]
        gathered = np.take_along_axis(P_rows, gather_idx, axis=2)
        logs = np.log(np.clip(gathered, EPS_PROBA, 1.0))
        if rest_mask is not None and self.mask_rest:
            mask = rest_mask.astype(bool)
            weight_matrix = row_weights[:, None]
            masked_logs = np.where(mask[:, None, :], logs, 0.0)
            weighted_logs = masked_logs * weight_matrix[:, None, :]
            sum_logs = weighted_logs.sum(axis=0)
            effective_weights = (mask * weight_matrix).sum(axis=0)
            zero_effective = effective_weights <= EPS_WEIGHT
            safe_weights = np.where(zero_effective, 1.0, effective_weights)
            aggregated = np.divide(
                sum_logs,
                safe_weights[None, :],
                out=np.full_like(sum_logs, -np.inf),
                where=~zero_effective[None, :],
            )
        else:
            weighted_logs = logs * row_weights[:, None, None]
            total_weight = max(row_weights.sum(), EPS_WEIGHT)
            aggregated = weighted_logs.sum(axis=0) / total_weight
        aggregated -= aggregated.max(axis=1, keepdims=True)
        exp_scores = np.exp(aggregated)
        denom = np.clip(exp_scores.sum(axis=1, keepdims=True), 1.0, None)
        probas = exp_scores / denom
        zero_mask = denom.squeeze() == 0
        if np.any(zero_mask):
            probas[zero_mask] = 1.0 / probas.shape[1]
        return probas


class LegacyAverageAggregator:
    def __init__(self, mask_rest: bool) -> None:
        self.mask_rest = mask_rest

    def aggregate(
        self,
        P_rows: np.ndarray,
        codebook: np.ndarray,
        row_weights: np.ndarray,
        *,
        rest_mask: np.ndarray | None,
    ) -> np.ndarray:
        gather_idx = codebook[:, None, :]
        gathered = np.take_along_axis(P_rows, gather_idx, axis=2)
        if rest_mask is not None and self.mask_rest:
            gathered = gathered * rest_mask[:, None, :]
            counts = (rest_mask.astype(float) * row_weights[:, None]).sum(axis=0)
        else:
            counts = np.full(codebook.shape[1], row_weights.sum())
        contributions = gathered * row_weights[:, None, None]
        aggregated = contributions.sum(axis=0)
        counts = np.where(counts == 0, 1.0, counts)
        averages = aggregated / counts[None, :]
        row_sum = averages.sum(axis=1, keepdims=True)
        denom = np.clip(row_sum, 1.0, None)
        averages /= denom
        zero_mask = row_sum.squeeze() == 0
        if np.any(zero_mask):
            averages[zero_mask] = 1.0 / averages.shape[1]
        return averages


def make_codebook_strategy(config: CodebookConfig) -> CodebookStrategy:
    if config.strategy == "legacy_rest":
        return LegacyRestCodebookStrategy(config)
    return BalancedClusterCodebookStrategy(config)


def make_row_weighter(config: RowWeightingConfig) -> RowWeighter:
    mode = config.resolved_mode()
    if mode is WeightMode.TRAIN_ENTROPY:
        return TrainEntropyRowWeighter(config.gamma)
    if mode is WeightMode.TRAIN_ACC:
        return TrainAccuracyRowWeighter(config.gamma)
    return NoopRowWeighter(config.gamma)


def make_aggregator(
    use_log: bool,
    *,
    mask_rest: bool,
) -> Aggregator:
    if use_log:
        return LogLikelihoodAggregator(mask_rest=mask_rest)
    return LegacyAverageAggregator(mask_rest=mask_rest)
