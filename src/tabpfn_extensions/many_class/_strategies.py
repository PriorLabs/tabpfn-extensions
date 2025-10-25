
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
from numpy.random import RandomState

from ._utils import (
    EPS_LOG,
    EPS_WEIGHT,
    CodebookQuality,
    normalize_weights,
    summarize_codebook,
)

logger = logging.getLogger(__name__)


class CodebookStrategy(Protocol):
    """Protocol for codebook generation strategies."""

    def generate(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: RandomState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...


class RowWeighter(Protocol):
    """Protocol for weighting ECOC rows."""

    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> RowWeightResult:
        ...


class Aggregator(Protocol):
    """Protocol for aggregating row predictions into final probabilities."""

    def aggregate(
        self,
        probabilities: np.ndarray,
        codebook: np.ndarray,
        row_weights: np.ndarray,
        *,
        rest_mask: np.ndarray | None,
        force_log_likelihood: bool,
    ) -> np.ndarray:
        ...


@dataclass
class CodebookConfig:
    strategy: str = "legacy_rest"
    retries: int = 50
    min_hamming_frac: float | None = None
    selection: str = "max_min_hamming"
    hamming_max_classes: int = 200
    legacy_filter_rest_train: bool = False


class WeightMode(Enum):
    NONE = "none"
    TRAIN_ENTROPY = "train_entropy"
    TRAIN_ACC = "train_acc"


@dataclass
class RowWeightingConfig:
    mode: WeightMode = WeightMode.NONE
    gamma: float = 1.0


@dataclass
class AggregationConfig:
    log_likelihood: bool = True
    legacy_mask_rest_log_agg: bool = True


@dataclass
class RowWeightResult:
    weight: float
    entropy: float | None
    accuracy: float | None


class NoOpRowWeighter:
    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> RowWeightResult:
        return RowWeightResult(weight=1.0, entropy=np.nan, accuracy=np.nan)


class TrainEntropyRowWeighter:
    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> RowWeightResult:
        if P_train.size == 0:
            return RowWeightResult(weight=EPS_WEIGHT, entropy=np.nan, accuracy=np.nan)
        P = np.clip(P_train, EPS_LOG, 1.0)
        entropy = float(-np.mean(np.sum(P * np.log(P), axis=1)))
        q = max(P.shape[1], 2)
        norm = max(math.log(q), EPS_LOG)
        ratio = min(1.0, entropy / norm)
        weight = max(EPS_WEIGHT, 1.0 - ratio)
        return RowWeightResult(weight=weight, entropy=entropy, accuracy=np.nan)


class TrainAccuracyRowWeighter:
    def weight(
        self, P_train: np.ndarray, y_train: np.ndarray, alphabet_size: int
    ) -> RowWeightResult:
        if P_train.size == 0:
            return RowWeightResult(weight=EPS_WEIGHT, entropy=np.nan, accuracy=np.nan)
        preds = np.argmax(P_train, axis=1)
        accuracy = float(np.mean(preds == y_train)) if P_train.shape[0] else 0.0
        counts = np.bincount(y_train, minlength=P_train.shape[1]).astype(float)
        total = counts.sum()
        if total > 0:
            chance = float(counts.max() / total)
        else:
            chance = 1.0 / max(P_train.shape[1], 1)
        weight = max(EPS_WEIGHT, accuracy - chance)
        return RowWeightResult(weight=weight, entropy=np.nan, accuracy=accuracy)


class DefaultAggregator:
    def __init__(self, config: AggregationConfig) -> None:
        self.config = config

    def aggregate(
        self,
        probabilities: np.ndarray,
        codebook: np.ndarray,
        row_weights: np.ndarray,
        *,
        rest_mask: np.ndarray | None,
        force_log_likelihood: bool,
    ) -> np.ndarray:
        weights = normalize_weights(row_weights)
        gather_idx = codebook[:, None, :]
        gathered = np.take_along_axis(probabilities, gather_idx, axis=2)
        use_log = force_log_likelihood or self.config.log_likelihood

        if use_log:
            log_values = np.log(np.clip(gathered, EPS_LOG, 1.0))
            if rest_mask is not None and self.config.legacy_mask_rest_log_agg:
                log_values = log_values * rest_mask[:, None, :]
            log_values = log_values * weights[:, None, None]
            aggregated = log_values.sum(axis=0)
            aggregated -= aggregated.max(axis=1, keepdims=True)
            exp_scores = np.exp(aggregated)
            denom = exp_scores.sum(axis=1, keepdims=True)
            probas = exp_scores / denom
            return probas

        mask = np.ones_like(codebook, dtype=float)
        if rest_mask is not None:
            mask = rest_mask.astype(float)
        weighted = gathered * weights[:, None, None] * mask[:, None, :]
        aggregated = weighted.sum(axis=0)
        counts = (mask * weights[:, None]).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            averages = aggregated / np.where(counts == 0, 1.0, counts)[None, :]
        averages[:, counts == 0] = 0.0
        row_sum = averages.sum(axis=1, keepdims=True)
        denom = np.where(row_sum == 0, 1.0, row_sum)
        averages /= denom
        zero_mask = row_sum.squeeze() == 0
        if np.any(zero_mask):
            averages[zero_mask] = 1.0 / codebook.shape[1]
        return averages


class LegacyRestCodebookStrategy:
    def __init__(self, config: CodebookConfig) -> None:
        self.config = config

    def generate(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: RandomState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if n_classes <= alphabet_size:
            raise ValueError("Legacy codebook requires n_classes > alphabet_size.")
        rest_code = alphabet_size - 1
        codes = list(range(alphabet_size - 1))
        if not codes:
            raise ValueError("alphabet_size must be at least 2 for codebook generation.")
        target = None
        if self.config.min_hamming_frac:
            target = math.ceil(self.config.min_hamming_frac * n_estimators)
        best_stats: dict[str, Any] | None = None
        best_codebook: np.ndarray | None = None
        best_quality: CodebookQuality | None = None
        attempts_used = 0
        for attempt in range(1, self.config.retries + 1):
            attempt_seed = int(rng.randint(0, np.iinfo(np.int32).max))
            local_rng = RandomState(attempt_seed)
            codebook = np.full((n_estimators, n_classes), rest_code, dtype=int)
            coverage = np.zeros(n_classes, dtype=int)
            for row in range(n_estimators):
                n_assignable = min(len(codes), n_classes)
                noisy = coverage + local_rng.uniform(0, 0.1, size=n_classes)
                selected = np.argsort(noisy)[:n_assignable]
                row_codes = local_rng.permutation(codes)[:n_assignable]
                codebook[row, selected] = row_codes
                coverage[selected] += 1
            if np.any(coverage == 0):
                logger.warning(
                    "Legacy codebook attempt %s failed to cover all classes.", attempt
                )
                continue
            stats, quality = summarize_codebook(
                codebook=codebook,
                coverage_count=coverage,
                alphabet_size=alphabet_size,
                strategy="legacy_rest",
                has_rest_symbol=True,
                rest_class_code=rest_code,
                attempt_count=attempt,
                max_classes=self.config.hamming_max_classes,
                selection=self.config.selection,
                attempt_seed=attempt_seed,
            )
            attempts_used = attempt
            logger.info(
                "Codebook attempt %d/%d (legacy_rest): min_hamming=%s mean_hamming=%s",
                attempt,
                self.config.retries,
                quality.min_distance,
                quality.mean_distance,
            )
            if best_quality is None or quality.as_key() > best_quality.as_key():
                best_quality = quality
                best_codebook = codebook
                best_stats = stats
            if (
                target is not None
                and quality.min_distance is not None
                and quality.min_distance >= target
            ):
                logger.info(
                    "Early exit after %d attempts: achieved min Hamming %s (target %s)",
                    attempt,
                    quality.min_distance,
                    target,
                )
                break
        if best_codebook is None or best_stats is None or best_quality is None:
            raise RuntimeError("Failed to generate a valid legacy codebook.")
        best_stats = best_stats.copy()
        best_stats["regeneration_attempts"] = attempts_used
        best_stats["best_min_pairwise_hamming_dist"] = best_quality.min_distance
        best_stats["mean_pairwise_hamming_dist"] = best_quality.mean_distance
        return best_codebook, best_stats


class BalancedClusterCodebookStrategy:
    def __init__(self, config: CodebookConfig) -> None:
        self.config = config

    def generate(
        self,
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        rng: RandomState,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if n_classes <= alphabet_size:
            raise ValueError("Balanced codebook requires n_classes > alphabet_size.")
        target = None
        if self.config.min_hamming_frac:
            target = math.ceil(self.config.min_hamming_frac * n_estimators)
        best_stats: dict[str, Any] | None = None
        best_codebook: np.ndarray | None = None
        best_quality: CodebookQuality | None = None
        attempts_used = 0
        class_indices = np.arange(n_classes)
        for attempt in range(1, self.config.retries + 1):
            attempt_seed = int(rng.randint(0, np.iinfo(np.int32).max))
            local_rng = RandomState(attempt_seed)
            codebook = np.zeros((n_estimators, n_classes), dtype=int)
            for row in range(n_estimators):
                local_rng.shuffle(class_indices)
                groups = np.array_split(class_indices, alphabet_size)
                for code, group in enumerate(groups):
                    codebook[row, group] = code
            coverage = np.full(n_classes, n_estimators, dtype=int)
            stats, quality = summarize_codebook(
                codebook=codebook,
                coverage_count=coverage,
                alphabet_size=alphabet_size,
                strategy="balanced_cluster",
                has_rest_symbol=False,
                rest_class_code=None,
                attempt_count=attempt,
                max_classes=self.config.hamming_max_classes,
                selection=self.config.selection,
                attempt_seed=attempt_seed,
            )
            attempts_used = attempt
            logger.info(
                "Codebook attempt %d/%d (balanced_cluster): min_hamming=%s mean_hamming=%s",
                attempt,
                self.config.retries,
                quality.min_distance,
                quality.mean_distance,
            )
            if best_quality is None or quality.as_key() > best_quality.as_key():
                best_quality = quality
                best_codebook = codebook
                best_stats = stats
            if (
                target is not None
                and quality.min_distance is not None
                and quality.min_distance >= target
            ):
                logger.info(
                    "Early exit after %d attempts: achieved min Hamming %s (target %s)",
                    attempt,
                    quality.min_distance,
                    target,
                )
                break
        if best_codebook is None or best_stats is None or best_quality is None:
            raise RuntimeError("Failed to generate a balanced codebook.")
        best_stats = best_stats.copy()
        best_stats["regeneration_attempts"] = attempts_used
        best_stats["best_min_pairwise_hamming_dist"] = best_quality.min_distance
        best_stats["mean_pairwise_hamming_dist"] = best_quality.mean_distance
        return best_codebook, best_stats


def build_codebook_strategy(config: CodebookConfig) -> CodebookStrategy:
    strategy = config.strategy
    if strategy == "legacy_rest":
        return LegacyRestCodebookStrategy(config)
    if strategy == "balanced_cluster":
        return BalancedClusterCodebookStrategy(config)
    raise ValueError(f"Unsupported codebook strategy: {strategy}")


def build_row_weighter(config: RowWeightingConfig) -> RowWeighter:
    mode = config.mode
    if mode == WeightMode.NONE:
        return NoOpRowWeighter()
    if mode == WeightMode.TRAIN_ENTROPY:
        return TrainEntropyRowWeighter()
    if mode == WeightMode.TRAIN_ACC:
        return TrainAccuracyRowWeighter()
    raise ValueError(f"Unsupported row weighting mode: {mode}")


def build_aggregator(config: AggregationConfig) -> Aggregator:
    return DefaultAggregator(config)
