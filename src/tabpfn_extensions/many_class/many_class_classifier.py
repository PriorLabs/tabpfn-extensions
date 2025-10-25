
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.misc.sklearn_compat import validate_data

from ._strategies import (
    AggregationConfig,
    Aggregator,
    CodebookConfig,
    RowWeightingConfig,
    WeightMode,
    build_aggregator,
    build_codebook_strategy,
    build_row_weighter,
)
from ._utils import (
    align_probabilities,
    apply_categorical_features_to_estimator,
    as_numpy,
    filter_fit_params_for_mask,
    make_row_usage_mask,
    normalize_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class RowRunResult:
    test_probabilities: np.ndarray
    train_probabilities: np.ndarray
    weight: float
    entropy: float | None
    accuracy: float | None
    support: int


@set_extension("many_class")
class ManyClassClassifier(ClassifierMixin, BaseEstimator):
    """Output-code multiclass strategy that extends base estimators beyond their class limit.

    Parameters mirror the original wrapper while grouping advanced options into
    lightweight configuration objects. Most tuning knobs now live in
    :class:`CodebookConfig`, :class:`RowWeightingConfig`, and
    :class:`AggregationConfig`.
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
        log_proba_aggregation: bool = True,
        codebook_config: CodebookConfig | str | None = None,
        row_weighting_config: RowWeightingConfig | WeightMode | str | None = None,
        aggregation_config: AggregationConfig | None = None,
    ) -> None:
        self.estimator = estimator
        self.alphabet_size = alphabet_size
        self.n_estimators = n_estimators
        self.n_estimators_redundancy = n_estimators_redundancy
        self.random_state = random_state
        self.verbose = verbose

        self.codebook_config = self._resolve_codebook_config(codebook_config)
        self.row_weighting_config = self._resolve_row_weighting_config(
            row_weighting_config
        )
        self.aggregation_config = self._resolve_aggregation_config(
            aggregation_config, log_proba_aggregation
        )
        self.log_proba_aggregation = self.aggregation_config.log_likelihood

        self._codebook_strategy = build_codebook_strategy(self.codebook_config)
        self._row_weighter = build_row_weighter(self.row_weighting_config)
        self._aggregator: Aggregator = build_aggregator(self.aggregation_config)

        self.fit_params_: dict[str, Any] | None = None
        self.code_book_: np.ndarray | None = None
        self.codebook_stats_: dict[str, Any] = {}
        self.estimators_: list[BaseEstimator] | None = None
        self.no_mapping_needed_: bool = False
        self.classes_: np.ndarray | None = None
        self.classes_index_: dict[Any, int] | None = None
        self.X_train: Any | None = None
        self.Y_train_per_estimator: np.ndarray | None = None
        self.alphabet_size_: int | None = None
        self._row_usage_mask: np.ndarray | None = None
        self.row_weights_: np.ndarray | None = None
        self.row_weights_raw_: np.ndarray | None = None
        self.row_train_support_: np.ndarray | None = None
        self.row_train_entropy_: np.ndarray | None = None
        self.row_train_acc_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _resolve_codebook_config(
        self, config: CodebookConfig | str | None
    ) -> CodebookConfig:
        if config is None:
            config = CodebookConfig()
        elif isinstance(config, str):
            config = CodebookConfig(strategy=config)
        elif not isinstance(config, CodebookConfig):
            raise TypeError("codebook_config must be a CodebookConfig or strategy name")
        if config.retries < 1:
            raise ValueError("codebook retries must be at least 1")
        if (
            config.min_hamming_frac is not None
            and not 0.0 <= config.min_hamming_frac <= 1.0
        ):
            raise ValueError("codebook_min_hamming_frac must lie in [0, 1]")
        if config.selection != "max_min_hamming":
            raise ValueError("Only 'max_min_hamming' selection is supported")
        if config.hamming_max_classes < 0:
            raise ValueError("codebook_hamming_max_classes must be non-negative")
        return CodebookConfig(
            strategy=config.strategy,
            retries=int(config.retries),
            min_hamming_frac=None
            if config.min_hamming_frac is None
            else float(config.min_hamming_frac),
            selection=config.selection,
            hamming_max_classes=int(config.hamming_max_classes),
            legacy_filter_rest_train=bool(config.legacy_filter_rest_train),
        )

    def _resolve_row_weighting_config(
        self, config: RowWeightingConfig | WeightMode | str | None
    ) -> RowWeightingConfig:
        if config is None:
            config = RowWeightingConfig()
        elif isinstance(config, RowWeightingConfig):
            config = RowWeightingConfig(mode=config.mode, gamma=config.gamma)
        elif isinstance(config, WeightMode):
            config = RowWeightingConfig(mode=config, gamma=1.0)
        elif isinstance(config, str):
            try:
                mode = WeightMode(config)
            except ValueError as exc:
                raise ValueError(
                    "row_weighting_config must be WeightMode or one of 'none', 'train_entropy', 'train_acc'"
                ) from exc
            config = RowWeightingConfig(mode=mode, gamma=1.0)
        else:
            raise TypeError(
                "row_weighting_config must be a RowWeightingConfig, WeightMode, or string"
            )
        gamma = float(config.gamma)
        if gamma <= 0:
            raise ValueError("row_weighting_gamma must be positive")
        return RowWeightingConfig(mode=config.mode, gamma=gamma)

    def _resolve_aggregation_config(
        self,
        config: AggregationConfig | None,
        log_proba_aggregation: bool,
    ) -> AggregationConfig:
        if config is None:
            return AggregationConfig(
                log_likelihood=bool(log_proba_aggregation),
                legacy_mask_rest_log_agg=True,
            )
        if not isinstance(config, AggregationConfig):
            raise TypeError("aggregation_config must be an AggregationConfig instance")
        return AggregationConfig(
            log_likelihood=bool(config.log_likelihood),
            legacy_mask_rest_log_agg=bool(config.legacy_mask_rest_log_agg),
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _set_verbosity(self) -> None:
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
        X_train: Any | None,
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

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _get_alphabet_size(self) -> int:
        provided = self.alphabet_size if self.alphabet_size is not None else None
        if provided is not None:
            return provided
        try:
            return int(self.estimator.max_num_classes_)
        except AttributeError:
            warnings.warn(
                "Could not infer alphabet_size from estimator; defaulting to 10.",
                UserWarning,
                stacklevel=2,
            )
            return 10

    def _get_n_estimators(self, n_classes: int, alphabet_size: int) -> int:
        if self.n_estimators is not None:
            return self.n_estimators
        if n_classes <= alphabet_size:
            return 1
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

    def _run_row(
        self,
        X_train_row: Any,
        y_train_row: np.ndarray,
        X_test: Any,
        *,
        categorical_features: list[int] | None,
        fit_params: dict[str, Any],
    ) -> RowRunResult:
        estimator = clone(self.estimator)
        apply_categorical_features_to_estimator(estimator, categorical_features)
        estimator.fit(X_train_row, y_train_row, **fit_params)
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError("Base estimator must implement predict_proba.")
        X_train_np = as_numpy(X_train_row)
        X_test_np = as_numpy(X_test)
        combined = np.concatenate([X_train_np, X_test_np], axis=0)
        proba_both = estimator.predict_proba(combined)
        classes_seen = getattr(estimator, "classes_", None)
        if classes_seen is None:
            raise AttributeError(
                "Base estimator must expose `classes_` after fitting to align probabilities."
            )
        aligned = align_probabilities(proba_both, classes_seen, self.alphabet_size_)
        n_train = X_train_np.shape[0]
        train_probs = aligned[:n_train]
        test_probs = aligned[n_train:]
        weight_result = self._row_weighter.weight(
            train_probs, y_train_row, self.alphabet_size_
        )
        weight = float(weight_result.weight)
        if self.row_weighting_config.gamma != 1.0:
            weight = max(weight, 0.0) ** self.row_weighting_config.gamma
        return RowRunResult(
            test_probabilities=test_probs,
            train_probabilities=train_probs,
            weight=weight,
            entropy=weight_result.entropy,
            accuracy=weight_result.accuracy,
            support=int(y_train_row.shape[0]),
        )

    # ------------------------------------------------------------------
    # Estimator API
    # ------------------------------------------------------------------
    def fit(self, X, y, **fit_params) -> ManyClassClassifier:
        self._set_verbosity()
        X, y = validate_data(self, X, y, ensure_all_finite=False)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_features_in_ = X.shape[1]

        rng = check_random_state(self.random_state)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        alphabet_size = self._get_alphabet_size()
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be >= 2.")
        self.alphabet_size_ = alphabet_size

        self.no_mapping_needed_ = n_classes <= alphabet_size
        self.codebook_stats_ = {}
        self.code_book_ = None
        self.estimators_ = None
        self.classes_index_ = None
        self.X_train = None
        self.Y_train_per_estimator = None
        self.fit_params_ = fit_params.copy() if fit_params else {}
        self._row_usage_mask = None
        self.row_weights_ = None
        self.row_weights_raw_ = None
        self.row_train_support_ = None
        self.row_train_entropy_ = None
        self.row_train_acc_ = None

        if n_classes == 0:
            raise ValueError("Cannot fit with no classes present.")

        if n_classes == 1:
            estimator = clone(self.estimator)
            apply_categorical_features_to_estimator(
                estimator, getattr(self, "categorical_features", None)
            )
            estimator.fit(X, y, **self.fit_params_)
            self.estimators_ = [estimator]
            self.code_book_ = np.zeros((1, 1), dtype=int)
            self.codebook_stats_ = {
                "n_classes": 1,
                "n_estimators": 1,
                "alphabet_size": 1,
                "strategy": "trivial",
            }
            return self

        if self.no_mapping_needed_:
            estimator = clone(self.estimator)
            apply_categorical_features_to_estimator(
                estimator, getattr(self, "categorical_features", None)
            )
            estimator.fit(X, y, **self.fit_params_)
            self.estimators_ = [estimator]
            if hasattr(estimator, "n_features_in_"):
                self.n_features_in_ = estimator.n_features_in_
            return self

        n_estimators = self._get_n_estimators(n_classes, alphabet_size)
        codebook, stats = self._codebook_strategy.generate(
            n_classes, n_estimators, alphabet_size, rng
        )
        self.code_book_ = codebook
        self.codebook_stats_ = stats
        self._log_codebook_stats(stats, tag="Codebook stats")
        self._row_usage_mask = make_row_usage_mask(
            codebook, rest_class_code=stats.get("rest_class_code")
        )
        self.classes_index_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.X_train = X
        y_indices = np.array([self.classes_index_[val] for val in y])
        self.Y_train_per_estimator = codebook[:, y_indices]
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._set_verbosity()
        check_is_fitted(self, ["classes_", "n_features_in_"])
        X = validate_data(self, X, ensure_all_finite=False)

        if self.no_mapping_needed_:
            if not self.estimators_:
                raise RuntimeError("Estimator not fitted. Call fit first.")
            return self.estimators_[0].predict_proba(X)

        if (
            self.X_train is None
            or self.Y_train_per_estimator is None
            or self.code_book_ is None
        ):
            raise RuntimeError("Fit must be called before predict_proba when mapping.")

        codebook_stats = self.codebook_stats_ or {}
        has_rest_symbol = bool(codebook_stats.get("has_rest_symbol", False))
        rest_code = codebook_stats.get("rest_class_code")
        if has_rest_symbol and rest_code is None:
            rest_code = self.alphabet_size_ - 1

        iterator = range(self.code_book_.shape[0])
        iterable = tqdm.tqdm(iterator, disable=(self.verbose < 2))
        categorical_features = getattr(self, "categorical_features", None)

        row_results: list[RowRunResult] = []
        raw_weights: list[float] = []
        entropies: list[float] = []
        accuracies: list[float] = []
        supports: list[int] = []

        for row_idx in iterable:
            y_row = self.Y_train_per_estimator[row_idx]
            mask = None
            X_train_row = self.X_train
            y_train_row = y_row
            if (
                has_rest_symbol
                and rest_code is not None
                and self.codebook_config.legacy_filter_rest_train
            ):
                candidate_mask = y_row != rest_code
                if np.any(candidate_mask):
                    mask = candidate_mask
                    X_train_row = self.X_train[mask]
                    y_train_row = y_row[mask]
                else:
                    uniform = np.full(
                        (X.shape[0], self.alphabet_size_),
                        1.0 / self.alphabet_size_,
                        dtype=np.float64,
                    )
                    row_results.append(
                        RowRunResult(
                            test_probabilities=uniform,
                            train_probabilities=np.empty((0, self.alphabet_size_)),
                            weight=1e-6,
                            entropy=np.nan,
                            accuracy=np.nan,
                            support=0,
                        )
                    )
                    raw_weights.append(1e-6)
                    entropies.append(np.nan)
                    accuracies.append(np.nan)
                    supports.append(0)
                    continue

            if y_train_row.size == 0:
                uniform = np.full(
                    (X.shape[0], self.alphabet_size_),
                    1.0 / self.alphabet_size_,
                    dtype=np.float64,
                )
                row_results.append(
                    RowRunResult(
                        test_probabilities=uniform,
                        train_probabilities=np.empty((0, self.alphabet_size_)),
                        weight=1e-6,
                        entropy=np.nan,
                        accuracy=np.nan,
                        support=0,
                    )
                )
                raw_weights.append(1e-6)
                entropies.append(np.nan)
                accuracies.append(np.nan)
                supports.append(0)
                continue

            fit_kwargs = filter_fit_params_for_mask(
                self.fit_params_, mask, n_samples=self.X_train.shape[0]
            )
            result = self._run_row(
                X_train_row,
                y_train_row,
                X,
                categorical_features=categorical_features,
                fit_params=fit_kwargs,
            )
            row_results.append(result)
            raw_weights.append(result.weight)
            entropies.append(np.nan if result.entropy is None else result.entropy)
            accuracies.append(np.nan if result.accuracy is None else result.accuracy)
            supports.append(result.support)

        prob_array = np.stack(
            [result.test_probabilities for result in row_results], axis=0
        )
        raw_weights_arr = np.asarray(raw_weights, dtype=float)
        normalized_weights = normalize_weights(raw_weights_arr)
        self.row_weights_raw_ = raw_weights_arr
        self.row_weights_ = normalized_weights
        self.row_train_support_ = np.asarray(supports, dtype=int)
        self.row_train_entropy_ = np.asarray(entropies, dtype=float)
        self.row_train_acc_ = np.asarray(accuracies, dtype=float)

        rest_mask = self._row_usage_mask if has_rest_symbol else None
        force_log = not has_rest_symbol
        if force_log and not self.aggregation_config.log_likelihood and self.verbose > 0:
            warnings.warn(
                "Using log-likelihood decoding for codebooks without a rest symbol (more accurate).",
                UserWarning,
                stacklevel=2,
            )

        probabilities = self._aggregator.aggregate(
            prob_array,
            self.code_book_,
            raw_weights_arr,
            rest_mask=rest_mask,
            force_log_likelihood=force_log,
        )
        self._log_shapes(
            X_train=self.X_train,
            X=X,
            Y_train_per_estimator=self.Y_train_per_estimator,
            proba_arr=prob_array,
        )
        return probabilities

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["classes_", "n_features_in_"])
        if self.no_mapping_needed_ and self.estimators_:
            return self.estimators_[0].predict(X)
        probas = self.predict_proba(X)
        if probas.shape[0] == 0:
            return np.array([], dtype=self.classes_.dtype)
        return self.classes_[np.argmax(probas, axis=1)]

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------
    def set_categorical_features(self, categorical_features: list[int]) -> None:
        self.categorical_features = categorical_features
        apply_categorical_features_to_estimator(self.estimator, categorical_features)
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
    def codebook_statistics_(self) -> dict[str, Any]:
        check_is_fitted(self, ["classes_"])
        if self.no_mapping_needed_:
            return {"message": "No codebook mapping was needed."}
        return dict(self.codebook_stats_)
