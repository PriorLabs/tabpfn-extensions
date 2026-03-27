from __future__ import annotations

import logging
import math
import warnings
from typing import Any

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
    CodebookConfig,
    RowWeightingConfig,
    WeightMode,
    make_aggregator,
    make_codebook_strategy,
    make_row_weighter,
)
from ._utils import RowRunResult, normalize_weights, run_row

logger = logging.getLogger(__name__)


@set_extension("many_class")
class ManyClassClassifier(BaseEstimator, ClassifierMixin):
    """Output-coding wrapper that enables TabPFN-style estimators to handle many classes."""

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        alphabet_size: int | None = None,
        n_estimators: int | None = None,
        n_estimators_redundancy: int = 4,
        random_state: int | None = None,
        verbose: int = 0,
        codebook_config: CodebookConfig | str | None = None,
        row_weighting_config: RowWeightingConfig | WeightMode | str | None = None,
        aggregation_config: AggregationConfig | None = None,
        n_jobs: int = 1,
        cache_preprocessing: bool = True,
    ) -> None:
        self.estimator = estimator
        self.alphabet_size = alphabet_size
        self.n_jobs = n_jobs
        self.cache_preprocessing = cache_preprocessing
        self.n_estimators = n_estimators
        self.n_estimators_redundancy = n_estimators_redundancy
        self.random_state = random_state
        self.verbose = verbose

        self.codebook_config = codebook_config
        self.row_weighting_config = row_weighting_config
        self.aggregation_config = aggregation_config

        self._codebook_config: CodebookConfig | None = None
        self._row_weighting_config: RowWeightingConfig | None = None
        self._aggregation_config: AggregationConfig | None = None
        self._codebook_strategy = None
        self._row_weighter = None

        # Multi-GPU pool state
        self._workers: list = []
        self._task_queues: list = []
        self._result_queue = None
        self._pool_alive = False

        # Attributes populated during fitting
        self.fit_params_: dict[str, Any] | None = None
        self._row_class_mask_: np.ndarray | None = None
        self.row_weights_: np.ndarray | None = None
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
            return CodebookConfig()
        if isinstance(config, CodebookConfig):
            return config
        return CodebookConfig(strategy=config)

    def _resolve_row_weighting_config(
        self, config: RowWeightingConfig | WeightMode | str | None
    ) -> RowWeightingConfig:
        if config is None:
            return RowWeightingConfig()
        if isinstance(config, RowWeightingConfig):
            return config
        return RowWeightingConfig(mode=config)

    # ------------------------------------------------------------------
    # Fitting utilities
    # ------------------------------------------------------------------
    def _get_alphabet_size(self) -> int:
        if self.alphabet_size is not None:
            return self.alphabet_size
        if hasattr(self.estimator, "max_num_classes_"):
            inferred = self.estimator.max_num_classes_
            if inferred is not None:
                return int(inferred)
        raise ValueError(
            "alphabet_size must be specified when base estimator has no limit"
        )

    def _get_n_estimators(self, n_classes: int, alphabet_size: int) -> int:
        if self.n_estimators is not None:
            return int(self.n_estimators)
        if alphabet_size <= 1:
            return max(n_classes, 1)
        log_cover = math.ceil(math.log(max(n_classes, 2), alphabet_size))
        cover = math.ceil(n_classes / max(alphabet_size - 1, 1))
        base = max(log_cover, cover)
        redundancy = max(1, int(self.n_estimators_redundancy))
        candidate = base * redundancy
        cap = max(base, 4 * max(log_cover, 1))
        return max(base, min(candidate, cap))

    def _set_verbosity(self) -> None:
        level = (
            logging.WARNING
            if self.verbose <= 0
            else logging.INFO
            if self.verbose == 1
            else logging.DEBUG
        )
        logger.setLevel(level)

    @staticmethod
    def _log_codebook_stats(stats: dict[str, Any], *, tag: str) -> None:
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
                    "best_min_pairwise_hamming_dist",
                    "regeneration_attempts",
                )
                if key in stats
            },
        )

    @staticmethod
    def _log_shapes(
        *,
        X_train: Any,
        X: Any,
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
    # Estimator API
    # ------------------------------------------------------------------
    def fit(self, X, y, **fit_params) -> ManyClassClassifier:
        self._set_verbosity()

        self._codebook_config = self._resolve_codebook_config(self.codebook_config)
        self._row_weighting_config = self._resolve_row_weighting_config(
            self.row_weighting_config
        )
        self._aggregation_config = (
            self.aggregation_config
            if self.aggregation_config is not None
            else AggregationConfig()
        )
        self._codebook_strategy = make_codebook_strategy(self._codebook_config)
        self._row_weighter = make_row_weighter(self._row_weighting_config)
        self.log_proba_aggregation = self._aggregation_config.log_likelihood

        X_validated, y_validated = validate_data(
            self,
            X,
            y,
            ensure_all_finite=False,
            dtype=None,
        )

        self.fit_params_ = dict(fit_params)
        self.classes_ = unique_labels(y_validated)
        self.alphabet_size_ = self._get_alphabet_size()
        if self.alphabet_size_ < 2:
            raise ValueError("alphabet_size must be >= 2.")

        self.n_features_in_ = X_validated.shape[1]
        if hasattr(X_validated, "columns"):
            self.feature_names_in_ = np.asarray(list(X_validated.columns), dtype=object)
        else:
            self.feature_names_in_ = None

        n_classes = len(self.classes_)
        rng = check_random_state(self.random_state)
        self.no_mapping_needed_ = n_classes <= self.alphabet_size_

        self.row_weights_ = None
        self.row_train_support_ = None
        self.row_train_entropy_ = None
        self.row_train_acc_ = None
        self._row_class_mask_ = None

        if self.no_mapping_needed_:
            estimator = self._clone_base_estimator()
            estimator.fit(X_validated, y_validated, **self.fit_params_)
            self.estimators_ = [estimator]
            self.code_book_ = None
            self.codebook_stats_ = {"strategy": "no_mapping"}
            self.classes_index_ = None
            self.X_train = None
            self.Y_train_per_estimator = None
            return self

        n_estimators = self._get_n_estimators(n_classes, self.alphabet_size_)
        codebook, stats = self._codebook_strategy.generate(
            n_classes, n_estimators, self.alphabet_size_, rng
        )
        self._log_codebook_stats(stats, tag="Codebook stats")

        self.code_book_ = codebook
        self.codebook_stats_ = stats
        self.estimators_ = None
        self.classes_index_ = {label: idx for idx, label in enumerate(self.classes_)}

        self.X_train = X_validated
        y_indices = np.array([self.classes_index_[label] for label in y_validated])
        self.Y_train_per_estimator = self.code_book_[:, y_indices]

        if stats.get("has_rest_symbol", False):
            rest_code = stats.get("rest_class_code")
            self._row_class_mask_ = self.code_book_ != rest_code
        else:
            self._row_class_mask_ = np.ones_like(self.code_book_, dtype=bool)

        return self

    # ------------------------------------------------------------------
    # Multi-GPU pool management
    # ------------------------------------------------------------------
    def start_pool(self) -> None:
        """Start persistent worker pool for multi-GPU inference.

        Call once before multiple predict_proba() invocations to amortize
        the TabPFN model loading cost across all calls. Each worker loads
        the model once and stays alive until stop_pool() is called.

        Requires n_jobs > 1 (set in __init__).
        """
        if self._pool_alive or self.n_jobs <= 1:
            return
        from ._parallel import start_pool

        self._workers, self._task_queues, self._result_queue = start_pool(
            self.n_jobs
        )
        self._pool_alive = True

    def stop_pool(self) -> None:
        """Stop the persistent worker pool."""
        if not self._pool_alive:
            return
        from ._parallel import stop_pool

        stop_pool(self._workers, self._task_queues, self._result_queue)
        self._workers, self._task_queues = [], []
        self._result_queue = None
        self._pool_alive = False

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["classes_", "n_features_in_"])
        self._set_verbosity()

        X_validated = validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
        )

        if getattr(self, "no_mapping_needed_", False):
            if not self.estimators_:
                raise RuntimeError("Estimator not fitted. Call fit first.")
            proba = self.estimators_[0].predict_proba(X_validated)
            self._log_shapes(
                X_train=None,
                X=X_validated,
                Y_train_per_estimator=None,
                proba_arr=proba,
            )
            return proba

        if self.code_book_ is None or self.Y_train_per_estimator is None:
            raise RuntimeError("Fit method did not initialize mapping structures.")

        n_est = self.code_book_.shape[0]
        has_rest = bool(self.codebook_stats_.get("has_rest_symbol", False))
        rest_code = self.codebook_stats_.get("rest_class_code") if has_rest else None
        categorical_features = getattr(self, "categorical_features", None)

        if self._pool_alive and self.n_jobs > 1:
            # ── Parallel path: dispatch batches to persistent GPU workers ──
            batches: list[list] = [[] for _ in range(self.n_jobs)]
            for i in range(n_est):
                row_codes = self.Y_train_per_estimator[i]
                mask = None
                if has_rest and self._codebook_config.legacy_filter_rest_train:
                    mask = (row_codes != rest_code)
                batches[i % self.n_jobs].append((i, row_codes, mask))

            n_sent = 0
            for g in range(self.n_jobs):
                if batches[g]:
                    self._task_queues[g].put({
                        "X_train": self.X_train,
                        "X_test": X_validated,
                        "rows": batches[g],
                        "alphabet_size": self.alphabet_size_,
                        "categorical_features": categorical_features,
                        "fit_params": self.fit_params_,
                        "cache_preprocessing": self.cache_preprocessing,
                    })
                    n_sent += 1

            all_results: dict[int, RowRunResult] = {}
            for _ in range(n_sent):
                r = self._result_queue.get(timeout=600)
                if r["status"] == "done":
                    all_results.update(r["results"])
                else:
                    raise RuntimeError(f"Parallel worker error: {r}")

            row_results = [all_results[i] for i in range(n_est)]
            proba_rows = np.stack(
                [rr.proba_test for rr in row_results], axis=0
            )
            weights = normalize_weights(
                np.asarray([rr.weight for rr in row_results], dtype=float)
            )

        else:
            # ── Sequential path (original behavior) ──
            iterator = range(n_est)
            iterable = tqdm.tqdm(iterator, disable=(self.verbose < 2))

            row_results_seq: list[RowRunResult] = []
            raw_weights: list[float] = []

            for row_idx in iterable:
                row_codes = self.Y_train_per_estimator[row_idx]
                mask = None
                if has_rest and self._codebook_config.legacy_filter_rest_train:
                    mask = row_codes != rest_code
                result = run_row(
                    self.estimator,
                    self.X_train,
                    row_codes,
                    X_validated,
                    alphabet_size=self.alphabet_size_,
                    categorical_features=categorical_features,
                    mask=mask,
                    fit_params=self.fit_params_,
                    row_weighter=self._row_weighter,
                )
                row_results_seq.append(result)
                raw_weights.append(result.weight)

            if not row_results_seq:
                raise RuntimeError(
                    "No ECOC rows were generated; check configuration."
                )

            row_results = row_results_seq
            proba_rows = np.stack(
                [result.proba_test for result in row_results], axis=0
            )
            weights = normalize_weights(
                np.asarray(raw_weights, dtype=float)
            )

        self.row_weights_ = weights
        self.row_train_support_ = np.asarray(
            [rr.support for rr in row_results], dtype=int
        )
        self.row_train_entropy_ = np.asarray(
            [np.nan if rr.entropy is None else float(rr.entropy) for rr in row_results],
            dtype=float,
        )
        self.row_train_acc_ = np.asarray(
            [np.nan if rr.accuracy is None else float(rr.accuracy) for rr in row_results],
            dtype=float,
        )

        rest_mask = None
        if has_rest and self._row_class_mask_ is not None:
            rest_mask = self._row_class_mask_.astype(float)

        use_log = self._aggregation_config.log_likelihood
        if not has_rest and not use_log:
            warnings.warn(
                "Using log-likelihood decoding for strategy without rest symbol.",
                UserWarning,
                stacklevel=2,
            )
            use_log = True

        aggregator = make_aggregator(
            use_log,
            mask_rest=has_rest and self._aggregation_config.legacy_mask_rest_log_agg,
        )

        probabilities = aggregator.aggregate(
            proba_rows,
            self.code_book_,
            weights,
            rest_mask=rest_mask,
        )

        self._log_shapes(
            X_train=self.X_train,
            X=X_validated,
            Y_train_per_estimator=self.Y_train_per_estimator,
            proba_arr=probabilities,
        )

        return probabilities

    def predict(self, X) -> np.ndarray:
        probas = self.predict_proba(X)
        if probas.size == 0:
            return np.array([], dtype=self.classes_.dtype)
        return self.classes_[np.argmax(probas, axis=1)]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _clone_base_estimator(self) -> BaseEstimator:
        cloned = clone(self.estimator)
        categorical = getattr(self, "categorical_features", None)
        if categorical is not None:
            if hasattr(cloned, "set_categorical_features"):
                cloned.set_categorical_features(categorical)
            elif hasattr(cloned, "categorical_features"):
                cloned.categorical_features = categorical
        return cloned

    def set_categorical_features(self, categorical_features: list[int]) -> None:
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
        return {"allow_nan": True}

    def __sklearn_tags__(self):  # type: ignore[override]
        tags = super().__sklearn_tags__()
        if isinstance(tags, dict):
            tags.setdefault("allow_nan", True)
            tags.setdefault("requires_fit", True)
            tags.setdefault("estimator_type", "classifier")
            input_tags = tags.get("input_tags")
            if isinstance(input_tags, dict):
                input_tags.setdefault("allow_nan", True)
            return tags

        # scikit-learn >=1.6 returns a Tags dataclass hierarchy.
        if hasattr(tags, "input_tags") and hasattr(tags.input_tags, "allow_nan"):
            tags.input_tags.allow_nan = True
        if hasattr(tags, "estimator_type"):
            tags.estimator_type = "classifier"
        if hasattr(tags, "requires_fit"):
            tags.requires_fit = True
        return tags

    @property
    def codebook_statistics_(self) -> dict[str, Any]:
        check_is_fitted(self, ["classes_"])
        if getattr(self, "no_mapping_needed_", False):
            return {"message": "No codebook mapping was needed."}
        return dict(getattr(self, "codebook_stats_", {}))
