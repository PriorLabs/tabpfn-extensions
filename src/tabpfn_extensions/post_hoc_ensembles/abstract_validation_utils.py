#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from tabpfn_extensions.scoring.scoring_utils import (
    CLF_LABEL_METRICS,
    score_classification,
    score_regression,
)

from .save_splitting import get_cv_split_for_data

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AbstractValidationUtils(ABC, BaseEstimator):
    _holdout_index_hit_list: np.ndarray | None = (
        None  # list of indices that were hit during repeated holdout validation
    )

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        n_repeats: int,
        seed: int,
        *,
        holdout_fraction: float = 0.33,
        validation_method: Literal["cv", "holdout"] = "cv",
        n_folds: int | None = None,
        time_limit: int | None = None,
        score_metric: str | None = None,
        cv_splitter: Optional[BaseCrossValidator] = None,
    ):
        """Abstract validation utilities for sklearn-like models.

        Args:
            estimators: list of tuples (name, estimator)
            n_repeats: number of repeats for getting OOF
            seed: random seed
            time_limit: If None, no time limit is set. Otherwise, this specifics time limit in seconds for early stopping cross-validation of the models.
                If the time limit set, the number of repeats is are the maximum amount of repeats. As a result, this code does not guarantee that to stop before
                the time_limit is over.
            score_metric: metric to use for scoring. If None, child class defines default on the fly.
            validation_method: method to use for validation. If "cv", (repeated) cross-validation is used. If "holdout", (repeated) holdout validation is used.
            n_folds: number of folds for getting OOF. Only used if validation_method is "cv"
            holdout_fraction: fraction of data to hold out for holdout validation. Only used if validation_method is "holdout"
        """
        self.estimators = estimators
        self.score_metric = score_metric
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.time_limit = time_limit
        self._start_time: int = 0
        self.classes_: np.ndarray | None = None
        self._repeats_seed = [
            int(self._rng.randint(0, int(np.iinfo(np.int32).max)))
            for _ in range(self.n_repeats)
        ]
        self._estimators = estimators.copy()  # internal copy for early stopping.
        self.validation_method = validation_method
        self.holdout_fraction = holdout_fraction
        self.cv_splitter = cv_splitter

        if self.cv_splitter: 
            self.n_repeats = 1
            self.n_folds = self.cv_splitter.get_n_splits()

    @property
    @abstractmethod
    def stratifed_split(self) -> bool:
        pass

    @abstractmethod
    def loss_function(self, y_true, y_pred_proba) -> float:
        pass

    @abstractmethod
    def _predict_oof(self, base_model, X) -> np.ndarray:
        pass

    @abstractmethod
    def _batch_predict_oof(self, X, configurations) -> np.ndarray:
        pass

    @abstractmethod
    def _proba_template(self, X, y) -> np.ndarray:
        pass

    @property
    def _is_classification(self) -> bool:
        return isinstance(self, AbstractValidationUtilsClassification)

    @property
    def _is_holdout(self) -> bool:
        return self.validation_method == "holdout"

    def _fill_predictions_in_place(
        self,
        *,
        model_i: int,
        base_model: Callable,
        oof_proba_list: list[np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        train_index: list[int],
        test_index: list[int],
        loss_per_estimator: list[list[float]] | None,
        holdout_index_hits: np.ndarray | None,
        _extra_processing: bool,
        split_i: int,
    ) -> None:
        fold_X_train, fold_X_test = X[train_index], X[test_index]
        fold_y_train, fold_y_test = y[train_index], y[test_index]
        # Note: Using the base_model directly without deepcopy for performance reasons

        # Default base models case
        base_model.fit(fold_X_train, fold_y_train)

        pred = self._predict_oof(base_model, fold_X_test)

        oof_proba_list[model_i][test_index] += pred

        if holdout_index_hits is not None:
            holdout_index_hits[test_index] += 1

        if loss_per_estimator is not None:
            loss_per_estimator[model_i].append(
                self.loss_function(fold_y_test, pred),
            )
        if _extra_processing:  # custom code for supporting extra use case (ignore)
            if not hasattr(self, "_extra_processing"):
                raise ValueError(
                    f"_extra_processing is True, but _extra_processing is not defined for the current class: {type(self)}.",
                )
            self._extra_processing(split_i, base_model, test_index)

    def yield_cv_data_iterator(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int, int, int, BaseEstimator, list[int], list[int], bool, bool]:
        """Yields data for cross-validation, handling both default and custom splitters.

        This iterator yields tuples containing the necessary information for a single
        cross-validation split, such as model index, split index, train/test indices, etc.

        When a custom `cv_splitter` is provided, this method assumes that the user
        wants to apply the same fold structure to each estimator. Therefore, the splits
        are generated once and reused for every model in `self.estimators`."""

        n_models = len(self.estimators)

        # TODO: Update documentation to make user aware that self.n_foldds and self.n_repeats are not used
        if self.cv_splitter:
            logger.info(f"Using provided CV splitter: {self.cv_splitter.__class__.__name__}")
            logger.info(f"Ignore n_folds parameter: {self.n_folds}")
            logger.info(f"Ignore n_repeats parameter: {self.n_repeats}, hardcoded to 1 in splitter")
            splits = list(self.cv_splitter.split(X, y))
            n_folds_per_rep = self.cv_splitter.get_n_splits()
            split_iterator = iter(splits)
            for model_i in range(n_models):
                # For a custom splitter, we reset the iterator for each model
                # This assumes the user wants to apply the same fold structure to each model
                split_iterator = iter(splits)
                for fold_i in range(n_folds_per_rep):
                    train_index, test_index = next(split_iterator)
                    is_last_fold = (fold_i + 1) == n_folds_per_rep
                    is_last_model = (model_i + 1) == n_models
                    yield (
                        model_i,
                        fold_i,
                        1, # Custom splitters don't have a native concept of repeats
                        self.estimators[model_i][1],
                        train_index,
                        test_index,
                        is_last_fold,
                        is_last_fold and is_last_model,
                    )
        else:
            holdout_validation = self._is_holdout
            _folds = self.n_folds if not holdout_validation else 1
            for repeat_i in range(self.n_repeats):
                
                # Prepare the splits for the CURRENT repeat
                splits_for_repeat = None
                if holdout_validation:
                    splitter = StratifiedShuffleSplit if self.stratifed_split else ShuffleSplit
                    # In holdout, we generate one split per repeat
                    splits_for_repeat = list(splitter(
                        n_splits=1,
                        test_size=self.holdout_fraction,
                        random_state=self._repeats_seed[repeat_i], # different seed for each repeat
                    ).split(X,y))
                else:
                    splits_for_repeat = get_cv_split_for_data(
                        x=X,
                        y=y,
                        stratified_split=self.stratifed_split,
                        n_splits=self.n_folds,
                        splits_seed=self._repeats_seed[repeat_i],
                        safety_shuffle=True,
                        auto_fix_stratified_splits=True,
                    )
                
                for model_i in range(n_models):
                    for fold_i in range(_folds):
                        train_index, test_index = splits_for_repeat[fold_i]
                        split_i = fold_i + repeat_i * _folds
                        is_last_fold = (fold_i + 1) == _folds
                        is_last_model = (model_i + 1) == n_models

                        logger.info(
                            f"Yielding data for model {self.estimators[model_i][0]} and split {split_i}."
                        )
                        yield (
                            model_i,
                            split_i,
                            repeat_i + 1,
                            self.estimators[model_i][1],
                            train_index,
                            test_index,
                            is_last_fold,
                            is_last_fold and is_last_model,
                        )

    def time_limit_reached(self) -> bool:
        """Check if the time limit for execution has been reached.

        Returns:
            bool: True if the time limit has been reached, False otherwise or if no time limit was set
        """
        if self.time_limit is None:
            return False
        return (time.time() - self._start_time) > self.time_limit

    def not_enough_time(self, current_repeat: int) -> bool:
        """Simple heuristic to stop cross-validation early if not enough time is left for another repeat.

        Args:
            current_repeat: The current repeat index

        Returns:
            bool: True if there likely isn't enough time for another repeat, False otherwise

        Note:
            This is a heuristic based on average time per repeat so far and may not be exact.
        """
        if self.time_limit is None:
            return False

        time_spent_so_far = time.time() - self._start_time
        avg_time_per_repeat = time_spent_so_far / current_repeat

        return (time_spent_so_far + avg_time_per_repeat) > self.time_limit

    def set_time_limit(self) -> None:
        """Initialize the timer for time-limited execution.

        Sets the start time for time limit tracking and logs the time limit info.
        This method should be called at the beginning of validation.
        """
        if self.time_limit is not None:
            self._start_time = time.time()
            logger.info(
                f"Set time limit to {self.time_limit} seconds. We will early stop validation if needed.",
            )

    def _impute_dropped_instances(
        self,
        *,
        proba_template: np.ndarray,
        oof_proba_list: list[np.ndarray],
        y: np.ndarray,
    ) -> list[np.ndarray]:
        fill_val = (
            np.mean(y) if proba_template.shape[1] == 1 else 1 / proba_template.shape[1]
        )

        for oof in oof_proba_list:
            # Only impute if any values were not filled (i.e., are NaN)
            if np.isnan(oof).any():
                logger.warning(
                    "Imputing instances that were dropped during splits (e.g., due to not enough instances per class).",
                )
                oof[np.isnan(oof).any(axis=1)] = fill_val

        # Fix dropped classes
        classes_ = np.unique(y)
        if len(classes_) != self.n_classes_:
            logger.warning(
                "Adding classes to OOF proba that were dropped during splits (e.g., due to not enough instances per class).",
            )
            missing_classes = np.setdiff1d(classes_, self.classes_)

            new_oof = []
            for oof in oof_proba_list:
                tmp_oof = np.full((oof.shape[0], len(classes_)), np.nan)
                for i, c in enumerate(classes_):
                    if c in missing_classes:
                        tmp_oof[:, i] = 0
                    else:
                        tmp_oof[:, i] = oof[
                            :,
                            np.where(self.classes_ == c)[0],
                        ].flatten()
                new_oof.append(tmp_oof)
            oof_proba_list = new_oof

        return oof_proba_list

    def get_oof_per_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        return_loss_per_estimator: bool = False,
        impute_dropped_instances: bool = True,
        _extra_processing: bool = False,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[float]]:
        """Get OOF predictions for each base model.

        Args:
            X: training data (features)
            y: training labels
            return_loss_per_estimator: if True, also return the loss per estimator.
            impute_dropped_instances: if True, impute instances that were dropped during the splits (e.g., due to not enough instances per class).
            _extra_processing:

        Returns: either only OOF predictions or OOF predictions and loss per estimator.
            If self.is_holdout is True, the OOF predictions can return NaN values for instances not covered during repeated holdout.
        """
        if self.validation_method == "cv":
            if self.n_folds is None:
                raise ValueError(
                    "Number of folds must be specified for cross-validation.",
                )
            logger.info(
                f"Starting {self.n_repeats}-repeated {self.n_folds}-fold cross-validation.",
            )
        elif self.validation_method == "holdout":
            if not isinstance(self.holdout_fraction, float):
                raise ValueError(
                    f"Holdout fraction must be specified as a float for holdout validation. Got: {self.holdout_fraction}",
                )
            logger.info(
                f"Starting {self.n_repeats}-repeated holdout validation with holdout_frac={self.holdout_fraction}.",
            )
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}")

        (
            oof_proba_list,
            backup_oof_proba_list,
            backup_loss_per_estimator,
            backup_run_repeats,
        ) = (None, None, None, None)
        use_backups = False  # only use backups if at least on repeat was done.
        loss_per_estimator = (
            [[] for _ in self.estimators] if return_loss_per_estimator else None
        )
        holdout_index_hits, holdout_index_hits_backup, holdout_index_hit_counts = (
            None,
            None,
            None,
        )

        # Get OOF predictions for each base model
        self.set_time_limit()
        ran_repeats = self.n_repeats

        for (
            model_i,
            split_i,
            current_repeat,
            base_model,
            train_index,
            test_index,
            check_for_model_early_stopping,
            check_for_repeat_early_stopping,
        ) in self.yield_cv_data_iterator(X, y):
            # Delayed setting classes_ to make sure that fixes for splits have been applied.
            if (self.classes_ is None) and self._is_classification:
                self.classes_ = np.unique(y[train_index])

            if oof_proba_list is None:
                proba_template = self._proba_template(X, y)
                oof_proba_list = [deepcopy(proba_template) for _ in self.estimators]

            to_pass_holdout_index_hits = None
            if (holdout_index_hits is None) and self._is_holdout:
                holdout_index_hits = np.zeros(X.shape[0], dtype=float)
                holdout_index_hit_counts = 0
            if self._is_holdout and holdout_index_hit_counts < current_repeat:
                to_pass_holdout_index_hits = holdout_index_hits
                holdout_index_hit_counts = current_repeat

            self._fill_predictions_in_place(
                model_i=model_i,
                base_model=base_model,
                oof_proba_list=oof_proba_list,
                X=X,
                y=y,
                train_index=train_index,
                test_index=test_index,
                loss_per_estimator=loss_per_estimator,
                holdout_index_hits=to_pass_holdout_index_hits,
                split_i=split_i,
                _extra_processing=_extra_processing,
            )

            if check_for_repeat_early_stopping:  # True after every repeat.
                ran_repeats = current_repeat

                # Save backups for model early stopping
                backup_oof_proba_list = deepcopy(oof_proba_list)
                backup_run_repeats = ran_repeats
                backup_loss_per_estimator = (
                    None if loss_per_estimator is None else deepcopy(loss_per_estimator)
                )
                holdout_index_hits_backup = (
                    None if holdout_index_hits is None else deepcopy(holdout_index_hits)
                )
                use_backups = True  # use backups after first repeat

                logger.info(
                    f"Finished validation repeat {current_repeat}/{self.n_repeats}.",
                )
                if self.time_limit_reached():
                    logger.info("Time limit reached. Stop repeating validation.")
                    break

                if self.not_enough_time(current_repeat=current_repeat):
                    logger.info(
                        "Likely not enough time left for another repeat. Stop repeating validation.",
                    )
                    break

            if check_for_model_early_stopping:
                if self.time_limit_reached():
                    logger.info("Time limit reached.")
                elif self.not_enough_time(
                    current_repeat=current_repeat * (model_i + 1),
                ):
                    logger.info("Likely not enough time left for another model.")
                else:
                    continue
                logger.info(
                    f"Stop validation of all models after {model_i + 1} models in repeat {current_repeat}.",
                )

                if use_backups:
                    logger.info("Use results from last repeat for all models!")
                    # Fallback to last repeat if time limit is reached and we did cv before.
                    oof_proba_list = backup_oof_proba_list
                    loss_per_estimator = backup_loss_per_estimator
                    holdout_index_hits = holdout_index_hits_backup
                    ran_repeats = backup_run_repeats
                else:
                    logger.info(
                        "As this is the first repeat, we trim down the models to all so-far run models!",
                    )
                    # Trim down OOF and losses to only have validated models.
                    ran_repeats = current_repeat
                    oof_proba_list = [
                        oof_proba_list[tmp_i] for tmp_i in range(model_i + 1)
                    ]
                    loss_per_estimator = (
                        [loss_per_estimator[tmp_i] for tmp_i in range(model_i + 1)]
                        if loss_per_estimator is not None
                        else None
                    )
                    self._estimators = self._estimators[: model_i + 1]
                break

        # Set empty OOF predictions to NaN
        for i, oof in enumerate(oof_proba_list):
            if self._is_classification:
                nan_mask = oof.sum(axis=1) == 0
            elif self._is_holdout:
                nan_mask = holdout_index_hits == 0
            else:
                # Regression without holdout. No need to set to NaN as all values must be filled (no dropped instances, no holdout).
                break
            oof_proba_list[i][nan_mask] = np.nan

        if self._is_holdout:
            self._holdout_index_hit_list = np.where(holdout_index_hits > 0)[0]

        # Sanity check
        #  - check that all OOF predictions sum up to the number of repeats for classification
        if self._is_classification:
            if self._is_holdout:
                holdout_index_hits[holdout_index_hits == 0] = np.nan
                if not all(
                    np.isclose(
                        oof.sum(axis=1),
                        holdout_index_hits,
                        equal_nan=True,
                    ).all()
                    for oof in oof_proba_list
                ):
                    raise ValueError(
                        "OOF predictions are not consistent over repeats for holdout! Something went wrong.",
                    )
            elif not all(
                np.isclose(
                    oof[~np.isnan(oof).any(axis=1)].sum(axis=1),
                    ran_repeats,
                ).all()
                for oof in oof_proba_list
            ):
                for i, oof in enumerate(oof_proba_list):
                    pass
                raise ValueError(
                    "OOF predictions are not consistent over repeats! Something went wrong.",
                )

        # Average OOF predictions over all repeats
        if ran_repeats > 1:
            for i, oof in enumerate(oof_proba_list):
                div_by = (
                    ran_repeats
                    if not self._is_holdout
                    else holdout_index_hits[:, None]
                    if self._is_classification
                    else holdout_index_hits
                )
                oof_proba_list[i] = oof / div_by

        # Imputation of dropped instances is only needed for stratified splits as we do not drop values otherwise.
        if impute_dropped_instances and self.stratifed_split and (not self._is_holdout):
            oof_proba_list = self._impute_dropped_instances(
                proba_template=proba_template,
                oof_proba_list=oof_proba_list,
                y=y,
            )

        if loss_per_estimator is not None:
            loss_per_estimator = [np.mean(losses) for losses in loss_per_estimator]
            return oof_proba_list, loss_per_estimator

        return oof_proba_list


class AbstractValidationUtilsClassification(AbstractValidationUtils, ClassifierMixin):
    n_classes_: int

    def loss_function(self, y_true, y_pred_proba):
        opt_metric = "log_loss" if self.score_metric is None else self.score_metric

        y_pred_is_labels = False
        if opt_metric in CLF_LABEL_METRICS:
            y_pred_proba = self.classes_.take(np.argmax(y_pred_proba, axis=1))
            y_pred_is_labels = True

        # negative to make score a loss
        return -score_classification(
            opt_metric,
            y_true,
            y_pred_proba,
            y_pred_is_labels=y_pred_is_labels,
        )

    def _batch_predict_oof(self, X, configurations):
        return self.tabpfn_model.custom_batch_predict_proba(
            X=X,
            configurations=configurations,
        )

    def _predict_oof(self, base_model, X):
        return base_model.predict_proba(X)

    @property
    def stratifed_split(self):
        return True

    def _proba_template(self, X, y):
        self.n_classes_ = len(self.classes_)
        return np.full((X.shape[0], self.n_classes_), 0, dtype=float)


class AbstractValidationUtilsRegression(AbstractValidationUtils, RegressorMixin):
    def loss_function(self, y_true, y_pred_proba):
        opt_metric = "rmse" if self.score_metric is None else self.score_metric
        return -score_regression(opt_metric, y_true, y_pred_proba)

    def _predict_oof(self, base_model, X):
        return base_model.predict(X)

    def _batch_predict_oof(self, X, configurations):
        return self.tabpfn_model.custom_batch_predict(
            X=X,
            configurations=configurations,
        )

    @property
    def stratifed_split(self):
        return False

    def _proba_template(self, X, y):
        return np.full((X.shape[0],), 0, dtype=float)
