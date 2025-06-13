#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.utils import check_random_state

from .abstract_validation_utils import (
    AbstractValidationUtils,
    AbstractValidationUtilsClassification,
    AbstractValidationUtilsRegression,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import BaseCrossValidator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def caruana_weighted(
    predictions: list[np.ndarray],
    labels: np.ndarray,
    seed,
    n_iterations,
    loss_function,
):
    """Caruana's ensemble selection with replacement."""
    # -- Init Vars
    num_input_models_ = len(predictions)
    ensemble = []  # type: list[np.ndarray]
    trajectory = []  # contains iteration best
    val_loss_over_iterations_ = []  # contains overall best
    order = []
    rand = check_random_state(seed)
    weighted_ensemble_prediction = np.zeros(predictions[0].shape, dtype=np.float64)

    fant_ensemble_prediction = np.zeros(
        weighted_ensemble_prediction.shape,
        dtype=np.float64,
    )

    for i in range(n_iterations):
        logger.debug(f"Iteration {i}")

        ens_size = len(ensemble)
        if ens_size > 0:
            np.add(
                weighted_ensemble_prediction,
                ensemble[-1],
                out=weighted_ensemble_prediction,
            )

        # -- Process Iteration Solutions
        losses = np.zeros((len(predictions)), dtype=np.float64)

        for j, pred in enumerate(predictions):
            np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
            np.multiply(
                fant_ensemble_prediction,
                (1.0 / float(ens_size + 1)),
                out=fant_ensemble_prediction,
            )
            losses[j] = loss_function(labels, fant_ensemble_prediction)

        if i == 0:
            model_losses = losses.copy()

        # -- Eval Iteration results
        all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
        best = rand.choice(all_best)  # break ties randomly
        ensemble_loss = losses[best]

        ensemble.append(predictions[best])
        trajectory.append(ensemble_loss)
        order.append(best)

        # Build Correct Validation loss list
        if not val_loss_over_iterations_:
            # Init
            val_loss_over_iterations_.append(ensemble_loss)
        elif val_loss_over_iterations_[-1] > ensemble_loss:
            # Improved
            val_loss_over_iterations_.append(ensemble_loss)
        else:
            # Not Improved
            val_loss_over_iterations_.append(val_loss_over_iterations_[-1])

        # -- Break for special cases
        #   - If we only have a pool of base models of size 1 (code found the single best model)
        #   - If we find a perfect ensemble/model, stop early
        if (len(predictions) == 1) or (ensemble_loss == 0):
            break

    indices_ = order
    trajectory_ = trajectory
    min_score = np.min(trajectory_)
    idx_best = trajectory_.index(min_score)
    indices_ = indices_[: idx_best + 1]
    n_iterations = idx_best + 1

    ensemble_members = Counter(indices_).most_common()
    weights = np.zeros((num_input_models_,), dtype=np.float64)

    for ensemble_member in ensemble_members:
        weight = float(ensemble_member[1]) / n_iterations
        weights[ensemble_member[0]] = weight

    if np.sum(weights) < 1:
        weights = weights / np.sum(weights)

    logger.info(f"Order of selections: {order}")
    logger.info(f"Val loss over iterations: {val_loss_over_iterations_}")
    logger.info(f"Model losses: {model_losses}")
    logger.info(f"Best weights: {weights}")

    return weights, val_loss_over_iterations_, model_losses


class GreedyWeightedEnsemble(AbstractValidationUtils):
    tune_temperature: bool = False

    def __init__(
        self,
        n_iterations: int,
        silo_top_n: int,
        model_family_per_estimator: list[str] | None = None,
    ):
        """GES implementation for sklearn-like models.

        Args:
            n_iterations: number of iterations for GES.
            silo_top_n: number of models for SiloTopN pruning.
            model_family_per_estimator: the model family for each estimator used for SiloTopN pruning.
        """
        self.n_iterations = n_iterations
        self.silo_top_n = silo_top_n
        self.model_family_per_estimator = model_family_per_estimator
        self._model_family_per_estimator = None  # in case of early stopping
        (
            self.ensemble,
            self.val_loss_over_iterations_,
            self.model_losses_,
            self.weights_,
        ) = (None, None, None, None)

    @abstractmethod
    def _build_ensemble(self, base_models, weights):
        pass

    def _tune_temperature_in_place(
        self,
        oof_proba_per_estimator: list[np.ndarray],
        y: np.ndarray,
    ) -> list[np.ndarray]:
        for model_i, (model_name, model) in enumerate(self._estimators):
            logger.info(f"Tuning temperature for model {model_name}")
            oof_proba_per_estimator[model_i] = self.tune_temperature_for_model_in_place(
                base_model=model,
                oof_predictions=oof_proba_per_estimator[model_i],
                y=y,
            )

    def get_weights(self, X, y):
        oof_proba = self.get_oof_per_estimator(X, y)
        self.model_family_per_estimator = (
            self.model_family_per_estimator
            if self.model_family_per_estimator is not None
            else ["X"] * len(self._estimators)
        )
        self._model_family_per_estimator = self.model_family_per_estimator[
            : len(self._estimators)
        ]

        if self._is_holdout:
            oof_proba = [oof[self._holdout_index_hit_list] for oof in oof_proba]
            y = y[self._holdout_index_hit_list]

        # If tune_temperature is True, we will tune the temperature for each model here before GES.
        if self.tune_temperature:
            self._tune_temperature_in_place(oof_proba_per_estimator=oof_proba, y=y)

        # Prune base models before GES
        data_for_pruning = []
        data_for_selection = {}
        assert (
            len(self._estimators)
            == len(oof_proba)
            == len(self._model_family_per_estimator)
        ), "All iterators must have the same length!"
        for (bm_name, bm_model), bm_oof_proba, bm_family in zip(
            self._estimators,
            oof_proba,
            self._model_family_per_estimator,
        ):
            bm_loss = self.loss_function(y, bm_oof_proba)
            data_for_selection[bm_name] = (bm_model, bm_oof_proba)
            data_for_pruning.append((bm_loss, bm_name, bm_family))
        bm_to_keep = _prune_to_silo_top_n(
            data_for_pruning,
            n=self.silo_top_n,
            maximize_metric=False,
        )

        tmp_estimators = []
        tmp_oof_proba = []
        for bm_name in bm_to_keep:
            bm_model, bm_oof_proba = data_for_selection[bm_name]
            tmp_estimators.append((bm_name, bm_model))
            tmp_oof_proba.append(bm_oof_proba)
        self._estimators = tmp_estimators
        oof_proba = tmp_oof_proba

        # Run GES
        (
            self.weights_,
            self.val_loss_over_iterations_,
            self.model_losses_,
        ) = caruana_weighted(
            oof_proba,
            y,
            self.seed + 62,
            n_iterations=self.n_iterations,
            loss_function=self.loss_function,
        )

        return self.weights_

    def fit(self, X, y):
        weights = self.get_weights(X, y)

        final_weights = []
        base_models = []
        assert len(self._estimators) == len(
            weights,
        ), "All iterators must have the same length!"
        for bm, weight in zip(self._estimators, weights):
            if weight != 0:
                final_weights.append(weight)
                base_models.append(bm)

        self.ensemble = self._build_ensemble(base_models, final_weights)
        self.ensemble.fit(X, y)  # (re-)fit on full data
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)


class GreedyWeightedEnsembleClassifier(
    GreedyWeightedEnsemble,
    AbstractValidationUtilsClassification,
):
    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        *,
        score_metric: str = "log_loss",
        time_limit: int | None = None,
        validation_method: Literal["cv", "holdout"] = "cv",
        holdout_fraction: float = 0.33,
        n_folds: int = 5,
        n_repeats: int = 1,
        seed: int = 42,
        n_iterations: int = 50,
        silo_top_n: int = 25,
        model_family_per_estimator: list[str] | None = None,
        cv_splitter: BaseCrossValidator | None = None,
    ):
        """Post Hoc Ensemble Classifier Specialized for TabPFN models.

        Args:
            estimators: list of tuples of (name, estimator) containing the base models to ensemble
            score_metric: the metric string to use for scoring the ensemble models
            time_limit: the time limit for the validation process
            validation_method: the method to use for validation
            holdout_fraction: the fraction of the data to use for holdout validation
            n_folds: the number of folds to use for cross-validation
            n_repeats: the number of repeats to use for validation
            seed: the random seed used to control randomness
            n_iterations: the number of iterations to use for the greedy ensemble search
            silo_top_n: the total number of models used for ensemble selection.
                In other words, the number to prune to when using SiloTopN.
            model_family_per_estimator: the model family for each estimator used for SiloTopN pruning.
        """
        GreedyWeightedEnsemble.__init__(
            self,
            n_iterations=n_iterations,
            silo_top_n=silo_top_n,
            model_family_per_estimator=model_family_per_estimator,
        )
        AbstractValidationUtilsClassification.__init__(
            self,
            estimators=estimators,
            score_metric=score_metric,
            n_folds=n_folds,
            n_repeats=n_repeats,
            seed=seed,
            time_limit=time_limit,
            holdout_fraction=holdout_fraction,
            validation_method=validation_method,
            cv_splitter=cv_splitter,
        )

    def _build_ensemble(self, base_models, weights):
        return VotingClassifier(estimators=base_models, voting="soft", weights=weights)


class GreedyWeightedEnsembleRegressor(
    GreedyWeightedEnsemble,
    AbstractValidationUtilsRegression,
):
    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        *,
        score_metric: str = "rmse",
        validation_method: Literal["cv", "holdout"] = "cv",
        n_folds: int = 5,
        holdout_fraction: float = 0.33,
        n_repeats: int = 1,
        time_limit: int | None = None,
        seed: int = 42,
        n_iterations: int = 50,
        silo_top_n: int = 25,
        model_family_per_estimator: list[str] | None = None,
        cv_splitter: BaseCrossValidator | None = None,
    ):
        """Post Hoc Ensemble Regressor Specialized for TabPFN models.

        Args:
            estimators: list of tuples of (name, estimator) containing the base models to ensemble
            score_metric: the metric string to use for scoring the ensemble models
            time_limit: the time limit for the validation process
            validation_method: the method to use for validation
            holdout_fraction: the fraction of the data to use for holdout validation
            n_folds: the number of folds to use for cross-validation
            n_repeats: the number of repeats to use for validation
            seed: the random seed used to control randomness
            n_iterations: the number of iterations to use for the greedy ensemble search
            silo_top_n: the total number of models used for ensemble selection.
                In other words, the number to prune to when using SiloTopN.
            model_family_per_estimator: the model family for each estimator used for SiloTopN pruning.
        """
        GreedyWeightedEnsemble.__init__(
            self,
            n_iterations=n_iterations,
            silo_top_n=silo_top_n,
            model_family_per_estimator=model_family_per_estimator,
        )
        AbstractValidationUtilsRegression.__init__(
            self,
            estimators=estimators,
            score_metric=score_metric,
            n_folds=n_folds,
            n_repeats=n_repeats,
            seed=seed,
            time_limit=time_limit,
            holdout_fraction=holdout_fraction,
            validation_method=validation_method,
            cv_splitter=cv_splitter,
        )

    def _build_ensemble(self, base_models, weights):
        return VotingRegressor(estimators=base_models, weights=weights)


def _prune_to_silo_top_n(
    base_models_data: list[tuple[float, str, str]],
    *,
    n: int,
    maximize_metric: bool,
) -> list[str]:
    """Prune to the top N models per silo based on validation score. A silo represents an algorithm family.

    Code Taken from Lennart's (unpublished) version of Assembled (https://github.com/ISG-Siegen/assembled)

    Args:
        base_models_data: list of tuples of (metric_value, model_name, algorithm_family) per base model
        n: number of models to keep
        maximize_metric: whether to maximize the metric or not
    """
    # -- Get silos
    # algorithm family (af) to list of base models of this family
    af_to_model = {}  # type: dict[str, list[tuple[float, str, str]]]
    for bm_metric_value, bm_name, bm_family in base_models_data:
        if bm_family not in af_to_model:
            af_to_model[bm_family] = []

        af_to_model[bm_family].append((bm_metric_value, bm_name, bm_family))

    # Get the minimal number of entire a silo can have (in case of equal distribution)
    min_silo_val = max(math.floor(n / len(af_to_model.keys())), 1)

    while sum(len(base_models) for base_models in af_to_model.values()) > n:
        # Find silos with too many values
        too_large_silos = [
            af
            for af, base_models in af_to_model.items()
            if len(base_models) > min_silo_val
        ]

        if not too_large_silos:
            break

        # For all silos with too many values, find and remove the model with the worst performance
        #   - This won't remove silos entirely, because silos with at least 1 element won't be too large
        #   - The first element of sorted() is the element with the worst performance (highest/lowest value)
        worst_model = sorted(
            [base_model for af in too_large_silos for base_model in af_to_model[af]],
            key=lambda x: x[0],  # sort by validation score/loss
            reverse=maximize_metric,  # determines if higher or lower is better
        )[-1]  # select the worst model (first element)
        af_to_model[worst_model[-1]].remove(worst_model)

    if sum(len(base_models) for base_models in af_to_model.values()) > n:
        # In this case, we have more silos than top_n (cat_to_model.keys() > top_n)
        # Moreover, at this point, all silos will only have 1 element in it.
        # To resolve this, we can simply return the top_n models over these silos
        # (other fallbacks like random for more diversity would work as well, but we think top is best for now)
        sort_rest = sorted(
            [
                base_model
                for base_models in af_to_model.values()
                for base_model in base_models
            ],  # all models
            key=lambda x: x[0],
            reverse=maximize_metric,
        )
        bm_names_mv_to_keep = [
            (name, metric_val) for metric_val, name, _ in sort_rest[-n:]
        ]
    else:
        bm_names_mv_to_keep = [
            (name, metric_val)
            for vals in af_to_model.values()
            for metric_val, name, _ in vals
        ]

    # Sort to have similar order to top n (also can be beneficial for ensembling)
    return [
        x[0]
        for x in sorted(
            bm_names_mv_to_keep,
            key=lambda m: m[1],
            reverse=maximize_metric,
        )
    ]
