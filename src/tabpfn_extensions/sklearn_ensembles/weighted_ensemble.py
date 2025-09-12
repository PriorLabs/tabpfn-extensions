#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from tabpfn_common_utils.telemetry import set_extension


@set_extension("sklearn_ensembles")
class WeightedAverageEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, weight_threshold=0.0, cv=5, n_max=None):
        self.estimators = estimators
        self.weight_threshold = weight_threshold
        self.n_max = n_max
        self.ensemble = None
        self.cv = cv

    def fit(self, X, y):
        scores = []
        for _name, clf in self.estimators:
            score = np.mean(cross_val_score(clf, X, y, cv=self.cv))
            scores.append(score)

        total_score = sum(scores)
        weights = [score / total_score for score in scores]

        # Prune classifiers with weights below the threshold
        pruned_classifiers = []
        pruned_weights = []
        for clf, weight in zip(self.estimators, weights):
            if weight >= self.weight_threshold:
                pruned_classifiers.append(clf)
                pruned_weights.append(weight)

        if self.n_max is not None:
            # sort pruned classifiers by weight
            pruned_classifiers = [
                clf
                for _, clf in sorted(
                    zip(pruned_weights, pruned_classifiers),
                    key=lambda pair: pair[0],
                )
            ]
            pruned_weights = sorted(pruned_weights)
            pruned_classifiers = pruned_classifiers[-self.n_max :]
            pruned_weights = pruned_weights[-self.n_max :]

        self.ensemble = VotingClassifier(
            estimators=pruned_classifiers,
            voting="soft",
            weights=pruned_weights,
        )
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)
