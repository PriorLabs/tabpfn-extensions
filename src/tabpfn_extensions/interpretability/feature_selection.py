#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import BaseCrossValidator


class _VerboseSFS(SequentialFeatureSelector):
    """Subclass of ``SequentialFeatureSelector`` that prints which feature it
    picked at each round, along with the CV score.

    Hook is the private ``_get_best_new_feature_score`` method. It's stable
    in current sklearn (1.6 / 1.7 / 1.8 all share the signature) but is
    private API — if sklearn renames or restructures it, this will break
    visibly (no per-round output) rather than silently. Verbosity then
    falls back to the pre/post CV-score prints in ``_feature_selection``.

    We deliberately do not override ``__init__``. sklearn's parameter
    introspection (``get_params`` / ``clone`` / ``_validate_params``)
    rejects subclasses whose ``__init__`` uses ``*args`` / ``**kwargs``
    or introduces parameters not in ``_parameter_constraints``. Instead
    the caller sets ``_verbose_feature_names`` as a plain attribute on
    the instance after construction; the iteration counter is lazy.
    """

    def _get_best_new_feature_score(  # type: ignore[override]
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
        current_mask: np.ndarray,
        **params: Any,
    ) -> tuple[int, float]:
        idx, score = super()._get_best_new_feature_score(
            estimator,
            X,
            y,
            cv,
            current_mask,
            **params,
        )
        self._verbose_iter = getattr(self, "_verbose_iter", 0) + 1
        names = getattr(self, "_verbose_feature_names", None)
        label = names[idx] if names is not None else idx
        # Direction matters for the message: forward "picks to add",
        # backward "picks to remove" — same _get_best_new_feature_score
        # method but the semantics differ.
        verb = "picked" if self.direction == "forward" else "dropped"
        print(  # noqa: T201 — intentional verbose-mode output
            f"  round {self._verbose_iter}/{self.n_features_to_select_}: "
            f"{verb} feature {label!r}, cv score = {score:.4f}",
        )
        return idx, score


@dataclass
class FeatureSelectionResult:
    """Result of running ``feature_selection``.

    Attributes:
        selector: The underlying fitted ``SequentialFeatureSelector``.
            Use it for ``.transform(X)`` to project to the selected
            columns, or for any sklearn-style downstream work.
        support_mask: Boolean array of shape ``(n_features,)`` — ``True``
            for the columns SFS picked.
        selected_indices: Integer indices of the selected columns, in
            ascending order.
        selected_names: Selected feature names, in the same order as
            ``selected_indices``. ``None`` iff ``feature_names`` wasn't
            passed.
        baseline_score_mean: Mean cross-validated score of ``estimator``
            on **all** features, using the same ``cv`` and ``scoring`` as
            the selection step.
        baseline_score_std: Standard deviation across CV folds for the
            baseline score.
        selected_score_mean: Mean cross-validated score of ``estimator``
            on the **selected** subset of features.
        selected_score_std: Standard deviation across CV folds for the
            selected-subset score.
    """

    selector: SequentialFeatureSelector
    support_mask: np.ndarray
    selected_indices: list[int]
    selected_names: list[str] | None
    baseline_score_mean: float
    baseline_score_std: float
    selected_score_mean: float
    selected_score_std: float


def feature_selection(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int | float | str,
    feature_names: list[str] | None = None,
    *,
    cv: int | BaseCrossValidator | Iterable = 5,
    scoring: str | Callable | None = None,
    direction: str = "forward",
    n_jobs: int | None = None,
    tol: float | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> FeatureSelectionResult:
    """Sequential feature selection wrapper around scikit-learn's SFS.

    Picks a subset of features that work well for ``estimator`` by
    repeatedly fitting ``estimator`` on candidate subsets and keeping the
    one that maximizes a cross-validated score. Forwards the relevant
    ``SequentialFeatureSelector`` hyperparameters; always computes the
    baseline (all-features) and selected (subset-only) CV scores so they
    are available on the returned object regardless of ``verbose``.

    Note that while we expose feature selection here, **TabPFN is very
    robust to noisy / uninformative features** in its native
    in-context-learning regime, so the *accuracy gain* from running this
    selector is often marginal — on real datasets the all-features
    baseline and the selected subset typically score within
    cross-validation noise of each other. The value of running selection
    on TabPFN is usually more in terms of interpretability.
    Note that other interpretability methods, such as SHAP, are also supported
    and are generally much faster because they can use the KV cache.

    Sequential feature selection is expensive: forward selection with
    ``n_features_to_select=k`` on ``d`` features uses on the order of
    ``cv * sum_{i=0..k-1} (d - i)`` model fits, plus 2 more for the
    baseline / selected CV scores. ``n_jobs`` parallelizes
    candidate-feature evaluation within each round; pass ``-1`` for all
    cores. Note that v3's KV cache does *not* help here — every candidate
    has a different ``X_train`` so the cache invalidates between fits.

    Args:
        estimator: The model to use for feature selection.
        X: Input features, shape ``(n_samples, n_features)``.
        y: Target values, shape ``(n_samples,)``.
        n_features_to_select: Number of features to keep. ``int`` for an
            absolute count, ``float`` for a fraction of the total, or
            ``"auto"`` to let ``tol`` decide (requires ``tol``).
        feature_names: Optional list of feature names. When provided, the
            returned ``FeatureSelectionResult`` carries the selected
            names under ``selected_names``.
        cv: Cross-validation folds — int (k-fold), CV generator, or
            iterable of splits. Default 5.
        scoring: Metric to maximize. ``str`` (e.g. ``"roc_auc"``,
            ``"neg_log_loss"``, ``"r2"``) or a callable. Default ``None``
            uses sklearn's per-estimator default (``accuracy`` for
            classifiers, ``r2`` for regressors).
        direction: ``"forward"`` (start empty, add features) or
            ``"backward"`` (start full, remove features). Backward is
            much more expensive but sometimes preferred when features
            are redundant.
        n_jobs: Parallelism over candidate features in each round.
            ``-1`` uses all cores. Default ``None`` is single-threaded.
        tol: Stop condition for auto selection — only used when
            ``n_features_to_select="auto"``. Forward selection stops
            when adding a feature improves the CV score by less than
            ``tol``.
        verbose: When ``True`` (default), print the pre- and
            post-selection CV scores and the names of the selected
            features. The scores are computed and returned regardless.
        **kwargs: Forwarded to ``SequentialFeatureSelector`` for forward
            compatibility with future sklearn options.

    Returns:
        FeatureSelectionResult: a dataclass with the fitted selector,
        the boolean support mask, the selected indices and names, and
        the baseline / selected CV scores.
    """
    if hasattr(estimator, "show_progress"):
        prev_show_progress = estimator.show_progress
        estimator.show_progress = False
        try:
            return _feature_selection(
                estimator,
                X,
                y,
                n_features_to_select=n_features_to_select,
                feature_names=feature_names,
                cv=cv,
                scoring=scoring,
                direction=direction,
                n_jobs=n_jobs,
                tol=tol,
                verbose=verbose,
                **kwargs,
            )
        finally:
            estimator.show_progress = prev_show_progress
    return _feature_selection(
        estimator,
        X,
        y,
        n_features_to_select=n_features_to_select,
        feature_names=feature_names,
        cv=cv,
        scoring=scoring,
        direction=direction,
        n_jobs=n_jobs,
        tol=tol,
        verbose=verbose,
        **kwargs,
    )


def _feature_selection(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_features_to_select: int | float | str,
    feature_names: list[str] | None,
    cv: int | BaseCrossValidator | Iterable,
    scoring: str | Callable | None,
    direction: str,
    n_jobs: int | None,
    tol: float | None,
    verbose: bool,
    **kwargs: Any,
) -> FeatureSelectionResult:
    """Internal implementation; ``feature_selection`` is the public entry."""
    scoring_desc = f"scoring={scoring!r}" if scoring is not None else "scoring=default"
    if verbose:
        print(  # noqa: T201
            f"Feature selection: direction={direction!r}, cv={cv}, "
            f"{scoring_desc}, n_features_to_select={n_features_to_select!r}"
        )

    # Baseline: how well does the model do with every feature available?
    baseline_scores = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    if verbose:
        print(  # noqa: T201
            f"Baseline CV score on all {X.shape[1]} features: "
            f"{baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}"
        )

    sfs_cls = _VerboseSFS if verbose else SequentialFeatureSelector
    sfs = sfs_cls(
        estimator,
        n_features_to_select=n_features_to_select,
        tol=tol,
        direction=direction,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        **kwargs,
    )
    if verbose:
        # Plain attributes, not constructor args — sklearn's get_params
        # introspects __init__ for parameter discovery and rejects extras.
        sfs._verbose_feature_names = feature_names  # type: ignore[attr-defined]
    sfs.fit(X, y)

    support_mask = sfs.get_support()
    selected_indices = [i for i, keep in enumerate(support_mask) if keep]
    selected_names = (
        [feature_names[i] for i in selected_indices]
        if feature_names is not None
        else None
    )

    selected_scores = cross_val_score(
        estimator,
        sfs.transform(X),
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    if verbose:
        display = selected_names if selected_names is not None else selected_indices
        print(  # noqa: T201
            f"Selected {sfs.n_features_to_select_} feature(s): {display}"
        )
        print(  # noqa: T201
            f"CV score on selected features: "
            f"{selected_scores.mean():.4f} ± {selected_scores.std():.4f}"
        )

    return FeatureSelectionResult(
        selector=sfs,
        support_mask=support_mask,
        selected_indices=selected_indices,
        selected_names=selected_names,
        baseline_score_mean=float(baseline_scores.mean()),
        baseline_score_std=float(baseline_scores.std()),
        selected_score_mean=float(selected_scores.mean()),
        selected_score_std=float(selected_scores.std()),
    )
