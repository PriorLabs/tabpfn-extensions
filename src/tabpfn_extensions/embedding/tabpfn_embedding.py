from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor


@set_extension("embedding")
class TabPFNEmbedding(TransformerMixin, BaseEstimator):
    """scikit-learn style transformer that extracts TabPFN embeddings.

    When ``n_fold >= 2``, ``fit`` produces out-of-fold (OOF) embeddings for the
    training data — the robust variant from "A Closer Look at TabPFN v2:
    Strength, Limitation, and Extension" (https://arxiv.org/abs/2502.17361) —
    and then refits a single model on the full training set for use on unseen
    data. The OOF embeddings are stored on ``train_embeddings_`` and returned
    by ``fit_transform``.

    ``transform(X)`` ALWAYS uses the final, full-data model — it does NOT
    return cached OOF embeddings, even when ``X`` happens to equal the
    training set. For OOF embeddings call ``fit_transform`` (or read
    ``train_embeddings_``).

    Note on output shape: ``transform`` returns a 3D array of shape
    ``(n_estimators, n_samples, embed_dim)``. It is not a drop-in input for
    ``sklearn.pipeline.Pipeline`` / ``ColumnTransformer`` — those expect 2D
    output. Pick an ensemble member (``embeds[0]``) or aggregate across
    ``axis=0`` before passing to a downstream 2D estimator.

    Parameters
    ----------
    n_fold : int, default=0
        Number of folds for cross-validation. ``0`` disables CV — the model
        is trained once on the entire training set and used for both train
        and unseen data.
    model : TabPFNClassifier or TabPFNRegressor, optional
        Pre-configured TabPFN estimator. When ``None``, the task is inferred
        from ``y`` at ``fit`` time and a warning is emitted.
    shuffle : bool, default=False
        Whether to shuffle the K-fold split. Independent of ``random_state``.
    random_state : int, optional
        Seed used by the K-fold split when ``shuffle=True``.
    tabpfn_clf : TabPFNClassifier, optional
        DEPRECATED. Use ``model=`` instead.
    tabpfn_reg : TabPFNRegressor, optional
        DEPRECATED. Use ``model=`` instead.

    Attributes:
    ----------
    model_ : TabPFNClassifier or TabPFNRegressor
        The fitted TabPFN model (cloned from ``model`` or auto-constructed).
        After ``fit`` with ``n_fold >= 2`` this is the model trained on the
        full training set.
    train_embeddings_ : np.ndarray
        Embeddings for the training set. For ``n_fold >= 2`` these are OOF
        embeddings aligned to the original sample order; for ``n_fold == 0``
        they are produced by the single full-data model.

    Examples:
    --------
    >>> from tabpfn_extensions.embedding import TabPFNEmbedding
    >>> embedding = TabPFNEmbedding(n_fold=5)
    >>> train_embeds = embedding.fit_transform(X_train, y_train)  # OOF
    >>> test_embeds = embedding.transform(X_test)                 # final model
    """

    def __init__(
        self,
        n_fold: int = 0,
        *,
        model: TabPFNClassifier | TabPFNRegressor | None = None,
        shuffle: bool = False,
        random_state: int | None = None,
        tabpfn_clf: TabPFNClassifier | None = None,
        tabpfn_reg: TabPFNRegressor | None = None,
    ) -> None:
        self.n_fold = n_fold
        self.model = model
        self.shuffle = shuffle
        self.random_state = random_state
        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg

    def _resolve_template(self, y: np.ndarray):
        """Return a fresh TabPFN model to use, warning on legacy kwargs."""
        legacy = self.tabpfn_clf if self.tabpfn_clf is not None else self.tabpfn_reg
        if legacy is not None:
            warnings.warn(
                "`tabpfn_clf` and `tabpfn_reg` are deprecated; pass `model=` "
                "instead (or omit it to auto-detect classifier vs regressor "
                "from y).",
                DeprecationWarning,
                stacklevel=3,
            )
        template = self.model if self.model is not None else legacy
        if template is not None:
            return clone(template)

        target_type = type_of_target(y)
        if target_type in ("binary", "multiclass"):
            inferred_cls = TabPFNClassifier
            inferred_name = "TabPFNClassifier"
        elif target_type == "continuous":
            inferred_cls = TabPFNRegressor
            inferred_name = "TabPFNRegressor"
        else:
            raise ValueError(
                f"Could not infer task from target type '{target_type}'. "
                "Pass an explicit `model=` (TabPFNClassifier or TabPFNRegressor).",
            )
        warnings.warn(
            f"No `model=` provided; inferred {inferred_name} from y "
            f"(type_of_target='{target_type}') with n_estimators=1. "
            "Integer-valued regression targets are detected as "
            "classification — pass `model=` explicitly to avoid silent "
            "misclassification of the task or to use a different "
            "n_estimators.",
            UserWarning,
            stacklevel=3,
        )
        return inferred_cls(n_estimators=1, random_state=self.random_state)

    def _compute_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        stratify: bool,
        shuffle: bool,
        random_state: int | None,
    ) -> np.ndarray:
        """Run K-fold and return OOF embeddings aligned to original order."""
        X = np.asarray(X)
        y = np.asarray(y)

        rs = random_state if shuffle else None
        if stratify:
            cv = StratifiedKFold(n_splits=self.n_fold, shuffle=shuffle, random_state=rs)
            splits = cv.split(X, y)
        else:
            cv = KFold(n_splits=self.n_fold, shuffle=shuffle, random_state=rs)
            splits = cv.split(X)

        chunks: list[np.ndarray] = []
        val_indices: list[np.ndarray] = []
        for train_idx, val_idx in splits:
            fold_model = clone(self.model_)
            fold_model.fit(X[train_idx], y[train_idx])
            chunks.append(fold_model.get_embeddings(X[val_idx], data_source="test"))
            val_indices.append(val_idx)

        oof = np.concatenate(chunks, axis=1)
        order = np.argsort(np.concatenate(val_indices))
        return oof[:, order, ...]

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabPFNEmbedding:
        if self.n_fold < 0 or self.n_fold == 1:
            raise ValueError("n_fold must be 0 (vanilla) or >= 2.")

        self.model_ = self._resolve_template(y)
        self._is_classifier_ = isinstance(self.model_, TabPFNClassifier)

        if self.n_fold == 0:
            self.model_.fit(X, y)
            self.train_embeddings_ = self.model_.get_embeddings(
                X,
                data_source="test",
            )
            return self

        self.train_embeddings_ = self._compute_oof(
            X,
            y,
            stratify=self._is_classifier_,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed unseen data ``X`` using the full-data model.

        Use this method when you have new, held-out data that was not part of
        training.  It always runs inference through ``model_`` (trained on the
        full training set) and never returns cached embeddings.

        If you want embeddings for the *training* data, prefer
        ``fit_transform(X_train, y_train)``, which yields out-of-fold
        embeddings for ``n_fold >= 2`` (avoiding label leakage) or reads
        ``train_embeddings_`` after a ``fit`` call.
        """
        check_is_fitted(self, "model_")
        return self.model_.get_embeddings(X, data_source="test")

    def fit_transform(  # type: ignore[override]
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit and return embeddings for the training data.

        For ``n_fold >= 2`` these are out-of-fold embeddings. For
        ``n_fold == 0`` they come from the single full-data model.
        """
        self.fit(X, y)
        return self.train_embeddings_

    def get_embeddings(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X: np.ndarray,
        data_source: str,
    ) -> np.ndarray:
        """DEPRECATED. Use ``fit_transform`` (OOF) or ``transform`` (unseen)."""
        warnings.warn(
            "TabPFNEmbedding.get_embeddings(X_train, y_train, X, data_source) "
            "is deprecated and will be removed in a future release. Use "
            "`embedding.fit_transform(X_train, y_train)` for OOF training "
            "embeddings or `embedding.fit(X_train, y_train).transform(X)` for "
            "unseen data. Note: the new `fit` path uses StratifiedKFold for "
            "classifiers (this legacy path uses plain KFold), so OOF numbers "
            "may differ.",
            DeprecationWarning,
            stacklevel=2,
        )
        template = (
            self.model
            if self.model is not None
            else (self.tabpfn_clf or self.tabpfn_reg)
        )
        if template is None:
            raise ValueError("No model has been set.")

        self.model_ = clone(template)
        self._is_classifier_ = isinstance(self.model_, TabPFNClassifier)

        if self.n_fold == 0:
            self.model_.fit(X_train, y_train)
            return self.model_.get_embeddings(X, data_source=data_source)
        if self.n_fold >= 2:
            if data_source == "test":
                self.model_.fit(X_train, y_train)
                return self.model_.get_embeddings(X, data_source=data_source)
            return self._compute_oof(
                X_train,
                y_train,
                stratify=False,
                shuffle=False,
                random_state=None,
            )
        raise ValueError("n_fold must be greater than 1.")


__all__ = ["TabPFNEmbedding"]
