#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Simple, transductive missing-value imputation with TabPFN.

A lightweight alternative to :class:`TabPFNUnsupervisedModel.impute` that models
each column the canonical way: treat the column being imputed as the target and
*every other column* as features (missing values in those feature columns are
left in place — TabPFN handles NaNs natively), fit on the rows where the target
is observed, and predict the rows where it is missing.

Unlike the permutation-ensembled approach in :mod:`unsupervised`, this does a
single ``fit`` + ``predict`` per column with missing values. It is transductive:
it takes one table and figures out reference vs. query rows itself, so no
separate complete reference set is required. By default each column is imputed
using only the original observed data, so the result is independent of column
order; pass ``use_imputed=True`` for chained, left-to-right imputation where a
column filled earlier feeds into later columns.

This module intentionally depends only on ``numpy`` and the package's own model
helpers — not on :class:`TabPFNUnsupervisedModel`.

Example usage:
    ```python
    import numpy as np
    from tabpfn_extensions.unsupervised import simple_impute

    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
    X_imputed = simple_impute(X)
    ```
"""

from __future__ import annotations

import numpy as np

# Import TabPFN models from extensions (which handles backend compatibility)
from tabpfn_extensions.utils import (  # type: ignore
    TabPFNClassifier,
    TabPFNRegressor,
    get_max_num_classes,
    infer_categorical_features,
)


def impute_column(
    X: np.ndarray,
    col: int,
    model: object,
    *,
    classification: bool = False,
) -> np.ndarray:
    """Impute column ``col`` of ``X`` (``np.nan`` = missing) in place.

    Uses every other column as features (feature NaNs left in — TabPFN handles
    them), fits ``model`` on the rows where ``col`` is observed, and predicts the
    rows where it is missing.

    Parameters:
        X: Data of shape ``(n_samples, n_features)`` with missing values as ``np.nan``.
        col: Index of the column to impute.
        model: A fitted-on-call TabPFN estimator (regressor, or classifier when
            ``classification=True``).
        classification: If True, treat the target as categorical (fit on integer
            labels and predict class labels).

    Returns:
        np.ndarray: ``X`` with column ``col`` filled (mutated in place and returned).
    """
    missing = np.isnan(X[:, col])
    if not missing.any():
        return X
    if missing.all():
        raise ValueError(f"Column {col} is entirely NaN and cannot be imputed.")

    features = [c for c in range(X.shape[1]) if c != col]
    # context = rows where the target column is observed (real labels)
    # query   = rows where the target column is missing
    X_ctx, y_ctx = X[~missing][:, features], X[~missing, col]
    X_query = X[missing][:, features]

    model.fit(X_ctx, y_ctx.astype(int) if classification else y_ctx)
    X[missing, col] = model.predict(X_query)
    return X


def simple_impute(
    X: np.ndarray,
    tabpfn_clf: object | None = None,
    tabpfn_reg: object | None = None,
    categorical_features: list[int] | None = None,
    use_imputed: bool = False,
) -> np.ndarray:
    """Impute all missing values (``np.nan``) in ``X``, one column at a time.

    For each column that contains missing values, fit a TabPFN model on the rows
    where that column is observed and predict the missing rows, using all other
    columns as features. Numerical columns use a regressor; categorical columns
    use a classifier.

    Parameters:
        X: Data of shape ``(n_samples, n_features)`` with missing values encoded
            as ``np.nan``. Converted to a float array (a copy is made; the input
            is not modified).
        tabpfn_reg: TabPFN regressor used for numerical columns. Defaults to
            ``TabPFNRegressor()`` when needed.
        tabpfn_clf: TabPFN classifier used for categorical columns. Defaults to
            ``TabPFNClassifier()`` when needed.
        categorical_features: Indices of categorical columns. If None, they are
            inferred from the data.
        use_imputed: How to treat already-imputed columns when imputing later
            ones.

            * ``False`` (default): every column is imputed using only the
              *original* observed data as features, so the result does not depend
              on column order.
            * ``True``: columns are filled left to right in place, so a column
              imputed earlier is used as a feature for later columns (chained;
              order-dependent).

    Returns:
        np.ndarray: A new array with all missing values imputed.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    if X.shape[1] < 2:
        raise ValueError(
            "Input data X must have at least 2 columns to perform imputation."
        )
    n_features = X.shape[1]

    categorical_features = infer_categorical_features(X, categorical_features)

    # X stays the frozen feature source. Results are collected into X_imputed.
    # With use_imputed=True we fill X_imputed in place so later columns see
    # earlier imputations; with use_imputed=False we impute each column on a
    # throwaway copy of the original and keep only that column, so the feature
    # source never changes and the outcome is independent of column order.
    X_imputed = X.copy()

    for col in range(n_features):
        if not np.isnan(X[:, col]).any():
            continue

        # A categorical column is imputed with the classifier only when one is
        # available and it can predict that many classes; otherwise fall back to
        # the regressor (mirrors TabPFNUnsupervisedModel's routing).
        use_clf = col in categorical_features
        if use_clf:
            if tabpfn_clf is None:
                tabpfn_clf = TabPFNClassifier()
            max_classes = get_max_num_classes(tabpfn_clf)
            n_unique = len(np.unique(X[~np.isnan(X[:, col]), col]))
            use_clf = max_classes is None or n_unique <= max_classes

        if use_clf:
            model, classification = tabpfn_clf, True
        else:
            if tabpfn_reg is None:
                tabpfn_reg = TabPFNRegressor()
            model, classification = tabpfn_reg, False

        if use_imputed:
            # Chained: read features from and write into the same accumulating array.
            impute_column(X_imputed, col, model, classification=classification)
        else:
            # Independent: impute on a fresh copy of the original data and keep
            # only this column, leaving the feature source (X) untouched.
            col_work = X.copy()
            impute_column(col_work, col, model, classification=classification)
            X_imputed[:, col] = col_work[:, col]

    return X_imputed
