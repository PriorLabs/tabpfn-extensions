import numpy as np
import torch
import pandas as pd

from typing import Any, Optional, Sequence, Tuple, Union

FeatureSpec = Union[int, str]

def coerce_X_y_to_numpy(
    X: Any,
    y: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[Sequence[str]]]:
    """
    Convert X and y to numpy arrays while preserving feature names if X is a DataFrame.
    """
    feature_names = None

    if pd is not None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_np = X.to_numpy()
    else:
        X_np = np.asarray(X)

    if pd is not None and isinstance(y, (pd.Series, pd.DataFrame)):
        y_np = np.asarray(y).reshape(-1)
    else:
        y_np = np.asarray(y).reshape(-1)

    return X_np, y_np, feature_names

def resolve_feature_index(
    j: FeatureSpec,
    feature_names: Optional[Sequence[str]],
    n_features: int,
) -> Tuple[int, Optional[str]]:
    """
    Resolve feature identifier j into integer index and optional name.
    """
    if isinstance(j, int):
        if j < 0 or j >= n_features:
            raise IndexError(
                f"Feature index j={j} out of bounds for n_features={n_features}."
            )
        name = feature_names[j] if feature_names is not None else None
        return j, name

    if isinstance(j, str):
        if feature_names is None:
            raise TypeError(
                "Feature name given but X has no column names. "
                "Pass a DataFrame or use an integer index."
            )
        if j not in feature_names:
            raise KeyError(f"Feature name '{j}' not found in X.")
        idx = feature_names.index(j)
        return idx, j

    raise TypeError("j must be int or str.")

def is_categorical(arr, max_unique=10):
    """
    Heuristic to determine whether a variable should be treated as categorical.

    Parameters
    ----------
    arr : array-like
        Input array.
    max_unique : int
        Maximum number of unique values for categorical treatment.

    Returns
    -------
    bool
        True if categorical, False otherwise.
    """
    arr = np.asarray(arr)
    uniq = np.unique(arr[~np.isnan(arr)])
    return len(uniq) <= max_unique


def logp_from_full_output(full_out, y_np):
    """
    Extract log predictive density from TabPFN 'full' prediction output.

    Parameters
    ----------
    full_out : dict
        Output from TabPFN predict(..., output_type="full").
    y_np : array-like
        Ground-truth targets.

    Returns
    -------
    np.ndarray
        Log predictive density for each observation.
    """
    criterion = full_out["criterion"]
    logits = full_out["logits"]

    y_torch = torch.as_tensor(
        y_np,
        device=logits.device,
        dtype=logits.dtype,
    ).view(*logits.shape[:-1])

    nll = criterion(logits, y_torch)
    return (-nll).detach().cpu().numpy().reshape(-1)


def logp_from_proba(probs, y_true, classes):
    """
    Compute log p(y_true | x) from class probabilities.

    probs: shape (n, C)
    y_true: shape (n,)
    classes: model.classes_
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([class_to_idx[y] for y in y_true])
    p = probs[np.arange(len(y_true)), idx]
    return np.log(np.clip(p, 1e-12, None))
