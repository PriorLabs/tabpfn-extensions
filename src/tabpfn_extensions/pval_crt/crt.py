from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError as err:
    raise ImportError(
        "pval_crt requires the full TabPFN package and does not support "
        "the tabpfn-client backend."
    ) from err

from tabpfn.constants import ModelVersion

from .utils import (
    coerce_X_y_to_numpy,
    is_categorical,
    logp_from_full_output,
    logp_from_proba,
    resolve_feature_index,
)


def tabpfn_crt(
    X: Any,
    y: Any,
    j: int | str | Sequence[int | str],
    *,
    B: int = 200,
    alpha: float = 0.05,
    test_size: float = 0.2,
    seed: int = 0,
    device: str | None = None,
    K: int = 100,
    max_unique_cat: int = 10,
    model_version: ModelVersion = ModelVersion.V2,
) -> dict[str, Any] | dict[int | str, dict[str, Any]]:
    """Conditional Randomization Test (CRT) using TabPFN.

    This function tests whether one or more features contain predictive
    information about the target variable y beyond the remaining covariates.

    For each feature X_j, the CRT compares:

        Observed statistic:
            Mean log predictive density of y given the observed X.

        Null statistic:
            Same quantity when X_j is replaced by samples drawn from
            p(X_j | X_-j).

    Efficiency
    ----------
    The predictive model p(y | X) is fit ONLY ONCE and reused across all
    tested features. Each feature requires fitting only the conditional
    model p(X_j | X_-j).

    Supports:
        • single feature testing
        • batch feature testing

    Returns:
    -------
    Single feature → result dict
    Multiple features → dict[feature → result dict]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------
    # Input normalization
    # ---------------------------
    X_np, y_np, feature_names = coerce_X_y_to_numpy(X, y)

    # ---------------------------
    # Train / evaluation split
    # ---------------------------
    # IMPORTANT:
    # We intentionally do NOT use stratified splitting. Conditional
    # Randomization Tests rely on IID sampling assumptions. Stratifying
    # by the target variable conditions the evaluation distribution on Y
    # and may invalidate CRT p-values.

    X_tr, X_ev, y_tr, y_ev = train_test_split(
        X_np,
        y_np,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=None,   # Explicitly enforced
    )

    # ---------------------------
    # Fit predictive model p(y|X) ONCE
    # ---------------------------
    y_is_cat = is_categorical(y_tr, max_unique_cat)
    ModelY = TabPFNClassifier if y_is_cat else TabPFNRegressor

    model_y = ModelY.create_default_for_version(
        model_version,
        device=device,
    )

    model_y.fit(X_tr, y_tr)

    # Pre-compute baseline log predictive density
    if y_is_cat:
        probs_plus = model_y.predict_proba(X_ev)
        logp_plus = logp_from_proba(probs_plus, y_ev, model_y.classes_)
    else:
        full_plus = model_y.predict(X_ev, output_type="full")
        logp_plus = logp_from_full_output(full_plus, y_ev)

    # ---------------------------
    # Observed T_obs
    # ---------------------------
    T_obs_global = np.mean(logp_plus)

    # ---------------------------
    # Multi-feature support
    # ---------------------------
    if isinstance(j, Sequence) and not isinstance(j, (str, bytes)):
        results = {}

        for feat in j:
            j_idx, feature_name = resolve_feature_index(
                feat, feature_names, X_np.shape[1]
            )

            res = _tabpfn_crt_single(
                X_tr=X_tr,
                X_ev=X_ev,
                y_ev=y_ev,
                model_y=model_y,
                T_obs=T_obs_global,
                j_idx=j_idx,
                feature_name=feature_name,
                B=B,
                alpha=alpha,
                device=device,
                K=K,
                max_unique_cat=max_unique_cat,
                y_is_cat=y_is_cat,
                seed=seed,
                model_version=model_version,
            )

            key = feature_name if feature_name is not None else j_idx
            results[key] = res

        return results

    # ---------------------------
    # Single feature
    # ---------------------------
    j_idx, feature_name = resolve_feature_index(j, feature_names, X_np.shape[1])

    return _tabpfn_crt_single(
        X_tr=X_tr,
        X_ev=X_ev,
        y_ev=y_ev,
        model_y=model_y,
        T_obs=T_obs_global,
        j_idx=j_idx,
        feature_name=feature_name,
        B=B,
        alpha=alpha,
        device=device,
        K=K,
        max_unique_cat=max_unique_cat,
        y_is_cat=y_is_cat,
        seed=seed,
        model_version=model_version,
    )

def _tabpfn_crt_single(
    *,
    X_tr,
    X_ev,
    y_ev,
    model_y,
    T_obs,
    j_idx,
    feature_name,
    B,
    alpha,
    device,
    K,
    max_unique_cat,
    y_is_cat,
    seed,
    model_version,
):
    """Execute the Conditional Randomization Test (CRT) for a single feature.

    This helper assumes that the main CRT preparation steps have already
    been completed, including:

        • Train/evaluation data split
        • Fitting the predictive model p(y | X)
        • Computing the observed test statistic T_obs

    The helper performs only the feature-specific portion of the CRT:

        1. Fits the conditional model p(X_j | X_-j)
        2. Generates B null datasets by resampling X_j from the conditional model
        3. Recomputes the predictive log-density using the fixed predictive model
        4. Estimates the CRT p-value

    Parameters
    ----------
    X_tr : ndarray of shape (n_train, n_features)
        Training feature matrix used to fit the conditional model p(X_j | X_-j).

    X_ev : ndarray of shape (n_eval, n_features)
        Evaluation feature matrix used for computing CRT statistics.

    y_ev : ndarray of shape (n_eval,)
        Evaluation targets used when computing predictive log densities.

    model_y : TabPFNClassifier or TabPFNRegressor
        Pre-fitted predictive model approximating p(y | X).
        This model is treated as fixed during CRT resampling.

    T_obs : float
        Observed test statistic computed from the original evaluation data.

    j_idx : int
        Index of the feature being tested.

    feature_name : str or None
        Optional feature name used for reporting and output labeling.

    B : int
        Number of CRT resamples used to approximate the null distribution.

    alpha : float
        Significance level used to determine rejection of the null hypothesis.

    device : {"cpu", "cuda"}
        Device used when fitting the conditional TabPFN model.

    K : int
        Number of quantile levels used when approximating continuous
        conditional distributions via TabPFN quantile regression.

    max_unique_cat : int
        Maximum number of unique values for a feature to be treated as categorical.

    y_is_cat : bool
        Indicates whether the target variable is modeled using classification.

    seed : int
        Base random seed used to generate feature-specific reproducible
        CRT resampling streams.

    Returns:
    -------
    dict
        Dictionary containing CRT results for the tested feature, including:

        p_value : float
            CRT p-value for conditional independence of X_j and y.

        reject_null : bool
            Whether the null hypothesis is rejected at level alpha.

        T_obs : float
            Observed test statistic.

        T_null : ndarray of shape (B,)
            Null distribution of the test statistic.

        interpretation : str
            Human-readable interpretation of the test result.

        y_is_categorical : bool
            Whether the target was modeled as categorical.

        xj_is_categorical : bool
            Whether the tested feature was modeled as categorical.

        feature_index : int
            Index of the tested feature.

        feature_name : str or None
            Optional name of the tested feature.

    Notes:
    -----
    • The predictive model p(y | X) is NOT refit during CRT resampling.
    • The test is right-tailed: larger predictive log-density indicates
      stronger evidence of feature relevance.
    • A feature-specific RNG stream is used to guarantee reproducibility
      independent of feature ordering.
    """
    rng_feature = np.random.RandomState(seed + j_idx)

    # ---------------------------
    # Model for Xj | X_-j
    # ---------------------------
    Xm_tr = np.delete(X_tr, j_idx, axis=1)
    Xm_ev = np.delete(X_ev, j_idx, axis=1)
    xj_tr = X_tr[:, j_idx]

    # ---------------------------
    # Fit conditional model p(X_j | X_-j)
    # ---------------------------
    xj_is_cat = is_categorical(xj_tr, max_unique_cat)
    ModelXJ = TabPFNClassifier if xj_is_cat else TabPFNRegressor

    model_xj = ModelXJ.create_default_for_version(
        model_version,
        device=device,
    )
    model_xj.fit(Xm_tr, xj_tr)

    # ---------------------------
    # Precompute conditional sampler
    # ---------------------------
    if not xj_is_cat:
        q_grid = np.linspace(0, 1, K)
        Q = np.asarray(model_xj.predict(Xm_ev,output_type="quantiles",quantiles=q_grid))
        if Q.shape[0] != K:
            Q = Q.T  # ensure (K, n_ev)
    else:
        probs = model_xj.predict_proba(Xm_ev)
        cdf = np.cumsum(probs, axis=1)

        if not np.all(np.isfinite(probs)):
            i = np.argwhere(~np.isfinite(probs))[0][0]
            raise ValueError(f"bad probs: non-finite probabilities found at row {i}: {probs[i]}")
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            i = np.argmax(np.abs(row_sums - 1.0))
            raise ValueError(
                f"probabilities not normalized: bad prob sum at row {i}: {row_sums[i]}, probs: {probs[i]}"
            )

    # ---------------------------
    # Null distribution
    # ---------------------------
    T_null = np.zeros(B)
    n_ev = X_ev.shape[0]
    X_ev_null = X_ev.copy()

    for b in range(B):
        if xj_is_cat:
            u = rng_feature.rand(n_ev, 1)
            idx = (u <= cdf).argmax(axis=1)
            xj_null = model_xj.classes_[idx]
        else:
            idx = rng_feature.randint(0, K, size=n_ev)
            xj_null = Q[idx, np.arange(n_ev)]

            bad = ~np.isfinite(xj_null)
            max_resample = 10
            n_try = 0

            while bad.any():
                if n_try >= max_resample:
                    raise RuntimeError(
                        f"CRT quantile sampling produced non-finite values after {max_resample} retries"
                    )

                # resample ONLY the bad positions
                idx_bad = rng_feature.randint(0, K, size=bad.sum())
                xj_null[bad] = Q[idx_bad, np.where(bad)[0]]

                bad = ~np.isfinite(xj_null)
                n_try += 1

        X_ev_null[:, j_idx] = np.asarray(xj_null)

        if y_is_cat:
            probs_null = model_y.predict_proba(X_ev_null)
            logp_null = logp_from_proba(probs_null, y_ev, model_y.classes_)
        else:
            full_null = model_y.predict(X_ev_null, output_type="full")
            logp_null = logp_from_full_output(full_null, y_ev)

        T_null[b] = np.mean(logp_null)

    # Compute right-tailed p-value
    p_value = float((1 + np.sum(T_null >= T_obs)) / (B + 1))

    # ---------------------------
    # Human-readable interpretation
    # ---------------------------
    reject = p_value <= alpha

    feat_label = feature_name if feature_name is not None else j_idx

    if reject:
        relevance_stmt = (
            f"Result: REJECT H0 at alpha = {alpha:.2f}.\n"
            f"Interpretation: The variable X[{feat_label}] provides information about the "
            f"target Y that is not explained by the remaining covariates."
        )
    else:
        relevance_stmt = (
            f"Result: FAIL TO REJECT H0 at alpha = {alpha:.2f}.\n"
            f"Interpretation: There is no evidence that the variable X[{feat_label}] provides additional "
            f"information about the target Y beyond the remaining covariates."
        )

    return {
        "p_value": float(p_value),
        "reject_null": bool(reject),
        "alpha": alpha,
        "T_obs": T_obs,
        "T_null": T_null,
        "interpretation": relevance_stmt,
        "y_is_categorical": y_is_cat,
        "xj_is_categorical": xj_is_cat,
        "B": B,
        "K": K if not xj_is_cat else None,
        "feature_index": int(j_idx),
        "feature_name": feature_name,
    }