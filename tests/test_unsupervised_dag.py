"""Tests for DAG-conditioned synthesis / imputation.

The pure-Python ordering / cycle / mutation tests run instantly and need no
TabPFN model. The end-to-end synthesis test uses ``FAST_TEST_MODE=1`` and
``n_estimators=1`` so it stays in the per-test budget on the existing CI.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised
from tabpfn_extensions.unsupervised.unsupervised import _resolve_dag_order

# ─────────────────────────── helper-level tests ──────────────────────────────


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_resolve_dag_order_basic():
    """Topological order respects parent → child."""
    dag = {0: [], 1: [0], 2: [1, 0]}
    ordered, full = _resolve_dag_order(dag, [0, 1, 2])
    assert ordered.index(0) < ordered.index(1)
    assert ordered.index(0) < ordered.index(2)
    assert ordered.index(1) < ordered.index(2)
    assert full == {0: [], 1: [0], 2: [1, 0]}


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_resolve_dag_order_partial():
    """Features with no explicit deps get empty parents."""
    ordered, full = _resolve_dag_order({2: [0, 1]}, [0, 1, 2, 3])
    assert ordered.index(2) > ordered.index(0)
    assert ordered.index(2) > ordered.index(1)
    assert set(ordered) == {0, 1, 2, 3}
    assert full[0] == []
    assert full[3] == []


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_resolve_dag_order_cycle_raises_valueerror():
    """Cycles surface as ValueError, not stdlib CycleError traceback."""
    with pytest.raises(ValueError, match="cycle"):
        _resolve_dag_order({0: [1], 1: [0]}, [0, 1])


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_resolve_dag_order_rejects_unknown_indices():
    """Out-of-range children/parents fail loudly, not via a later IndexError."""
    with pytest.raises(ValueError, match="unknown feature indices"):
        _resolve_dag_order({0: [], 1: [9]}, [0, 1, 2])
    with pytest.raises(ValueError, match="unknown feature indices"):
        _resolve_dag_order({5: [0]}, [0, 1, 2])


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_resolve_dag_order_does_not_mutate_caller_dict():
    """Filling in empty deps must not write into the caller's dict."""
    user_dag = {2: [0, 1]}
    snapshot = {k: list(v) for k, v in user_dag.items()}
    _resolve_dag_order(user_dag, [0, 1, 2, 3])
    assert user_dag == snapshot, "caller's DAG dict was mutated"


# ─────────────────────────── public API tests ────────────────────────────────


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_impute_rejects_dag_with_condition_on_all_features():
    """The two modes are mutually exclusive — must raise a clear ValueError."""
    X = torch.tensor(np.random.rand(5, 3), dtype=torch.float32)
    model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(n_estimators=1),
        tabpfn_reg=TabPFNRegressor(n_estimators=1),
    )
    model.fit(X)
    with pytest.raises(ValueError, match="mutually exclusive"):
        model.impute_(
            X,
            condition_on_all_features=True,
            dag={0: [], 1: [0], 2: [0, 1]},
        )


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_data_with_dag(monkeypatch):
    """End-to-end: generation with a DAG returns the right shape."""
    monkeypatch.setenv("FAST_TEST_MODE", "1")

    X = np.random.rand(5, 3).astype(np.float32)
    X_tensor = torch.tensor(X)
    model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(n_estimators=1),
        tabpfn_reg=TabPFNRegressor(n_estimators=1),
    )
    model.fit(X_tensor)

    # 0 is independent; 1 depends on 0; 2 depends on 0 and 1
    dag = {0: [], 1: [0], 2: [0, 1]}
    synthetic_X = model.generate_synthetic_data(n_samples=5, dag=dag)

    assert isinstance(synthetic_X, torch.Tensor)
    assert synthetic_X.shape == (5, 3)


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_impute_with_dag(monkeypatch):
    """End-to-end: imputation with a DAG fills every NaN and keeps the shape.

    Uses a different DAG shape from the synthesis test above — a fork/diamond
    (two children of a shared root that both feed a final feature) rather than
    a simple chain — to exercise a column conditioned on multiple parents.
    """
    monkeypatch.setenv("FAST_TEST_MODE", "1")

    rng = np.random.default_rng(0)
    X = rng.random((6, 4)).astype(np.float32)
    model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(n_estimators=1),
        tabpfn_reg=TabPFNRegressor(n_estimators=1),
    )
    # Fit on the complete data; impute a copy with missing values (the
    # recommended workflow — the fitting context stays fully observed).
    model.fit(torch.tensor(X))

    X_missing = X.copy()
    X_missing[1, 2] = np.nan
    X_missing[4, 3] = np.nan
    X_missing[5, 1] = np.nan

    # 0 is a root; 1 and 2 both depend on 0; 3 depends on both 1 and 2.
    dag = {0: [], 1: [0], 2: [0], 3: [1, 2]}
    imputed = model.impute(torch.tensor(X_missing), dag=dag)

    assert isinstance(imputed, torch.Tensor)
    assert imputed.shape == (6, 4)
    assert not torch.isnan(imputed).any()
