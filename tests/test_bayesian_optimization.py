from __future__ import annotations

import pytest
import torch

bo = pytest.importorskip(
    "tabpfn_extensions.bayesian_optimization",
    reason="bayesian_optimization requires the full TabPFN package",
)

from tabpfn import TabPFNRegressor  # noqa: E402

if not hasattr(TabPFNRegressor, "fit_with_differentiable_input"):
    pytest.skip(
        "bayesian_optimization requires tabpfn>=8.1.0 "
        "(fit_with_differentiable_input)",
        allow_module_level=True,
    )

DIM = 3


@pytest.fixture(scope="module")
def train_data():
    torch.manual_seed(0)
    train_x = torch.rand(8, DIM)
    train_y = train_x.sum(dim=1)
    return train_x, train_y


def make_regressor() -> TabPFNRegressor:
    return TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        random_state=0,
        inference_precision=torch.float32,
        differentiable_input=True,
    )


def test_expected_improvement_is_nonnegative_and_differentiable(train_data):
    train_x, train_y = train_data
    reg = make_regressor()
    reg.fit_with_differentiable_input(train_x, train_y)

    x = torch.rand(4, DIM, requires_grad=True)
    ei = bo.expected_improvement(reg, x, train_y.max().item())

    assert ei.shape == (4,)
    assert torch.all(ei >= 0)

    ei.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_propose_next_point_stays_in_unit_cube(train_data):
    train_x, train_y = train_data
    reg = make_regressor()

    next_x = bo.propose_next_point(
        reg,
        train_x,
        train_y,
        n_candidates=16,
        top_k=2,
        n_refine_steps=1,
    )

    assert next_x.shape == (DIM,)
    assert not next_x.requires_grad
    assert torch.all(next_x >= 0)
    assert torch.all(next_x <= 1)
