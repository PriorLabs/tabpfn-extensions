#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Bayesian optimization with TabPFN as the surrogate model.

TabPFN predicts a full bar distribution over the target, so acquisition
functions like Expected Improvement (EI) can be computed in closed form from
the predicted logits — no Gaussian assumption needed. This follows the
PFNs4BO approach (Mueller et al., ICML 2023, https://arxiv.org/abs/2305.17535,
https://github.com/automl/PFNs4BO), using the TabPFN foundation model as the
surrogate.

The regressor must be the full TabPFN package (>=8.1.0), constructed with
``differentiable_input=True`` so gradients flow from the acquisition value
back to the candidate coordinates. The search domain is the unit hypercube
``[0, 1]^d``; scale your inputs accordingly. EI maximizes the objective;
negate your objective values to minimize.
"""

from __future__ import annotations

import torch

try:
    from tabpfn import TabPFNRegressor
except ImportError as err:
    raise ImportError(
        "bayesian_optimization requires the full TabPFN package (>=8.1.0) and "
        "does not support the tabpfn-client backend. Install with "
        "'pip install \"tabpfn-extensions[bayesian_optimization]\"'."
    ) from err


def expected_improvement(
    reg: TabPFNRegressor,
    x: torch.Tensor,
    best_f: float,
) -> torch.Tensor:
    """EI over ``best_f`` for a batch of points, differentiable w.r.t. ``x``.

    ``forward`` returns bar-distribution logits as [N_borders, N_samples];
    after transposing, ``raw_space_bardist_.ei`` integrates the improvement
    over the predicted distribution in closed form. Because the raw-space
    borders are an affine rescaling of the z-normalized ones, the logits can
    be used with the raw-space criterion directly and ``best_f`` is passed in
    the original (unnormalized) target space.

    Args:
        reg: A fitted ``TabPFNRegressor`` with ``differentiable_input=True``.
        x: Candidate points of shape ``(n, d)``.
        best_f: Best objective value observed so far, in the original
            (unnormalized) target space.

    Returns:
        EI values of shape ``(n,)``.
    """
    averaged_logits, _outputs, _borders = reg.forward(x, use_inference_mode=True)
    logits = averaged_logits.transpose(0, 1).float()
    return reg.raw_space_bardist_.ei(logits, best_f, maximize=True)


def propose_next_point(
    reg: TabPFNRegressor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    n_candidates: int = 512,
    top_k: int = 4,
    n_refine_steps: int = 8,
    refine_lr: float = 0.05,
) -> torch.Tensor:
    """One acquisition round: screen random candidates, refine the best by EI ascent.

    Fits ``reg`` on the observations via ``fit_with_differentiable_input``,
    screens ``n_candidates`` uniform random points in ``[0, 1]^d`` with EI in
    a single forward pass, then refines the ``top_k`` most promising ones by
    gradient ascent on EI w.r.t. the candidate coordinates.

    Args:
        reg: A ``TabPFNRegressor`` constructed with ``differentiable_input=True``.
        train_x: Observed inputs of shape ``(n, d)``, scaled to ``[0, 1]^d``.
        train_y: Observed objective values of shape ``(n,)``.
        n_candidates: Number of random candidates screened per round.
        top_k: Number of candidates refined by gradient ascent on EI.
        n_refine_steps: Gradient steps on the candidate coordinates.
        refine_lr: Learning rate for the refinement steps.

    Returns:
        The proposed next point of shape ``(d,)``, detached.
    """
    dim = train_x.shape[1]
    device = train_x.device
    reg.fit_with_differentiable_input(train_x, train_y)
    best_f = train_y.max().item()

    # Stage 1: screen a cheap batch of random candidates in one forward pass.
    with torch.no_grad():
        cand_x = torch.rand(n_candidates, dim, device=device)
        ei = expected_improvement(reg, cand_x, best_f)
        top_x = cand_x[ei.topk(min(top_k, n_candidates)).indices]

    # Stage 2: gradient ascent on EI w.r.t. the candidate coordinates.
    refine_x = top_x.clone().requires_grad_(requires_grad=True)
    optimizer = torch.optim.Adam([refine_x], lr=refine_lr)
    for _ in range(n_refine_steps):
        optimizer.zero_grad()
        loss = -expected_improvement(reg, refine_x, best_f).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            refine_x.clamp_(0.0, 1.0)  # stay inside the search domain

    with torch.no_grad():
        ei_refined = expected_improvement(reg, refine_x, best_f)
        return refine_x[ei_refined.argmax()].detach()
