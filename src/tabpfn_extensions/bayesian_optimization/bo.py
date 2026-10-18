#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Bayesian optimization with TabPFN as the surrogate model.

TabPFN predicts a full bar distribution over the target, so acquisition
functions like Expected Improvement (EI) can be computed in closed form from
the predicted logits — no Gaussian assumption needed. This follows the
PFNs4BO approach (Mueller et al., ICML 2023, https://arxiv.org/abs/2305.17535,
https://github.com/automl/PFNs4BO), using the TabPFN foundation model as the
surrogate.

The regressor must be the full TabPFN package (>=8.1.0), because the
acquisitions read the predicted bar distribution. ``propose_next_point``
maximises EI over the unit hypercube ``[0, 1]^d`` with a gradient-free
sampled optimiser, so an ordinary ``TabPFNRegressor`` is enough; scale your
inputs accordingly. Only the optional gradient refinement
(``n_refine_steps > 0``) and ``expected_improvement`` itself need a regressor
constructed with ``differentiable_input=True``. EI maximizes the objective;
negate your objective values to minimize.
"""

from __future__ import annotations

import copy

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
    # The raw-space bar distribution can live on a different device than the
    # logits (TabPFN builds it on cpu in the differentiable-input path even
    # when the model runs on cuda). It is an nn.Module, so .to() moves its
    # borders buffer in place and the EI computation never mixes devices.
    bardist = reg.raw_space_bardist_.to(logits.device)
    return bardist.ei(logits, best_f, maximize=True)


def _aligned(out):
    """Return ``(logits, bardist)`` guaranteed to share device *and* dtype.

    TabPFN can hand back float64 bar-distribution buffers alongside logits we
    cast to float32, and the criterion's own arithmetic then multiplies the two
    together — ``variance`` does ``probs @ bucket_mean_of_square`` and raises
    "addmv input tensors must have the same dtype". Casting the module as well
    as the tensor keeps every acquisition path safe, not just the ones that
    happen to avoid the mixed-dtype matmul today.

    The criterion is the regressor's own module and ``nn.Module.to`` moves it
    in place, so it is copied first: moving the original to cpu would break the
    regressor's next ``predict`` on a cuda or mps device.
    """
    logits = out["logits"].detach().float().cpu()
    bardist = copy.deepcopy(out["criterion"]).to(device=logits.device).float()
    return logits, bardist


def _fit(reg: TabPFNRegressor, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
    """Fit through whichever path the regressor supports."""
    if getattr(reg, "differentiable_input", False):
        reg.fit_with_differentiable_input(train_x, train_y)
    else:
        reg.fit(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy())


def _ei_no_grad(reg: TabPFNRegressor, x: torch.Tensor, best_f: float) -> torch.Tensor:
    """EI for a candidate batch without gradients, for either kind of regressor.

    A ``differentiable_input=True`` regressor goes through ``forward``; a plain
    one through ``predict(output_type="full")``, which is also the path that
    lets it run in TabPFN's default (autocast) precision and with the full
    ensemble.
    """
    if getattr(reg, "differentiable_input", False):
        with torch.no_grad():
            return expected_improvement(reg, x, best_f)
    out = reg.predict(x.detach().cpu().numpy(), output_type="full")
    logits, bardist = _aligned(out)
    return bardist.ei(logits, best_f, maximize=True).to(x.device)


def propose_next_point(
    reg: TabPFNRegressor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    n_candidates: int = 2048,
    top_k: int = 16,
    n_local_rounds: int = 3,
    n_local: int = 32,
    local_sigma: float = 0.1,
    n_refine_steps: int = 0,
    refine_lr: float = 0.05,
) -> torch.Tensor:
    """One acquisition round: maximise exact EI over ``[0, 1]^d``.

    Fits ``reg`` on the observations, then maximises Expected Improvement with a
    gradient-free sampled optimiser: ``n_candidates`` uniform points are scored
    in one forward pass, and the ``top_k`` are refined for ``n_local_rounds``
    rounds by drawing ``n_local`` Gaussian proposals around each (standard
    deviation ``local_sigma`` in unit-cube coordinates, halved every round),
    clipped to the box, and keeping the best of parent and proposals.

    This is the optimiser used for TabPFN in our continuous BoTorch benchmark
    (eight synthetic functions, 32 seeds, 250 evaluations, protocol of Ament et
    al., NeurIPS 2023), where an 8-estimator TabPFN v3 with exact EI had the
    best mean rank at 250 evaluations. The same benchmark showed that BoTorch's
    GP under this sampled optimiser matches the GP under its own L-BFGS-B
    ``optimize_acqf``, so the surrogate, not the inner optimiser, is what
    separates the methods. Because nothing here needs gradients, ``reg`` can be
    an ordinary ``TabPFNRegressor`` with its full ensemble.

    ``n_refine_steps > 0`` adds the PFNs4BO-style polish from earlier versions:
    gradient ascent on EI w.r.t. the candidate coordinates, applied to the
    refined ``top_k`` after the sampled rounds. It requires a regressor built
    with ``differentiable_input=True`` and did not improve on the sampled
    optimiser in the benchmark, so it is off by default.

    Args:
        reg: A ``TabPFNRegressor``. Any construction works for the default
            sampled optimiser; ``differentiable_input=True`` is required only
            when ``n_refine_steps > 0``.
        train_x: Observed inputs of shape ``(n, d)``, scaled to ``[0, 1]^d``.
        train_y: Observed objective values of shape ``(n,)``.
        n_candidates: Uniform random candidates scored in the first pass.
        top_k: Candidates carried into local refinement.
        n_local_rounds: Rounds of Gaussian refinement around each survivor.
        n_local: Proposals drawn per survivor per round.
        local_sigma: Standard deviation of the proposals in the first round,
            in unit-cube coordinates; halved each round.
        n_refine_steps: Gradient-ascent steps on EI after the sampled rounds
            (``0`` disables; needs ``differentiable_input=True``).
        refine_lr: Learning rate of the gradient-ascent steps.

    Returns:
        The proposed next point of shape ``(d,)``, detached.
    """
    dim = train_x.shape[1]
    device = train_x.device
    _fit(reg, train_x, train_y)
    best_f = train_y.max().item()

    # Stage 1: screen a cheap batch of uniform candidates in one pass.
    cand_x = torch.rand(n_candidates, dim, device=device)
    ei = _ei_no_grad(reg, cand_x, best_f)
    keep = ei.topk(min(top_k, n_candidates)).indices
    top_x, top_ei = cand_x[keep], ei[keep]

    # Stage 2: Gaussian refinement around each survivor, shrinking every round.
    sigma = local_sigma
    for _ in range(n_local_rounds):
        noise = torch.randn(top_x.shape[0], n_local, dim, device=device) * sigma
        local_x = (top_x.unsqueeze(1) + noise).clamp_(0.0, 1.0)
        local_ei = _ei_no_grad(reg, local_x.reshape(-1, dim), best_f)
        local_ei = local_ei.reshape(top_x.shape[0], n_local)
        best_local = local_ei.argmax(dim=1)
        improved = local_ei.max(dim=1).values > top_ei
        picked = local_x[torch.arange(top_x.shape[0], device=device), best_local]
        top_x = torch.where(improved.unsqueeze(1), picked, top_x)
        top_ei = torch.maximum(local_ei.max(dim=1).values, top_ei)
        sigma *= 0.5

    if n_refine_steps <= 0:
        return top_x[top_ei.argmax()].detach()

    # Stage 3 (optional): gradient ascent on EI w.r.t. the candidate coordinates.
    if not getattr(reg, "differentiable_input", False):
        raise ValueError(
            "n_refine_steps > 0 needs a TabPFNRegressor constructed with "
            "differentiable_input=True; the default sampled optimiser does not.",
        )
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
