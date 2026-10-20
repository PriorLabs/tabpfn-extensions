#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Bayesian optimization with TabPFN as the surrogate model.

TabPFN predicts a full bar distribution over the target, so Expected
Improvement (EI) is computed exactly from the predicted logits -- no Gaussian
assumption, no moment matching. This follows the PFNs4BO idea (Mueller et al.,
ICML 2023, https://arxiv.org/abs/2305.17535) with the TabPFN foundation model
as the surrogate.

Each iteration calls ``tabpfn_extensions.bayesian_optimization.propose_next_point``,
which fits TabPFN on the points evaluated so far, scores a large batch of
random candidates with exact EI in one forward pass, and refines the best of
them by sampling around them with a shrinking Gaussian. It needs no gradients,
so an ordinary ``TabPFNRegressor`` with its full ensemble is used.

The settings here are the ones we validated on the continuous BoTorch suite
(eight synthetic functions, 32 seeds, 250 evaluations each, protocol of Ament
et al., NeurIPS 2023): an 8-estimator TabPFN v3 with exact EI had the best
mean rank at 250 evaluations against BoTorch's GP with EI and LogEI. The split
is by function shape. On multimodal functions with steep, narrow valleys
(Michalewicz, Ackley) TabPFN leads by an order of magnitude; on smooth
single-basin functions (Branin, Hartmann-3) the GP is ahead early and wins on
final precision at regrets around 1e-5. Michalewicz-2 below is one where
TabPFN is ahead at this budget on most initial designs, so the difference is
visible in a single run. Do not read one seed as a benchmark; the cookbook
repeats the experiment over seeds.

For comparison the loop also runs BoTorch's default GP with LogEI and random
search, all three from the same initial design.

NOTE: This example requires the extras of the bayesian_optimization
extension: pip install "tabpfn-extensions[bayesian_optimization]". It needs
the full TabPFN implementation, version 8.4.0 or later, and does not work with
the TabPFN client, because it reads the predicted bar distribution. BoTorch
provides the GP baseline and the benchmark objective.

Runs in about three minutes on CPU with tabpfn 8.4 (the TabPFN rounds
dominate; 9.0 is roughly twice as slow per round), faster on a CUDA GPU. Keep
TabPFN on cpu or cuda -- Apple's mps backend is an order of magnitude slower
here.
"""

import os

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Michalewicz
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange

from tabpfn import TabPFNRegressor
from tabpfn_extensions.bayesian_optimization import propose_next_point

# Under FAST_TEST_MODE=1 (set by the CI example tests) the workload shrinks so
# the script only has to prove it runs end to end.
FAST_TEST_MODE = os.environ.get("FAST_TEST_MODE") == "1"

N_INIT = 10  # scrambled-Sobol points to seed every method
N_BO_STEPS = 3 if FAST_TEST_MODE else 40  # one objective evaluation each
N_ESTIMATORS = 1 if FAST_TEST_MODE else 8  # TabPFN ensemble size
SAMPLER = (
    {"n_candidates": 128, "top_k": 4, "n_local": 8}  # tiny, for the smoke test
    if FAST_TEST_MODE
    else {}  # library defaults: 2048 candidates, top 16, 3 rounds of 32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"Device: {device}")

# Michalewicz-2: a 2-D benchmark with steep, narrow valleys on [0, pi]^2 where
# a stationary GP kernel struggles. EI maximizes, so we negate it; the optimum
# becomes +1.8013. Moved to the evaluation device: its bounds buffer must live
# where the candidate tensors do, or input validation fails.
objective = Michalewicz(dim=2, negate=True).to(device)
LO, HI = objective.bounds[0], objective.bounds[1]
print(f"Objective optimum: {objective.optimal_value:.4f}\n")


def evaluate(unit_x: torch.Tensor) -> torch.Tensor:
    """Evaluate the objective on points given in the unit cube.

    The BO code searches ``[0, 1]^d``; the benchmark lives on its own box, so
    every candidate is mapped affinely onto the problem's bounds first.
    """
    return objective(LO + unit_x * (HI - LO))


# --------------------------------------------------------------------------
# Three methods. Each starts from the same initial design and returns every
# observed objective value, initial points first.
# --------------------------------------------------------------------------
def run_tabpfn(init_x: torch.Tensor, init_y: torch.Tensor) -> torch.Tensor:
    reg = TabPFNRegressor(n_estimators=N_ESTIMATORS, device=device, random_state=0)
    train_x, train_y = init_x.clone(), init_y.clone()
    for step in trange(N_BO_STEPS, desc="TabPFN-EI"):
        next_x = propose_next_point(reg, train_x, train_y, **SAMPLER)
        next_y = evaluate(next_x.unsqueeze(0))
        train_x = torch.cat([train_x, next_x.unsqueeze(0)])
        train_y = torch.cat([train_y, next_y])
        print(
            f"  step {step + 1:2d}: evaluated f={next_y.item():7.4f} "
            f"| best so far f={train_y.max().item():7.4f}",
        )
    return train_y


def run_gp_logei(init_x: torch.Tensor, init_y: torch.Tensor) -> torch.Tensor:
    """BoTorch's default GP with LogEI."""
    dim = init_x.shape[1]
    unit_bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.double), torch.ones(dim, dtype=torch.double)],
    )
    train_x = init_x.double().cpu()
    train_y = init_y.double().cpu().unsqueeze(-1)
    for _ in trange(N_BO_STEPS, desc="GP-LogEI "):
        gp = SingleTaskGP(  # RBF kernel with ARD by default
            train_x,
            train_y,
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqf = LogExpectedImprovement(gp, best_f=train_y.max())
        next_x, _ = optimize_acqf(
            acqf,
            bounds=unit_bounds,
            q=1,
            num_restarts=16,
            raw_samples=512,
        )
        next_y = evaluate(next_x.to(device).float()).double().cpu().unsqueeze(-1)
        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])
    return train_y.squeeze(-1).float().to(device)


def run_random(init_x: torch.Tensor, init_y: torch.Tensor) -> torch.Tensor:
    extra = torch.rand(N_BO_STEPS, init_x.shape[1], device=device)
    return torch.cat([init_y, evaluate(extra)])


# Shared initial design: one scrambled-Sobol draw that every method starts from,
# so the curves only diverge once the acquisition strategy takes over. The
# initial design matters a lot at this budget -- over four Sobol seeds with
# tabpfn 8.4 on CPU, TabPFN reached the optimum within 50 evaluations on three
# and the GP on two -- which is why a single run is not a benchmark.
sobol = torch.quasirandom.SobolEngine(objective.dim, scramble=True, seed=2)
init_x = sobol.draw(N_INIT).to(device)
init_y = evaluate(init_x)

results = {
    "TabPFN-EI": run_tabpfn(init_x, init_y),
    "GP-LogEI (BoTorch)": run_gp_logei(init_x, init_y),
    "Random search": run_random(init_x, init_y),
}

# Best-so-far at a few budgets, and the final simple regret (optimum - best).
checkpoints = [
    n for n in (N_INIT, 20, 30, 40, N_INIT + N_BO_STEPS) if n <= N_INIT + N_BO_STEPS
]
checkpoints = sorted(set(checkpoints))
print("\nBest objective found after n evaluations (higher is better):")
print(
    f"{'':20s}" + "".join(f"{f'n={n}':>10s}" for n in checkpoints) + f"{'regret':>12s}"
)
for name, y in results.items():
    best = torch.cummax(y, dim=0).values
    row = "".join(f"{best[n - 1].item():10.4f}" for n in checkpoints)
    regret = objective.optimal_value - best[-1].item()
    print(f"{name:20s}{row}{regret:12.2e}")
print(f"\n(optimum: {objective.optimal_value:.4f})")
