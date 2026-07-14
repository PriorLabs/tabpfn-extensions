#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Bayesian optimization with TabPFN as the surrogate model.

TabPFN predicts a full bar distribution over the target, so acquisition
functions like Expected Improvement (EI) can be computed in closed form from
the predicted logits — no Gaussian assumption needed. This follows the
PFNs4BO approach (Mueller et al., ICML 2023, https://arxiv.org/abs/2305.17535,
https://github.com/automl/PFNs4BO), using the TabPFN foundation model as the
surrogate.

Each iteration of the loop below calls
``tabpfn_extensions.bayesian_optimization.propose_next_point``, which:

1. Fits TabPFN on the points evaluated so far via
   ``fit_with_differentiable_input`` (tensors in, gradients preserved).
2. Scores a batch of random candidates with EI in a single forward pass.
3. Refines the most promising candidates by gradient *ascent on EI itself* —
   ``differentiable_input=True`` lets gradients flow from the acquisition
   value back to the candidate coordinates.

The loop then evaluates the objective at the proposed point.

NOTE: This example requires the extras of the bayesian_optimization
extension: pip install "tabpfn-extensions[bayesian_optimization]". It needs
the full TabPFN implementation, version 8.1.0 or later, and will not work
with the TabPFN client because it needs the differentiable torch inference
path. BoTorch is only used for the benchmark objective.

Runs in well under a minute on CPU; faster on a CUDA GPU.
"""

import torch
from botorch.test_functions import Hartmann
from tqdm import trange

from tabpfn import TabPFNRegressor
from tabpfn_extensions.bayesian_optimization import propose_next_point

N_INIT = 10  # random points to seed the surrogate
N_BO_STEPS = 25  # BO iterations (one objective evaluation each)
N_CANDIDATES = 512  # random candidates screened per iteration
TOP_K = 4  # candidates refined by gradient ascent on EI
N_REFINE_STEPS = 8  # gradient steps on the candidate coordinates
REFINE_LR = 0.05

# Hartmann-6: a classic 6D benchmark on [0, 1]^6 where random search does
# poorly. EI maximizes, so we negate it; the optimum becomes +3.32237.
objective = Hartmann(dim=6, negate=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"Device: {device}")
print(f"Objective optimum: {objective.optimal_value:.4f}\n")

reg = TabPFNRegressor(
    n_estimators=1,
    device=device,
    random_state=0,
    inference_precision=torch.float32,
    differentiable_input=True,
)

# Seed the surrogate with random evaluations.
train_x = torch.rand(N_INIT, objective.dim, device=device)
train_y = objective(train_x)

for step in trange(N_BO_STEPS, desc="BO steps"):
    next_x = propose_next_point(
        reg,
        train_x,
        train_y,
        n_candidates=N_CANDIDATES,
        top_k=TOP_K,
        n_refine_steps=N_REFINE_STEPS,
        refine_lr=REFINE_LR,
    )
    next_y = objective(next_x.unsqueeze(0))
    train_x = torch.cat([train_x, next_x.unsqueeze(0)])
    train_y = torch.cat([train_y, next_y])
    print(
        f"  step {step + 1:2d}: evaluated f={next_y.item():7.4f} "
        f"| best so far f={train_y.max().item():7.4f}",
    )

# Random-search baseline with the same total evaluation budget.
rand_y = objective(torch.rand(N_INIT + N_BO_STEPS, objective.dim, device=device))

print(f"\nBest found by TabPFN-BO:     {train_y.max().item():.4f}")
print(f"Best found by random search: {rand_y.max().item():.4f}")
print(f"(optimum:                    {objective.optimal_value:.4f})")
