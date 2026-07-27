import os

import numpy as np

from tabpfn_extensions.pval_crt import tabpfn_crt

# Under FAST_TEST_MODE=1 (set by the CI example tests) the workload shrinks so
# the example finishes quickly; the p-values are correspondingly coarser.
FAST_TEST_MODE = os.environ.get("FAST_TEST_MODE") == "1"


def make_synthetic(n=300, seed=0):
    rng = np.random.RandomState(seed)

    X = rng.randn(n, 5)
    y = 2 * X[:, 0] - X[:, 1] + rng.randn(n)

    return X, y


if __name__ == "__main__":
    X, y = make_synthetic(n=100 if FAST_TEST_MODE else 300)

    print("\nRunning CRT for ALL features (batched):\n")

    results = tabpfn_crt(
        X=X,
        y=y,
        j=list(range(X.shape[1])),
        B=10 if FAST_TEST_MODE else 100,
        alpha=0.05,
        seed=0,
        K=5 if FAST_TEST_MODE else 50,
    )

    for feature, res in results.items():
        print(
            f"Feature {feature}: p-value = {res['p_value']:.4f}, reject = {res['reject_null']}"
        )
