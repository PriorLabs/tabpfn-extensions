import numpy as np
from tabpfn_extensions.pval_crt import tabpfn_crt


def make_synthetic(n=300, seed=0):
    rng = np.random.RandomState(seed)

    X = rng.randn(n, 5)
    y = 2 * X[:, 0] - X[:, 1] + rng.randn(n)

    return X, y


if __name__ == "__main__":
    X, y = make_synthetic()

    print("\nRunning CRT for each feature:\n")

    for j in range(X.shape[1]):
        res = tabpfn_crt(
            X=X,
            y=y,
            j=j,
            B=100,
            alpha=0.05,
            seed=0,
            K=50,
        )

        print(f"Feature {j}: p-value = {res['p_value']:.4f}, reject = {res['reject_null']}")
