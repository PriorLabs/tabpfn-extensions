import numpy as np
from tabpfn_extensions.pval_crt import tabpfn_crt


def make_synthetic(n=300, seed=0):
    rng = np.random.RandomState(seed)

    X = rng.randn(n, 5)
    y = 2 * X[:, 0] - X[:, 1] + rng.randn(n)

    return X, y


if __name__ == "__main__":
    X, y = make_synthetic()

    print("\nRunning CRT for ALL features (batched):\n")

    results = tabpfn_crt(
        X=X,
        y=y,
        j=list(range(X.shape[1])),  
        B=100,
        alpha=0.05,
        seed=0,
        K=50,
    )

    for feature, res in results.items():
        print(f"Feature {feature}: p-value = {res['p_value']:.4f}, reject = {res['reject_null']}")
