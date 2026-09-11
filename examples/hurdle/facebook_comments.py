"""Compare mean and median predictions on UCI Facebook Comment Volume.

Data: Singh, K. (2015), https://doi.org/10.24432/C5Q886 (CC BY 4.0).
The target is the number of comments in the next H hours. Training variant 1
has 55.1% zero targets. We sample the published train and test sets separately.
Root mean squared error evaluates means. For an absolute-error objective,
the example also compares median predictions using mean absolute error.
Both comparisons include plain TabPFN and a constant training-target baseline.
Results depend on the sample and model version.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from tabpfn_extensions import AutoHurdleRegressor, TabPFNClassifier, TabPFNRegressor


def main(archive: Path, n_train: int, n_test: int, seed: int) -> None:
    """Download the public data if needed, then fit and compare both models."""
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            "https://archive.ics.uci.edu/static/public/363/"
            "facebook%2Bcomment%2Bvolume%2Bdataset.zip",
            timeout=60,
        ) as response:
            archive.write_bytes(response.read())
    with ZipFile(archive) as dataset:
        train = pd.read_csv(
            dataset.open("Dataset/Training/Features_Variant_1.csv"), header=None
        ).sample(n=n_train, random_state=seed)
        test = pd.read_csv(
            dataset.open("Dataset/Testing/Features_TestSet.csv"), header=None
        ).sample(n=n_test, random_state=seed)
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    print(f"Rows: {len(train)} train, {len(test)} test")
    print(
        f"Zero targets: {(y_train == 0).mean():.1%} train, {(y_test == 0).mean():.1%} test"
    )
    baseline = TabPFNRegressor(n_estimators=1, random_state=seed)
    hurdle = AutoHurdleRegressor(
        classifier=TabPFNClassifier(n_estimators=1, random_state=seed),
        regressor=baseline,
    ).fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    print(f"Hurdle enabled: {hurdle.hurdle_}")
    mean_predictions = {
        "Constant mean": np.full(len(test), np.mean(y_train)),
        "TabPFN mean": baseline.predict(X_test, output_type="mean"),
        "Hurdle mean": hurdle.predict(X_test, output_type="mean"),
    }
    for label, prediction in mean_predictions.items():
        print(f"{label:20s} RMSE: {root_mean_squared_error(y_test, prediction):.4f}")
    median_predictions = {
        "Constant median": np.full(len(test), np.median(y_train)),
        "TabPFN median": baseline.predict(X_test, output_type="median"),
        "Hurdle median": hurdle.predict(X_test, output_type="median"),
    }
    for label, prediction in median_predictions.items():
        print(f"{label:20s} MAE: {mean_absolute_error(y_test, prediction):.4f}")


if __name__ == "__main__":
    fast = os.environ.get("FAST_TEST_MODE") == "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("downloads/facebook_comments.zip")
    )
    parser.add_argument("--n-train", type=int, default=100 if fast else 1000)
    parser.add_argument("--n-test", type=int, default=50 if fast else 1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.archive, args.n_train, args.n_test, args.seed)
