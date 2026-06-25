#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Generate synthetic data for a dataset with categorical and numerical columns.

This example shows how to tell ``GenerateSyntheticDataExperiment`` which columns
are categorical via the ``categorical_features`` argument, and then checks that the
generated categorical columns resemble the input: they should not contain category
values that were absent from the input, and their frequencies should be similar.
"""

import matplotlib.pyplot as plt
import numpy as np

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment


def make_mixed_dataset(n_samples=400, seed=0):
    """Build a dataset with two categorical columns and two numerical columns.

    The numerical columns depend on the categorical ones, so there is joint
    structure for the model to learn.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.30, 0.18, 0.07])
    tier = rng.choice([0, 1, 2], size=n_samples, p=[0.60, 0.30, 0.10])
    spend = rng.normal(10 + 3 * region, 2.0)
    age = rng.normal(40 + 5 * tier, 8.0)

    X = np.column_stack([region, tier, spend, age]).astype(np.float32)
    attribute_names = ["region", "tier", "spend", "age"]
    categorical_features = [0, 1]
    return X, attribute_names, categorical_features


def report_categorical_fidelity(data, attribute_names, categorical_features):
    """Print, per categorical column, the input categories and any new ones."""
    real = data[data["real_or_synthetic"] == "Actual samples"]
    synthetic = data[data["real_or_synthetic"] == "Generated samples"]
    for idx in categorical_features:
        col = attribute_names[idx]
        real_categories = set(real[col].round().astype(int).unique())
        synthetic_categories = set(synthetic[col].round().astype(int).unique())
        novel = sorted(synthetic_categories - real_categories)
        print(
            f"{col}: input categories={sorted(real_categories)}, "
            f"categories in synthetic data not seen in input={novel}",
        )


def plot_real_vs_synthetic(data, attribute_names, categorical_features):
    """Plot real vs synthetic per column: bars for categorical, histograms otherwise."""
    real = data[data["real_or_synthetic"] == "Actual samples"]
    synthetic = data[data["real_or_synthetic"] == "Generated samples"]

    fig, axes = plt.subplots(
        1, len(attribute_names), figsize=(4 * len(attribute_names), 4)
    )
    for idx, (ax, col) in enumerate(zip(axes, attribute_names, strict=True)):
        if idx in categorical_features:
            real_freq = real[col].round().astype(int).value_counts(normalize=True)
            synth_freq = synthetic[col].round().astype(int).value_counts(normalize=True)
            categories = sorted(set(real_freq.index) | set(synth_freq.index))
            positions = np.arange(len(categories))
            width = 0.4
            ax.bar(
                positions - width / 2,
                [real_freq.get(c, 0) for c in categories],
                width,
                label="real",
            )
            ax.bar(
                positions + width / 2,
                [synth_freq.get(c, 0) for c in categories],
                width,
                label="synthetic",
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
            ax.set_ylabel("frequency")
        else:
            ax.hist(real[col], bins=20, density=True, alpha=0.5, label="real")
            ax.hist(synthetic[col], bins=20, density=True, alpha=0.5, label="synthetic")
            ax.set_ylabel("density")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.legend()

    fig.tight_layout()
    plt.show()


def main():
    X, attribute_names, categorical_features = make_mixed_dataset()

    model_unsupervised = TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(n_estimators=2),
        tabpfn_reg=TabPFNRegressor(n_estimators=2),
    )

    experiment = GenerateSyntheticDataExperiment(task_type="unsupervised")
    experiment.run(
        tabpfn=model_unsupervised,
        X=X,
        y=np.array([]),
        attribute_names=attribute_names,
        indices=list(range(X.shape[1])),
        categorical_features=categorical_features,
        n_samples=X.shape[0],
        should_plot=False,  # we draw our own real-vs-synthetic comparison below
    )

    print(
        "Categorical features used by the model:",
        model_unsupervised.categorical_features,
    )
    report_categorical_fidelity(experiment.data, attribute_names, categorical_features)
    plot_real_vs_synthetic(experiment.data, attribute_names, categorical_features)


if __name__ == "__main__":
    main()
