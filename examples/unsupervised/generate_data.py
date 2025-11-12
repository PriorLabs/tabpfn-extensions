#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel


def plotting_helper_create_combined_dataframe(
    real_samples, synthetic_samples, feature_names
):
    """Return a tidy DataFrame with real and generated samples."""
    real_df = pd.DataFrame(real_samples.numpy(), columns=feature_names)
    real_df["real_or_synthetic"] = "Actual samples"

    synthetic_df = pd.DataFrame(
        synthetic_samples.detach().numpy(),
        columns=feature_names,
    )
    synthetic_df["real_or_synthetic"] = "Generated samples"

    return pd.concat([real_df, synthetic_df], ignore_index=True)


def plotting_helper_plot_pairgrid(combined_df):
    """Mirror the experiment helper's PairGrid visualisation."""
    plot_sample_count = combined_df["real_or_synthetic"].value_counts().min()
    plot_df = (
        combined_df.groupby("real_or_synthetic", group_keys=False)
        .apply(lambda df: df.sample(n=plot_sample_count, random_state=42))
        .reset_index(drop=True)
    )

    grid = sns.PairGrid(plot_df, hue="real_or_synthetic", diag_sharey=False)
    grid.map_diag(sns.histplot, common_norm=True)
    grid.map_offdiag(sns.scatterplot, s=2, alpha=0.5)
    grid.add_legend()
    plt.show()


def main():
    feature_indices = [0, 1]  # Select features to analyse (first two columns)

    breast_cancer = load_breast_cancer(return_X_y=False)
    X = breast_cancer["data"]
    attribute_names = breast_cancer["feature_names"]

    feature_names = [attribute_names[i] for i in feature_indices]
    X_tensor = torch.tensor(X[:, feature_indices], dtype=torch.float32)

    clf = TabPFNClassifier(n_estimators=3)
    reg = TabPFNRegressor(n_estimators=3)
    model_unsupervised = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    model_unsupervised.fit(X_tensor)

    multiplier = 3.0  # fraction of samples to be generated
    n_samples = int(X_tensor.shape[0] * multiplier)
    synthetic_samples = model_unsupervised.generate_synthetic_data(
        n_samples=n_samples,
        t=1.0,  # Default temperature used during sampling
        n_permutations=3,  # Number of permutations to average, more yields better results but slower runtime
    )

    combined_df = plotting_helper_create_combined_dataframe(
        X_tensor, synthetic_samples, feature_names
    )

    print("Combined dataset with real and synthetic samples:")
    print(combined_df.head())

    plotting_helper_plot_pairgrid(combined_df)


if __name__ == "__main__":
    main()
