#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel


def load_training_features(feature_indices):
    """Return selected training features and their human-readable names."""

    breast_cancer = load_breast_cancer(return_X_y=False)
    X, y = breast_cancer["data"], breast_cancer["target"]
    attribute_names = breast_cancer["feature_names"]

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=42,
    )

    selected_feature_names = [attribute_names[i] for i in feature_indices]
    X_tensor = torch.tensor(X_train[:, feature_indices], dtype=torch.float32)
    return X_tensor, selected_feature_names


def build_unsupervised_model():
    """Instantiate TabPFN models that work with both TabPFN and TabPFN-client."""

    clf = TabPFNClassifier(n_estimators=3)
    reg = TabPFNRegressor(n_estimators=3)
    return TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)


def generate_samples(model, X_tensor, *, multiplier=3, t=1.0, n_permutations=3):
    """Sample synthetic data directly from the fitted model."""

    n_samples = X_tensor.shape[0] * multiplier
    return model.generate_synthetic_data(
        n_samples=n_samples,
        t=t,  # Default temperature used during sampling
        n_permutations=n_permutations,  # Matches the experiment helper default
    )


def create_combined_dataframe(real_samples, synthetic_samples, feature_names):
    """Return a tidy DataFrame with real and generated samples."""

    real_df = pd.DataFrame(real_samples.numpy(), columns=feature_names)
    real_df["real_or_synthetic"] = "Actual samples"

    synthetic_df = pd.DataFrame(
        synthetic_samples.detach().numpy(),
        columns=feature_names,
    )
    synthetic_df["real_or_synthetic"] = "Generated samples"

    return pd.concat([real_df, synthetic_df], ignore_index=True)


def plot_pairgrid(combined_df):
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

    X_tensor, feature_names = load_training_features(feature_indices)

    model_unsupervised = build_unsupervised_model()
    model_unsupervised.set_categorical_features([])  # All features are numerical here
    model_unsupervised.fit(X_tensor)

    synthetic_samples = generate_samples(
        model_unsupervised,
        X_tensor,
        multiplier=3,
        t=1.0,
        n_permutations=3,
    )

    combined_df = create_combined_dataframe(X_tensor, synthetic_samples, feature_names)

    print("Combined dataset with real and synthetic samples:")
    print(combined_df.head())

    plot_pairgrid(combined_df)


if __name__ == "__main__":
    main()
