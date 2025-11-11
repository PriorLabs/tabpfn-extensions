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

# Load the breast cancer dataset
breast_cancer = load_breast_cancer(return_X_y=False)
X, y = breast_cancer["data"], breast_cancer["target"]
attribute_names = breast_cancer["feature_names"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)

# Use parameters that work with both TabPFN and TabPFN-client
clf = TabPFNClassifier(n_estimators=3)
reg = TabPFNRegressor(n_estimators=3)

# Initialize unsupervised model
model_unsupervised = TabPFNUnsupervisedModel(
    tabpfn_clf=clf,
    tabpfn_reg=reg,
)

# Select features for analysis (e.g., first two features)
feature_indices = [0, 1]
selected_feature_names = [attribute_names[i] for i in feature_indices]

# Prepare tensors for fitting the unsupervised model
X_tensor = torch.tensor(X_train[:, feature_indices], dtype=torch.float32)

# For this dataset all features are numerical, so we leave the categorical list empty.
model_unsupervised.set_categorical_features([])

# Fit the model on the selected features
model_unsupervised.fit(X_tensor)

# Generate synthetic samples directly from the fitted model
synthetic_samples = model_unsupervised.generate_synthetic_data(
    n_samples=X_tensor.shape[0] * 3,  # Generate 3x original samples
    t=1.0,  # Default temperature used during sampling
    n_permutations=3,  # Match default number of permutations in the experiment helper
)

# Combine real and synthetic samples for analysis or plotting
real_df = pd.DataFrame(X_tensor.numpy(), columns=selected_feature_names)
real_df["real_or_synthetic"] = "Actual samples"

synthetic_df = pd.DataFrame(
    synthetic_samples.detach().numpy(),
    columns=selected_feature_names,
)
synthetic_df["real_or_synthetic"] = "Generated samples"

combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

print("Combined dataset with real and synthetic samples:")
print(combined_df.head())

# Mirror the PairGrid plot from the experiment helper for quick visual inspection.
# To keep the classes balanced in the figure, downsample to the smaller class size.
plot_sample_count = min(len(real_df), len(synthetic_df))
plot_df = pd.concat(
    [
        real_df.sample(n=plot_sample_count, random_state=42),
        synthetic_df.sample(n=plot_sample_count, random_state=42),
    ],
    ignore_index=True,
)

g = sns.PairGrid(plot_df, hue="real_or_synthetic", diag_sharey=False)
g.map_diag(sns.histplot, common_norm=True)
g.map_offdiag(sns.scatterplot, s=2, alpha=0.5)
g.add_legend()
plt.show()
