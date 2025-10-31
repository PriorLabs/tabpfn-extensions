#!/usr/bin/env python3
"""Tutorial: Augment real-world data with TabEBM

This script replicates the functionality of the notebook but uses sklearn
instead of TabCamel for data loading and preprocessing.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tabpfn_extensions.tabebm.tabebm import TabEBM, seed_everything


def load_adult_dataset(dataset_id):
    """Load the adult dataset using OpenML ID."""
    print("Loading adult dataset from OpenML...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    X = data.data
    y = data.target

    print(f"Dataset shape: {X.shape}")
    print(f"Target classes: {y.unique()}")
    print(f"Feature names: {list(X.columns)}")

    return X, y


def identify_feature_types(X):
    """Identify categorical and numerical features."""
    categorical_features = []
    numerical_features = []

    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    return categorical_features, numerical_features


def subsample_data(X, y, sample_size, random_state):
    """Subsample the dataset to simulate low-sample-size scenario."""
    print(f"Subsampling dataset to {sample_size} samples...")

    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=sample_size, stratify=y, random_state=random_state
    )

    print(f"Subsampled dataset shape: {X_sub.shape}")
    print(f"Class distribution: {pd.Series(y_sub).value_counts().sort_index()}")

    return X_sub, y_sub


def preprocess_data(
    X_train, X_test, y_train, y_test, categorical_features, numerical_features
):
    """Preprocess the data using sklearn transformers."""
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    # Fit and transform features
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_processed = label_encoder.fit_transform(y_train)
    y_test_processed = label_encoder.transform(y_test)

    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")
    print(f"Unique target labels: {np.unique(y_train_processed)}")

    return X_train_processed, X_test_processed, y_train_processed, y_test_processed


def main():
    """Main execution function."""
    # Set random seed
    seed_everything(42)

    # Load the dataset
    X, y = load_adult_dataset(dataset_id=45068)

    # Subsample to simulate low-sample-size scenario
    X_sub, y_sub = subsample_data(X, y, sample_size=200, random_state=42)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Identify feature types
    categorical_features, numerical_features = identify_feature_types(X_train)

    # Preprocess the data
    X_train_processed, X_test_processed, y_train_processed, y_test_processed = (
        preprocess_data(
            X_train, X_test, y_train, y_test, categorical_features, numerical_features
        )
    )

    print("\n" + "=" * 50)
    print("GENERATING SYNTHETIC DATA WITH TabEBM")
    print("=" * 50)

    # Fit TabEBM and generate synthetic samples
    tabebm = TabEBM()
    # Generate 50 synthetic samples per class
    data_syn = tabebm.generate(X_train_processed, y_train_processed, num_samples=50)

    # Combine the synthetic samples with the real samples
    X_syn = np.concatenate(list(data_syn.values()))
    y_syn = np.concatenate(
        [np.full(len(data_syn[f"class_{i}"]), i) for i in range(len(data_syn.keys()))]
    )

    X_train_augmented = np.concatenate([X_train_processed, X_syn])
    y_train_augmented = np.concatenate([y_train_processed, y_syn])

    print(f"Original training set size: {len(X_train_processed)}")
    print(f"Synthetic data size: {len(X_syn)}")
    print(f"Augmented training set size: {len(X_train_augmented)}")

    print("\n" + "=" * 50)
    print("TRAINING DOWNSTREAM PREDICTORS")
    print("=" * 50)

    # Train model on original dataset (only real data)
    print("\nTraining vanilla model on original data...")
    model_vanilla = KNeighborsClassifier()
    model_vanilla.fit(X_train_processed, y_train_processed)

    # Train model on augmented dataset (real + synthetic data)
    print("Training augmented model on real + synthetic data...")
    model_augmented = KNeighborsClassifier()
    model_augmented.fit(X_train_augmented, y_train_augmented)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Evaluate the predictive accuracy
    y_pred_vanilla = model_vanilla.predict(X_test_processed)
    y_pred_augmented = model_augmented.predict(X_test_processed)

    acc_vanilla = balanced_accuracy_score(y_test_processed, y_pred_vanilla) * 100
    acc_augmented = balanced_accuracy_score(y_test_processed, y_pred_augmented) * 100

    print(f"Vanilla model's balanced accuracy: {acc_vanilla:.2f}%")
    print(f"Augmented model's balanced accuracy: {acc_augmented:.2f}%")
    print(f"Improvement: {acc_augmented - acc_vanilla:.2f} percentage points")

    if acc_augmented > acc_vanilla:
        print("✓ Data augmentation with TabEBM improved model performance!")
    else:
        print("✗ Data augmentation with TabEBM did not improve model performance.")


if __name__ == "__main__":
    main()
