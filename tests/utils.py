from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

try:
    from hyperopt import hp

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

#######################
# Testing Utilities
#######################


def get_small_test_search_space():
    """Create a minimal search space for testing purposes.

    This provides a much faster search space for HPO tests and post-hoc ensembles.
    """
    if not HYPEROPT_AVAILABLE:
        return None

    return {
        # Simplified model config options
        "model_type": hp.choice(
            "model_type",
            ["single"],
        ),  # Only use single model (not dt_pfn)
        "n_estimators": hp.choice(
            "n_estimators",
            [1],
        ),  # Minimal ensemble repeats
        # Model hyperparameters - only test minimal options
        "average_before_softmax": hp.choice("average_before_softmax", [True]),
        "softmax_temperature": hp.choice("softmax_temperature", [0.9]),
        # Simplified preprocessing options
        "inference_config/FINGERPRINT_FEATURE": hp.choice(
            "FINGERPRINT_FEATURE",
            [False],
        ),
        "inference_config/PREPROCESS_TRANSFORMS": hp.choice(
            "PREPROCESS_TRANSFORMS",
            [
                [
                    {
                        # Use "name" parameter as expected by TabPFN PreprocessorConfig
                        "name": "none",
                        "global_transformer_name": None,
                        "subsample_features": -1,
                        "categorical_name": "none",
                        "append_original": False,
                    },
                ],
            ],
        ),
        "inference_config/POLYNOMIAL_FEATURES": hp.choice(
            "POLYNOMIAL_FEATURES",
            ["no"],  # Use "no" to match the TabPFN 2.0.6 expected values
        ),
        "inference_config/OUTLIER_REMOVAL_STD": hp.choice(
            "OUTLIER_REMOVAL_STD",
            [None],
        ),
        "inference_config/SUBSAMPLE_SAMPLES": hp.choice("SUBSAMPLE_SAMPLES", [None]),
    }


class DatasetGenerator:
    """Utility class for generating datasets for testing TabPFN.

    Generates various types of datasets for testing machine learning models:
    - Basic random datasets (numpy or pandas)
    - Classification and regression problems
    - Datasets with missing values
    - Datasets with text features
    - Datasets with correlations
    """

    def __init__(self, seed: int = 42):
        """Initialize the dataset generator with a specific random seed."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def make_numpy_dataset(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        as_pandas: bool = False,
        feature_names: list[str] = None,
    ):
        """Generate a basic dataset with random values.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            as_pandas: Whether to return pandas DataFrame
            feature_names: Custom column names for DataFrame

        Returns:
            X: Features as numpy array or pandas DataFrame
        """
        X = self.rng.rand(n_samples, n_features)

        if as_pandas:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            else:
                # Make sure we have enough names
                while len(feature_names) < n_features:
                    feature_names.append(f"feature_{len(feature_names)}")

            X = pd.DataFrame(X, columns=feature_names[:n_features])

        return X

    def add_correlations(self, X):
        """Add correlations between features in a dataset.

        Args:
            X: Input dataset (will be modified in-place)

        Returns:
            X: Modified dataset with correlations
        """
        n_samples, n_features = X.shape

        # Add a linear correlation between features
        if n_features >= 2:
            X[:, 1] = 0.8 * X[:, 0] + 0.2 * self.rng.randn(n_samples)

        # Add a non-linear relationship
        if n_features >= 4:
            X[:, 3] = np.sin(X[:, 2] * 3) + 0.1 * self.rng.randn(n_samples)

        return X

    def add_missing_values(self, X, missing_rate=0.1):
        """Add missing values to a dataset.

        Args:
            X: Input dataset
            missing_rate: Fraction of values to set as missing

        Returns:
            X_with_nan: Dataset with missing values
        """
        X_with_nan = X.copy()

        # Create random mask for missing values
        mask = self.rng.rand(*X.shape) < missing_rate
        X_with_nan[mask] = np.nan

        return X_with_nan

    def generate_data(
        self,
        task_type: str = "classification",
        n_samples: int = 100,
        n_features: int = 5,
        n_classes: int = 2,
        as_pandas: bool = False,
        data_type: str = "basic",
    ):
        """Generate a dataset for classification or regression.

        Args:
            task_type: 'classification' or 'regression'
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes (classification only)
            as_pandas: Whether to return pandas DataFrame
            data_type: Dataset type - 'basic', 'correlated', or 'missing'

        Returns:
            X: Features
            y: Target values
        """
        # Generate features
        X = self.make_numpy_dataset(n_samples, n_features)

        # Apply the requested modifications
        if data_type == "correlated":
            X = self.add_correlations(X)
        elif data_type == "missing":
            X = self.add_missing_values(X)
            # Impute missing values for testing
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)

        # Generate target variable
        if task_type == "classification":
            if n_classes == 2:
                # Binary classification
                y = (X[:, 0] > 0.5).astype(int)
            else:
                # Multi-class classification
                if n_features >= 2:
                    # Use the first two features with random centroids
                    centroids = self.rng.rand(n_classes, 2)
                    distances = np.zeros((n_samples, n_classes))

                    for i in range(n_classes):
                        distances[:, i] = np.sqrt(
                            (X[:, 0] - centroids[i, 0]) ** 2
                            + (X[:, 1] - centroids[i, 1]) ** 2,
                        )

                    y = np.argmin(distances, axis=1)
                else:
                    y = self.rng.randint(0, n_classes, size=n_samples)

        else:  # regression
            if n_features >= 3:
                # Non-linear relationship
                y = (
                    2 * X[:, 0]
                    + 1.5 * X[:, 1] ** 2
                    + self.rng.normal(0, 0.1, size=n_samples)
                )
            else:
                # Linear relationship
                coefficients = self.rng.randn(n_features)
                y = X.dot(coefficients) + self.rng.normal(0, 0.1, size=n_samples)

        # Convert to pandas if requested
        if as_pandas:
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name="target")

        return X, y

    def generate_classification_data(self, **kwargs):
        """Generate a classification dataset."""
        kwargs["task_type"] = "classification"
        return self.generate_data(**kwargs)

    def generate_regression_data(self, **kwargs):
        """Generate a regression dataset."""
        kwargs["task_type"] = "regression"
        return self.generate_data(**kwargs)

    def generate_text_dataset(
        self,
        n_samples: int = 100,
        task_type: str = "classification",
    ):
        """Generate a dataset with text features.

        Args:
            n_samples: Number of samples
            task_type: 'classification' or 'regression'

        Returns:
            X_original: DataFrame with text features
            y: Target values
        """
        # Generate numerical features
        X_numeric = self.make_numpy_dataset(n_samples, n_features=3)

        # Create DataFrame
        df = pd.DataFrame(X_numeric, columns=["numeric_0", "numeric_1", "numeric_2"])

        # Add text features
        categories = ["low", "medium", "high"]
        df["text_cat1"] = [
            categories[int(val * len(categories)) % len(categories)]
            for val in df["numeric_0"]
        ]

        categories2 = ["red", "green", "blue", "yellow"]
        df["text_cat2"] = [
            categories2[int(val * len(categories2)) % len(categories2)]
            for val in df["numeric_1"]
        ]

        # Create target variable
        if task_type == "classification":
            y = (df["numeric_0"] > 0.5).astype(int)
        else:  # regression
            y = df["numeric_0"] * 10 + self.rng.normal(0, 0.1, size=n_samples)

        # Remove feature used for target
        X_original = df.drop("numeric_0", axis=1)

        # Explicitly mark categorical columns with pandas category dtype for better detection
        X_original["text_cat1"] = X_original["text_cat1"].astype("category")
        X_original["text_cat2"] = X_original["text_cat2"].astype("category")

        return X_original, y

    def generate_missing_values_dataset(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        missing_rate: float = 0.1,
        task_type: str = "classification",
    ):
        """Generate a dataset with missing values.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            missing_rate: Fraction of values to set as missing
            task_type: 'classification' or 'regression'

        Returns:
            X_missing: Dataset with missing values
            y: Target values
        """
        # Generate complete data
        X_complete = self.make_numpy_dataset(n_samples, n_features)

        # Add correlations to make imputation more effective
        X_complete = self.add_correlations(X_complete)

        # Create target before adding missing values
        if task_type == "classification":
            y = (X_complete[:, 0] > 0.5).astype(int)
        else:  # regression
            y = X_complete[:, 0] * 10 + self.rng.normal(0, 0.1, size=n_samples)

        # Add missing values
        X_missing = self.add_missing_values(X_complete, missing_rate)

        return X_missing, y

    def generate_mixed_types_dataset(
        self,
        n_samples: int = 100,
        n_numerical: int = 3,
        n_categorical: int = 2,
    ):
        """Generate a dataset with mixed numerical and categorical features.

        Args:
            n_samples: Number of samples
            n_numerical: Number of numerical features
            n_categorical: Number of categorical features

        Returns:
            df: DataFrame with mixed types
        """
        # Generate numerical features
        X_num = self.rng.rand(n_samples, n_numerical)

        # Generate categorical features (as strings)
        X_cat = np.zeros((n_samples, n_categorical), dtype=object)
        for i in range(n_categorical):
            n_categories = 3  # Fixed number of categories for simplicity
            categories = [f"cat_{i}_{j}" for j in range(n_categories)]
            X_cat[:, i] = self.rng.choice(categories, size=n_samples)

        # Create DataFrames and combine
        df_num = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(n_numerical)])
        df_cat = pd.DataFrame(X_cat, columns=[f"cat_{i}" for i in range(n_categorical)])

        return pd.concat([df_num, df_cat], axis=1)


@pytest.fixture
def dataset_generator():
    """Create a dataset generator with a fixed seed."""
    return DatasetGenerator(seed=42)
