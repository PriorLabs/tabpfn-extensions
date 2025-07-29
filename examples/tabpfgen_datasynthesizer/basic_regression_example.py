"""Basic Regression Example with TabPFGen Data Synthesizer

This example demonstrates how to use TabPFGen for synthetic data generation
in regression tasks, using TabPFGen's built-in features.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Import TabPFN Extensions
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from tabpfn_extensions.tabpfgen_datasynthesizer.utils import (
    calculate_synthetic_quality_metrics,
)


def main():
    """Run basic regression example."""
    print("=== TabPFGen Regression Example ===\n")

    # Load diabetes dataset
    print("Loading diabetes dataset...")
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes().feature_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Target range: [{y_train.min():.1f}, {y_train.max():.1f}]")

    # Initialize TabPFGen synthesizer
    print("\nInitializing TabPFGen synthesizer...")
    synthesizer = TabPFNDataSynthesizer(
        n_sgld_steps=300,  # Good balance for regression
        device="auto",
    )

    # Generate synthetic regression data
    print("\nGenerating synthetic regression data...")
    n_synthetic = 150
    X_synth, y_synth = synthesizer.generate_regression(
        X_train,
        y_train,
        n_samples=n_synthetic,
        use_quantiles=True,  # Important for regression quality
        visualize=True,  # Use TabPFGen's built-in visualization
        feature_names=list(feature_names),
    )

    print(f"\nGenerated {len(X_synth)} synthetic samples")
    print(f"Synthetic target range: [{y_synth.min():.1f}, {y_synth.max():.1f}]")

    # Combine original and synthetic data
    from tabpfn_extensions.tabpfgen_datasynthesizer.utils import combine_datasets

    X_augmented, y_augmented = combine_datasets(
        X_train, y_train, X_synth, y_synth, strategy="append"
    )

    print(f"Combined dataset: {len(X_augmented)} samples")
    print(f"Combined target range: [{y_augmented.min():.1f}, {y_augmented.max():.1f}]")

    # Calculate quality metrics
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA QUALITY METRICS")
    print("=" * 60)

    quality_metrics = calculate_synthetic_quality_metrics(
        X_train, X_synth, y_train, y_synth
    )

    print("\nFeature quality metrics:")
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Statistical comparison
    print("\nStatistical comparison:")
    print(f"Original data - Mean: {np.mean(X_train):.3f}, Std: {np.std(X_train):.3f}")
    print(f"Synthetic data - Mean: {np.mean(X_synth):.3f}, Std: {np.std(X_synth):.3f}")
    print("Target correlation preservation:")

    # Check target correlations
    orig_target_corr = []
    synth_target_corr = []

    for i in range(X_train.shape[1]):
        orig_corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        synth_corr = np.corrcoef(X_synth[:, i], y_synth)[0, 1]
        orig_target_corr.append(orig_corr)
        synth_target_corr.append(synth_corr)

    print(
        f"Average target correlation - Original: {np.mean(np.abs(orig_target_corr)):.3f}"
    )
    print(
        f"Average target correlation - Synthetic: {np.mean(np.abs(synth_target_corr)):.3f}"
    )

    correlation_preservation = 1 - np.mean(
        np.abs(np.array(orig_target_corr) - np.array(synth_target_corr))
    )
    print(f"Correlation preservation score: {correlation_preservation:.3f}")

    print("\n✅ Synthetic regression data generation completed successfully!")


if __name__ == "__main__":
    main()
