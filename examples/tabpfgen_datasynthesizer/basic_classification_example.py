"""
Basic Classification Example with TabPFGen Data Synthesizer

This example demonstrates how to use TabPFGen for synthetic data generation
in classification tasks, leveraging the actual TabPFGen package features.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import TabPFN Extensions
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from tabpfn_extensions.tabpfgen_datasynthesizer.utils import analyze_class_distribution

def main():
    """Run basic classification example."""
    print("=== TabPFGen Classification Example ===\n")
    
    # Load breast cancer dataset
    print("Loading breast cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer().feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} samples")
    
    # Analyze original distribution
    analyze_class_distribution(y_train, "Original Training Data")
    
    # Initialize TabPFGen synthesizer
    print("\nInitializing TabPFGen synthesizer...")
    synthesizer = TabPFNDataSynthesizer(
        n_sgld_steps=300,  # Reduced for faster demo
        device='auto'
    )
    
    # Generate synthetic data using TabPFGen's built-in methods
    print("\nGenerating synthetic classification data...")
    n_synthetic = 200
    X_synth, y_synth = synthesizer.generate_classification(
        X_train, y_train,
        n_samples=n_synthetic,
        balance_classes=True,  # This balances only the synthetic samples
        visualize=True,  # Use TabPFGen's built-in visualization
        feature_names=list(feature_names)
    )
    
    print(f"\nGenerated {len(X_synth)} synthetic samples")
    analyze_class_distribution(y_synth, "Synthetic Data")
    
    # Combine original and synthetic data
    from tabpfn_extensions.tabpfgen_datasynthesizer.utils import combine_datasets
    X_augmented, y_augmented = combine_datasets(
        X_train, y_train, X_synth, y_synth, strategy='append'
    )
    
    analyze_class_distribution(y_augmented, "Augmented Training Data")
    
    print("\n✅ Synthetic data generation completed successfully!")
    
    # Calculate quality metrics
    from tabpfn_extensions.tabpfgen_datasynthesizer.utils import calculate_synthetic_quality_metrics
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA QUALITY METRICS")
    print("="*60)
    
    quality_metrics = calculate_synthetic_quality_metrics(
        X_train, X_synth, y_train, y_synth
    )
    
    for metric, value in quality_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()