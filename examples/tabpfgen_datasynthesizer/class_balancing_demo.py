"""
Dataset Balancing Demo with TabPFGen's balance_dataset Method

This example demonstrates the new balance_dataset method in TabPFGen v0.1.3+
for automatically balancing imbalanced classification datasets.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Import TabPFN Extensions
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from tabpfn_extensions.tabpfgen_datasynthesizer.utils import analyze_class_distribution
# Calculate quality metrics for both approaches
from tabpfn_extensions.tabpfgen_datasynthesizer.utils import calculate_synthetic_quality_metrics

def create_imbalanced_dataset():
    """Create a highly imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],  # Highly imbalanced: 70%, 20%, 10%
        random_state=42
    )
    return X, y

def main():
    """Run dataset balancing demonstration."""
    print("=== TabPFGen Dataset Balancing Demo ===\n")
    
    # Create imbalanced dataset
    print("Creating highly imbalanced dataset...")
    X, y = create_imbalanced_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} samples")
    
    # Analyze original imbalanced distribution
    original_analysis = analyze_class_distribution(y_train, "Original Imbalanced Training Data")
    
    # Initialize TabPFGen synthesizer
    print("\nInitializing TabPFGen synthesizer...")
    synthesizer = TabPFNDataSynthesizer(
        n_sgld_steps=400,  # Good balance of quality and speed
        device='auto'
    )
    
    print("\n" + "="*70)
    print("AUTOMATIC BALANCING (to majority class size)")
    print("="*70)
    
    # Use TabPFGen's balance_dataset method - automatic balancing
    X_synth_auto, y_synth_auto, X_balanced_auto, y_balanced_auto = synthesizer.balance_dataset(
        X_train, y_train,
        visualize=True,  # Use TabPFGen's built-in visualization
        feature_names=[f'feature_{i}' for i in range(X_train.shape[1])]
    )
    
    balanced_analysis_auto = analyze_class_distribution(y_balanced_auto, "Auto-Balanced Dataset")
    
    print("\n" + "="*70)
    print("CUSTOM TARGET BALANCING (1000 samples per class)")
    print("="*70)
    
    # Use TabPFGen's balance_dataset method - custom target
    X_synth_custom, y_synth_custom, X_balanced_custom, y_balanced_custom = synthesizer.balance_dataset(
        X_train, y_train,
        target_per_class=1000,  # Custom target
        visualize=True,
        feature_names=[f'feature_{i}' for i in range(X_train.shape[1])]
    )
    
    balanced_analysis_custom = analyze_class_distribution(y_balanced_custom, "Custom-Balanced Dataset (target=1000)")
    
    balanced_analysis_custom = analyze_class_distribution(y_balanced_custom, "Custom-Balanced Dataset (target=1000)")
    
    # Quality analysis
    print("\n" + "="*70)
    print("BALANCING EFFECTIVENESS SUMMARY")
    print("="*70)
    
    print(f"\nOriginal dataset imbalance ratio: {original_analysis['imbalance_ratio']:.1f}:1")
    print(f"Auto-balanced imbalance ratio: {balanced_analysis_auto['imbalance_ratio']:.1f}:1")
    print(f"Custom-balanced imbalance ratio: {balanced_analysis_custom['imbalance_ratio']:.1f}:1")
    
    print(f"\nData size summary:")
    print(f"Original training: {len(X_train)} samples")
    print(f"Auto-balanced: {len(X_balanced_auto)} samples (+{len(X_synth_auto)} synthetic)")
    print(f"Custom-balanced: {len(X_balanced_custom)} samples (+{len(X_synth_custom)} synthetic)")
    
    
    print(f"\nSynthetic data quality metrics:")
    print(f"Auto-balanced approach:")
    quality_auto = calculate_synthetic_quality_metrics(X_train, X_synth_auto, y_train, y_synth_auto)
    for metric, value in quality_auto.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nCustom-balanced approach:")
    quality_custom = calculate_synthetic_quality_metrics(X_train, X_synth_custom, y_train, y_synth_custom)
    for metric, value in quality_custom.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n✅ Dataset balancing demo completed successfully!")

if __name__ == "__main__":
    main()

