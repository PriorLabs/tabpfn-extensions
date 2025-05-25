# TabPFGen Data Synthesizer Examples

This directory contains examples demonstrating how to use the TabPFGen Data Synthesizer extension for TabPFN.

The TabPFGen Data Synthesizer extension integrates [TabPFGen](https://github.com/sebhaan/TabPFGen) with the TabPFN ecosystem, enabling synthetic tabular data generation with automatic dataset balancing capabilities.

Author: Sebastian Haan

## Key Features

- **Synthetic Data Generation**: Support for both classification and regression tasks
- ** Automatic Dataset Balancing**: Built-in imbalanced dataset handling
- **Built-in Visualizations**: Uses TabPFGen's comprehensive visualization suite
- **Quality Assessment**: Comprehensive synthetic data quality metrics

## Examples

### 1. Basic Classification Example
```bash
python basic_classification_example.py
```

**Demonstrates:**
- Loading and analyzing datasets
- Generating synthetic classification data
- Using TabPFGen's built-in visualizations
- Quality assessment metrics

### 2. Dataset Balancing Demo
```bash
python class_balancing_demo.py
```

**Demonstrates:**
- Creating imbalanced datasets
- Using TabPFGen's new `balance_dataset()` method
- Automatic vs. custom target balancing
- Effectiveness analysis

### 3. Basic Regression Example
```bash
python basic_regression_example.py
```

**Demonstrates:**
- Synthetic regression data generation
- Quantile-based sampling
- Target correlation preservation
- Statistical quality comparisons


## Installation Requirements

```bash
# Install TabPFN (choose one)
pip install tabpfn              # For local inference
pip install tabpfn-client       # For cloud-based inference

# Install TabPFGen (v0.1.3+)
pip install tabpfgen>=0.1.3

# Install TabPFN Extensions
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
```

## Quick Start

### Basic Generation
```python
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Initialize synthesizer
synthesizer = TabPFNDataSynthesizer(n_sgld_steps=300)

# Generate synthetic data with TabPFGen's visualizations
X_synth, y_synth = synthesizer.generate_classification(
    X, y, n_samples=100, visualize=True
)
```

### Dataset Balancing
```python
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from sklearn.datasets import make_classification

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=3, 
                         n_informative=3, n_redundant=1,
                         weights=[0.7, 0.2, 0.1], random_state=42)

# Initialize synthesizer
synthesizer = TabPFNDataSynthesizer(n_sgld_steps=300)

# Balance automatically
X_synth, y_synth, X_balanced, y_balanced = synthesizer.balance_dataset(
    X, y, visualize=True
)

print(f"Original: {len(X)} samples")
print(f"Balanced: {len(X_balanced)} samples") 
print(f"Added: {len(X_synth)} synthetic samples")
```

### Quality Assessment
```python
from tabpfn_extensions.tabpfgen_datasynthesizer.utils import (
    validate_tabpfn_data,
    analyze_class_distribution,
    calculate_synthetic_quality_metrics
)

# Validate data for TabPFN compatibility
is_valid, message = validate_tabpfn_data(X, y)
print(f"Validation: {message}")

# Analyze class distribution
analysis = analyze_class_distribution(y, "Dataset Name")

# Calculate quality metrics
quality = calculate_synthetic_quality_metrics(X, X_synth, y, y_synth)
```

## Parameters

### TabPFNDataSynthesizer Parameters

- `n_sgld_steps` (int, default=500): Number of SGLD iterations for generation
- `sgld_step_size` (float, default=0.01): Step size for SGLD updates  
- `sgld_noise_scale` (float, default=0.01): Scale of noise in SGLD
- `device` (str, default='auto'): Computing device ('cpu', 'cuda', or 'auto')

### balance_dataset() Parameters

- `target_per_class` (int, optional): Custom target samples per class
- `visualize` (bool, default=False): Enable TabPFGen's built-in visualizations
- `feature_names` (list, optional): Feature names for visualization

### Generation Parameters

- `n_samples` (int): Number of synthetic samples to generate
- `balance_classes` (bool, default=True): Balance only synthetic samples
- `use_quantiles` (bool, default=True): Quantile-based sampling for regression
- `visualize` (bool, default=False): Enable visualization plots

## Important Notes

### About balance_classes vs balance_dataset()

- **`balance_classes=True`**: Only balances the synthetic samples generated
- **`balance_dataset()`**: Balances the entire dataset by generating synthetic samples for minority classes

### Balancing Results

The final class distribution may be **approximately balanced** rather than perfectly balanced. This is due to TabPFN's label refinement process, which prioritizes data quality and realism over exact class counts.

## Tips for Best Results

1. **SGLD Steps**: Use 300-500 steps for good quality; 500+ for production
2. **Device**: Use 'cuda' for significant speedup on GPU systems
3. **Validation**: Always validate data compatibility with `validate_tabpfn_data()`
4. **Balancing**: Use `balance_dataset()` for imbalanced datasets
5. **Quality Check**: Monitor synthetic data quality with built-in metrics

## Troubleshooting

### Common Issues

1. **TabPFGen Import Error**: 
   ```bash
   pip install tabpfgen>=0.1.3
   ```

2. **Memory Issues**: Reduce `n_samples` or `n_sgld_steps`

3. **Generation Quality**: Increase `n_sgld_steps` or adjust step size

4. **Imbalanced Results**: Use `balance_dataset()` instead of `generate_classification()`

### Performance Optimization

- **Development**: Use 100-300 SGLD steps for faster iteration
- **Production**: Use 500+ SGLD steps for best quality
- **GPU**: Enable with `device='cuda'` for 5-10x speedup
- **Batch Processing**: Generate larger batches rather than multiple small ones

