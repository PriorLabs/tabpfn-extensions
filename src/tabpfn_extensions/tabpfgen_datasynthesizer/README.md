# TabPFGen Data Synthesizer Extension

A TabPFN extension for synthetic tabular data generation using [TabPFGen](https://github.com/sebhaan/TabPFGen). 

Author: Sebastian Haan

## Motivation

While there are many tools available for generating synthetic images or text, creating realistic tabular data that preserves the statistical properties and relationships of the original dataset has been more challenging.

Generating synthetic tabular data is particularly useful in scenarios where:

1. You have limited real data but need more samples for training
2. You can't share real data due to privacy concerns
3. You need to balance an imbalanced dataset
4. You want to test how your models would perform with more data

What makes TabPFGen interesting is that it's built on the TabPFN transformer architecture and doesn't require additional training. It includes built-in visualization tools to help you verify the quality of the generated data by comparing distributions, feature correlations, and other important metrics between the real and synthetic datasets.


## Key Features

- Energy-based synthetic data generation
- Support for both classification and regression tasks
- Automatic dataset balancing for imbalanced classes
- Class-balanced sampling option
- Comprehensive visualization tools
- Built on TabPFN transformer architecture
- No additional training required

## Requirements

- **Python 3.10+** (due to TabPFGen dependency)
- TabPFN Extensions framework

> **Note**: While tabpfn-extensions supports Python 3.9+, this specific extension requires Python 3.10+ due to its TabPFGen dependency. 

## Installation

```bash
# Ensure Python 3.10+
python --version  # Should show 3.10 or higher

# Install TabPFN (choose one)
pip install tabpfn              # For local inference  
pip install tabpfn-client       # For cloud-based inference

# Python 3.10+ users who want every extension including TabPFGen
pip install "tabpfn-extensions[all,tabpfgen_datasynthesizer]"

# Or install only the tabpfgen_datasynthesizer extension
pip install "tabpfn-extensions[tabpfgen_datasynthesizer]"
```

## 🚀 Quick Start

### Basic Synthetic Data Generation

```python
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from sklearn.datasets import load_breast_cancer

# Load example data
X, y = load_breast_cancer(return_X_y=True)

# Initialize synthesizer
synthesizer = TabPFNDataSynthesizer(n_sgld_steps=500)

# Generate synthetic classification data
X_synth, y_synth = synthesizer.generate_classification(
    X, y, 
    n_samples=100,
    balance_classes=True,  # Only balances synthetic samples
    visualize=True         # TabPFGen's built-in visualization
)

# Generate synthetic regression data  
from sklearn.datasets import load_diabetes

# Load regression example dataset
X, y = load_diabetes(return_X_y=True)

X_synth, y_synth = synthesizer.generate_regression(
    X, y,
    n_samples=150,
    use_quantiles=True,
    visualize=True
)
```

### Automatic Dataset Balancing

Automatically balance imbalanced datasets:

```python
from tabpfn_extensions.tabpfgen_datasynthesizer import TabPFNDataSynthesizer
from sklearn.datasets import make_classification

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=3, 
                         n_informative=3, n_redundant=1,
                         weights=[0.7, 0.2, 0.1], random_state=42)

print("Original class distribution:")
print("Class 0: 700 samples (70.0%)")
print("Class 1: 200 samples (20.0%)")
print("Class 2: 100 samples (10.0%)")

# Initialize synthesizer
synthesizer = TabPFNDataSynthesizer(n_sgld_steps=500)

# Balance dataset automatically
X_synth, y_synth, X_balanced, y_balanced = synthesizer.balance_dataset(
    X, y, visualize=True
)

print(f"Original dataset: {len(X)} samples")
print(f"Synthetic samples: {len(X_synth)} samples")
print(f"Balanced dataset: {len(X_balanced)} samples")
# Final distribution approximately balanced!
```

## 📊 API Reference

### TabPFNDataSynthesizer

Main class for synthetic data generation:

```python
TabPFNDataSynthesizer(
    n_sgld_steps=500,        # SGLD iterations
    sgld_step_size=0.01,     # Step size
    sgld_noise_scale=0.01,   # Noise scale
    device='auto'            # 'cpu', 'cuda', or 'auto' 
)
```

**Key Methods:**

#### `balance_dataset()` ⭐ NEW
```python
X_synth, y_synth, X_combined, y_combined = synthesizer.balance_dataset(
    X, y,
    target_per_class=None,    # Auto-detect majority class size
    visualize=False,
    feature_names=None
)
```

**Returns:**
- `X_synth, y_synth`: Synthetic data only
- `X_combined, y_combined`: Original + synthetic data

#### `generate_classification()`
```python
X_synth, y_synth = synthesizer.generate_classification(
    X, y,
    n_samples,
    balance_classes=True,     # Balance only synthetic samples
    visualize=False,
    feature_names=None
)
```

#### `generate_regression()`
```python
X_synth, y_synth = synthesizer.generate_regression(
    X, y,
    n_samples,
    use_quantiles=True,
    visualize=False,
    feature_names=None
)
```

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


### Utility Functions

```python
from tabpfn_extensions.tabpfgen_datasynthesizer.utils import (
    validate_tabpfn_data,           # Check TabPFN compatibility
    analyze_class_distribution,      # Analyze class balance
    calculate_synthetic_quality_metrics,  # Quality assessment
    combine_datasets                # Combine original + synthetic
)

# Validate data
is_valid, message = validate_tabpfn_data(X, y)

# Analyze distribution
analysis = analyze_class_distribution(y, "Dataset Name")

# Calculate quality
quality = calculate_synthetic_quality_metrics(X_orig, X_synth, y_orig, y_synth)

# Combine datasets  
X_combined, y_combined = combine_datasets(
    X_orig, y_orig, X_synth, y_synth, 
    strategy='append'  # 'append', 'replace', or 'balanced'
)
```

## 🎯 Use Cases

### 1. Imbalanced Dataset Balancing
Perfect for datasets with class imbalance:

```python
# Detect imbalance
is_valid, message = validate_tabpfn_data(X, y)
if "imbalanced" in message:
    # Auto-balance
    _, _, X_balanced, y_balanced = synthesizer.balance_dataset(X, y)
```

### 2. Data Augmentation
Increase training data size:

```python

X_synth, y_synth = synthesizer.generate_classification(
    X_train, y_train, n_samples=int(len(X_train) * 0.5)
)
X_augmented, y_augmented = combine_datasets(
    X_train, y_train, X_synth, y_synth, strategy='append'
)
```

### 3. Quality Assessment
Monitor synthetic data quality:

```python
quality_metrics = calculate_synthetic_quality_metrics(
    X_orig, X_synth, y_orig, y_synth
)

for metric, value in quality_metrics.items():
    print(f"{metric}: {value:.4f}")
```

## 📈 Examples

The `examples/` directory contains comprehensive demonstrations:

1. **`basic_classification_example.py`** - Standard classification workflow
2. **`basic_regression_example.py`** - Regression data generation
3. **`class_balancing_demo.py`** - Showcase of `balance_dataset()` method  


```bash
cd examples/tabpfgen_datasynthesizer/
python basic_classification_example.py
```

## ⚡ Troubleshooting

### Common Issues

1. **TabPFGen Import Error**: 
   ```bash
   pip install tabpfgen>=0.1.4
   ```

2. **Memory Issues**: Reduce `n_samples` or `n_sgld_steps`

3. **Generation Quality**: Increase `n_sgld_steps` or adjust step size

4. **Imbalanced Results**: Use `balance_dataset()` instead of `generate_classification()`

### Performance Optimization

- **Development**: Use 100-300 SGLD steps for faster iteration
- **Production**: Use 500+ SGLD steps for best quality
- **GPU**: Enable with `device='cuda'` for 5-10x speedup
- **Batch Processing**: Generate larger batches rather than multiple small ones


## 🔍 Important Notes

### balance_classes vs balance_dataset()

- **`balance_classes=True`**: Only balances the generated synthetic samples
- **`balance_dataset()`**: Balances entire dataset by generating minority class samples

### Approximate Balancing

Final class distributions may be **approximately balanced** rather than perfectly balanced due to TabPFN's label refinement process, which prioritizes data quality over exact counts.

## 📚 Citation

@software{haan2025tabpfgen,
  author = {Haan, Sebastian},
  title = {TabPFGen: Synthetic Tabular Data Generation with TabPFN},
  url = {https://github.com/sebhaan/TabPFGen},
  year = {2025}
}

## 📄 License

Apache License 2.0 - same as TabPFN Extensions.
