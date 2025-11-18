# TabPFN Extensions Examples

This directory contains examples for using TabPFN extensions. Each extension has its own subdirectory with examples demonstrating its usage.

## Example Organization

```
examples/
├── classifier_as_regressor/   # Using TabPFN classifiers for regression
├── hpo/                      # Hyperparameter optimization
├── interpretability/         # SHAP and feature selection
├── many_class/               # Handling many classes
├── multioutput/              # Multi-output regression and classification
├── post_hoc_ensembles/       # Post-hoc ensembles
├── rf_pfn/                   # Random Forest + TabPFN combinations
└── unsupervised/             # Unsupervised learning
```

## Running Examples

Most examples can be run directly:

```bash
python examples/many_class/many_class_classifier_example.py
```

## Important Guidelines

1. **Do not wrap example code in main blocks**, but rather directly add it. This way our example tests can pick them up and ensure they work correctly.

2. **TabPFN Compatibility**: Examples should work with both TabPFN implementations when possible:
   - TabPFN Package - Full PyTorch implementation for local inference
   - TabPFN Client - Lightweight API client for cloud-based inference

3. **Flexible Imports**: Use this pattern for imports to support both backends:
   ```python
   try:
       # Try standard TabPFN package first
       from tabpfn import TabPFNClassifier, TabPFNRegressor
   except ImportError:
       # Fall back to TabPFN client
       from tabpfn_client import TabPFNClassifier, TabPFNRegressor
   ```

4. **Example Datasets**: Use small built-in datasets from scikit-learn. No external data downloads required.

5. **Documentation**: Include clear comments explaining what the example demonstrates and how it works.

## Contributing New Examples

When adding examples for new extensions:

1. Create a new subdirectory with your extension name
2. Include a README.md with description and usage instructions
3. Make examples self-contained and easy to run
4. Clearly document any requirements or limitations

See the existing examples for reference.
