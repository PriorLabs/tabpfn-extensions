# Contributing to TabPFN Extensions

Welcome to TabPFN Extensions! This repository is a collection of community-contributed packages that extend and enhance TabPFN, a foundation model for tabular data. We welcome all contributions and aim to make the process as simple as possible.

Before developing a feature / opening a PR, please open a GitHub issue and describe the bug or feature request.
This allows us to discuss the proposal and helps avoid unnecessary work.

## 📋 Quick Start

1. **Setup your development environment**:

```bash
# Clone the repository
git clone https://github.com/PriorLabs/tabpfn-extensions.git
cd tabpfn-extensions

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Lightweight install for extension development
pip install -e ".[dev]"  # Only installs base requirements

# OR full install with all dependencies (may take longer)
pip install -e ".[dev,all]"
```

2. **Create your extension package**:

```bash
mkdir -p src/tabpfn_extensions/your_package examples/your_package tests/
```

3. **Start coding**:
   - Implement your extension in `src/tabpfn_extensions/your_package/`
   - Create examples in `examples/your_package/`
   - Write tests in `tests/test_your_package.py`

4. **Test your code**:

```bash
# Quick test run for your package only
FAST_TEST_MODE=1 pytest tests/test_your_package.py -v

# Even faster test with debug mode (smaller datasets)
TABPFN_DEBUG=1 FAST_TEST_MODE=1 pytest tests/test_your_package.py -v

# Full test of your package (before submitting PR)
pytest tests/test_your_package.py -v

# Run all tests (optional, CI will do this anyway)
pytest
```

## 🔄 TabPFN Compatibility

TabPFN Extensions supports two different TabPFN implementations:

1. **TabPFN Package** - Full PyTorch implementation for local inference
2. **TabPFN Client** - Lightweight API client for cloud-based inference

To ensure your extension works with both backends:

1. **Import TabPFN in a flexible way**:

```python
try:
    # Try standard TabPFN package first
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:
    # Fall back to TabPFN client
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
```

2. **Use common parameters** that are available in both implementations.

3. **Add appropriate test markers** in your tests:
   - `client_compatible` if your extension works with TabPFN client
   - `local_compatible` if it requires the full TabPFN package

## 📁 Repository Structure

TabPFN Extensions uses a modular structure where each contribution lives in its own subpackage:

```
tabpfn-extensions/
├── src/
│   └── tabpfn_extensions/
│       └── your_package/       # Your extension code
├── examples/
│   └── your_package/           # Usage examples
└── tests/
    └── test_your_package.py    # Tests for your extension
```

## 📝 Code Guidelines

### Requirements

- Python 3.9+ compatibility
- Support for both TabPFN and TabPFN-client when possible
- Minimize dependencies and document them in pyproject.toml
- Follow scikit-learn API conventions when appropriate
- Document all public functions and classes

### Documentation

Each extension should include:

#### Essential:
- Basic docstrings explaining what functions/classes do
- At least one example script in the examples directory

#### Recommended:
- Google style format docstrings for key functions
- Type hints for main functions and parameters
- Comments explaining non-obvious code sections

### Testing

- Write tests using pytest
- For quick test development, use `FAST_TEST_MODE=1 pytest tests/test_your_package.py`
- Only run long tests (like full dataset tests) when `FAST_TEST_MODE` is not set
- Use appropriate markers: `client_compatible`, `local_compatible`
- Use the fixtures provided in conftest.py for TabPFN instances
- Focus on testing your core functionality first, not edge cases

## ✅ Contribution Checklist

### Minimum Requirements
- [ ] Added extension under src/tabpfn_extensions/
- [ ] Included at least one example in examples/
- [ ] Added required dependencies to pyproject.toml
- [ ] Written basic tests that pass with FAST_TEST_MODE=1

### Before Submitting PR
- [ ] Documented public functions and classes
- [ ] Ensured compatibility with both TabPFN backends (when possible)
- [ ] All tests pass: `pytest tests/test_your_package.py`

### Advanced/Optional
- [ ] Added type hints for main functions
- [ ] Run linting with ruff: `ruff check src/tabpfn_extensions/your_package/`
- [ ] Run type checking: `mypy src/tabpfn_extensions/your_package/`

## 📜 Legal & Security

- Only contribute code you have rights to
- Don't include sensitive or private data
- All contributions must be Apache 2.0 licensed
- No machine learning models or large datasets in the repository

## 📦 Example Extension Structure

Here's a minimal example of a complete extension:

```
src/tabpfn_extensions/sample_extension/
├── __init__.py
├── core.py
└── utils.py

examples/sample_extension/
├── basic_usage.py
└── advanced_features.py

tests/
└── test_sample_extension.py
```

## 🧪 Versioning and Releases

We follow [Semantic Versioning](https://semver.org/), with version numbers in the format MAJOR.MINOR.PATCH:

- MAJOR: Incompatible API changes
- MINOR: New functionality in a backwards compatible manner
- PATCH: Backwards compatible bug fixes

## 🛠️ Need Help?

- Join our [Discord](https://discord.com/channels/1285598202732482621/) for questions
- Open an issue for bug reports
- Check out existing extensions for examples
