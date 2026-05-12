# TabPFN Extensions

[![PyPI version](https://badge.fury.io/py/tabpfn-extensions.svg)](https://badge.fury.io/py/tabpfn-extensions)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/automl/tabpfn-client)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

> [!WARNING]
>
> #### Experimental Code Notice
> Please note that the extensions in this repository are experimental.
> -   They are less rigorously tested than the core `tabpfn` library.
> -   APIs are subject to change without notice in future releases.
> We welcome your feedback and contributions to help improve and stabilize them!

## Interactive Notebook Tutorial
> [!TIP]
>
> Dive right in with our interactive Colab notebook! It's the best way to get a hands-on feel for TabPFN, walking you through installation, classification, and regression examples.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

## Installation

```bash
# Clone and install the repository
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
```

## Available Extensions

- **interpretability**: Explain TabPFN predictions with SHAP values and feature selection
- **many_class**: Handle classification problems with more classes than your TabPFN checkpoint supports
- **classifier_as_regressor**: Use TabPFN's classifier for regression tasks
- **unsupervised**: Data generation and outlier detection
- **embedding**: Get TabPFN's internal dense sample embeddings
- **tabebm**: Data augmentation using TabPFN-based Energy-Based Models
- **pval_crt**: Statistical feature relevance testing (p-values)
- **post_hoc_ensembles** *(deprecated)*: `AutoTabPFN*` — improve performance with model combination via AutoGluon. Scheduled for removal in a future release.
- **hpo** *(deprecated)*: `TunedTabPFN*` — automatic hyperparameter tuning for TabPFN via Hyperopt. Scheduled for removal in a future release.

See the [Documentation](#documentation) section below for guides, examples, and per-extension READMEs.

### Backend Options

Many TabPFN Extensions works with two TabPFN implementations:

1. ** TabPFN Package** - Full PyTorch implementation for local inference:
   ```bash
   pip install tabpfn
   ```

2. ** TabPFN Client** - Lightweight API client for cloud-based inference:
   ```bash
   pip install tabpfn-client
   ```

Choose the backend that fits your needs - most extensions work with either option!

Exceptions to this are **post_hoc_ensembles** *(deprecated)* and **embedding**, which only work with the local `tabpfn` package.

## Documentation

Documentation for `tabpfn-extensions` is spread across several sources. If you are new to the project, the [examples](#examples) are usually the fastest way to get started; for deeper conceptual guides, see the [TabPFN Docs pages](#tabpfn-docs-pages).

### Examples

Runnable scripts and notebooks for extensions and general use cases live in the [`examples/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples) directory of this repository:

- [`embedding/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/embedding) — access TabPFN's internal dense sample embeddings
- [`interpretability/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/interpretability) — SHAP values, partial dependence plots, feature selection
- [`many_class/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/many_class) — classification with more classes than your checkpoint supports
- [`pval_crt/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/pval_crt) — statistical feature relevance testing
- [`survival/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/survival) — survival analysis
- [`tabebm/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/tabebm) — data augmentation via TabEBM
- [`unsupervised/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/unsupervised) — data generation, imputation, and outlier detection
- [`hpo/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/hpo) *(deprecated)* — `TunedTabPFN*` automatic hyperparameter tuning
- [`phe/`](https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/phe) *(deprecated)* — `AutoTabPFN*` post-hoc ensembles

### TabPFN Docs pages

In-depth guides for selected extensions are available on [docs.priorlabs.ai](https://docs.priorlabs.ai):

- [Many-class](https://docs.priorlabs.ai/extensions/many-class)

### Per-extension READMEs

Some extensions ship a dedicated README alongside their source code:

- [`interpretability/`](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/interpretability/README.md)
- [`pval_crt/`](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/pval_crt/README.md)
- [`tabebm/`](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/tabebm/README.md)
- [`hpo/`](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/hpo/README.md) *(deprecated)*
- [`post_hoc_ensembles/`](https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/post_hoc_ensembles/README.md) *(deprecated)*

### Interactive notebook

The main TabPFN demo notebook also covers several extensions — in particular the [unsupervised](https://github.com/PriorLabs/tabpfn-extensions/tree/main/src/tabpfn_extensions/unsupervised) and [interpretability](https://github.com/PriorLabs/tabpfn-extensions/tree/main/src/tabpfn_extensions/interpretability) extensions:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Telemetry

For details on telemetry, please see our [Telemetry Reference](https://github.com/PriorLabs/TabPFN/blob/main/TELEMETRY.md) and our [Privacy Policy](https://priorlabs.ai/privacy_policy/).

## For Contributors

Interested in adding your own extension? We welcome contributions!

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the project's environment, so install that first.

```bash
# Clone and set up for development
git clone https://github.com/PriorLabs/tabpfn-extensions.git
cd tabpfn-extensions
uv sync
source .venv/bin/activate

# If you add optional dependencies for your extension in pyproject.toml, install them
# like this
uv sync --extra [your extension name]

# Test your extension with fast mode
FAST_TEST_MODE=1 pytest tests/test_your_extension.py -v
```

See our [Contribution Guide](CONTRIBUTING.md) for more details.

[![Contributors](https://contrib.rocks/image?repo=priorlabs/tabpfn-extensions)](https://github.com/priorlabs/tabpfn-extensions/graphs/contributors)

---
Built with ❤️ by the TabPFN community
