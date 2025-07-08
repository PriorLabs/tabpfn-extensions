# TabPFN Extensions ⚡

[![PyPI version](https://badge.fury.io/py/tabpfn-extensions.svg)](https://badge.fury.io/py/tabpfn-extensions)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/automl/tabpfn-client)

<img src="tabpfn_summary.webp" width="650" alt="TabPFN Summary">

## 🛠️ Available Extensions

- **post_hoc_ensembles**: Improve performance with model combination
- **interpretability**: Explain TabPFN predictions with SHAP values and feature selection
- **many_class**: Handle classification with more classes than TabPFN's default limit
- **classifier_as_regressor**: Use TabPFN's classifier for regression tasks
- **hpo**: Automatic hyperparameter tuning for TabPFN
- **rf_pfn**: Combine TabPFN with decision trees and random forests
- **unsupervised**: Data generation and outlier detection
- **embedding**: Get TabPFNs internal dense sample embeddings

Detailed documentation for each extension is available in the respective module directories.

## ⚙️ Installation

```bash
# Clone and install the repository
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
```

### 🔄 Backend Options

TabPFN Extensions works with two TabPFN implementations:

1. **🖥️ TabPFN Package** - Full PyTorch implementation for local inference:
   ```bash
   pip install tabpfn
   ```

2. **☁️ TabPFN Client** - Lightweight API client for cloud-based inference:
   ```bash
   pip install tabpfn-client
   ```

Choose the backend that fits your needs - most extensions work with either option!

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🧑‍💻 For Contributors

Interested in adding your own extension? We welcome contributions!

```bash
# Clone and set up for development
git clone https://github.com/PriorLabs/tabpfn-extensions.git
cd tabpfn-extensions

# Lightweight dev setup (fast)
pip install -e ".[dev]"

# Test your extension with fast mode
FAST_TEST_MODE=1 pytest tests/test_your_extension.py -v
```

See our [Contribution Guide](CONTRIBUTING.md) for more details.

[![Contributors](https://contrib.rocks/image?repo=priorlabs/tabpfn-extensions)](https://github.com/priorlabs/tabpfn-extensions/graphs/contributors)

## 📦 Repository Structure

Each extension lives in its own subpackage:

```
tabpfn-extensions/
├── src/
│   └── tabpfn_extensions/
│       └── your_package/      # Extension implementation
├── examples/
│   └── your_package/          # Usage examples
└── tests/
    └── your_package/          # Tests
```

---

Built with ❤️ by the TabPFN community


---

## 🚀 TabPFN Workflow Diagram

```mermaid
graph TD
    %% 1. DEFINE ALL STYLES
    classDef main fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef decision fill:#e0ffff,stroke:#333,stroke-width:2px;
    classDef terminal fill:#f0fff0,stroke:#28a745,stroke-width:2px;

    %% 2. DEFINE THE GRAPH STRUCTURE
    start((Start)) --> gpu_check{GPU available?};
    gpu_check -- No --> cpu_only_options["[Use TabPFN Client backend](https://github.com/automl/TabPFN) or<br/>[Local Version](https://github.com/automl/TabPFN)"];
    gpu_check -- Yes --> task_type{"Type of task?"};

    task_type -- Unsupervised --> unsupervised_type{"What kind of<br/>unsupervised task?"};
    unsupervised_type --> imputation(Imputation);
    unsupervised_type --> data_gen("Data Generation");
    unsupervised_type --> density("Density Estimation");
    unsupervised_type --> embedding("Get Embeddings");

    task_type -- "Prediction Problem" --> text_check{"Contains Text Data?"};
    text_check -- Yes --> api_backend["Consider using our API client as<br/>TabPFN backend.<br/>Natively understands text."];
    text_check -- No --> ts_check{"Time-Series Data?"};

    ts_check -- Yes --> ts_features["Consider TabPFN-Time-Series features"];
    ts_check -- No --> sample_size_check{"> 10,000 samples?"};

    ts_features --> sample_size_check;
    api_backend --> sample_size_check;

    sample_size_check -- No --> class_check{"> 10 classes?"};
    sample_size_check -- Yes --> subsample["TabPFN subsample<br/>samples 10,000"];

    class_check -- No --> rfpfn("RF-PFN");
    class_check -- Yes --> many_class("Many Class");

    subsample --> finetune_check{"Need to Finetune?"};
    rfpfn --> finetune_check;
    many_class --> finetune_check;

    finetune_check -- Yes --> finetuning("Finetuning");
    finetune_check -- No --> interpretability_check{"Need Interpretability?"};

    finetuning --> performance_check{"Performance not<br/>good enough?"};
    interpretability_check -- Yes --> shapley("Shapley Values for TabPFN");
    interpretability_check -- No --> performance_check;
    shapley --> performance_check;

    performance_check -- No --> congrats((Congrats!));
    performance_check -- Yes --> tuning_options("Tuning Options");
    tuning_options --> more_estimators("More estimators on TabPFN");
    tuning_options --> hpo("HPO for TabPFN");
    tuning_options --> post_hoc("Post-Hoc-Ensemble<br/>(AutoTabPFN)");

    %% 3. APPLY STYLES TO NODES
    class start,gpu_check,task_type,unsupervised_type,text_check,ts_check,sample_size_check,class_check,finetune_check,interpretability_check,performance_check decision;
    class congrats terminal;
    class cpu_only_options,imputation,data_gen,density,embedding,api_backend,ts_features,subsample,rfpfn,many_class,finetuning,shapley,more_estimators,hpo,post_hoc,tuning_options main;