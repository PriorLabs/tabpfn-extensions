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
*Note: The nodes in the workflow diagram above are clickable and link to relevant documentation and examples.*

```mermaid
graph LR
    %% 1. DEFINE STYLES
    classDef start_node fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#333;
    classDef end_node fill:#f0fff0,stroke:#28a745,stroke-width:2px;
    classDef process_node fill:#e0f7fa,stroke:#007bff,stroke-width:2px,color:#333;
    classDef decision_node fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#333;

    %% 2. DEFINE GRAPH STRUCTURE
    subgraph "⚙️ 1. Setup"
        start((Start)) --> gpu_check{GPU?};
        gpu_check -- Yes --> local_version("Use TabPFN<br/>(local PyTorch)");
        gpu_check -- No --> api_client("Use TabPFN-Client<br/>(cloud API)");
    end

    %% Main Branching Point
    task_type{"What is your task?"}
    local_version --> task_type
    api_client --> task_type

    %% Define End Node early for linking
    end_node((Workflow Complete ✨));

    %% Unsupervised Path
    subgraph "🔮 2a. Unsupervised Tasks"
        unsupervised_type{"Select Task"};
        unsupervised_type --> data_gen("Data Generation");
        unsupervised_type --> density("Outlier Detection");
        unsupervised_type --> embedding("Get Embeddings");
    end
    data_gen --> end_node;
    density --> end_node;
    embedding --> end_node;

    %% Supervised Path (Series of Subgraphs)
    subgraph "🎯 2b. Supervised Pipeline"
        data_check{"Data Checks"};
        model_choice{"Samples > 10k or<br/>Classes > 10?"}
        
        data_check -- "Text Data?" --> api_backend_note["Note: API client has<br/>native text support"];
        api_backend_note --> model_choice;
        data_check -- "Time-Series?" --> ts_features["Use Time-Series<br/>Features"];
        ts_features --> model_choice;
        data_check -- "Tabular" --> model_choice;
    end

    subgraph "⚙️ 3. Model Selection"
        rfpfn("RF-PFN");
        subsample("Subsample Data");
        many_class("Many-Class Method");
        
        model_choice -- "No" --> rfpfn;
        model_choice -- "Yes, >10k samples" --> subsample;
        model_choice -- "Yes, >10 classes" --> many_class;
    end

    subgraph "🛠️ 4. Post-Training Steps"
        finetune_check{"Need Finetuning?"};
        finetuning("Finetuning");
        interpretability_check{"Need Interpretability?"};
        shapley("Explain with SHAP");
        
        finetune_check -- Yes --> finetuning;
        finetune_check -- No --> interpretability_check;
        finetuning --> interpretability_check;
        interpretability_check -- Yes --> shapley;
    end

    subgraph "🚀 5. Performance Tuning"
        performance_check{"Performance OK?"};
        hpo("HPO");
        post_hoc("Post-Hoc Ensembling");

        performance_check -- Yes --> end_node;
        performance_check -- No --> hpo;
        performance_check -- No --> post_hoc;
        hpo --> end_node;
        post_hoc --> end_node;
    end
    
    %% 3. LINK SUBGRAPHS AND PATHS
    task_type -- Prediction --> data_check;
    task_type -- Unsupervised --> unsupervised_type;

    rfpfn --> finetune_check;
    subsample --> finetune_check;
    many_class --> finetune_check;
    
    interpretability_check -- No --> performance_check;
    shapley --> performance_check;

    %% 4. APPLY STYLES
    class start,end_node start_node;
    class gpu_check,task_type,unsupervised_type,data_check,model_choice,finetune_check,interpretability_check,performance_check decision_node;
    class local_version,api_client,data_gen,density,embedding,api_backend_note,ts_features,rfpfn,subsample,many_class,finetuning,shapley,hpo,post_hoc process_node;

    %% 5. ADD CLICKABLE LINKS (abbreviated for clarity)
    click local_version "https://github.com/PriorLabs/TabPFN" "TabPFN Backend Options" _blank

