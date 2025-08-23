# TabPFN Extensions ⚡

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
> #### 🧪 Experimental Code Notice
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

## ⚙️ Installation

```bash
# Clone and install the repository
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"
```

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


## 📊 TabPFN Workflow
```mermaid
---
config:
  theme: 'default'
  themeVariables:
    edgeLabelBackground: 'white'
---
graph LR
    %% 1. DEFINE COLOR SCHEME & STYLES
    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333;
    classDef start_node fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#333;
    classDef process_node fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#333;
    classDef decision_node fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#333;

    style Infrastructure fill:#fff,stroke:#ccc,stroke-width:5px;
    style Unsupervised fill:#fff,stroke:#ccc,stroke-width:5px;
    style Data fill:#fff,stroke:#ccc,stroke-width:5px;
    style Performance fill:#fff,stroke:#ccc,stroke-width:5px;
    style Interpretability fill:#fff,stroke:#ccc,stroke-width:5px;

    %% 2. DEFINE GRAPH STRUCTURE
    subgraph Infrastructure
        start((Start)) --> gpu_check["GPU available?"];
        gpu_check -- Yes --> local_version["Use TabPFN<br/>(local PyTorch)"];
        gpu_check -- No --> api_client["Use TabPFN-Client<br/>(cloud API)"];
        task_type["What is<br/>your task?"]
    end

    local_version --> task_type
    api_client --> task_type

    end_node((Workflow<br/>Complete));

    subgraph Unsupervised
        unsupervised_type["Select<br/>Unsupervised Task"];
        unsupervised_type --> imputation["Imputation"]
        unsupervised_type --> data_gen["Data<br/>Generation"];
        unsupervised_type --> density["Outlier<br/>Detection"];
        unsupervised_type --> embedding["Get<br/>Embeddings"];
    end


    subgraph Data
        data_check["Data Checks"];
        model_choice["Samples > 10k or<br/>Classes > 10?"]
        data_check -- "Table Contains Text Data?" --> api_backend_note["Note: API client has<br/>native text support"];
        api_backend_note --> model_choice;
        data_check -- "Time-Series Data?" --> ts_features["Use Time-Series<br/>Features"];
        ts_features --> model_choice;
        data_check -- "Purely Tabular" --> model_choice;
        model_choice -- "No" --> rfpfn["RF-PFN"];
        model_choice -- "Yes, >10k samples" --> subsample["Subsample<br/>Data"];
        model_choice -- "Yes, >10 classes" --> many_class["Many-Class<br/>Method"];
    end

    subgraph Performance
        finetune_check["Need<br/>Finetuning?"];
        performance_check["Need Even Better Performance?"];
        tuning_complete["Tuning Complete"];

        finetune_check -- Yes --> finetuning["Finetuning"];
        finetune_check -- No --> performance_check;

        finetuning --> performance_check;


        performance_check -- No --> tuning_complete;
        performance_check -- Yes --> hpo["HPO"];
        performance_check -- Yes --> post_hoc["Post-Hoc<br/>Ensembling"];
        performance_check -- Yes --> more_estimators["More<br/>Estimators"];

        hpo --> tuning_complete;
        post_hoc --> tuning_complete;
        more_estimators --> tuning_complete;
    end

    subgraph Interpretability

        tuning_complete --> interpretability_check;

        interpretability_check["Need<br/>Interpretability?"];

        interpretability_check -- Yes --> shapley["Explain with<br/>SHAP"];
        interpretability_check -- No --> end_node;

        shapley --> end_node;

    end

    %% 3. LINK SUBGRAPHS AND PATHS
    task_type -- "Prediction" --> data_check;
    task_type -- "Unsupervised" --> unsupervised_type;

    rfpfn --> finetune_check;
    subsample --> finetune_check;
    many_class --> finetune_check;

    %% 4. APPLY STYLES
    class start,end_node start_node;
    class local_version,api_client,imputation,data_gen,density,embedding,api_backend_note,ts_features,rfpfn,subsample,many_class,finetuning,shapley,hpo,post_hoc,more_estimators process_node;
    class gpu_check,task_type,unsupervised_type,data_check,model_choice,finetune_check,interpretability_check,performance_check decision_node;
    class tuning_complete process_node;

    %% 5. ADD CLICKABLE LINKS (RESTORED FROM ORIGINAL)
    click local_version "https://github.com/PriorLabs/TabPFN" "TabPFN Backend Options" _blank
    click api_client "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Client" _blank
    click api_backend_note "https://github.com/PriorLabs/tabpfn-client" "TabPFN API Backend" _blank
    click unsupervised_type "https://github.com/PriorLabs/tabpfn-extensions" "TabPFN Extensions" _blank
    click imputation "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/imputation.py" "TabPFN Imputation Example" _blank
    click data_gen "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py" "TabPFN Data Generation Example" _blank
    click density "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/density_estimation_outlier_detection.py" "TabPFN Density Estimation/Outlier Detection Example" _blank
    click embedding "https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/embedding" "TabPFN Embedding Example" _blank
    click ts_features "https://github.com/PriorLabs/tabpfn-time-series" "TabPFN Time-Series Example" _blank
    click rfpfn "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/rf_pfn/rf_pfn_example.py" "RF-PFN Example" _blank
    click many_class "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/many_class/many_class_classifier_example.py" "Many Class Example" _blank
    click finetuning "https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py" "Finetuning Example" _blank
    click shapley "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shap_example.py" "Shapley Values Example" _blank
    click post_hoc "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/phe/phe_example.py" "Post-Hoc Ensemble Example" _blank
    click hpo "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/hpo/tuned_tabpfn.py" "HPO Example" _blank
    click subsample "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/large_datasets/large_datasets_example.py" "Large Datasets Example" _blank
```


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

---

Built with ❤️ by the TabPFN community
