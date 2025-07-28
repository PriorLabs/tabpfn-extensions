# AutoTabPFN Ensembles

This module provides an AutoGluon-powered system for building robust, high-performance Post-Hoc Ensembles (PHE) of TabPFN models. It automates the process of hyperparameter tuning and ensembling to achieve state-of-the-art results on tabular data tasks.


## Overview

This system leverages AutoGluon's powerful ensembling capabilities to combine multiple TabPFN models. It works by performing a random search across the TabPFN hyperparameter space to generate a diverse set of model configurations. These configurations are then used to train individual TabPFN models, and AutoGluon builds a weighted ensemble from the best performers.

The core process is as follows:
1.  A random search is conducted across the TabPFN hyperparameter space.
2.  A specified number of configurations (`n_ensemble_models`) are sampled from this search.
3.  AutoGluon trains a TabPFN model for each sampled configuration.
4.  AutoGluon builds a final, optimized ensemble model from these individual models.

---

## Core Parameters

While the system provides a comprehensive scikit-learn compatible interface, control is centered around a few key parameters.

### AutoGluon Control Parameters

These parameters manage the AutoGluon ensembling process:

* `presets`: Controls the trade-off between training time and predictive accuracy. Options range from `'medium_quality'` to `'best_quality'`.
* `phe_init_args`: A dictionary of arguments passed directly to the AutoGluon `TabularPredictor` constructor for advanced customization.
* `phe_fit_args`: A dictionary of arguments passed to the AutoGluon `TabularPredictor.fit()` method to control the training process.

### TabPFN Model Parameters

These parameters are applied to the underlying TabPFN models within the ensemble.

* `n_estimators` (int, default=16): The number of internal transformers to ensemble within *each* individual TabPFN model. Higher values can significantly improve performance but also increase computational resource usage.
* `balance_probabilities` (bool, default=False): If `True`, balances the output probabilities from TabPFN. This can be highly beneficial for classification tasks with imbalanced classes. This parameter is applied uniformly to all models in the ensemble and is not part of the random hyperparameter search.
* `ignore_pretraining_limits` (bool, default=False): If `True`, this bypasses TabPFN's built-in limits on dataset size (10000 samples) and feature count (500). **Warning:** Use this with caution, as model performance is not guaranteed and may be poor when exceeding these recommended limits.


---

## Getting Started

### Installation

Please ensure all dependencies are installed.

```bash
pip install "tabpfn-extensions[post_hoc_ensembles]"