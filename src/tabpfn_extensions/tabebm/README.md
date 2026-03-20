# TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models

<h2 align="center">
  <img src="https://s2.loli.net/2024/10/01/uJjKCNfqhFcXyPM.png" height="200px">
</h2>

<div align="center">

[![Arxiv-Paper](https://img.shields.io/badge/Arxiv-Paper-olivegreen)](https://arxiv.org/abs/2409.16118)
[![License: Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg)](https://github.com/andreimargeloiu/TabEBM/blob/master/LICENSE)

[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreimargeloiu/TabEBM/blob/main/tutorials/tutorial3_augment_real_world_data.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Downloads](https://static.pepy.tech/badge/tabebm)](https://pypi.org/project/tabebm/)

</div>

Official code for the paper ["TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models"](https://arxiv.org/abs/2409.16118), published in the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).

Authored by [Andrei Margeloiu\*](https://www.linkedin.com/in/andreimargeloiu/), [Xiangjian Jiang\*](https://silencex12138.github.io/), [Nikola Simidjievski](https://simidjievskin.github.io/), [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/), University of Cambridge, UK

**Primary Maintainer:** Xiangjian Jiang

## 📌 Overview

![image-20241001125640288](https://s2.loli.net/2024/10/01/5loipZJdOrtVS3Q.png)

**TL;DR:** We introduce a high-performance tabular data augmentation method that is fast, requires no additional training, and can be applied to any downstream predictive model. The optimized implementation features advanced caching, GPU acceleration, and memory-efficient SGLD sampling.

**Abstract:** Data collection is often difficult in critical fields such as medicine, physics, and chemistry. As a result, classification methods usually perform poorly with these small datasets, leading to weak predictive performance. Increasing the training set with additional synthetic data, similar to data augmentation in images, is commonly believed to improve downstream classification performance. However, current tabular generative methods that learn either the joint distribution p(x,y) p(\mathbf{x}, y) or the class-conditional distribution p(x∣y) p(\mathbf{x} \mid y) often overfit on small datasets, resulting in poor-quality synthetic data, usually worsening classification performance compared to using real data alone. To solve these challenges, we introduce TabEBM, a novel class-conditional generative method using Energy-Based Models (EBMs). Unlike existing methods that use a shared model to approximate all class-conditional densities, our key innovation is to create distinct EBM generative models for each class, each modelling its class-specific data distribution individually. This approach creates robust energy landscapes, even in ambiguous class distributions. Our experiments show that TabEBM generates synthetic data with higher quality and better statistical fidelity than existing methods. When used for data augmentation, our synthetic data consistently improves the classification performance across diverse datasets of various sizes, especially small ones.

## 📖 Citation

For attribution in academic contexts, please cite this work as:

```
@article{margeloiu2024tabebm,
	title={TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models},
	author={Andrei Margeloiu and Xiangjian Jiang and Nikola Simidjievski and Mateja Jamnik},
	journal={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
	year={2024},
}
```

## 🔑 Features

- **Optimized TabEBM Implementation**: High-performance TabEBM with advanced caching, memory optimizations, and GPU acceleration
- **Fast Synthetic Data Generation**: Generate synthetic tabular data with minimal configuration and improved speed
- **TabPFN-v2 Integration**: Seamless integration with the latest TabPFN-v2 for enhanced gradient-based sampling
- **Memory-Efficient SGLD**: Optimized Stochastic Gradient Langevin Dynamics with pre-computed noise tensors
- **Comprehensive Tutorials**: Three interactive notebooks covering data generation, real-world augmentation, and density analysis

### 🔥 Performance Optimizations

- **Model Caching**: Intelligent caching system to avoid redundant model training across classes
- **Vectorized Operations**: Optimized tensor operations and reduced device transfers for faster computation
- **Stratified Sampling**: Smart subsampling for large datasets while maintaining class balance
- **Gradient Computation**: Enhanced gradient-based sampling with TabPFN-v2's energy landscapes
- **Memory Management**: Pre-allocated tensors and efficient memory usage patterns
