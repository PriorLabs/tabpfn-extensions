# TabPFN Embeddings

The `TabPFNEmbedding` class is a utility for extracting embeddings from `TabPFNClassifier` or `TabPFNRegressor` models. It supports both standard training (vanilla embeddings) and K-fold cross-validation for embedding extraction. The embeddings can be used to enhance the performance of downstream machine learning models.

## Key Features

- **Vanilla Embeddings**: Extract embeddings by training on the entire dataset when `n_fold=0`.
- **K-Fold Cross-Validation**: Improve embedding effectiveness by using K-fold cross-validation when `n_fold>0`, as proposed in the paper *"A Closer Look at TabPFN v2: Strength, Limitation, and Extension"*.

## Usage

To use `TabPFNEmbedding`, initialize it with either a `TabPFNClassifier` or `TabPFNRegressor` and specify the number of folds (`n_fold`). Then, call the `get_embeddings` method to generate embeddings for your dataset.

### Data Source Parameter

`data_source` determines whether embeddings from the training or test tokens are
returned. See the docstring of ``TabPFNEmbedding.get_embeddings`` for details on
the options and examples of how to combine embeddings when using an ensemble.



## Citing This Work

If you use this utility in your research, please cite the following paper:

```bibtex
@misc{ye2025closerlooktabpfnv2,
      title={A Closer Look at TabPFN v2: Strength, Limitation, and Extension},
      author={Han-Jia Ye and Si-Yang Liu and Wei-Lun Chao},
      year={2025},
      eprint={2502.17361},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.17361},
}
```
