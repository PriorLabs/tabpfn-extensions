# TabPFN Embeddings

`TabPFNEmbedding` is a scikit-learn style transformer that extracts embeddings from `TabPFNClassifier` or `TabPFNRegressor`. It supports both vanilla embeddings and K-fold cross-validated (out-of-fold) embeddings, which can be used to enhance the performance of downstream machine learning models.

## Key Features

- **Vanilla embeddings**: with `n_fold=0`, the model is trained once on the entire training set and used to embed both train and test data.
- **K-fold cross-validation**: with `n_fold>=2`, out-of-fold (OOF) embeddings are produced for the training data (the robust method from *"A Closer Look at TabPFN v2: Strength, Limitation, and Extension"*). A final model is also trained on the full training set and used to embed unseen data.
- **scikit-learn-style API**: `fit` / `transform` / `fit_transform`.

## Usage

```python
from tabpfn_extensions.embedding import TabPFNEmbedding

embedding = TabPFNEmbedding(n_fold=5)
train_embeds = embedding.fit_transform(X_train, y_train)  # OOF embeddings
test_embeds = embedding.transform(X_test)                 # full-data model
```

`transform(X)` always uses the final full-data model. It does **not** return
cached OOF embeddings, even when `X` happens to equal the training set — use
`fit_transform(X_train, y_train)` (or read `embedding.train_embeddings_`)
when you want OOF.

**Output shape.** `transform` returns a 3D array of shape
`(n_estimators, n_samples, embed_dim)`. This is not a drop-in input for
`sklearn.pipeline.Pipeline` / `ColumnTransformer`, which expect 2D arrays —
select an ensemble member (`embeds[0]`) or aggregate across `axis=0` before
passing the embeddings to a downstream 2D estimator.

> The legacy `TabPFNEmbedding.get_embeddings(...)` method and the
> `tabpfn_clf=` / `tabpfn_reg=` constructor kwargs are deprecated;
> prefer the `fit` / `transform` interface and `model=` in new code. The
> new `fit` path also uses `StratifiedKFold` for classifiers (the legacy
> path used plain `KFold`), so OOF numbers may differ.

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
