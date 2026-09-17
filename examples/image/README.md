# Pictures as TabPFN columns

`TabPFNWithImages` lets a TabPFN estimator take pictures alongside tabular fields. Each cell of a declared image column holds one image, as the path of the image file, absolute or relative to the working directory, as a base64 string (a `data:image/...;base64,` prefix is fine), as the file's bytes or as a PIL image. At `fit` the column is replaced by `n_components` numeric features, 30 by default: the CLS embedding of a frozen [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), standardised and reduced by a PCA fit on the training rows. At `predict` the same encoder and PCA are applied. The wrapped estimator only ever sees numbers, so the local `tabpfn` package and the TabPFN client both work. `ImageTransformer` is the same step as a standalone scikit-learn transformer, for pipelines of your own.

## Setup

```bash
pip install "tabpfn-extensions[image]"
```

The encoder's weights are gated on the Hugging Face Hub: accept the license once on the model page, then run `hf auth login` or set `HF_TOKEN`.

## Usage

```python
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.image import TabPFNWithImages

model = TabPFNWithImages(
    TabPFNClassifier(categorical_features_indices=[1]),
    image_features_indices=[3],  # positions in X, like categorical_features_indices
)
model.fit(X_train, y_train)  # X is a DataFrame; column 3 holds paths, base64, bytes or PIL images
proba = model.predict_proba(X_test)
```

Positions in `image_features_indices` and in the estimator's `categorical_features_indices` both refer to the frame you pass in; the wrapper moves the categorical positions to the expanded frame itself. A path is read only when its batch is embedded, so a column of paths costs no more memory than the paths; a table that holds file names next to a folder becomes one in a line, `X["photo"] = [str(folder / name) for name in X["photo"]]`. A cell that is missing or cannot be decoded as an image is refused by row rather than guessed at, at fit and at predict alike. The encoder is downloaded on first use and cached for the process; it is not stored on the fitted model, which pickles without it.

## Example

`tabpfn_with_images.py` fits the Amazon Bestseller task of [MulTaBench](https://arxiv.org/abs/2605.10616): the log price of a best-selling product from its photo, rank, page and ratings. It prints the RMSE of three regressors, on the numeric fields alone, on the picture alone and on both. The archive is fetched with `kagglehub` (`pip install kagglehub`, no Kaggle account needed). Under `FAST_TEST_MODE=1` the script runs on a 300-row sample.
