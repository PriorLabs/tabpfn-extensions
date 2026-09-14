#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Predict from a picture and a few tabular fields, together.

An image goes into TabPFN as a base64 string in a DataFrame cell, in a column
declared through `image_features_indices`. `TabPFNWithImages` replaces each such
column by the PCA-reduced CLS embedding of a frozen DINOv3 ViT-S/16 before the
tabular model sees it, and moves any declared categorical positions to match. The
script fits three regressors, on the tabular fields alone, on the picture alone and
on both, and prints each R2.

The data is the Amazon Bestseller task of MulTaBench (https://arxiv.org/abs/2605.10616):
the log price of a best-selling product from its photo, rank, page and ratings,
3488 rows. The archive downloads through kagglehub without a Kaggle account. The
encoder's weights are gated: accept the license once at
https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m and log in with
`hf auth login` or `HF_TOKEN`. A GPU is recommended for the full set; TabPFN's CPU
sample guard applies to the tabular model as it always does.

    pip install "tabpfn-extensions[image]" kagglehub
    python tabpfn_with_images.py
"""

import base64
import json
import os
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNRegressor
from tabpfn_extensions.image import TabPFNWithImages

# Under FAST_TEST_MODE=1 (set by the CI example tests) the workload shrinks so
# the example finishes quickly; results are correspondingly rougher.
FAST_TEST_MODE = os.environ.get("FAST_TEST_MODE") == "1"

root = Path(kagglehub.dataset_download("chico89/multabench-amazon-bestseller"))
meta = json.loads((root / "metadata.json").read_text())
data = pd.read_csv(root / "data.csv")
if FAST_TEST_MODE:
    data = data.sample(n=300, random_state=0).reset_index(drop=True)
y = data.pop(meta["target"])
# Each cell of the image column is the picture file, base64-encoded.
data["image"] = [
    base64.b64encode((root / path).read_bytes()).decode("ascii")
    for path in data.pop(meta["image_col"])
]
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.25, random_state=0
)

tabular = [column for column in data.columns if column != "image"]
for name, columns in [
    ("tabular only", tabular),
    ("picture only", ["image"]),
    ("tabular + picture", [*tabular, "image"]),
]:
    image_indices = [i for i, column in enumerate(columns) if column == "image"]
    model = TabPFNWithImages(TabPFNRegressor(), image_features_indices=image_indices)
    model.fit(X_train[columns], y_train)
    print(f"R2, {name}: {r2_score(y_test, model.predict(X_test[columns])):.3f}")
