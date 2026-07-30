"""Visualize TabPFN's decoder head as a label-vote over training rows.

TabPFN classifies with an attention-based retrieval head: each query attends to
the training rows and predicts the attention-weighted average of their labels.
``get_decoder_readout`` recovers those per-training-row attention weights (they
sum to 1 per query), so a prediction can be read as *which* training points voted
for it and how strongly.

This script fits a TabPFNClassifier and, for four queries spanning the confidence
range, draws the readout with ``plot_decoder_readout`` over two 2D projections to
contrast what the head keys on:

* the raw feature space, where the vote weights spread across classes because
  nearby rows are not cleanly separated;
* TabPFN's target-conditioned embeddings, where the head's votes concentrate on
  the query's own class because that is the space it measures distance in.

Dataset: breast cancer (binary classification).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.interpretability import (
    class_vote,
    get_decoder_readout,
    plot_decoder_readout,
)

QUERY_TITLES = [
    "confident negative",
    "borderline (leans negative)",
    "borderline (leans positive)",
    "confident positive",
]


def pick_queries(p: np.ndarray) -> list[int]:
    """Four test rows spanning confident-negative to confident-positive."""
    order = np.argsort(p)
    split = np.clip(np.searchsorted(p[order], 0.5), 1, len(order) - 1)
    return [order[0], order[split - 1], order[split], order[-1]]


data = load_breast_cancer()
X, y, class_names = data.data, data.target, list(data.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=220,
    test_size=150,
    random_state=0,
    stratify=y,
)

clf = TabPFNClassifier(random_state=0)
clf.fit(X_train, y_train)

# Readout: attention weights from each test row over the training rows.
weights, train_idx = get_decoder_readout(clf, X_test)  # (n_test, n_train)
votes, classes = class_vote(weights, y_train)  # (n_test, 2), sums to 1 per row
p_pos = votes[:, 1]  # readout P(positive class)

queries = pick_queries(p_pos)

# TabPFN's target-conditioned train/test embeddings, averaged over the ensemble.
train_emb = clf.get_embeddings(X_test, data_source="train").mean(axis=0)
test_emb = clf.get_embeddings(X_test, data_source="test").mean(axis=0)

# Raw feature space: the head's votes spread across classes because nearby rows
# are not cleanly separated.
fig_features = plot_decoder_readout(
    weights,
    queries,
    X_train,
    X_test,
    y_train,
    class_names,
    y_test=y_test,
    query_titles=QUERY_TITLES,
    title="TabPFN decoder-head readout over raw features",
)
fig_features.savefig("decoder_readout_features.png", dpi=150, bbox_inches="tight")

# Target-conditioned embeddings: the head votes by distance in this space, so the
# weights concentrate on the query's own class.
fig_embeddings = plot_decoder_readout(
    weights,
    queries,
    X_train,
    X_test,
    y_train,
    class_names,
    y_test=y_test,
    embeddings=(train_emb, test_emb),
    query_titles=QUERY_TITLES,
    title="TabPFN decoder-head readout over target-conditioned embeddings",
)
fig_embeddings.savefig("decoder_readout_embeddings.png", dpi=150, bbox_inches="tight")

plt.show()
