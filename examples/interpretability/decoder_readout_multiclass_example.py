"""Visualize TabPFN's decoder readout on a 4-class problem.

The multi-class analogue of ``decoder_readout_example.py``. ``get_decoder_readout``
recovers the per-training-row attention weights (summing to 1 per test row) and
``class_vote`` collapses them by training label into a class distribution, so a
prediction reads as a vote: ``argmax`` of the class votes is the predicted class.

``plot_decoder_readout`` draws, for one confident query per class, lines from the
query to its top-attended training rows, colored by the row's class and scaled by
vote weight, with the query star colored by the predicted class. As in the binary
example the readout is shown over two 2D projections:

* the raw feature space, where a query's votes scatter across neighboring classes;
* TabPFN's target-conditioned embeddings, where the votes collapse onto the query's
  own class because the head measures distance in that space.

Dataset: a synthetic 4-class classification problem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.interpretability import (
    class_vote,
    get_decoder_readout,
    plot_decoder_readout,
)

X, y = make_classification(
    n_samples=370,
    n_features=8,
    n_informative=6,
    n_redundant=0,
    n_classes=4,
    n_clusters_per_class=1,
    class_sep=1.1,
    random_state=0,
)
class_names = [f"class {c}" for c in range(4)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=220, test_size=150, random_state=0, stratify=y
)

clf = TabPFNClassifier(random_state=0)
clf.fit(X_train, y_train)

weights, train_idx = get_decoder_readout(clf, X_test)  # (n_test, n_train)
votes, classes = class_vote(weights, y_train)  # (n_test, 4), sums to 1 per row

# One confident query per class: the test row whose readout most favors that class.
queries = [int(np.argmax(votes[:, ci])) for ci in range(len(classes))]
query_titles = [f"confident {class_names[c]}" for c in classes]

# TabPFN's target-conditioned train/test embeddings, averaged over the ensemble.
train_emb = clf.get_embeddings(X_test, data_source="train").mean(axis=0)
test_emb = clf.get_embeddings(X_test, data_source="test").mean(axis=0)

fig_features = plot_decoder_readout(
    weights,
    queries,
    X_train,
    X_test,
    y_train,
    class_names,
    y_test=y_test,
    query_titles=query_titles,
    title="TabPFN decoder-head readout over raw features",
)
fig_features.savefig("decoder_readout_mc_features.png", dpi=150, bbox_inches="tight")

fig_embeddings = plot_decoder_readout(
    weights,
    queries,
    X_train,
    X_test,
    y_train,
    class_names,
    y_test=y_test,
    embeddings=(train_emb, test_emb),
    query_titles=query_titles,
    title="TabPFN decoder-head readout over target-conditioned embeddings",
)
fig_embeddings.savefig(
    "decoder_readout_mc_embeddings.png", dpi=150, bbox_inches="tight"
)

plt.show()
