"""Visualize TabPFN's decoder head as a label-vote over training rows.

TabPFN classifies with an attention-based retrieval head: each query attends to
the training rows and predicts the attention-weighted average of their labels.
``get_decoder_readout`` recovers those per-training-row attention weights (they
sum to 1 per query), so a prediction can be read as *which* training points voted
for it and how strongly.

This script fits a TabPFNClassifier, projects the model's training embeddings to
2D (UMAP, falling back to PCA), and for four queries spanning the confidence
range draws the readout: lines from the query to its most-attended training rows,
colored by the row's class and scaled by its vote weight. Summing the red (positive
class) weights gives the readout probability, which matches ``predict_proba``.

Dataset: breast cancer (binary classification).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.interpretability import class_vote, get_decoder_readout

TOP_K = 20  # how many top-voting training rows to draw per query
BLUE, RED = "#2c7fb8", "#d6404e"  # negative / positive class

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


def pick_queries(p: np.ndarray) -> list[int]:
    """Four test rows spanning confident-negative to confident-positive."""
    order = np.argsort(p)
    lean_neg = order[np.searchsorted(p[order], 0.5) - 1]
    lean_pos = order[np.searchsorted(p[order], 0.5)]
    return [order[0], lean_neg, lean_pos, order[-1]]


queries = pick_queries(p_pos)

# 2D projection of the model's training embeddings (averaged over the ensemble).
train_emb = clf.get_embeddings(X_test, data_source="train").mean(axis=0)
query_emb = clf.get_embeddings(X_test[queries], data_source="test").mean(axis=0)

try:
    from umap import UMAP

    reducer = UMAP(n_components=2, random_state=0)
    proj_name = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=2)
    proj_name = "PCA"

Z_train = reducer.fit_transform(train_emb)
Z_query = reducer.transform(query_emb)


def class_density(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray) -> None:
    """Soft KDE background, one contour set per class."""
    for label, cmap in ((0, "Blues"), (1, "Reds")):
        pts = Z_train[y_train == label]
        density = gaussian_kde(pts.T)(np.vstack([xx.ravel(), yy.ravel()]))
        ax.contourf(xx, yy, density.reshape(xx.shape), levels=6, cmap=cmap, alpha=0.18)


pad = 0.08 * np.ptp(Z_train, axis=0)
lo, hi = Z_train.min(axis=0) - pad, Z_train.max(axis=0) + pad
xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))

fig, axes = plt.subplots(2, 2, figsize=(15, 13))
titles = [
    "confident negative",
    "borderline (leans negative)",
    "borderline (leans positive)",
    "confident positive",
]

for pos, (ax, q, title) in enumerate(zip(axes.ravel(), queries, titles, strict=True)):
    class_density(ax, xx, yy)
    for label, color in ((0, BLUE), (1, RED)):
        m = y_train == label
        ax.scatter(*Z_train[m].T, s=12, color=color, alpha=0.35, linewidths=0)

    w = weights[q]
    top = np.argsort(w)[-TOP_K:]
    w_max = w[top].max()
    qx, qy = Z_query[pos]
    for j in top:
        color = RED if y_train[j] == 1 else BLUE
        frac = w[j] / w_max
        ax.plot(
            [qx, Z_train[j, 0]],
            [qy, Z_train[j, 1]],
            color=color,
            lw=1 + 6 * frac,
            alpha=0.25 + 0.6 * frac,
            zorder=2,
        )
        ax.scatter(
            *Z_train[j], s=90, color=color, edgecolor="white", linewidths=0.8, zorder=3
        )

    pred = int(p_pos[q] >= 0.5)
    ax.scatter(
        qx,
        qy,
        marker="*",
        s=650,
        color=(RED if pred else "#dddddd"),
        edgecolor="black",
        linewidths=1.6,
        zorder=4,
    )
    ax.set(xticks=[], yticks=[])
    ax.set_title(
        f"{title}\nreadout P({class_names[1]}) = Σ(red weights) = {p_pos[q]:.2f}"
        f"  ·  true = {class_names[y_test[q]]}"
        f"  ·  top-{TOP_K} hold {w[top].sum():.0%} of the vote",
        fontsize=11,
    )

handles = [
    plt.Line2D([], [], color=RED, lw=4, label=f"{class_names[1]} vote"),
    plt.Line2D([], [], color=BLUE, lw=4, label=f"{class_names[0]} vote"),
    plt.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor=RED,
        markersize=10,
        label=f"{class_names[1]} patient",
    ),
    plt.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor=BLUE,
        markersize=10,
        label=f"{class_names[0]} patient",
    ),
    plt.Line2D(
        [],
        [],
        marker="*",
        color="w",
        markerfacecolor="#999999",
        markeredgecolor="black",
        markersize=18,
        label="query",
    ),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)
fig.suptitle(
    f"TabPFN decoder-head readout — the prediction as a label-vote over training "
    f"rows ({proj_name} of embeddings, decoder attention weights, Σ = 1)",
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
plt.show()
