#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
"""Read out TabPFN's classification head as a label-vote over training rows.

TabPFN classifies with an attention-based retrieval head (``ManyClassDecoder``):
each test row attends to the training rows, and the prediction is the average of
their one-hot labels weighted by that attention. The prediction is therefore a
weighted vote, and ``P(class c)`` for a test row is the sum of its attention
weights over the training rows whose label is ``c``.

``get_decoder_readout`` recovers those per-training-row attention weights, so you
can see *which* training points drive a prediction and by how much. For each test
row the weights sum to 1 (averaged over the decoder's attention heads and over the
ensemble members). Collapsing them by training label with ``class_vote`` reproduces
the model's ``predict_proba`` up to the head's log-clamping, at the classifier's
default ``softmax_temperature=0.9`` and ``balance_probabilities=False``. Both
settings are applied to the decoder's logits *after* this readout, so a
non-default ``softmax_temperature`` or ``balance_probabilities=True`` will make
``class_vote`` diverge further from ``predict_proba``.

Only the local ``tabpfn`` backend is supported: the client/API backend does not
expose the model internals this reads from. Row subsampling
(``TabPFNClassifier(..., subsample_samples=...)``) is not supported, since the
weight columns would no longer align to a single set of training rows.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from sklearn.base import BaseEstimator


def _find_decoder(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the ``ManyClassDecoder`` submodule of a fitted TabPFN model."""
    for module in model.modules():
        if type(module).__name__ == "ManyClassDecoder":
            return module
    raise RuntimeError(
        "No ManyClassDecoder found in the model. get_decoder_readout only "
        "supports TabPFN classification models.",
    )


def _row_attention_weights(
    decoder: torch.nn.Module,
    train_embeddings: torch.Tensor,  # (B, N, E)
    test_embeddings: torch.Tensor,  # (B, M, E)
) -> torch.Tensor:
    """Per-train-row attention weights (B, M, N), averaged over heads.

    Replays the decoder's query/key projection, optional softmax scaling and
    scaled-dot-product softmax over the training rows. Mirrors the internal
    forward pass but returns the attention distribution itself rather than the
    label-weighted average, so ``weights[..., n]`` is the vote mass on train row
    ``n`` and rows sum to 1.
    """
    B, M, _ = test_embeddings.shape
    N = train_embeddings.shape[1]
    head_dim, num_heads = decoder.head_dim, decoder.num_heads

    q = decoder.q_projection(test_embeddings).view(B, M, num_heads, head_dim)
    if train_embeddings.dtype != q.dtype:
        train_embeddings = train_embeddings.to(q.dtype)
    k = decoder.k_projection(train_embeddings).view(B, N, num_heads, head_dim)
    if decoder.softmax_scaling_layer is not None:
        q = decoder.softmax_scaling_layer(q, N)

    scores = torch.einsum("bmhd,bnhd->bhmn", q, k).float() / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    return attn.mean(dim=1)  # average over heads -> (B, M, N)


def get_decoder_readout(
    estimator: BaseEstimator,
    X: np.ndarray,
    *,
    average_over_estimators: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the decoder-head attention weights over training rows.

    Args:
        estimator: A fitted local ``TabPFNClassifier``.
        X: Test inputs, shape ``(n_test, n_features)``.
        average_over_estimators: If True (default), average the weights over the
            preprocessing ensemble members, returning one weight matrix. If
            False, keep the per-member weights on a leading axis.

    Returns:
        ``(weights, train_indices)``.

        ``weights`` has shape ``(n_test, n_train)`` when
        ``average_over_estimators`` is True, else
        ``(n_estimators, n_test, n_train)``. Along the training axis the weights
        are non-negative and sum to 1 for each test row: ``weights[i, j]`` is the
        attention test row ``i`` pays to training row ``j``.

        ``train_indices`` has shape ``(n_train,)`` and indexes the columns of
        ``weights`` into the rows of the data the model was fit on (i.e.
        ``weights[:, k]`` refers to training row ``train_indices[k]``).
    """
    check_is_fitted(estimator)
    try:
        model = estimator.model_
    except AttributeError as err:
        raise TypeError(
            "get_decoder_readout requires a local tabpfn TabPFNClassifier; the "
            "estimator does not expose a `model_` (the client/API backend is not "
            "supported).",
        ) from err
    except ValueError as err:
        raise NotImplementedError(
            "get_decoder_readout does not support multi-model ensembles.",
        ) from err

    subsample_samples = estimator.inference_config_.SUBSAMPLE_SAMPLES
    if subsample_samples is not None:
        raise NotImplementedError(
            "get_decoder_readout does not support row subsampling "
            f"(inference_config SUBSAMPLE_SAMPLES={subsample_samples!r}): each "
            "estimator would attend over a different subset of training rows, so "
            "the weight columns would no longer align to a single set of rows. "
            "Refit with SUBSAMPLE_SAMPLES=None.",
        )

    decoder = _find_decoder(model)
    captured: list[np.ndarray] = []

    def hook(module: torch.nn.Module, args: tuple) -> None:
        train_embeddings, test_embeddings = args[0], args[1]
        weights = _row_attention_weights(module, train_embeddings, test_embeddings)
        captured.append(weights.detach().to(torch.float32).cpu().numpy())

    handle = decoder.register_forward_pre_hook(hook)
    try:
        estimator.predict(X)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "The decoder head was never invoked during predict; cannot read out "
            "attention weights.",
        )

    if any(w.shape[-2] != len(X) for w in captured):
        raise NotImplementedError(
            "get_decoder_readout does not support test-row chunking: the decoder "
            "ran on partial test batches (e.g. fit_mode='fit_with_cache' with more "
            "than max_batched_test_rows rows), so the captured weights no longer "
            "map to a single (n_test, n_train) matrix. Predict on fewer rows or "
            "refit with fit_mode='fit_preprocessors'.",
        )

    weights = np.concatenate(captured, axis=0)  # (n_estimators, n_test, n_train)
    train_indices = np.arange(weights.shape[-1])
    if average_over_estimators:
        weights = weights.mean(axis=0)
    return weights, train_indices


def class_vote(
    weights: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-row readout weights into a per-class vote.

    Sums the attention weights within each training label, turning the readout
    into a class distribution. Averaged over the ensemble, this reproduces the
    model's ``predict_proba`` up to the head's log-clamping, at the classifier's
    default ``softmax_temperature=0.9`` and ``balance_probabilities=False``
    (both are applied downstream of this readout, so non-default values widen
    the gap to ``predict_proba``).

    Args:
        weights: Readout weights ``(n_test, n_train)`` from ``get_decoder_readout``.
        y_train: Training labels aligned to the weight columns, shape ``(n_train,)``.
        classes: Class order for the output columns. Defaults to the sorted unique
            labels of ``y_train``.

    Returns:
        ``(votes, classes)`` where ``votes`` has shape ``(n_test, n_classes)`` and
        each row sums to 1, and ``classes`` is the class order of its columns.
    """
    y_train = np.asarray(y_train)
    if classes is None:
        classes = np.unique(y_train)
    votes = np.stack([weights[:, y_train == c].sum(axis=1) for c in classes], axis=1)
    return votes, classes


_NEG_COLOR, _POS_COLOR = "#2c7fb8", "#d6404e"  # negative / positive class


def _project_2d(
    train_vecs: np.ndarray,
    query_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Project train and query vectors to 2D (UMAP, falling back to PCA)."""
    try:
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=0)
        name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        warnings.warn(
            "umap-learn is not installed; falling back to PCA for the 2D "
            "projection. Install umap-learn for a projection that better "
            "separates the classes.",
            stacklevel=2,
        )
        reducer = PCA(n_components=2)
        name = "PCA"

    return reducer.fit_transform(train_vecs), reducer.transform(query_vecs), name


def plot_decoder_readout(
    weights: np.ndarray,
    queries: list[int],
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    class_names: list[str],
    *,
    y_test: np.ndarray | None = None,
    embeddings: tuple[np.ndarray, np.ndarray] | None = None,
    query_titles: list[str] | None = None,
    title: str = "TabPFN decoder-head readout",
    top_k: int = 20,
) -> Figure:
    """Draw the binary decoder readout for a set of queries over a 2D projection.

    Each panel places one query and its ``top_k`` most-attended training rows on a
    2D projection of the rows, drawing a line from the query to each attended row
    colored by the row's class and scaled by its vote weight. The projection uses
    ``embeddings`` (a ``(train_vecs, test_vecs)`` pair, e.g. TabPFN's
    target-conditioned embeddings from ``get_embeddings``) when given, else UMAP
    (falling back to PCA) over the raw ``train_features``/``test_features``.
    Contrasting the two shows what the head keys on: distance in the embedding
    space, where votes concentrate on the query's own class, versus the raw feature
    space, where that locality is weaker.

    All test-row arrays (``weights``, ``test_features``, the test vectors in
    ``embeddings``, ``y_test``) span the full test set; ``queries`` indexes into
    them to select the rows to draw.

    Binary classification only; ``class_names[1]`` is treated as the positive class.

    Args:
        weights: Readout weights ``(n_test, n_train)`` from ``get_decoder_readout``.
        queries: Indices into the test set of the rows to draw, one per panel.
        train_features: Raw training features ``(n_train, n_features)``.
        test_features: Raw test features ``(n_test, n_features)``; the queried rows
            are selected internally.
        y_train: Training labels aligned to the weight columns, ``(n_train,)``.
        class_names: ``[negative_name, positive_name]``.
        y_test: Optional test labels ``(n_test,)``; when given, each panel is
            annotated with the query's true class.
        embeddings: Optional ``(train_vecs, test_vecs)`` with ``test_vecs`` spanning
            the full test set, projected instead of the raw features.
        query_titles: Optional per-panel labels, aligned to ``queries``.
        title: Figure title; the projection name is appended.
        top_k: Number of top-voting training rows to draw per query.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    train_vecs, test_vecs = (
        embeddings if embeddings is not None else (train_features, test_features)
    )
    Z_train, Z_query, proj_name = _project_2d(train_vecs, test_vecs[queries])
    p_pos = class_vote(weights, y_train)[0][:, 1]  # readout P(positive class)

    def class_density(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray) -> None:
        """Soft KDE background, one contour set per class."""
        for label, cmap in ((0, "Blues"), (1, "Reds")):
            pts = Z_train[y_train == label]
            density = gaussian_kde(pts.T)(np.vstack([xx.ravel(), yy.ravel()]))
            ax.contourf(
                xx, yy, density.reshape(xx.shape), levels=6, cmap=cmap, alpha=0.18
            )

    pad = 0.08 * np.ptp(Z_train, axis=0)
    lo, hi = Z_train.min(axis=0) - pad, Z_train.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))

    n = len(queries)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 6.5 * nrows), squeeze=False
    )
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")

    for pos, q in enumerate(queries):
        ax = axes[pos]
        class_density(ax, xx, yy)
        for label, color in ((0, _NEG_COLOR), (1, _POS_COLOR)):
            m = y_train == label
            ax.scatter(*Z_train[m].T, s=12, color=color, alpha=0.35, linewidths=0)

        w = weights[q]
        top = np.argsort(w)[-top_k:]
        w_max = w[top].max()
        qx, qy = Z_query[pos]
        for j in top:
            color = _POS_COLOR if y_train[j] == 1 else _NEG_COLOR
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
                *Z_train[j],
                s=90,
                color=color,
                edgecolor="white",
                linewidths=0.8,
                zorder=3,
            )

        pred = int(p_pos[q] >= 0.5)
        ax.scatter(
            qx,
            qy,
            marker="*",
            s=650,
            color=(_POS_COLOR if pred else "#dddddd"),
            edgecolor="black",
            linewidths=1.6,
            zorder=4,
        )
        ax.set(xticks=[], yticks=[])

        parts = [f"readout P({class_names[1]}) = Σ(vote) = {p_pos[q]:.2f}"]
        if y_test is not None:
            parts.append(f"true = {class_names[y_test[q]]}")
        parts.append(f"top-{top_k} hold {w[top].sum():.0%} of the vote")
        prefix = "" if query_titles is None else f"{query_titles[pos]}\n"
        ax.set_title(prefix + "  ·  ".join(parts), fontsize=11)

    handles = [
        plt.Line2D([], [], color=_POS_COLOR, lw=4, label=f"{class_names[1]} vote"),
        plt.Line2D([], [], color=_NEG_COLOR, lw=4, label=f"{class_names[0]} vote"),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=_POS_COLOR,
            markersize=10,
            label=f"{class_names[1]} sample",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=_NEG_COLOR,
            markersize=10,
            label=f"{class_names[0]} sample",
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
    fig.suptitle(f"{title} ({proj_name})", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig
