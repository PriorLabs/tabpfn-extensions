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
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
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
