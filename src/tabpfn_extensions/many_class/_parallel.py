"""Multi-GPU parallel dispatch for ManyClassClassifier.

Persistent workers load the base estimator once per GPU and process batches
of ECOC sub-problems. Includes y-swap optimization: first sub-problem runs
full fit() (caching X preprocessing), subsequent sub-problems only replace
y_train in cached ensemble members, skipping ~500ms of redundant
preprocessing per row.

This is safe because preprocessor.fit_transform(X_train, feature_schema) does
not take y as input (see preprocessing/transform.py). Labels are handled
separately via config.class_permutation[y].
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch.multiprocessing as mp
from sklearn.base import clone

from ._utils import (
    EPS_WEIGHT,
    RowRunResult,
    align_probabilities,
    apply_categorical_features,
    as_numpy,
    filter_fit_params_for_mask,
)

logger = logging.getLogger(__name__)


def _worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """Persistent worker: load estimator once, process batches with y-swap.

    The estimator template is received via the first 'init' message,
    ensuring the worker uses the exact same estimator as the sequential path.
    """
    # Set CUDA device BEFORE importing torch to ensure proper GPU masking
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch  # noqa: E402

    template = None

    result_queue.put({"status": "ready", "gpu_id": gpu_id})

    while True:
        msg = task_queue.get()
        if msg is None or msg.get("cmd") == "stop":
            result_queue.put({"status": "stopped", "gpu_id": gpu_id})
            break

        cmd = msg.get("cmd", "batch")

        if cmd == "init":
            # Receive the user's estimator and clone it as template
            template = msg["estimator"]
            result_queue.put({"status": "init_done", "gpu_id": gpu_id})
            continue

        # cmd == "batch" (default)
        if template is None:
            result_queue.put({
                "status": "error",
                "gpu_id": gpu_id,
                "error": "Worker not initialized. Send 'init' first.",
            })
            continue

        X_train = msg["X_train"]
        X_test = msg["X_test"]
        rows = msg["rows"]  # list of (row_idx, y_codes, mask_or_None)
        alphabet_size = msg["alphabet_size"]
        categorical_features = msg.get("categorical_features")
        fit_params = msg.get("fit_params") or {}
        cache_preprocessing = msg.get("cache_preprocessing", True)
        row_weighter = msg.get("row_weighter")

        results: dict[int, RowRunResult] = {}
        cached = None

        for idx, (row_idx, y_codes, mask) in enumerate(rows):
            X_row = X_train[mask] if mask is not None else X_train
            y_row = y_codes[mask] if mask is not None else y_codes

            if len(y_row) == 0:
                n_test = as_numpy(X_test).shape[0]
                results[row_idx] = RowRunResult(
                    proba_test=np.full((n_test, alphabet_size), 1.0 / alphabet_size),
                    proba_train=np.empty((0, alphabet_size)),
                    weight=EPS_WEIGHT,
                    support=0,
                    entropy=None,
                    accuracy=None,
                )
                continue

            # Filter fit_params by mask (consistent with sequential run_row)
            filtered_params = filter_fit_params_for_mask(
                fit_params, mask, n_samples=len(X_train)
            )

            try:
                if idx == 0 or not cache_preprocessing or cached is None:
                    # Full fit: preprocessing + forward pass (~880ms)
                    cached = clone(template)
                    apply_categorical_features(cached, categorical_features)
                    cached.fit(X_row, y_row, **filtered_params)
                else:
                    # Y-swap: reuse cached preprocessing, replace y only (~380ms)
                    for em in cached.executor_.ensemble_members:
                        perm = em.config.class_permutation
                        y_new = perm[y_row] if perm is not None else y_row
                        if isinstance(em.y_train, torch.Tensor):
                            em.y_train = torch.tensor(
                                y_new,
                                dtype=torch.long,
                                device=em.y_train.device,
                            )
                        else:
                            em.y_train = np.asarray(y_new, dtype=np.int64)

                X_train_np = as_numpy(X_row)
                X_test_np = as_numpy(X_test)
                proba_both = cached.predict_proba(
                    np.concatenate([X_train_np, X_test_np], axis=0)
                )
                aligned = align_probabilities(
                    proba_both, cached.classes_, alphabet_size
                )
                n_train = X_train_np.shape[0]
                proba_train = aligned[:n_train]
                proba_test = aligned[n_train:]

                # Compute weight and diagnostics (same as sequential run_row)
                weight = 1.0
                entropy = None
                accuracy = None
                if row_weighter is not None:
                    try:
                        w, diag = row_weighter.weight(
                            proba_train, y_row, alphabet_size
                        )
                        weight = float(w)
                        entropy = diag.get("entropy") if diag else None
                        accuracy = diag.get("accuracy") if diag else None
                    except Exception:
                        pass

                results[row_idx] = RowRunResult(
                    proba_test=proba_test,
                    proba_train=proba_train,
                    weight=weight,
                    support=len(y_row),
                    entropy=entropy,
                    accuracy=accuracy,
                )
            except Exception as exc:
                # Fallback: full fit on y-swap error
                logger.debug(
                    "Worker %d: y-swap failed for row %d, falling back to full fit: %s",
                    gpu_id, row_idx, exc,
                )
                cached = clone(template)
                apply_categorical_features(cached, categorical_features)
                cached.fit(X_row, y_row, **filtered_params)
                X_train_np = as_numpy(X_row)
                X_test_np = as_numpy(X_test)
                proba_both = cached.predict_proba(
                    np.concatenate([X_train_np, X_test_np], axis=0)
                )
                aligned = align_probabilities(
                    proba_both, cached.classes_, alphabet_size
                )
                n_train = X_train_np.shape[0]
                results[row_idx] = RowRunResult(
                    proba_test=aligned[n_train:],
                    proba_train=aligned[:n_train],
                    weight=1.0,
                    support=len(y_row),
                    entropy=None,
                    accuracy=None,
                )

        result_queue.put({"status": "done", "gpu_id": gpu_id, "results": results})


def start_pool(n_gpus: int) -> tuple[list, list, mp.Queue]:
    """Start persistent worker pool. Returns (workers, task_queues, result_queue)."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    task_queues: list[mp.Queue] = []
    workers: list[mp.Process] = []
    for i in range(n_gpus):
        tq = ctx.Queue()
        task_queues.append(tq)
        p = ctx.Process(target=_worker, args=(i, tq, result_queue), daemon=True)
        p.start()
        workers.append(p)
    for _ in range(n_gpus):
        r = result_queue.get(timeout=120)
        if r["status"] != "ready":
            raise RuntimeError(f"Worker startup failed: {r}")
    return workers, task_queues, result_queue


def stop_pool(
    workers: list, task_queues: list, result_queue: mp.Queue
) -> None:
    """Stop persistent worker pool."""
    for tq in task_queues:
        tq.put({"cmd": "stop"})
    for _ in range(len(workers)):
        try:
            result_queue.get(timeout=30)
        except Exception:
            pass
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
