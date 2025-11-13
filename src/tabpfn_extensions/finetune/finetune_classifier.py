#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import copy
import logging
from functools import partial
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.getLogger().setLevel(logging.INFO)


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Evaluates the model's performance on the validation set."""
    eval_classifier = clone_model_for_evaluation(
        classifier,
        eval_config,
        TabPFNClassifier,
    )
    eval_classifier.fit(X_train, y_train)

    try:
        probabilities = eval_classifier.predict_proba(X_val)  # type: ignore
        roc_auc = (
            roc_auc_score(
                y_val,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )
            if len(np.unique(y_val)) > 2
            else roc_auc_score(
                y_val,
                probabilities[:, 1],
            )
        )
        log_loss_score = log_loss(y_val, probabilities)
    except (ValueError, RuntimeError, AttributeError) as e:
        logging.warning(f"An error occurred during evaluation: {e}")
        roc_auc, log_loss_score = np.nan, np.nan

    return roc_auc, log_loss_score  # pyright: ignore[reportReturnType]


class FinetunedTabPFNClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for fine-tuning the TabPFNClassifier.

    This class encapsulates the fine-tuning loop, allowing you to fine-tune
    TabPFN on a specific dataset using the familiar .fit() and .predict() API.

    Parameters:
    ----------
    device: tuple[torch.device, ...]
        The devices determined to be used.
        The devices are determined based on the `device` argument to the constructor, and
        the devices available on the system. If multiple devices are listed, currently only
        the first is used for inference.

    epochs : int, default=5
        The total number of passes through the fine-tuning data.

    learning_rate : float, default=1e-5
        The learning rate for the Adam optimizer. A small value is crucial
        for stable fine-tuning.

    n_inference_context_samples : int, default=10_000
        The total number of samples to use for creating each meta-dataset, which
        is then split into a context set and a query set for each step of the fine-tuning loop.
        This same context size is used during final inference as well.

    finetune_split_ratio : float, default=0.2
        The proportion of each meta-dataset to use for calculating the loss.
        The rest is used as the context for the model.

    meta_batch_size : int, default=1
        The number of meta-datasets to process in a single optimization step.
        Currently, this should be kept at 1.

    random_state : int, default=42
        Seed for reproducibility of data splitting and model initialization.

    early_stopping : bool, default=True
        Whether to use early stopping based on validation ROC AUC performance.

    patience : int, default=3
        Number of epochs to wait for improvement before early stopping.

    min_delta : float, default=1e-4
        Minimum change in ROC AUC to be considered as an improvement.

    grad_clip_value : float, default=1.0
        Maximum norm for gradient clipping. If None, gradient clipping is disabled.
        Gradient clipping helps stabilize training by preventing exploding gradients.

    **kwargs : dict
        Additional keyword arguments to pass to the underlying TabPFNClassifier,
        such as `n_estimators`.
    """

    def __init__(
        self,
        device: tuple[torch.device, ...] | str = "auto",
        epochs: int = 5,
        learning_rate: float = 1e-5,
        n_inference_context_samples: int = 10_000,
        finetune_split_ratio: float = 0.2,
        meta_batch_size: int = 1,
        random_state: int = 42,
        early_stopping: bool = True,
        patience: int = 3,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_inference_context_samples = n_inference_context_samples
        self.finetune_split_ratio = finetune_split_ratio
        self.meta_batch_size = meta_batch_size
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip_value = grad_clip_value
        self.kwargs = kwargs

        assert self.meta_batch_size == 1, "meta_batch_size must be 1 for finetuning"

    def fit(self, X: np.ndarray, y: np.ndarray) -> FinetunedTabPFNClassifier:
        """Fine-tunes the TabPFN model on the provided training data.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        self.y_ = y

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.finetune_split_ratio,
            random_state=self.random_state,
            stratify=y,
        )

        # Calculate the context size used during finetuning
        n_finetuning_fit_predict_context_samples = min(
            self.n_inference_context_samples,
            len(y_train),
        )

        classifier_config = {
            **self.kwargs,
            "ignore_pretraining_limits": True,
            "device": self.device,
            "random_state": self.random_state,
        }

        self.finetuned_classifier_ = TabPFNClassifier(
            **classifier_config,
            fit_mode="batched",
            differentiable_input=False,
        )
        self.finetuned_classifier_._initialize_model_variables()
        self.finetuned_classifier_.softmax_temperature_ = (
            self.finetuned_classifier_.softmax_temperature
        )

        eval_config = {
            **classifier_config,
            "inference_config": {
                "SUBSAMPLE_SAMPLES": n_finetuning_fit_predict_context_samples,  # Passing this to the dataloader causes an error, so we set eval config separately from the classifier config.
            },
        }

        optimizer = Adam(
            self.finetuned_classifier_.model_.parameters(),  # type: ignore
            lr=self.learning_rate,
        )
        loss_function = torch.nn.CrossEntropyLoss()

        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        use_amp = self.device == "cuda" and torch.cuda.is_available()
        scaler: GradScaler | None = GradScaler("cuda") if use_amp else None

        logging.info("--- 🚀 Starting Fine-tuning ---")

        best_roc_auc = -np.inf
        patience_counter = 0
        best_model = None

        for epoch in range(self.epochs):
            # Regenerate datasets each epoch with a different random_state to ensure
            # diversity in context/query pairs across epochs. This prevents the model
            # from seeing the exact same splits in every epoch, which could reduce
            # training signal diversity.
            training_splitter = partial(
                train_test_split,
                test_size=self.finetune_split_ratio,
                random_state=self.random_state + epoch,
            )

            training_datasets = self.finetuned_classifier_.get_preprocessed_datasets(
                X_raw=X_train,
                y_raw=y_train,
                split_fn=training_splitter,
                max_data_size=n_finetuning_fit_predict_context_samples,
                equal_split_size=False,
            )

            finetuning_dataloader = DataLoader(
                training_datasets,
                batch_size=self.meta_batch_size,
                collate_fn=meta_dataset_collator,
                shuffle=True,
            )

            progress_bar = tqdm(
                finetuning_dataloader,
                desc=f"Finetuning Epoch {epoch + 1}/{self.epochs}",
            )
            for (
                X_context_batch,
                X_query_batch,
                y_context_batch,
                y_query_batch,
                cat_ixs,
                confs,
            ) in progress_bar:
                ctx = set(np.unique(y_context_batch))
                qry = set(np.unique(y_query_batch))
                if not qry.issubset(ctx):
                    logging.warning(
                        f"Skipping batch: query labels {qry} are not a subset of context labels {ctx}",
                    )
                    continue

                if (
                    X_context_batch[0].shape[1] + X_query_batch[0].shape[1]
                    != n_finetuning_fit_predict_context_samples
                ):
                    actual_size = (
                        X_context_batch[0].shape[1] + X_query_batch[0].shape[1]
                    )
                    logging.warning(
                        f"Skipping batch: total batch size {actual_size} does not match n_finetuning_fit_predict_context_samples {n_finetuning_fit_predict_context_samples}",
                    )
                    continue

                optimizer.zero_grad()

                self.finetuned_classifier_.fit_from_preprocessed(
                    X_preprocessed=X_context_batch,
                    y_preprocessed=y_context_batch,
                    cat_ix=cat_ixs,
                    configs=confs,
                )

                if use_amp and scaler is not None:
                    with autocast("cuda"):
                        predictions = self.finetuned_classifier_.forward(
                            X_query_batch,
                            return_logits=True,
                        )

                        loss = loss_function(predictions, y_query_batch.to(self.device))

                    scaler.scale(loss).backward()

                    if self.grad_clip_value is not None:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(
                            self.finetuned_classifier_.model_.parameters(),  # type: ignore
                            self.grad_clip_value,
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = self.finetuned_classifier_.forward(
                        X_query_batch,
                        return_logits=True,
                    )

                    loss = loss_function(predictions, y_query_batch.to(self.device))

                    loss.backward()

                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_classifier_.model_.parameters(),  # type: ignore
                            self.grad_clip_value,
                        )

                    optimizer.step()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            roc_auc, log_loss_score = evaluate_model(
                self.finetuned_classifier_,
                eval_config,
                X_train,  # pyright: ignore[reportArgumentType]
                y_train,  # pyright: ignore[reportArgumentType]
                X_val,  # pyright: ignore[reportArgumentType]
                y_val,  # pyright: ignore[reportArgumentType]
            )

            logging.info(
                f"📊 Epoch {epoch + 1} Evaluation | Val ROC: {roc_auc:.4f}, Val Log Loss: {log_loss_score:.4f}\n",
            )

            scheduler.step()

            if self.early_stopping and not np.isnan(roc_auc):
                if roc_auc > best_roc_auc + self.min_delta:
                    best_roc_auc = roc_auc
                    patience_counter = 0
                    best_model = copy.deepcopy(self.finetuned_classifier_)
                else:
                    patience_counter += 1
                    logging.info(
                        f"⚠️  No improvement for {patience_counter} epochs. Best ROC AUC: {best_roc_auc:.4f}",
                    )

                if patience_counter >= self.patience:
                    logging.info(
                        f"🛑 Early stopping triggered. Best ROC AUC: {best_roc_auc:.4f}",
                    )
                    if best_model is not None:
                        self.finetuned_classifier_ = best_model
                    break

        logging.info("--- ✅ Fine-tuning Finished ---")

        finetuned_inference_classifier = clone_model_for_evaluation(
            original_model=self.finetuned_classifier_,  # type: ignore
            eval_init_args=eval_config,
            model_class=TabPFNClassifier,
        )
        self.finetuned_inference_classifier_ = finetuned_inference_classifier
        self.finetuned_inference_classifier_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_inference_classifier_.fit(self.X_, self.y_)  # type: ignore

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)

        return self.finetuned_inference_classifier_.predict_proba(X)  # type: ignore

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class for X.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)

        return self.finetuned_inference_classifier_.predict(X)  # type: ignore
