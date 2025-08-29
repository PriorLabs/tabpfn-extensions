import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils.validation import check_is_fitted
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import logging
from typing import Any
import tempfile
from pathlib import Path

from tabpfn import TabPFNClassifier
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator
from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model


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
            classifier, eval_config, TabPFNClassifier
        )
        eval_classifier.fit(X_train, y_train)

        try:
            probabilities = eval_classifier.predict_proba(X_val) # type: ignore
            roc_auc = roc_auc_score(
                y_val, probabilities, multi_class="ovr", average="weighted"
            ) if len(np.unique(y_val)) > 2 else roc_auc_score(
                y_val, probabilities[:, 1]
            )
            log_loss_score = log_loss(y_val, probabilities)
        except Exception as e:
            logging.warning(f"An error occurred during evaluation: {e}")
            roc_auc, log_loss_score = np.nan, np.nan

        return roc_auc, log_loss_score


class FinetunedTabPFNClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for fine-tuning the TabPFNClassifier.

    This class encapsulates the fine-tuning loop, allowing you to fine-tune
    TabPFN on a specific dataset using the familiar .fit() and .predict() API.

    Parameters
    ----------
    device : str, default='cuda'
        The device to run the model on, e.g., 'cuda' or 'cpu'.

    epochs : int, default=5
        The total number of passes through the fine-tuning data.

    learning_rate : float, default=1e-5
        The learning rate for the Adam optimizer. A small value is crucial
        for stable fine-tuning.

    meta_dataset_size : int, default=1024
        The total number of samples to use for creating each meta-dataset, which
        is then split into a context set and a fine-tuning set. This should be
        less than or equal to the number of samples in the training data.

    finetune_split_ratio : float, default=0.3
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

    **kwargs : dict
        Additional keyword arguments to pass to the underlying TabPFNClassifier,
        such as `n_estimators`.
    """
    def __init__(self,
                 device: str = 'cuda',
                 epochs: int = 5,
                 learning_rate: float = 1e-5,
                 meta_dataset_size: int = 1024,
                 finetune_split_ratio: float = 0.3,
                 meta_batch_size: int = 1,
                 random_state: int = 42,
                 early_stopping: bool = True,
                 patience: int = 3,
                 min_delta: float = 1e-4,
                 **kwargs: Any):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.meta_dataset_size = meta_dataset_size
        self.finetune_split_ratio = finetune_split_ratio
        self.meta_batch_size = meta_batch_size
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.kwargs = kwargs

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FinetunedTabPFNClassifier":
        """
        Fine-tunes the TabPFN model on the provided training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
    
        self.X_ = X
        self.y_ = y

        classifier_config = {
        "ignore_pretraining_limits": True,
        "device": self.device,
        "n_estimators": self.kwargs.get("n_estimators", 8),
        "random_state": self.random_state,
        "inference_precision": torch.float32,
        }
        

        # 2. Initialize the base TabPFNClassifier
        self.finetuned_classifier_ = TabPFNClassifier(**classifier_config, fit_mode="batched", differentiable_input=False)
        # Required to access model parameters for the optimizer
        self.finetuned_classifier_._initialize_model_variables()

        eval_config = {
            **classifier_config,
             "inference_config": {
                "SUBSAMPLE_SAMPLES": self.kwargs.get("n_inference_context_samples", 10000)
            }
        }

        # 3. Prepare data for the fine-tuning loop
        # This splitter function will be applied to the training data to create
        # (context, finetune) pairs for each step of the loop.
        splitter = partial(
            train_test_split,
            test_size=self.finetune_split_ratio,
            random_state=self.random_state,
            stratify=y
        )

        X_train, X_val, y_train, y_val = splitter(X, y)

        # Cap the meta-dataset size by the total number of training samples
        effective_meta_dataset_size = min(self.meta_dataset_size, len(y_train))

        training_datasets = self.finetuned_classifier_.get_preprocessed_datasets(
            X_train, y_train, splitter, effective_meta_dataset_size
        )

        finetuning_dataloader = DataLoader(
            training_datasets,
            batch_size=self.meta_batch_size,
            collate_fn=meta_dataset_collator,
            shuffle=True
        )

        # 4. Setup optimizer and loss function
        optimizer = Adam(self.finetuned_classifier_.model_.parameters(), lr=self.learning_rate)  # type: ignore
        loss_function = torch.nn.CrossEntropyLoss()


        # 5. Fine-tuning loop
        print("--- 🚀 Starting Fine-tuning ---")
        
        # Early stopping variables
        best_roc_auc = -np.inf
        patience_counter = 0
        best_model_path = None
        
        for epoch in range(self.epochs):
            progress_bar = tqdm(
                finetuning_dataloader,
                desc=f"Finetuning Epoch {epoch + 1}/{self.epochs}"
            )
            for (
                X_context_batch, # Context features
                X_query_batch,  # Query features
                y_context_batch, # Context labels
                y_query_batch,  # Query labels
                cat_ixs,
                confs,
            ) in progress_bar:

                # Skip batches where a split results in missing classes,
                # which can cause errors in loss calculation.
                if len(np.unique(y_context_batch)) != len(np.unique(y_query_batch)):
                    continue

                optimizer.zero_grad()

                # Provide the context set to the model
                self.finetuned_classifier_.fit_from_preprocessed(
                    X_context_batch, y_context_batch, cat_ixs, confs
                )

                # Get predictions (logits) on the fine-tuning set
                predictions = self.finetuned_classifier_.forward(X_query_batch, return_logits=True)

                # Calculate loss
                loss = loss_function(predictions, y_query_batch.to(self.device))

                # Backpropagation
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            roc_auc, log_loss_score = evaluate_model(self.finetuned_classifier_, eval_config, X_train, y_train, X_val, y_val)
            
            print(
                f"📊 Epoch {epoch + 1} Evaluation | Val ROC: {roc_auc:.4f}, Val Log Loss: {log_loss_score:.4f}\n"
            )
            
            # Early stopping logic
            if self.early_stopping and not np.isnan(roc_auc):
                if roc_auc > best_roc_auc + self.min_delta:
                    best_roc_auc = roc_auc
                    patience_counter = 0
                    # Save the best model using TabPFN's official save function
                    with tempfile.NamedTemporaryFile(suffix='.tabpfn_fit', delete=False) as tmp_file:
                        best_model_path = Path(tmp_file.name)
                        save_fitted_tabpfn_model(self.finetuned_classifier_, best_model_path)
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement for {patience_counter} epochs. Best ROC AUC: {best_roc_auc:.4f}")
                
                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered. Best ROC AUC: {best_roc_auc:.4f}")
                    # Restore the best model using TabPFN's official load function
                    if best_model_path is not None:
                        self.finetuned_classifier_ = load_fitted_tabpfn_model(best_model_path, device=self.device)
                        # Clean up the temporary file
                        best_model_path.unlink(missing_ok=True)
                    break

        print("--- ✅ Fine-tuning Finished ---")
        
        # Clean up temporary file if early stopping didn't trigger
        if best_model_path is not None and best_model_path.exists():
            best_model_path.unlink(missing_ok=True)
        
        # Set fit_mode to standard mode for prediction and fit the model with full training data
        # This ensures the model is ready for prediction without warnings
        self.finetuned_classifier_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_classifier_.fit(self.X_, self.y_)  # type: ignore
            
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)

        # The model is already fitted with the full training data in the fit method
        return self.finetuned_classifier_.predict_proba(X)  # type: ignore

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)

        # The model is already fitted with the full training data in the fit method
        return self.finetuned_classifier_.predict(X)  # type: ignore
