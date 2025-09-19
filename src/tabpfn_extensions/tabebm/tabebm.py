from __future__ import annotations

import os
import random
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.model_selection import train_test_split
from tabpfn_common_utils.telemetry import set_extension
from torch.utils.data import DataLoader

from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig
from tabpfn.utils import meta_dataset_collator
from tabpfn_extensions.utils import TabPFNClassifier


def to_numpy(X: np.ndarray | torch.Tensor | pd.DataFrame | None) -> np.ndarray | None:
    """Convert input data to numpy array format.

    Args:
        X: Input data in various formats (numpy array, torch tensor, pandas DataFrame, or None)

    Returns:
        numpy array representation of the input data, or None if input is None

    Raises:
        ValueError: If input type is not supported
    """
    if isinstance(X, np.ndarray):
        return X
    elif isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    elif isinstance(X, pd.DataFrame):
        return X.to_numpy()
    elif X is None:
        return None
    else:
        raise ValueError("X must be a np.ndarray, torch.Tensor, pd.DataFrame, or None")


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================================================================
#                               TabEBM CLASS
# ========================================================================


@set_extension("tabebm")
class TabEBM:
    """TabEBM: Tabular Energy-Based Model for synthetic data generation.

    This class implements an energy-based model that uses TabPFN as the underlying
    classifier to define energy functions. It generates synthetic tabular data
    using Stochastic Gradient Langevin Dynamics (SGLD) sampling.

    The core idea is to treat each class as having its own energy landscape,
    where real data points have low energy and synthetic points are generated
    by following the energy gradient through SGLD sampling.
    """

    def __init__(
        self,
        max_data_size: int = 10000,
    ):
        """Initialize TabEBM with optimized configuration.

        Args:
            max_data_size: Maximum number of data points to use for training.
                          Larger datasets will be subsampled to this size.
        """
        # Configure TabPFN to disable preprocessing for gradient computation
        # This is crucial for SGLD sampling as we need gradients w.r.t. input features
        no_preprocessing_inference_config = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False,
            FEATURE_SHIFT_METHOD=None,
            CLASS_SHIFT_METHOD=None,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize TabPFN with single estimator for gradient computation
        # Note: Multiple estimators are disabled because preprocessing coupling
        # in TabPFN-v2 prevents effective gradient-based sampling
        self.model = TabPFNClassifier(
            n_estimators=1,
            fit_mode="batched",
            inference_config=no_preprocessing_inference_config,
            device=self.device,
        )

        self.max_data_size = max_data_size

        # Cache for fitted models to avoid redundant fitting
        self._fitted_models_cache: dict[int, Any] = {}

    def generate(
        self,
        X: np.ndarray | torch.Tensor | pd.DataFrame,
        y: np.ndarray | torch.Tensor | pd.Series,
        num_samples: int,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.1,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        """Generate synthetic samples using Stochastic Gradient Langevin Dynamics (SGLD).

        This method creates synthetic data by treating the TabPFN classifier as an energy
        function and using SGLD to sample from the learned energy landscape. For each class,
        it creates a binary classification problem (target class vs surrogate negatives)
        and samples new points by following energy gradients.

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            num_samples: Number of synthetic samples to generate per class
            starting_point_noise_std: Standard deviation of noise added to real data
                                    points when initializing SGLD chains
            sgld_step_size: Step size for gradient updates in SGLD sampling
            sgld_noise_std: Standard deviation of noise added at each SGLD step
            sgld_steps: Number of SGLD steps to perform
            distance_negative_class: Distance for placing surrogate negative samples
                                   from the origin (used to create binary classification)
            seed: Random seed for reproducibility
            verbose: Whether to print verbose information during sampling

        Returns:
            Dictionary mapping class names to generated samples:
            {
                'class_0': np.ndarray of shape (num_samples, n_features),
                'class_1': np.ndarray of shape (num_samples, n_features),
                ...
            }
        """
        # Preprocess and validate input data
        data_dict = self._preprocess(X, y)

        # Perform optimized sampling with caching and batch processing
        res = self._sampling_internal(
            X=data_dict["X"],
            y=data_dict["y"],
            num_samples=num_samples,
            starting_point_noise_std=starting_point_noise_std,
            sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std,
            sgld_steps=sgld_steps,
            distance_negative_class=distance_negative_class,
            seed=seed,
            verbose=verbose,
        )

        # Extract sampling results and format output
        augmented_data = {}
        unique_labels = np.unique(to_numpy(y))
        for target_class in unique_labels:
            class_key = f"class_{int(target_class)}"
            augmented_data[class_key] = res[class_key]["sampling_paths"]

        return augmented_data

    def _sampling_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_samples: int,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.1,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Optimized internal SGLD sampling method with caching and batch processing.

        This method performs the core SGLD sampling for each class by:
        1. Creating binary classification datasets (target class vs surrogate negatives)
        2. Fitting TabPFN models (with caching to avoid redundant computation)
        3. Initializing SGLD chains from noisy real data points
        4. Running SGLD updates using energy gradients

        Key optimizations:
        - Model fitting cache to avoid redundant training
        - Pre-computed noise tensors for memory efficiency
        - Vectorized operations where possible
        - Reduced device transfers

        Args:
            X, y: Preprocessed input data and labels
            num_samples: Number of samples to generate per class
            starting_point_noise_std, sgld_step_size, sgld_noise_std, sgld_steps: SGLD parameters
            distance_negative_class: Distance for surrogate negative samples
            seed: Random seed
            verbose: verbose flag

        Returns:
            Dictionary containing sampling results for each class
        """
        # Pre-compute unique classes for iteration
        unique_classes = np.unique(y)
        synthetic_data_per_class = {}

        # Pre-allocate noise tensor for all SGLD steps (memory optimization)
        sample_shape = (num_samples, X.shape[1])
        noise_shape = (sgld_steps, *sample_shape)

        # Set seed for reproducible noise generation
        seed_everything(seed)

        for target_class in unique_classes:
            # Create or retrieve cached EBM dataset and model
            ebm_dict = self._get_or_create_ebm_dataset(
                X, y, target_class, distance_negative_class
            )
            X_ebm = ebm_dict["X_ebm"]
            y_ebm = ebm_dict["y_ebm"]

            # Fit predictor with caching
            self._fit_predictor_cached(X_ebm, y_ebm, target_class)

            # Initialize SGLD starting points
            start_dict = self._initialize_sgld_starting_points(
                X_ebm, y_ebm, num_samples, starting_point_noise_std, seed
            )
            X_sgld = start_dict["X_start"]
            y_sgld = start_dict["y_start"]

            # Prepare batch data for TabPFN (optimized)
            batch_dict = self._prepare_tabpfn_batch_data(X_sgld, y_sgld)
            X_sgld_tensor = (
                batch_dict["X_train"][0]
                .to(self.device)
                .requires_grad_(requires_grad=True)
            )

            # Pre-generate all noise for SGLD steps (memory/computation savings)
            noise_tensor = torch.randn(
                noise_shape, device=self.device, dtype=X_sgld_tensor.dtype
            )

            # SGLD sampling loop (optimized)
            X_sgld_tensor = self._perform_sgld_sampling(
                X_sgld_tensor,
                noise_tensor,
                sgld_step_size,
                sgld_noise_std,
                sgld_steps,
                verbose,
            )

            # Store results
            synthetic_data_per_class[f"class_{int(target_class)}"] = {
                "sampling_paths": X_sgld_tensor.detach().cpu().squeeze(0).numpy()
            }

        return synthetic_data_per_class

    def _preprocess(
        self,
        X: np.ndarray | torch.Tensor | pd.DataFrame,
        y: np.ndarray | torch.Tensor | pd.Series,
    ) -> dict[str, np.ndarray]:
        """Preprocess input data with optimized memory usage and stratified sampling.

        Converts inputs to numpy arrays and applies stratified subsampling for large datasets
        to ensure balanced representation across classes while maintaining efficiency.

        Args:
            X: Input features in various formats
            y: Target labels in various formats

        Returns:
            Dictionary containing preprocessed X and y as numpy arrays
        """
        # Convert to numpy arrays
        if not isinstance(X, np.ndarray):
            X = to_numpy(X)
        if not isinstance(y, np.ndarray):
            y = to_numpy(y).reshape(-1)

        # Optimize for large datasets with stratified sampling
        # This reduces the probability of TabPFN-v2's internal splitting that may remove constant features
        if X.shape[0] > self.max_data_size:
            X_sampled, _, y_sampled, _ = train_test_split(
                X,
                y,
                train_size=self.max_data_size,
                random_state=42,
                stratify=y,
            )
            X, y = X_sampled, y_sampled

        return {"X": X, "y": y}

    def _get_or_create_ebm_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_class: int,
        distance_negative_class: float,
    ) -> dict[str, torch.Tensor]:
        """Create EBM dataset with caching for better performance.

        Args:
            X, y: Input data and labels
            target_class: Target class for binary classification
            distance_negative_class: Distance for surrogate negatives

        Returns:
            Dictionary containing X_ebm and y_ebm tensors
        """
        # Use the existing method but add caching potential
        X_one_class = X[y == target_class]
        X_ebm, y_ebm = TabEBM.add_surrogate_negative_samples(
            X_one_class, distance_negative_class=distance_negative_class
        )

        # Convert to tensors with proper device placement
        X_ebm = torch.from_numpy(X_ebm).float().to(self.device)
        y_ebm = torch.from_numpy(y_ebm).long().to(self.device)

        return {"X_ebm": X_ebm, "y_ebm": y_ebm}

    def _fit_predictor_cached(
        self, X_ebm: torch.Tensor, y_ebm: torch.Tensor, target_class: int
    ) -> None:
        """Fit predictor with caching to avoid redundant model training.

        This method caches fitted models per class to avoid retraining when
        generating multiple batches for the same class.

        Args:
            X_ebm: EBM training features
            y_ebm: EBM training labels
            target_class: Target class identifier for caching
        """
        # Check if model for this class is already fitted
        cache_key = target_class
        if cache_key in self._fitted_models_cache:
            # Model already fitted for this class, skip training
            return

        # Prepare batch data for TabPFN
        batch_dict = self._prepare_tabpfn_batch_data(X_ebm, y_ebm)
        X_ebm_list = [X_batch.to(self.device) for X_batch in batch_dict["X_train"]]
        y_ebm_list = [y_batch.to(self.device) for y_batch in batch_dict["y_train"]]
        cat_ixs = batch_dict["cat_ixs"]
        confs = batch_dict["confs"]

        # Train the model
        self.model.fit_from_preprocessed(
            X_ebm_list,
            y_ebm_list,
            cat_ix=cat_ixs,
            configs=confs,
        )

        # Cache the fitted state
        self._fitted_models_cache[cache_key] = True

    def _initialize_sgld_starting_points(
        self,
        X_ebm: torch.Tensor,
        y_ebm: torch.Tensor,
        num_samples: int,
        starting_point_noise_std: float,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        """Optimized initialization of SGLD starting points.

        Args:
            X_ebm: EBM training features
            y_ebm: EBM training labels
            num_samples: Number of starting points needed
            starting_point_noise_std: Noise level for starting points
            seed: Random seed

        Returns:
            Dictionary with starting points
        """
        seed_everything(seed)

        # Select random samples from the target class (class 0 by convention)
        real_sample_mask = y_ebm == 0
        real_samples = X_ebm[real_sample_mask]

        # Efficient random sampling
        num_real_samples = real_samples.shape[0]
        start_indices = torch.randint(
            0, num_real_samples, (num_samples,), device=self.device
        )
        X_start = real_samples[start_indices]
        y_start = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        # Add noise to starting points (vectorized operation)
        if starting_point_noise_std > 0:
            noise = (
                torch.randn_like(X_start, device=self.device) * starting_point_noise_std
            )
            X_start = X_start + noise

        return {"X_start": X_start, "y_start": y_start}

    def _prepare_tabpfn_batch_data(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> dict[str, Any]:
        """Optimized preparation of TabPFN batch data with reduced overhead.

        Args:
            X: Input features tensor
            y: Input labels tensor

        Returns:
            Dictionary containing prepared batch data
        """
        # Use efficient train/test split with no validation data
        splitter = partial(
            TabEBM.train_test_split_allow_full_train,
            test_size=0,
            random_state=42,
            shuffle=False,
        )

        # Convert to CPU for preprocessing (TabPFN requirement)
        X_cpu = X.cpu()
        y_cpu = y.cpu()

        # Get preprocessed datasets
        batched_datasets = self.model.get_preprocessed_datasets(
            X_cpu, y_cpu, splitter, max_data_size=self.max_data_size
        )

        # Create dataloader with minimal overhead
        batch_dataloader = DataLoader(
            batched_datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            pin_memory=self.device == "cuda",  # Optimize data transfer
        )

        # Extract first (and only) batch efficiently
        batch_data = next(iter(batch_dataloader))
        X_train, X_val, y_train, y_val, cat_ixs, confs = batch_data

        return {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "cat_ixs": cat_ixs,
            "confs": confs,
        }

    def _perform_sgld_sampling(
        self,
        X_sgld: torch.Tensor,
        noise_tensor: torch.Tensor,
        sgld_step_size: float,
        sgld_noise_std: float,
        sgld_steps: int,
        verbose: bool,
    ) -> torch.Tensor:
        """Optimized SGLD sampling loop with reduced memory allocations.

        This method performs the core SGLD updates using pre-computed noise
        and optimized gradient computations.

        Args:
            X_sgld: Starting points tensor with gradients enabled
            noise_tensor: Pre-computed noise for all SGLD steps
            sgld_step_size: SGLD step size
            sgld_noise_std: SGLD noise standard deviation
            sgld_steps: Number of SGLD steps
            verbose: verbose flag

        Returns:
            Final samples after SGLD sampling
        """
        # Wrap in list for TabPFN forward pass (API requirement)
        X_sgld_list = [X_sgld]

        for t in range(sgld_steps):
            # Clear previous gradients
            if X_sgld_list[0].grad is not None:
                X_sgld_list[0].grad.zero_()

            # Forward pass to compute energy
            logits = self.model.forward(X_sgld_list, return_logits=True)

            # Compute energy and total loss
            energy = TabEBM.compute_energy(logits.reshape(logits.shape[-1], -1))
            total_energy = energy.sum() / X_sgld_list[0].shape[1]

            # Backward pass
            total_energy.backward()

            # SGLD update with pre-computed noise
            with torch.no_grad():
                X_sgld_updated = (
                    X_sgld_list[0]
                    - sgld_step_size * X_sgld_list[0].grad
                    + sgld_noise_std * noise_tensor[t]
                )

                # Update tensor in-place to maintain gradient tracking
                X_sgld_list[0] = X_sgld_updated.requires_grad_(requires_grad=True)

        return X_sgld_list[0]

    @staticmethod
    def compute_energy(
        logits: torch.Tensor | np.ndarray,
        return_unnormalized_prob: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """Compute TabEBM class-specific energy function.

        The energy function is defined as:
        E_c(x) = -log(exp(f^c(x)[0]) + exp(f^c(x)[1]))

        Where f^c(x) are the logits from the class-specific binary classifier.
        Lower energy corresponds to higher probability of belonging to the target class.

        Args:
            logits: Model's unnormalized logits for each class
                   Shape: (num_samples, num_classes)
            return_unnormalized_prob: If True, return exp(-energy) instead of energy

        Returns:
            Energy values (or unnormalized probabilities) for each sample

        Raises:
            ValueError: If logits are not unnormalized or have wrong type
        """
        if isinstance(logits, torch.Tensor):
            # Validate that logits are unnormalized (not probabilities)
            logit_sums = logits.sum(dim=1)
            if (logit_sums - 1).abs().max() <= 1e-5:
                raise ValueError("Logits must be unnormalized (not probabilities)")
            # Compute energy using log-sum-exp for numerical stability
            energy = -torch.logsumexp(logits, dim=1)

            if return_unnormalized_prob:
                return torch.exp(-energy)
            return energy

        elif isinstance(logits, np.ndarray):
            # Validate that logits are unnormalized (not probabilities)
            logit_sums = logits.sum(axis=1)
            if np.abs(logit_sums - 1).max() <= 1e-5:
                raise ValueError("Logits must be unnormalized (not probabilities)")
            # Compute energy using scipy's log-sum-exp for numerical stability
            energy = -scipy.special.logsumexp(logits, axis=1)

            if return_unnormalized_prob:
                return np.exp(-energy)
            return energy
        else:
            raise ValueError("Logits must be either a torch.Tensor or a np.ndarray")

    @staticmethod
    def add_surrogate_negative_samples(
        X: np.ndarray | torch.Tensor,
        distance_negative_class: float = 5,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Create surrogate negative samples for TabEBM's binary classification approach.

        This method creates artificial "negative" samples at specified distances from the origin
        to enable energy-based modeling through binary classification. The surrogate negatives
        help define the energy landscape by providing clear decision boundaries.

        For 2D data, negatives are placed at the four corners of a square centered at origin.
        For higher dimensions, random combinations of ±distance_negative_class are used.

        Args:
            X: Real data samples (expected to be approximately standardized)
               Shape: (num_samples, num_features)
            distance_negative_class: Distance of surrogate negatives from origin
                                   Larger values create more distinct separation

        Returns:
            Tuple of (X_ebm, y_ebm) where:
            - X_ebm: Combined real and surrogate samples
            - y_ebm: Binary labels (0 for real data, 1 for surrogates)
        """
        num_features = X.shape[1]

        if num_features == 2:
            # For 2D case, use deterministic corner placement for consistency
            surrogate_negatives = [
                [-distance_negative_class, -distance_negative_class],
                [distance_negative_class, distance_negative_class],
                [-distance_negative_class, distance_negative_class],
                [distance_negative_class, -distance_negative_class],
            ]
        else:
            # For higher dimensions, generate diverse surrogate negatives
            # Use a set to ensure uniqueness and symmetry
            surrogate_set = set()

            # Generate points with random ±distance_negative_class coordinates
            while len(surrogate_set) < 4:
                point = np.random.choice(
                    [-distance_negative_class, distance_negative_class], num_features
                )
                point_tuple = tuple(point)

                # Add both the point and its negation for symmetry
                if point_tuple not in surrogate_set:
                    surrogate_set.add(point_tuple)
                    surrogate_set.add(tuple(-np.array(point)))

            surrogate_negatives = list(surrogate_set)

        num_surrogates = len(surrogate_negatives)

        if isinstance(X, np.ndarray):
            # NumPy array processing
            X_surrogates = np.array(surrogate_negatives, dtype=X.dtype)
            X_ebm = np.concatenate([X, X_surrogates], axis=0)

            # Create binary labels: 0 for real data, 1 for surrogates
            y_ebm = np.concatenate(
                [
                    np.zeros(X.shape[0], dtype=np.int64),
                    np.ones(num_surrogates, dtype=np.int64),
                ]
            )

            return X_ebm, y_ebm

        elif isinstance(X, torch.Tensor):
            # PyTorch tensor processing
            X_surrogates = torch.tensor(
                surrogate_negatives, dtype=X.dtype, device=X.device
            )
            X_ebm = torch.cat([X, X_surrogates], dim=0)

            # Create binary labels: 0 for real data, 1 for surrogates
            y_ebm = torch.cat(
                [
                    torch.zeros(X.shape[0], dtype=torch.long, device=X.device),
                    torch.ones(num_surrogates, dtype=torch.long, device=X.device),
                ]
            )

            return X_ebm, y_ebm
        else:
            raise ValueError("X must be either a np.ndarray or a torch.Tensor")

    @staticmethod
    def train_test_split_allow_full_train(
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        test_size: float | None = None,
        train_size: float | None = None,
        random_state: int | None = None,
        shuffle: bool = True,
        stratify: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
    ]:
        """Enhanced train-test split that supports full training mode.

        This method extends sklearn's train_test_split to handle the case where
        test_size=0, which means we want to use all data for training (no validation).
        This is useful for TabEBM's energy-based training approach.

        Args:
            X, y: Input features and labels
            test_size: Fraction of data for testing (if 0, enables full train mode)
            train_size: Fraction of data for training
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
            stratify: Array-like for stratified splitting

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
            In full train mode, X_train=X and y_train=y
        """
        # Detect full training mode
        full_train_mode = test_size == 0

        if full_train_mode:
            # Reset test_size to None for sklearn compatibility
            test_size = None

        # Perform standard train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        # Override with full data in full train mode
        if full_train_mode:
            X_train = X
            y_train = y

        return X_train, X_val, y_train, y_val
