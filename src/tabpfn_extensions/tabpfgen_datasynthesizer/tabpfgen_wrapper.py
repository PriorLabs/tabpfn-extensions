"""Streamlined wrapper around TabPFGen for integration with TabPFN Extensions ecosystem."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

try:
    from tabpfgen import TabPFGen
    from tabpfgen.visuals import (
        visualize_classification_results,
        visualize_regression_results,
    )

    TABPFGEN_AVAILABLE = True
except ImportError:
    TABPFGEN_AVAILABLE = False
    TabPFGen = None


class TabPFNDataSynthesizer:
    """Streamlined wrapper around TabPFGen for synthetic tabular data generation.

    This class provides a clean interface to TabPFGen functionality with
    sensible defaults optimized for TabPFN workflows. It relies heavily on
    the actual TabPFGen package features and built-in visualizations.

    Parameters
    ----------
    n_sgld_steps : int, default=500
        Number of SGLD iterations for generation
    sgld_step_size : float, default=0.01
        Step size for SGLD updates
    sgld_noise_scale : float, default=0.01
        Scale of noise in SGLD
    device : str, default='auto'
        Computing device ('cpu', 'cuda', or 'auto')

    Examples:
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> synthesizer = TabPFNDataSynthesizer(n_sgld_steps=300)
    >>> X_synth, y_synth = synthesizer.generate_classification(X, y, n_samples=100)
    """

    def __init__(
        self,
        n_sgld_steps: int = 500,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        device: str = "auto",
    ):
        if not TABPFGEN_AVAILABLE:
            raise ImportError(
                "TabPFGen is required but not installed. "
                "Install it with: pip install tabpfgen"
            )

        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.device = device

        # Initialize TabPFGen generator
        self.generator = TabPFGen(
            n_sgld_steps=n_sgld_steps,
            sgld_step_size=sgld_step_size,
            sgld_noise_scale=sgld_noise_scale,
            device=device,
        )

    def generate_classification(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_samples: int,
        balance_classes: bool = True,
        visualize: bool = False,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data using TabPFGen.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training labels
        n_samples : int
            Number of synthetic samples to generate
        balance_classes : bool, default=True
            Whether to generate balanced class distributions
        visualize : bool, default=False
            Whether to create TabPFGen's built-in visualization plots
        feature_names : list, optional
            Names of features for visualization

        Returns:
        -------
        X_synth : ndarray of shape (n_samples, n_features)
            Generated synthetic features
        y_synth : ndarray of shape (n_samples,)
            Generated synthetic labels
        """
        # Convert inputs to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)

        # Generate synthetic data using TabPFGen
        X_synth, y_synth = self.generator.generate_classification(
            X, y, n_samples=n_samples, balance_classes=balance_classes
        )

        # Use TabPFGen's built-in visualization if requested
        if visualize and TABPFGEN_AVAILABLE:
            try:
                visualize_classification_results(
                    X, y, X_synth, y_synth, feature_names=feature_names
                )
            except (ImportError, AttributeError, ValueError, TypeError) as e:
                warnings.warn(f"TabPFGen visualization failed: {e}", stacklevel=2)

        return X_synth, y_synth

    def generate_regression(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_samples: int,
        use_quantiles: bool = True,
        visualize: bool = False,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression data using TabPFGen.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        n_samples : int
            Number of synthetic samples to generate
        use_quantiles : bool, default=True
            Whether to use quantile-based sampling
        visualize : bool, default=False
            Whether to create TabPFGen's built-in visualization plots
        feature_names : list, optional
            Names of features for visualization

        Returns:
        -------
        X_synth : ndarray of shape (n_samples, n_features)
            Generated synthetic features
        y_synth : ndarray of shape (n_samples,)
            Generated synthetic targets
        """
        # Convert inputs to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)

        # Generate synthetic data using TabPFGen
        X_synth, y_synth = self.generator.generate_regression(
            X, y, n_samples=n_samples, use_quantiles=use_quantiles
        )

        # Use TabPFGen's built-in visualization if requested
        if visualize and TABPFGEN_AVAILABLE:
            try:
                visualize_regression_results(
                    X, y, X_synth, y_synth, feature_names=feature_names
                )
            except (ImportError, AttributeError, ValueError, TypeError) as e:
                warnings.warn(f"TabPFGen visualization failed: {e}", stacklevel=2)

        return X_synth, y_synth

    def balance_dataset(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        target_per_class: int | None = None,
        visualize: bool = False,
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Balance classification dataset using TabPFGen's balance_dataset method.

        This method uses TabPFGen's new balance_dataset functionality to automatically
        generate synthetic samples for minority classes, bringing them up to the
        majority class size or a specified target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training labels
        target_per_class : int, optional
            Target number of samples per class. If None, balances to majority class size
        visualize : bool, default=False
            Whether to create TabPFGen's built-in visualization plots
        feature_names : list, optional
            Names of features for visualization

        Returns:
        -------
        X_synth : ndarray
            Generated synthetic features only
        y_synth : ndarray
            Generated synthetic labels only
        X_combined : ndarray
            Combined dataset features (original + synthetic)
        y_combined : ndarray
            Combined dataset labels (original + synthetic)

        Notes:
        -----
        The final class distribution may be approximately balanced rather than
        perfectly balanced due to TabPFN's label refinement process, which
        prioritizes data quality and realism over exact class counts.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Show original class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Use TabPFGen's balance_dataset method
        if target_per_class is None:
            # Balance to majority class size automatically
            X_synth, y_synth, X_combined, y_combined = self.generator.balance_dataset(
                X, y
            )
        else:
            # Balance to specified target per class
            X_synth, y_synth, X_combined, y_combined = self.generator.balance_dataset(
                X, y, target_per_class=target_per_class
            )

        # Show results

        # Show final distribution
        final_unique, final_counts = np.unique(y_combined, return_counts=True)

        # Use TabPFGen's built-in visualization if requested
        if visualize and TABPFGEN_AVAILABLE:
            try:
                visualize_classification_results(
                    X, y, X_synth, y_synth, feature_names=feature_names
                )
            except (ImportError, AttributeError, ValueError, TypeError) as e:
                warnings.warn(f"TabPFGen visualization failed: {e}", stacklevel=2)

        return X_synth, y_synth, X_combined, y_combined
