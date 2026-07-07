#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import copy
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from tabpfn_extensions.benchmarking import Experiment

DEFAULT_HEIGHT = 6


def _map_categorical_to_subset(categorical_features, indices):
    """Map categorical column indices (original X space) to selected-subset positions.

    Args:
        categorical_features: Column indices in the original ``X`` space, or ``None``.
            Any sequence type (list, tuple, numpy array, torch tensor) is accepted.
        indices: The selected columns, as a list of ints in subset order.

    Returns:
        list[int]: Positions within ``indices`` of the selected categorical columns;
            indices not present in ``indices`` are dropped.
    """
    if categorical_features is None:
        return []
    subset_position = {col: pos for pos, col in enumerate(indices)}
    return [
        subset_position[int(c)]
        for c in categorical_features
        if int(c) in subset_position
    ]


class EmbeddingUnsupervisedExperiment(Experiment):
    """This class is used to run experiments on synthetic toy functions."""

    name = "EmbeddingUnsupervisedExperiment"

    def _plot(self, ax, **kwargs):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        # Instantialte tsne, specify cosine metric
        lower_dim = TSNE(random_state=0, n_iter=10000, metric="cosine")
        lower_dim = PCA(n_components=2)

        scaler = StandardScaler()

        # Fit and transform
        embeddings2d = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(self.emb)),
        )
        # Scatter points, set alpha low to make points translucent
        ax[0].scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=1 + self.y_test.numpy())
        ax[0].set_title("Embedded data + PCA")
        ax[0].set_xlabel("PCA 1")
        ax[0].set_ylabel("PCA 2")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Fit and transform
        embeddings2d = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(self.X_test)),
        )
        # Scatter points, set alpha low to make points translucent
        ax[1].scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=1 + self.y_test.numpy())
        ax[1].set_title("Original data + PCA")
        ax[1].set_xlabel("PCA 1")
        ax[1].set_ylabel("PCA 2")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    def plot(self, **kwargs):
        # Set figsize
        fig, ax = plt.subplots(2, figsize=(DEFAULT_HEIGHT, DEFAULT_HEIGHT))
        fig.tight_layout()

        self._plot(ax, **kwargs)

    def run(self, tabpfn, **kwargs):
        assert kwargs.get("dataset") is not None, "Dataset must be provided"
        dataset = kwargs.get("dataset")

        self.X, self.y = dataset.x, dataset.y
        # split into train & test
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.5,
            random_state=42,
        )

        tabpfn.fit(self.X_train, self.y_train)
        self.emb = tabpfn.get_embeddings(
            self.X_test,
            per_column=kwargs.get("per_column", False),
        )

        self.plot()


class GenerateSyntheticDataExperiment(Experiment):
    """This class is used to run experiments on generating synthetic data."""

    name = "GenerateSyntheticDataExperiment"

    def plot(self, **kwargs):
        # Create a grid of jointplots using PairGrid
        g = sns.PairGrid(self.data, hue="real_or_synthetic", diag_sharey=False)
        g.map_diag(sns.histplot, common_norm=True)
        g.map_offdiag(sns.scatterplot, s=2, alpha=0.5)
        g.add_legend()

    def run(self, tabpfn, *, categorical_features=None, should_plot=True, **kwargs):
        """Generate synthetic data and store it on the experiment instance.

        The synthetic data is stored on the following instance property:
            - synthetic_X: array of generated data, shape (n_samples, n_features)
        The following properties are also set
            - data_real: input X data as a DataFrame, potentially resampled
            - data_synthetic: synthetic_X as a DataFrame, potentially resampled
            - data: data_real and data_synthetic concatenated
        If one of data_real or data_synthetic has fewer rows, it is resampled with
        replacement so both have max(n_input_samples, n_samples) rows.
        data_real, data_synthetic, and data have an additional real_or_synthetic column
        that indicates if the data is real or synthetic.

        Args:
            tabpfn: A ``TabPFNUnsupervisedModel`` used to learn the joint
                distribution of the selected features and sample synthetic rows.
            categorical_features: Column indices of ``X`` (same index space as
                ``indices``) to treat as categorical. Indices not present in
                ``indices`` are ignored. Defaults to ``None``, in which case the
                model auto-detects categorical columns at ``fit`` time.
            should_plot: Whether to render the pairwise plot. Defaults to ``True``.
            **kwargs: Keyword arguments controlling the run:
                X: Input data array of shape ``(n_input_samples, n_features)``.
                y: Targets (unused for unsupervised generation; may be empty).
                attribute_names: Column names for every column in ``X``.
                indices: Column indices of ``X`` to model. Defaults to all columns.
                temp: Sampling temperature. Defaults to ``1.0``.
                n_samples: Number of synthetic rows to generate. Defaults to
                    ``X.shape[0]``.
                n_permutations: Number of feature-order permutations to average.
                    Defaults to ``3``.
                dag: Optional causal DAG passed to the generator.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X, y = copy.deepcopy(kwargs.get("X")), copy.deepcopy(kwargs.get("y"))
            attribute_names = kwargs.get("attribute_names")

            indices = [int(i) for i in kwargs.get("indices", range(X.shape[1]))]

            temp = kwargs.get("temp", 1.0)
            n_samples = kwargs.get("n_samples", X.shape[0])
            n_permutations = kwargs.get("n_permutations", 3)
            dag = kwargs.get("dag")

            self.X, self.y = X, y
            self.X = self.X[:, indices]
            self.feature_names = [attribute_names[i] for i in indices]
            categorical_features = _map_categorical_to_subset(
                categorical_features,
                indices,
            )
            tabpfn.set_categorical_features(categorical_features)
            tabpfn.fit(self.X)

            self.synthetic_X = tabpfn.generate_synthetic_data(
                n_samples=n_samples,
                t=temp,
                n_permutations=n_permutations,
                dag=dag,
            )

            data_real = pd.DataFrame(
                {
                    **dict(
                        zip(
                            self.feature_names,
                            [self.X[:, i] for i in range(self.X.shape[1])],
                            strict=True,
                        ),
                    ),
                    "real_or_synthetic": "Actual samples",
                },
            )
            data_synthetic = pd.DataFrame(
                {
                    **dict(
                        zip(
                            self.feature_names,
                            [
                                self.synthetic_X[:, i]
                                for i in range(self.synthetic_X.shape[1])
                            ],
                            strict=True,
                        ),
                    ),
                    "real_or_synthetic": "Generated samples",
                },
            )
            self.data_real = data_real
            self.data_synthetic = data_synthetic
            if self.data_real.shape[0] < self.data_synthetic.shape[0]:
                self.data_real = self.data_real.sample(
                    n=self.data_synthetic.shape[0],
                    replace=True,
                )
            elif self.data_synthetic.shape[0] < self.data_real.shape[0]:
                self.data_synthetic = self.data_synthetic.sample(
                    n=self.data_real.shape[0],
                    replace=True,
                )
            self.data = pd.concat([self.data_real, self.data_synthetic])

            if should_plot:
                self.plot()


class OutlierDetectionUnsupervisedExperiment(Experiment):
    """This class is used to run experiments for outlier detection."""

    name = "OutlierDetectionUnsupervisedExperiment"

    def plot(self):
        # Create a grid of jointplots using PairGrid
        g = sns.PairGrid(self.data, vars=self.feature_names)
        g.map_upper(sns.scatterplot, s=5, alpha=0.5, hue=self.data["log_p"])
        g.map_lower(sns.scatterplot, s=5, alpha=0.5, hue=self.data["log_p_rank"])
        g.add_legend()

    def plot_two(self, **kwargs):
        outlier_thresh_p = kwargs.get("outlier_thresh_p", 0.02)
        outlier_thresh_p_1 = kwargs.get("outlier_thresh_p_1", 0.1)

        # np.quantile returns NaN if any rank position falls on -inf (since
        # interpolation across -inf yields -inf - -inf = NaN). Clamp -inf to
        # the finite minimum just for the quantile computation; the original
        # log_p series is preserved for bucketing, where x < thresh keeps
        # -inf rows correctly classified as Low.
        log_p_series = self.data["log_p"]
        finite_mask = np.isfinite(log_p_series)
        if finite_mask.any() and not finite_mask.all():
            finite_floor = float(log_p_series[finite_mask].min())
            log_p_for_quantile = log_p_series.where(finite_mask, finite_floor)
        else:
            log_p_for_quantile = log_p_series

        outlier_thresh = np.quantile(log_p_for_quantile, outlier_thresh_p)
        outlier_thresh_1 = np.quantile(log_p_for_quantile, outlier_thresh_p_1)

        def outlier_f(x, thresh_0, thresh_1):
            if np.isnan(x):
                return np.nan
            if x < thresh_0:
                return f"Low ({round(100 * (outlier_thresh_p), 2)} Percentile)"
            if x < thresh_1:
                return f"Medium ({round(100 * (outlier_thresh_p_1), 2)} Percentile)"
            return "High"

        self.data["outlier"] = self.data["log_p"].map(
            partial(outlier_f, thresh_0=outlier_thresh, thresh_1=outlier_thresh_1),
        )
        # Oversample the data with outlier = True
        oversample_low = self.data[
            self.data["outlier"].map(lambda x: "Low" in x)
        ].sample(frac=1 / (outlier_thresh_p), replace=True)
        oversample_med = self.data[
            self.data["outlier"].map(lambda x: "Medium" in x)
        ].sample(frac=1 / (outlier_thresh_p_1), replace=True)
        data_ = pd.concat(
            [
                self.data[self.data["outlier"].map(lambda x: "High" in x)],
                oversample_low,
                oversample_med,
            ],
        )
        fig, ax = plt.subplots(figsize=(DEFAULT_HEIGHT, DEFAULT_HEIGHT))
        sns.scatterplot(
            data=data_,
            x=self.feature_names[0],
            y=self.feature_names[1],
            hue="outlier",
            s=50,
            alpha=0.5,
            ax=ax,
        )

        ax.set_title("outlier detection")

        ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            title="Estimated data log(density)",
            fontsize="small",
            title_fontsize="small",
            borderpad=0.6,
            handletextpad=0.5,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_alpha(1)
        fig.tight_layout()

        return ax

    def run(
        self,
        tabpfn,
        overwrite_baseline_cache=False,
        overwrite_tabpfn_cache=True,
        *,
        categorical_features=None,
        should_plot=True,
        **kwargs,
    ):
        """Estimate per-sample outlier scores for the selected features.

        Args:
            tabpfn: A ``TabPFNUnsupervisedModel`` used to estimate sample density.
            overwrite_baseline_cache: Unused placeholder kept for API symmetry.
            overwrite_tabpfn_cache: Unused placeholder kept for API symmetry.
            categorical_features: Column indices of ``X`` (same index space as
                ``indices``) to treat as categorical. Indices not present in
                ``indices`` are ignored. Defaults to ``None``, in which case the
                model auto-detects categorical columns at ``fit`` time.
            should_plot: Whether to render the density plot. Defaults to ``True``.
            **kwargs: Keyword arguments controlling the run:
                X: Input data array of shape ``(n_samples, n_features)``.
                y: Targets (unused; may be empty).
                attribute_names: Column names for every column in ``X``.
                indices: Column indices of ``X`` to model. Defaults to all columns.
                n_permutations: Number of feature-order permutations to average.
                    Defaults to ``3``.

        Returns:
            dict: Mapping with key ``"log_p"`` holding the per-sample log-density
                (lower values indicate more likely outliers).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X, _y = copy.deepcopy(kwargs.get("X")), copy.deepcopy(kwargs.get("y"))
            attribute_names = kwargs.get("attribute_names")

            indices = [int(i) for i in kwargs.get("indices", range(X.shape[1]))]
            n_permutations = kwargs.get("n_permutations", 3)

            self.X = X
            self.X = self.X[:, indices]
            self.feature_names = [attribute_names[i] for i in indices]
            categorical_features = _map_categorical_to_subset(
                categorical_features,
                indices,
            )
            tabpfn.set_categorical_features(categorical_features)

            tabpfn.fit(self.X)
            self.log_p = tabpfn.outliers(self.X, n_permutations=n_permutations)

            log_p_rank = self.log_p.argsort().argsort()

            self.data = pd.DataFrame(
                torch.cat(
                    [self.log_p[:, np.newaxis], log_p_rank[:, np.newaxis], self.X],
                    dim=1,
                ).numpy(),
                columns=["log_p", "log_p_rank", *self.feature_names],
            )

            if should_plot:
                try:
                    # We don't need to import the module directly here
                    # since plot_two() will do the import
                    self.plot_two()
                except ImportError:
                    # Skip plotting if matplotlib is not available
                    pass

            return {"log_p": self.log_p.numpy()}
