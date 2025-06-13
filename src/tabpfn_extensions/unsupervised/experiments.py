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

    def run(self, tabpfn, **kwargs):
        """:param tabpfn:
        :param kwargs:
            indices: list of indices from X features to use
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X, y = copy.deepcopy(kwargs.get("X")), copy.deepcopy(kwargs.get("y"))
            attribute_names = kwargs.get("attribute_names")

            indices = kwargs.get("indices", list(range(X.shape[1])))

            temp = kwargs.get("temp", 1.0)
            n_samples = kwargs.get("n_samples", X.shape[0])
            
            n_permutations = kwargs.get("n_permutations", 1)
            dag = kwargs.get("dag", None)

            self.X, self.y = X, y
            self.X = self.X[:, indices]
            old_features_names = attribute_names
            self.feature_names = [attribute_names[i] for i in indices]
            # generate subset of categorical indices
            categorical_features = [
                self.feature_names.index(name)
                for name in old_features_names
                if name in self.feature_names
            ]
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

            self.plot()


class OutlierDetectionUnsupervisedExperiment(Experiment):
    """This class is used to run experiments for outlier detection."""

    name = "OutlierDetectionUnsupervisedExperiment"

    def plot(self):
        # Create a grid of jointplots using PairGrid
        g = sns.PairGrid(self.data, vars=self.feature_names)
        g.map_upper(sns.scatterplot, s=5, alpha=0.5, hue=self.data["p"])
        g.map_lower(sns.scatterplot, s=5, alpha=0.5, hue=self.data["p_rank"])
        g.add_legend()

    def plot_two(self, **kwargs):
        outlier_thresh_p = kwargs.get("outlier_thresh_p", 0.98)
        outlier_thresh = np.quantile(
            self.data["p"][self.data["p"] > 0],
            outlier_thresh_p,
        )

        outlier_thresh_p_1 = kwargs.get("outlier_thresh_p_1", 0.9)
        outlier_thresh_1 = np.quantile(
            self.data["p"][self.data["p"] > 0],
            outlier_thresh_p_1,
        )

        def outlier_f(x, thresh_0, thresh_1):
            if np.isnan(x):
                return np.nan
            if x > thresh_0:
                return f"Low ({round(100 * (1 - outlier_thresh_p), 2)} Percentile)"
            if x > thresh_1:
                return f"Medium ({round(100 * (1 - outlier_thresh_p_1), 2)} Percentile)"
            return "High"

        self.data["outlier"] = self.data["p"].map(
            partial(outlier_f, thresh_0=outlier_thresh, thresh_1=outlier_thresh_1),
        )
        # Oversample the data with outlier = True
        oversample_low = self.data[
            self.data["outlier"].map(lambda x: "Low" in x)
        ].sample(frac=1 / (1 - outlier_thresh_p), replace=True)
        oversample_med = self.data[
            self.data["outlier"].map(lambda x: "Medium" in x)
        ].sample(frac=1 / (1 - outlier_thresh_p_1), replace=True)
        data_ = pd.concat(
            [
                self.data[self.data["outlier"].map(lambda x: "High" in x)],
                oversample_low,
                oversample_med,
            ],
        )
        g = sns.JointGrid(
            data=data_,
            hue="outlier",
            x=self.feature_names[0],
            y=self.feature_names[1],
            height=DEFAULT_HEIGHT,
        )

        g.fig.suptitle("Data Density Estimation")
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)  # Reduce plot to make room

        g.plot_joint(sns.scatterplot, s=50, alpha=0.5)
        g.plot_marginals(sns.histplot, kde=True, stat="density")

        # Remove the original legend created by plot_joint
        g.ax_joint.get_legend().remove()

        # Create a new legend on the joint plot axis with no frame and no title
        handles, labels = g.ax_joint.get_legend_handles_labels()
        leg = g.ax_joint.legend(
            handles=handles,
            labels=labels,
            loc="upper right",
            title="Estimated density (percentile)",
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_alpha(1)  # Make the legend background completely opaque

        return g

    def run(
        self,
        tabpfn,
        overwrite_baseline_cache=False,
        overwrite_tabpfn_cache=True,
        **kwargs,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X, _y = copy.deepcopy(kwargs.get("X")), copy.deepcopy(kwargs.get("y"))
            attribute_names = kwargs.get("attribute_names")

            indices = kwargs.get("indices", list(range(X.shape[1])))
            n_permutations = kwargs.get("n_permutations", 3)

            self.X = X
            self.X = self.X[:, indices]
            old_features_names = attribute_names
            self.feature_names = [attribute_names[i] for i in indices]
            # generate subset of categorical indices
            categorical_features = [
                self.feature_names.index(name)
                for name in old_features_names
                if name in self.feature_names
            ]
            tabpfn.set_categorical_features(categorical_features)

            tabpfn.fit(self.X)
            self.p = tabpfn.outliers(self.X, n_permutations=n_permutations)

            p_rank = self.p.argsort().argsort()

            self.data = pd.DataFrame(
                torch.cat(
                    [self.p[:, np.newaxis], p_rank[:, np.newaxis], self.X],
                    dim=1,
                ).numpy(),
                columns=["p", "p_rank", *self.feature_names],
            )

            if kwargs.get("should_plot", True):
                try:
                    # We don't need to import the module directly here
                    # since plot_two() will do the import
                    self.plot_two()
                except ImportError:
                    # Skip plotting if matplotlib is not available
                    pass

            return {"outlier_scores": self.p.numpy()}
