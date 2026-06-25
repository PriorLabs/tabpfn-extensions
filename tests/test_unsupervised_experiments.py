"""Tests for the unsupervised experiment wrappers in ``experiments.py``.

These cover the categorical-feature handling of ``GenerateSyntheticDataExperiment``
and ``OutlierDetectionUnsupervisedExperiment`` (regression test for issue #323,
where every selected feature was force-marked categorical and numerical columns
were generated as integers).

The fast tests use a lightweight recording stand-in so they need no TabPFN
weights; one end-to-end test uses a real model under ``FAST_TEST_MODE=1``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

# experiments.py imports matplotlib/seaborn at import time; skip the whole module
# if those optional plotting dependencies are not installed.
experiments = pytest.importorskip("tabpfn_extensions.unsupervised.experiments")
GenerateSyntheticDataExperiment = experiments.GenerateSyntheticDataExperiment
OutlierDetectionUnsupervisedExperiment = (
    experiments.OutlierDetectionUnsupervisedExperiment
)


class _RecordingUnsupervisedModel:
    """Stand-in for ``TabPFNUnsupervisedModel`` that needs no TabPFN weights.

    It records the categorical-feature indices the experiment passes to
    ``set_categorical_features`` so the index-translation logic can be tested in
    isolation.
    """

    def __init__(self):
        self.categorical_features: list[int] = []
        self._fitted_X = None

    def set_categorical_features(self, categorical_features):
        self.categorical_features = list(categorical_features)

    def fit(self, X):
        self._fitted_X = X
        return self

    def generate_synthetic_data(self, n_samples, t=1.0, n_permutations=3, dag=None):
        n_features = self._fitted_X.shape[1]
        return np.random.default_rng(0).normal(size=(n_samples, n_features))

    def outliers(self, X, n_permutations=3):
        return torch.zeros(X.shape[0], dtype=torch.float32)


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_experiment_uses_supplied_categorical_features():
    """Only the caller-designated column is marked categorical (issue #323)."""
    X = np.column_stack(
        [
            np.repeat(np.arange(4), 30),  # 120 rows, 4 unique -> categorical
            np.random.default_rng(0).normal(size=120),  # continuous -> numerical
        ],
    ).astype(np.float32)

    model = _RecordingUnsupervisedModel()
    experiment = GenerateSyntheticDataExperiment(task_type="unsupervised")
    experiment.run(
        tabpfn=model,
        X=X,
        y=np.array([]),
        attribute_names=["categorical", "numerical"],
        indices=[0, 1],
        categorical_features=[0],
        n_samples=10,
        should_plot=False,
    )

    # Previously the experiment force-marked every selected column categorical.
    assert model.categorical_features == [0]


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_experiment_defaults_to_no_categorical_features():
    """With no override, nothing is force-marked categorical (model auto-detects)."""
    X = np.column_stack(
        [
            np.repeat(np.arange(4), 30),
            np.random.default_rng(0).normal(size=120),
        ],
    ).astype(np.float32)

    model = _RecordingUnsupervisedModel()
    experiment = GenerateSyntheticDataExperiment(task_type="unsupervised")
    experiment.run(
        tabpfn=model,
        X=X,
        y=np.array([]),
        attribute_names=["categorical", "numerical"],
        indices=[0, 1],
        n_samples=10,
        should_plot=False,
    )

    assert model.categorical_features == []


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_experiment_translates_original_indices_to_subset():
    """categorical_features given in original X space map to the selected subset."""
    X = np.column_stack(
        [
            np.random.default_rng(0).normal(size=120),  # col 0
            np.random.default_rng(1).normal(size=120),  # col 1 (not selected)
            np.repeat(np.arange(4), 30),  # col 2 categorical
        ],
    ).astype(np.float32)

    model = _RecordingUnsupervisedModel()
    experiment = GenerateSyntheticDataExperiment(task_type="unsupervised")
    experiment.run(
        tabpfn=model,
        X=X,
        y=np.array([]),
        attribute_names=["a", "b", "c"],
        indices=[2, 0],  # original col 2 lands at subset position 0
        categorical_features=[2, 1],  # col 1 is not selected -> ignored
        n_samples=10,
        should_plot=False,
    )

    assert model.categorical_features == [0]


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_outlier_experiment_uses_supplied_categorical_features():
    """OutlierDetection shares the same fix: respect the supplied categorical list."""
    X = torch.tensor(
        np.column_stack(
            [
                np.repeat(np.arange(4), 30),
                np.random.default_rng(0).normal(size=120),
            ],
        ).astype(np.float32),
    )

    model = _RecordingUnsupervisedModel()
    experiment = OutlierDetectionUnsupervisedExperiment(task_type="unsupervised")
    result = experiment.run(
        tabpfn=model,
        X=X,
        y=torch.zeros(X.shape[0]),
        attribute_names=["categorical", "numerical"],
        indices=[0, 1],
        categorical_features=[0],
        should_plot=False,
    )

    assert model.categorical_features == [0]
    assert "log_p" in result


@pytest.mark.client_compatible
@pytest.mark.local_compatible
def test_generate_synthetic_experiment_keeps_numerical_features_numerical(monkeypatch):
    """End-to-end: a numerical column must not be generated as integers (issue #323)."""
    monkeypatch.setenv("FAST_TEST_MODE", "1")

    rng = np.random.default_rng(0)
    X = np.column_stack(
        [
            np.repeat(np.arange(4), 15),  # 60 rows, 4 unique -> categorical
            rng.normal(size=60),  # continuous -> numerical
        ],
    ).astype(np.float32)

    clf = TabPFNClassifier(n_estimators=1, random_state=0)
    reg = TabPFNRegressor(n_estimators=1, random_state=0)
    model = unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)

    experiment = GenerateSyntheticDataExperiment(task_type="unsupervised")
    experiment.run(
        tabpfn=model,
        X=X,
        y=np.array([]),
        attribute_names=["categorical", "numerical"],
        indices=[0, 1],
        categorical_features=[0],
        n_samples=20,
        n_permutations=1,
        should_plot=False,
    )

    # The bug marked both columns categorical; the numerical one must stay numerical.
    assert 0 in model.categorical_features
    assert 1 not in model.categorical_features

    synthetic = experiment.data[
        experiment.data["real_or_synthetic"] == "Generated samples"
    ]
    numerical = synthetic["numerical"].to_numpy()
    assert not np.allclose(numerical, np.round(numerical))
