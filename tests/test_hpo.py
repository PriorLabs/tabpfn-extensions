"""Tests for the TabPFN hyperparameter optimization (HPO) implementation.

This file tests the HPO implementations in tabpfn_extensions.hpo.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FAST_TEST_MODE
from utils import get_small_test_search_space

# Try to import the HPO module, but skip tests if hyperopt is not available
try:
    from hyperopt.pyll.stochastic import sample

    from tabpfn_extensions.hpo import TunedTabPFNClassifier, TunedTabPFNRegressor
    from tabpfn_extensions.hpo.search_space import get_param_grid_hyperopt

except ImportError:
    pytest.skip(
        "hyperopt not installed. Install with 'pip install \"tabpfn-extensions[hpo]\"'",
        allow_module_level=True,
    )

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

except ImportError:
    pytest.skip(
        "tabpfn core library not installed. Cannot run compatibility tests.",
        allow_module_level=True,
    )

from test_base_tabpfn import BaseClassifierTests, BaseRegressorTests

# Using get_small_test_search_space from tests.utils - all HPO modules should use it

NUM_CONFIGS_TO_TEST = 100


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTunedTabPFNClassifier(BaseClassifierTests):
    """Test TunedTabPFNClassifier using the BaseClassifierTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a HPO-based TabPFN classifier as the estimator."""
        n_trials = 3 if FAST_TEST_MODE else 10  # Very limited trials for fast testing

        # Use minimal search space in fast test mode
        search_space = get_small_test_search_space() if FAST_TEST_MODE else None

        return TunedTabPFNClassifier(
            n_trials=n_trials,
            metric="accuracy",
            random_state=42,
            search_space=search_space,
        )

    @pytest.mark.skip(reason="Tuned TabPFN models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for HPO."""
        pass

    @pytest.mark.skip(
        reason="TunedTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass


class TestTunedTabPFNClassifierROCAUC(BaseClassifierTests):
    """Test TunedTabPFNClassifier with ROC AUC metric."""

    @pytest.fixture
    def estimator(self, tabpfn_classifier):
        """Provide a HPO-based TabPFN classifier as the estimator."""
        n_trials = 3 if FAST_TEST_MODE else 10  # Very limited trials for fast testing

        # Use minimal search space in fast test mode
        search_space = get_small_test_search_space() if FAST_TEST_MODE else None

        return TunedTabPFNClassifier(
            n_trials=n_trials,
            metric="roc_auc",
            random_state=42,
            search_space=search_space,
        )


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestTunedTabPFNRegressor(BaseRegressorTests):
    """Test TunedTabPFNRegressor using the BaseRegressorTests framework."""

    @pytest.fixture
    def estimator(self, tabpfn_regressor):
        """Provide a HPO-based TabPFN regressor as the estimator."""
        n_trials = 2  # Very limited trials for fast testing

        # Use minimal search space in fast test mode
        search_space = get_small_test_search_space() if FAST_TEST_MODE else None

        return TunedTabPFNRegressor(
            n_trials=n_trials,
            metric="rmse",
            random_state=42,
            search_space=search_space,
        )

    @pytest.mark.skip(reason="Tuned TabPFN models take too long for this test")
    def test_with_various_datasets(self, estimator, dataset_generator):
        """Skip test with various datasets as it takes too long for HPO."""
        pass

    @pytest.mark.skip(
        reason="TunedTabPFN needs additional work to pass all sklearn estimator checks",
    )
    def test_passes_estimator_checks(self, estimator):
        pass


@pytest.mark.local_compatible
@pytest.mark.client_compatible
class TestSearchSpaceCompatibility:
    """Tests to ensure the search space is compatible with core TabPFN models."""

    @pytest.mark.parametrize("task_type", ["multiclass", "regression"])
    def test_search_space_instantiates_core_tabpfn_models(self, task_type):
        """Tests if parameters drawn from the hyperopt search space can be
        used to instantiate core TabPFNClassifier or TabPFNRegressor models.
        This verifies compatibility between the HPO search space and the
        TabPFN core library's expected configuration.
        """
        full_search_space = get_param_grid_hyperopt(task_type)
        rng = np.random.default_rng(42)

        for i in range(NUM_CONFIGS_TO_TEST):
            sample_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

            sampled_config = sample(full_search_space, rng=sample_rng)

            inference_config_params = {}
            model_params = {}
            for k, v in sampled_config.items():
                if k.startswith("inference_config/"):
                    inference_config_params[k.split("/")[-1]] = v
                else:
                    model_params[k] = v

            # Handle model_type and max_depth which are specific to TunedTabPFN and not core TabPFN
            model_type = model_params.pop("model_type", "single")
            max_depth = model_params.pop("max_depth", None)

            if "model_path" not in model_params:
                model_params["model_path"] = "auto"

            try:
                if task_type == "multiclass":
                    if model_type == "single":
                        model_instance = TabPFNClassifier(
                            **model_params,
                            inference_config=inference_config_params,
                            device="cpu",
                        )
                    elif model_type == "dt_pfn":
                        from tabpfn_extensions.rf_pfn import (
                            DecisionTreeTabPFNClassifier,
                        )

                        base_clf = TabPFNClassifier(
                            **model_params,
                            inference_config=inference_config_params,
                            device="cpu",
                        )
                        model_instance = DecisionTreeTabPFNClassifier(
                            tabpfn=base_clf, max_depth=max_depth
                        )
                    else:
                        pytest.fail(
                            f"Unsupported model_type for multiclass: {model_type}"
                        )

                elif task_type == "regression":
                    if model_type == "single":
                        model_instance = TabPFNRegressor(
                            **model_params,
                            inference_config=inference_config_params,
                            device="cpu",
                        )
                    elif model_type == "dt_pfn":
                        from tabpfn_extensions.rf_pfn import DecisionTreeTabPFNRegressor

                        base_reg = TabPFNRegressor(
                            **model_params,
                            inference_config=inference_config_params,
                            device="cpu",
                        )
                        model_instance = DecisionTreeTabPFNRegressor(
                            tabpfn=base_reg, max_depth=max_depth
                        )
                    else:
                        pytest.fail(
                            f"Unsupported model_type for regression: {model_type}"
                        )
                else:
                    pytest.fail(f"Unknown task type: {task_type}")

                assert (
                    model_instance is not None
                ), "Model instance should not be None after instantiation attempt."

            except (TypeError, ValueError, RuntimeError, AssertionError) as e:
                pytest.fail(
                    f"Test failed for sample {i + 1} ({task_type} task).\n"
                    f"Sampled parameters: model_params: {model_params}, "
                    f"inference_config: {inference_config_params}.\n"
                    f"Error: {type(e).__name__}: {e}"
                )
