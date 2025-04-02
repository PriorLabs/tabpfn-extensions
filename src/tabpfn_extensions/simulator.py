import threading
import contextlib
from typing import Literal
import numpy as np
import time
from unittest import mock
import warnings
import logging
import functools
import os

from tabpfn_client.client import ServiceClient
from tabpfn_client.tabpfn_common_utils.expense_estimation import estimate_duration


USE_TABPFN_LOCAL = False  # os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"
CLIENT_COST_ESTIMATION_LATENCY_OFFSET = 1.0


# Use thread-local variables to keep track of the current mode, simulated costs and time.
_thread_local = threading.local()


# Block of small helper functions to access and modify the thread-local
# variables used for mock prediction in a simple and unified way.
def get_mock_cost() -> float:
    return getattr(_thread_local, "cost", 0.0)


def increment_mock_cost(value: float):
    setattr(_thread_local, "cost", get_mock_cost() + value)


def set_mock_cost(value: float = 0.0):
    setattr(_thread_local, "cost", value)


def get_mock_time() -> float:
    return getattr(_thread_local, "mock_time")


def set_mock_time(value: float):
    setattr(_thread_local, "mock_time", value)


def increment_mock_time(seconds: float):
    set_mock_time(get_mock_time() + seconds)


def mock_fit(cls, X, y, config=None):
    return("mock_id")


def mock_predict(
    cls,
    X_test,
    task: Literal["classification", "regression"],
    train_set_uid: str,
    X_train,
    y_train,
    config=None,
    predict_params=None,
):
    """
    Mock function for prediction, which can be called instead of the real
    prediction function. Outputs random results in the expacted format and
    keeps track of the simulated cost and time.
    """
    if X_train is None or y_train is None:
        raise ValueError(
            "X_train and y_train must be provided in mock mode during prediction."
        )

    duration = estimate_duration(
        num_rows=X_train.shape[0] + X_test.shape[0],
        num_features=X_test.shape[1],
        task=task,
        tabpfn_config=config,
        latency_offset=CLIENT_COST_ESTIMATION_LATENCY_OFFSET,  # To slightly overestimate (safer)
    )
    increment_mock_time(duration)

    cost = (
        (X_train.shape[0] + X_test.shape[0])
        * X_test.shape[1]
        * config.get("n_estimators", 4 if task == "classification" else 8)
    )
    increment_mock_cost(cost)

    # Return random result in the correct format
    if task == "classification":
        if (
            not predict_params["output_type"]
            or predict_params["output_type"] == "preds"
        ):
            return np.random.rand(X_test.shape[0])
        elif predict_params["output_type"] == "probas":
            probs = np.random.rand(X_test.shape[0], len(np.unique(y_train)))
            return probs / probs.sum(axis=1, keepdims=True)

    elif task == "regression":
        if not predict_params["output_type"] or predict_params["output_type"] == "mean":
            return np.random.rand(X_test.shape[0])
        elif predict_params["output_type"] == "full":
            return {
                "logits": np.random.rand(X_test.shape[0], 5000),
                "mean": np.random.rand(X_test.shape[0]),
                "median": np.random.rand(X_test.shape[0]),
                "mode": np.random.rand(X_test.shape[0]),
                "quantiles": np.random.rand(3, X_test.shape[0]),
                "borders": np.random.rand(5001),
                "ei": np.random.rand(X_test.shape[0]),
                "pi": np.random.rand(X_test.shape[0]),
            }


@contextlib.contextmanager
def mock_mode():
    """
    Context manager that enables mock mode in the current thread.
    """
    set_mock_cost(0.0)
    start_time = time.time()
    set_mock_time(start_time)

    # Store original logging levels for all loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())  # Add root logger
    original_levels = {logger: logger.level for logger in loggers}

    if USE_TABPFN_LOCAL:
        from tabpfn.classifier import TabPFNClassifier
        from tabpfn.regressor import TabPFNRegressor
        original_fit_classification = getattr(TabPFNClassifier, "fit")
        original_fit_regressor = getattr(TabPFNRegressor, "fit")
        original_predict_classification = getattr(TabPFNClassifier, "predict")
        original_predict_regressor = getattr(TabPFNRegressor, "predict")
        setattr(TabPFNClassifier, "fit", classmethod(mock_fit))
        setattr(TabPFNClassifier, "predict", classmethod(mock_predict))
        setattr(TabPFNRegressor, "fit", classmethod(mock_fit))
        setattr(TabPFNRegressor, "predict", classmethod(mock_predict))
    else:
        from tabpfn_client.service_wrapper import InferenceClient
        original_fit = getattr(InferenceClient, "fit")
        original_predict = getattr(InferenceClient, "predict")
        setattr(InferenceClient, "fit", classmethod(mock_fit))
        setattr(InferenceClient, "predict", classmethod(mock_predict))

    # Suppress all warnings and logging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Set all loggers to ERROR level
        for logger in loggers:
            logger.setLevel(logging.ERROR)

        with mock.patch("time.time", side_effect=get_mock_time):
            try:
                yield lambda: (get_mock_time() - start_time, get_mock_cost())
            finally:
                if USE_TABPFN_LOCAL:
                    from tabpfn.classifier import TabPFNClassifier
                    from tabpfn.regressor import TabPFNRegressor
                    setattr(TabPFNClassifier, "fit", original_fit_classification)
                    setattr(TabPFNClassifier, "predict", original_predict_classification)
                    setattr(TabPFNRegressor, "fit", original_fit_regressor)
                    setattr(TabPFNRegressor, "predict", original_predict_regressor)
                else:
                    from tabpfn_client.service_wrapper import InferenceClient
                    setattr(InferenceClient, "fit", original_fit)
                    setattr(InferenceClient, "predict", original_predict)
                
                # Restore original logging levels
                for logger in loggers:
                    logger.setLevel(original_levels[logger])


def simulate_first(func):
    """
    Decorator that first runs the decorated function in mock mode to simulate its credit usage.
    If user has enough credits, function is then executed for real.
    """
    from tabpfn_client import get_access_token

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mock_mode() as get_simulation_results:
            func(*args, **kwargs)
            time_estimate, credit_estimate = get_simulation_results()
        access_token = get_access_token()
        api_usage = ServiceClient.get_api_usage(access_token)

        if (
            not api_usage["usage_limit"] == -1
            and api_usage["usage_limit"] - api_usage["current_usage"] < credit_estimate
        ):
            raise RuntimeError(
                f"Not enough credits left. Estimated credit usage: {credit_estimate}, credits left: {api_usage['usage_limit'] - api_usage['current_usage']}"
            )
        else:
            print("Enough credits left.")

        return func(*args, **kwargs)

    return wrapper
