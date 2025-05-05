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

from tabpfn_common_utils.expense_estimation import estimate_duration


CLIENT_COST_ESTIMATION_LATENCY_OFFSET = 1.0


# Use thread-local variables to keep track of the current mode, simulated costs and time.
_thread_local = threading.local()


# Block of small helper functions to access and modify the thread-local
# variables used for mock prediction in a simple and unified way.
def get_is_local_tabpfn() -> bool:
    return getattr(_thread_local, "use_local_tabpfn", True)


def set_is_local_tabpfn():
    """Figure out whether local TabPFN or client is used and set thread-local variable."""
    use_local_env = os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"

    try:
        from tabpfn import TabPFNClassifier as LocalTabPFNClassifier
    except ImportError:
        LocalTabPFNClassifier = None

    use_local = use_local_env and LocalTabPFNClassifier is not None
    setattr(_thread_local, "use_local_tabpfn", use_local)


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


# Block of functions that will replace the actual fit and predict functions in mock mode.
def mock_fit_local(self, X, y, config=None):
    # Store train data, as it is needed for mocking prediction correctly. The client
    # already does this internally.
    self.X_train = X
    self.y_train = y
    return("mock_id")


def mock_fit_client(cls, X, y, config=None):
    return("mock_id")


def mock_predict_local(self, X_test):
    """Wrapper for being able to distinguish between predict and predict_proba."""
    return mock_predict_proba_local(self, X_test, from_classifier_predict=True)


def mock_predict_proba_local(self, X_test, from_classifier_predict=False):
    """
    Wrapper for mock_predict to set the correct arguments for local prediction. The client
    already does this internally.
    """
    task = "classification" if self.__class__.__name__ == "TabPFNClassifier" else "regression"
    config = {"n_estimators": self.n_estimators}
    params = {}
    if task == "classification":
        params["output_type"] = "preds" if from_classifier_predict else "probas"
    else:
        params["output_type"] = "mean"
    return mock_predict(self, X_test, task, "dummy", self.X_train, self.y_train, config, params)


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
        latency_offset=0 if get_is_local_tabpfn() else CLIENT_COST_ESTIMATION_LATENCY_OFFSET,  # To slightly overestimate (safer)
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
    loggers.append(logging.getLogger())
    original_levels = {logger: logger.level for logger in loggers}

    # Overwrite actual fit and predict functions with mock functions
    if get_is_local_tabpfn():
        from tabpfn import TabPFNClassifier
        from tabpfn import TabPFNRegressor
        original_fit_classification = getattr(TabPFNClassifier, "fit")
        original_fit_regressor = getattr(TabPFNRegressor, "fit")
        original_predict_classification = getattr(TabPFNClassifier, "predict")
        original_predict_proba_classification = getattr(TabPFNClassifier, "predict_proba")
        original_predict_regressor = getattr(TabPFNRegressor, "predict")
        setattr(TabPFNClassifier, "fit", mock_fit_local)
        setattr(TabPFNClassifier, "predict", mock_predict_local)
        setattr(TabPFNClassifier, "predict_proba", mock_predict_proba_local)
        setattr(TabPFNRegressor, "fit", mock_fit_local)
        setattr(TabPFNRegressor, "predict", mock_predict_proba_local)
    else:
        from tabpfn_client.service_wrapper import InferenceClient
        original_fit = getattr(InferenceClient, "fit")
        original_predict = getattr(InferenceClient, "predict")
        setattr(InferenceClient, "fit", classmethod(mock_fit_client))
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
                if get_is_local_tabpfn():
                    from tabpfn.classifier import TabPFNClassifier
                    from tabpfn.regressor import TabPFNRegressor
                    setattr(TabPFNClassifier, "fit", original_fit_classification)
                    setattr(TabPFNClassifier, "predict", original_predict_classification)
                    setattr(TabPFNClassifier, "predict_proba", original_predict_proba_classification)
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
    Decorator that first runs the decorated function in mock mode to simulate its duration
    and credit usage. If client is used, only executes function if enough credits are available.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        set_is_local_tabpfn()
        with mock_mode() as get_simulation_results:
            func(*args, **kwargs)
            time_estimate, credit_estimate = get_simulation_results()
        
        if not get_is_local_tabpfn():
            from tabpfn_client.client import ServiceClient
            from tabpfn_client import get_access_token
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

        print(f"Estimated duration: {time_estimate:.1f} seconds {'(on GPU)' if get_is_local_tabpfn() else ''}")

        return func(*args, **kwargs)

    return wrapper
