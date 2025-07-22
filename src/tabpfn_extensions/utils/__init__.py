# Create alias for test_utils
from . import test_utils
from .simulator import simulate_first
from .utils import (
    ClientTabPFNClassifier,
    ClientTabPFNRegressor,
    LocalTabPFNClassifier,
    LocalTabPFNRegressor,
    TabPFNClassifier,
    TabPFNRegressor,
    get_device,
    get_tabpfn_models,
    infer_categorical_features,
    is_tabpfn,
    product_dict,
    softmax,
)

__all__ = [
    "get_tabpfn_models",
    "is_tabpfn",
    "test_utils",
    "simulate_first",
    "get_device",
    "TabPFNClassifier",
    "TabPFNRegressor",
    "LocalTabPFNClassifier",
    "LocalTabPFNRegressor",
    "ClientTabPFNClassifier",
    "ClientTabPFNRegressor",
    "infer_categorical_features",
    "softmax",
    "product_dict",
]
