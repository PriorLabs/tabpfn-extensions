from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabpfn-extensions")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from .utils import test_utils, get_tabpfn_models, is_tabpfn, simulate_first

# Get the TabPFN models with our wrappers applied
TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()

__all__ = ["test_utils", "TabPFNClassifier", "TabPFNRegressor", "is_tabpfn", "simulate_first"]
