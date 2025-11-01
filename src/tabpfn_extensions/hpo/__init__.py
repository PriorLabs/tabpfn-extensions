# Check if hyperopt is available
try:
    import importlib.util

    HYPEROPT_AVAILABLE = importlib.util.find_spec("hyperopt") is not None
    if not HYPEROPT_AVAILABLE:
        raise ImportError("hyperopt not found")
except ImportError:
    HYPEROPT_AVAILABLE = False
    import warnings

    warnings.warn(
        "hyperopt not installed. HPO extensions will not be available. "
        "Install with 'pip install \"tabpfn-extensions[hpo]\"'",
        ImportWarning,
        stacklevel=2,
    )

# Only import if hyperopt is available
if HYPEROPT_AVAILABLE:
    from .search_space import (
        get_param_grid_hyperopt,
    )
    from .tuned_tabpfn import TunedTabPFNClassifier, TunedTabPFNRegressor

    __all__ = [
        "get_param_grid_hyperopt",
        "TunedTabPFNClassifier",
        "TunedTabPFNRegressor",
        "HYPEROPT_AVAILABLE",
    ]
else:
    # Define placeholder classes when hyperopt is not available
    class TunedTabPFNClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TunedTabPFNClassifier requires hyperopt. "
                "Install with 'pip install \"tabpfn-extensions[hpo]\"'"
            )

    class TunedTabPFNRegressor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TunedTabPFNRegressor requires hyperopt. "
                "Install with 'pip install \"tabpfn-extensions[hpo]\"'"
            )

    __all__ = [
        "TunedTabPFNClassifier",
        "TunedTabPFNRegressor",
        "HYPEROPT_AVAILABLE",
    ]
