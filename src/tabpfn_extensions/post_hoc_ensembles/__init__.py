# Check if autogluon is available
try:
    import importlib.util

    AUTOGLUON_TABULAR_AVAILABLE = (
        importlib.util.find_spec("autogluon.tabular") is not None
    )
except ImportError:
    AUTOGLUON_TABULAR_AVAILABLE = False
    import warnings

    warnings.warn(
        "autogluon.tabular not installed. Post hoc ensembling extensions will not be available. "
        "Install with 'pip install \"tabpfn-extensions[post_hoc_ensembles]\"'",
        ImportWarning,
        stacklevel=2,
    )

if AUTOGLUON_TABULAR_AVAILABLE:
    from .sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

    __all__ = [
        "AutoTabPFNClassifier",
        "AutoTabPFNRegressor",
    ]
else:
    __all__ = ["AUTOGLUON_TABULAR_AVAILABLE"]
