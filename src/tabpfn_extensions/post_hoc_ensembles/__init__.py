import importlib.util
import warnings

# Check if the optional dependency 'autogluon.tabular' is installed.
AUTOGLUON_TABULAR_AVAILABLE = importlib.util.find_spec("autogluon.tabular") is not None

if AUTOGLUON_TABULAR_AVAILABLE:
    # If it's installed, import and expose the relevant classes.
    from .sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

    __all__ = [
        "AutoTabPFNClassifier",
        "AutoTabPFNRegressor",
    ]
else:
    # If it's not installed, issue a warning and expose only the flag.
    warnings.warn(
        "autogluon.tabular not installed. Post hoc ensembling will not be available. "
        'Install with: pip install "tabpfn-extensions[post_hoc_ensembles]"',
        ImportWarning,
        stacklevel=2,
    )
    __all__ = ["AUTOGLUON_TABULAR_AVAILABLE"]
