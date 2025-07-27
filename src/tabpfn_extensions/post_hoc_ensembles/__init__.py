import importlib.util
import warnings

# Check if the optional dependency 'autogluon.tabular' is installed.
try:
    importlib.util.find_spec("autogluon.tabular")
    AUTOGLUON_TABULAR_AVAILABLE = importlib.util.find_spec("autogluon.tabular") is not None
except ModuleNotFoundError:
    AUTOGLUON_TABULAR_AVAILABLE = False

if AUTOGLUON_TABULAR_AVAILABLE:
    # If it's installed, import and expose the relevant classes.
    from .sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

    __all__ = [
        "AutoTabPFNClassifier",
        "AutoTabPFNRegressor",
    ]
else:
    warnings.warn(
        "autogluon.tabular not installed. Post hoc ensembling will not be available. "
        'Install with: pip install "tabpfn-extensions[post_hoc_ensembles]"',
        ImportWarning,
        stacklevel=2,
    )

    # Define a helpful error message to be reused.
    _error_msg = (
        "autogluon.tabular is not installed, which is required by this class. "
        'Please install with: pip install "tabpfn-extensions[post_hoc_ensembles]"'
    )

    # Create a stub class that raises an error only when instantiated.
    class AutoTabPFNClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError(_error_msg)

    # Do the same for the regressor for consistency.
    class AutoTabPFNRegressor:
        def __init__(self, *args, **kwargs):
            raise ImportError(_error_msg)

    # Now, export the names of the stubs and the availability flag.
    __all__ = [
        "AutoTabPFNClassifier",
        "AutoTabPFNRegressor",
        "AUTOGLUON_TABULAR_AVAILABLE",
    ]
