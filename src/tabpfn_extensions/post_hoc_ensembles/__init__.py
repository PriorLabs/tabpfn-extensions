# Check if autogluon is available
try:
    from autogluon.tabular import TabularPredictor
    from autogluon.tabular.models import TabPFNV2Model
    AUTOGLUON_TABULAR_AVAILABLE = True
except ImportError:
    AUTOGLUON_TABULAR_AVAILABLE = False
    import warnings

    warnings.warn(
        "Latest version of autogluon.tabular is not installed. Post hoc  "
        "ensembling extensions will not be available. Make sure to install "
        "the latest version of autogluon.tabular. "
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
