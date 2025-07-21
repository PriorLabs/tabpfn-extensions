try:
    from .sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor
except ImportError:
    raise ImportError(
        "Please install tabpfn-extensions with the 'post_hoc_ensembles' extra: pip install 'tabpfn-extensions[post_hoc_ensembles]'",
    )

__all__ = [
    "AutoTabPFNClassifier",
    "AutoTabPFNRegressor",
]
