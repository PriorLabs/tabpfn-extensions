try:
    from . import feature_selection, pdp, shapiq
    from .feature_selection import FeatureSelectionResult
except ImportError:
    raise ImportError(
        "Please install tabpfn-extensions with the 'interpretability' extra: pip install 'tabpfn-extensions[interpretability]'",
    )
__all__ = ["feature_selection", "shapiq", "pdp", "FeatureSelectionResult"]
