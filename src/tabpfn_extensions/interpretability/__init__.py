try:
    from . import feature_selection, shap, shapiq, pdp
except ImportError:
    raise ImportError(
        "Please install tabpfn-extensions with the 'interpretability' extra: pip install 'tabpfn-extensions[interpretability]'",
    )
__all__ = ["feature_selection", "shap", "shapiq", "pdp"]
