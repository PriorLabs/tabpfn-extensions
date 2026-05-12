try:
    from . import feature_selection, pdp, shap, shapiq
    from .shap import shapiq_to_shap_explanation
except ImportError:
    raise ImportError(
        "Please install tabpfn-extensions with the 'interpretability' extra: pip install 'tabpfn-extensions[interpretability]'",
    )
__all__ = ["feature_selection", "shapiq", "pdp", "shap", "shapiq_to_shap_explanation"]
