# INFO: The survival prediction work is work-in-progress and still unoptimized,
#  TabPFN survival may provide inconsistent improvements.
#
# Requires scikit-survival (install manually with `pip install scikit-survival`).

from .survival import SurvivalTabPFN

__all__ = ["SurvivalTabPFN"]
