# INFO: The survival prediction work is work-in-progress and still unoptimized,
#  TabPFN survival may provide inconsistent improvements.
#
#  Concept and code developed by Robert Hatch based on his solution at the
#  Kaggle CIBMTR - Equity in post-HCT Survival Predictions Competition.
#
# Requires scikit-survival (install manually with `pip install scikit-survival`).

from .survival import SurvivalTabPFN

__all__ = ["SurvivalTabPFN"]
