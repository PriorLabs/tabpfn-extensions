"""TabPFGen Data Synthesizer Extension for TabPFN

This extension integrates TabPFGen for synthetic tabular data generation
with the TabPFN ecosystem, providing seamless workflows for data augmentation,
class balancing, and performance improvement.

Requirements:
    - Python 3.10+
    - TabPFGen >= 0.1.4

Note: While tabpfn-extensions supports Python 3.9+, this extension specifically
requires Python 3.10+ due to the underlying TabPFGen package requirements.

Citation:
- TabPFGen package: https://github.com/sebhaan/TabPFGen
"""

# Check Python version before any other imports
import sys

if sys.version_info < (3, 10):
    raise ImportError(
        "TabPFGen Data Synthesizer requires Python 3.10+ "
        f"(current: {sys.version_info.major}.{sys.version_info.minor})."
        "Please upgrade Python or use other TabPFN extensions."
    )

from .tabpfgen_wrapper import TabPFNDataSynthesizer
from .utils import combine_datasets, validate_tabpfn_data

__version__ = "0.1.0"
__author__ = "Sebastian Haan"

__all__ = ["TabPFNDataSynthesizer", "validate_tabpfn_data", "combine_datasets"]
