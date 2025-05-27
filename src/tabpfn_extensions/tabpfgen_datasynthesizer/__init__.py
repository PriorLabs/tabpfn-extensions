"""TabPFGen Data Synthesizer Extension for TabPFN

This extension integrates TabPFGen for synthetic tabular data generation
with the TabPFN ecosystem, providing seamless workflows for data augmentation,
class balancing, and performance improvement.

Citation:
- TabPFGen package: https://github.com/sebhaan/TabPFGen
"""

from .tabpfgen_wrapper import TabPFNDataSynthesizer
from .utils import combine_datasets, validate_tabpfn_data

__version__ = "0.1.0"
__author__ = "Sebastian Haan"

__all__ = ["TabPFNDataSynthesizer", "validate_tabpfn_data", "combine_datasets"]
