from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabpfn-extensions")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Create alias for test_utils
# Import third party extensions
from tabpfn_common_utils.telemetry.interactive import opt_in

from . import test_utils
from .embedding import TabPFNEmbedding
from .hpo import TunedTabPFNClassifier, TunedTabPFNRegressor
from .many_class import ManyClassClassifier
from .post_hoc_ensembles import AutoTabPFNClassifier, AutoTabPFNRegressor
from .unsupervised import TabPFNUnsupervisedModel

# Import utilities and wrapped TabPFN classes
from .utils import TabPFNClassifier, TabPFNRegressor, is_tabpfn

__all__ = [
    "test_utils",
    "TabPFNClassifier",
    "TabPFNRegressor",
    "is_tabpfn",
    "TabPFNEmbedding",
    "ManyClassClassifier",
    "TabPFNUnsupervisedModel",
    "AutoTabPFNClassifier",
    "AutoTabPFNRegressor",
    "TunedTabPFNClassifier",
    "TunedTabPFNRegressor",
]

# Prompt the user to opt in for our newsletter
opt_in()
