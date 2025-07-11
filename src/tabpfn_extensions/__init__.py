from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabpfn-extensions")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Create alias for test_utils
from . import test_utils
from .autogluon import AutogluonTabPFNClassifier, AutogluonTabPFNRegressor
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
    "AutogluonTabPFNClassifier",
    "AutogluonTabPFNRegressor",
]
