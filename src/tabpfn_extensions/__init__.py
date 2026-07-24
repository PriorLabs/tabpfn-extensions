from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabpfn-extensions")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Import third party extensions
from tabpfn_common_utils.telemetry.interactive import opt_in

from .embedding import TabPFNEmbedding
from .many_class import ManyClassClassifier
from .unsupervised import TabPFNUnsupervisedModel

# Import utilities and wrapped TabPFN classes
from .utils import TabPFNClassifier, TabPFNRegressor, is_tabpfn

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "is_tabpfn",
    "TabPFNEmbedding",
    "ManyClassClassifier",
    "TabPFNUnsupervisedModel",
]

# Prompt the user to opt in for our newsletter
opt_in()
