from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("your_package")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from .utils import is_tabpfn
from .utils import TabPFNRegressor, TabPFNClassifier

from . import utils_todo
