"""By importing this file you setup you configuration for the cluster.

You set this with the environment variable `TABPFN_CLUSTER_SETUP`. If there
is no such variable, it will use the local setup.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Union

global base_path
global log_folder
global neptune_project_prefix
global openml_path
global kaggle_cache_path
global model_string_config
global local_model_path
global container_path
global default_partition


def set_cluster_settings(setup: str | None = None):
    global base_path
    global log_folder
    global neptune_project_prefix
    global openml_path
    global kaggle_cache_path
    global model_string_config
    global local_model_path
    global container_path
    global default_partition

    if setup is None:
        setup = os.environ.get("TABPFN_CLUSTER_SETUP")

    neptune_project_prefix = "PriorLabs/tabpfn-pretraining"

    root_dir = Path(__file__).parent.resolve()
    local_model_path = root_dir / "model_cache"
    default_partition = "unknown"

    if setup == "FREIBURG" or setup == "FREIBURG_AADLOGIN":
        print("Using Freiburg Cluster Setup")
        # For baselines, PROJECT_SETUP_BASE_PATH="/work/dlclarge1/hollmann-PFN_Tabular/baselines"
        base_path = os.path.join(
            os.environ.get(  # either environment OS variable or default value
                "PROJECT_SETUP_BASE_PATH",
                default="/work/dlclarge2/hollmann-TabPFN/base_feb24",
            )
        )
        log_folder = os.path.join("/work/dlclarge2/hollmann-LogSpace")
        openml_path = "/work/dlclarge2/hollmann-TabPFN/openml"
        kaggle_cache_path = "/work/dlclarge1/hollmann-PFN_Tabular/kaggle_cache"
        model_string_config = "CLUSTER"
        container_path = (
            "/work/dlclarge1/hollmann-PFN_Tabular/singularity_container_cache"
        )
    elif setup == "JUWELS":
        base_path = "/p/scratch/ccstdl/tabpfn-basepath"
        log_folder = "/p/scratch/ccstdl/tabpfn-logs"
        openml_path = "/p/scratch/ccstdl/tabpfn-openml"
        kaggle_cache_path = "/p/scratch/ccstdl/tabpfn-kaggle"
        model_string_config = "CLUSTER"
        container_path = "/p/scratch/ccstdl/tabpfn-singularity"
    elif setup == "HOREKA":
        base_path = "/home/hk-project-p0021863/fr_sm1188/tabpfn-data/basepath"
        log_folder = "/home/hk-project-p0021863/fr_sm1188/tabpfn-data/logs"
        openml_path = "/home/hk-project-p0021863/fr_sm1188/tabpfn-data/openml"
        kaggle_cache_path = "/home/hk-project-p0021863/fr_sm1188/tabpfn-data/kaggle"
        model_string_config = "CLUSTER"
        container_path = "/home/hk-project-p0021863/fr_sm1188/tabpfn-data/singularity"
    elif setup == "HELIX":
        print("Using HELIX Cluster Setup")
        base_path = os.path.join("/home/fr/fr_fr/fr_sm1188/tabpfn_results")
        log_folder = os.path.join("/home/fr/fr_fr/fr_sm1188/logs")
        openml_path = "/home/fr/fr_fr/fr_sm1188/openml"
        kaggle_cache_path = "/home/fr/fr_fr/fr_sm1188/kaggle_cache"
        model_string_config = "CLUSTER"
        container_path = None
    elif setup == "CHARITE":
        print("Using Charite Cluster Setup")
        base_path = os.path.join("regression")
        log_folder = os.path.join(base_path, "log_test/")
        openml_path = "~/.cache/openml"
        kaggle_cache_path = None  # TODO: set this to your local path
        model_string_config = "CLUSTER"
        container_path = None
    elif setup == "FREIBURG_AADLOGIN":
        openml_path = os.path.join("~/.cache/openml")
        base_path = os.path.join("~/tabpfn_results")
        kaggle_cache_path = None  # TODO: set this to your local path
        model_string_config = "CLUSTER"
        container_path = None
    elif setup == "UNITTEST":
        base_path = os.path.join("tests/local-results")
        os.makedirs(base_path, exist_ok=True)
        log_folder = os.path.join(base_path, "log_test/")
        os.makedirs(log_folder, exist_ok=True)
        neptune_project_prefix = "PriorLabs/PyTest"
        openml_path = f"/tmp/openml_cache_{str(random.randint(0, 100000))}"
        kaggle_cache_path = None  # TODO: set this to your local path
        model_string_config = "LOCAL"
        container_path = None
    elif setup == "LOCAL" or setup is None:
        # Local setup for usage of the repo on, e.g., a laptop.
        model_string_config = "LOCAL"
        root_dir = Path(__file__).parent.parent.resolve()
        print("root_dir", root_dir)
        openml_path = root_dir / "openml_cache"
        base_path = root_dir / "results"
        kaggle_cache_path = root_dir / "kaggle_cache"
        container_path = None
        log_folder = root_dir / "logs"
        default_partition = "gpua100"
    elif setup == "NEMO":
        root_dir = Path("/work/ws/nemo/fr_lp1082-tabpfn-0").resolve()
        base_path = root_dir / "output"
        log_folder = root_dir / "logs"
        openml_path = root_dir / "openml"
        kaggle_cache_path = root_dir / "kaggle_cache"
        model_string_config = "CLUSTER_NEMO"
        container_path = root_dir / "singularity_container_cache"
    else:
        raise ValueError(f"Unknown Cluster Setup {setup=}")

    DEFAULT_OPENML_PATH = (
        Path("/work") / "dlclarge1" / "hollmann-PFN_Tabular" / "openml"
    )

    print("base_path", base_path)

    def set_openml_config_path(path: Union[str, Path] = DEFAULT_OPENML_PATH) -> None:
        """Sets the path where openml stores its cache.

        :param path: The path to the openml cache
        """
        path = Path(path).absolute()
        if not path.exists():
            path.mkdir(parents=True)
        # Future proofing for openml>=0.14.0
        # See https://github.com/openml/automlbenchmark/pull/579/files
        try:
            openml.config.set_cache_directory(str(path))  # type: ignore
        except AttributeError:
            openml.config.set_root_cache_directory(str(path))  # type: ignore

    os.makedirs(f"{base_path}/results", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/multiclass", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/regression", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/survival", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/quantile_regression", exist_ok=True)
    os.makedirs(f"{base_path}/results/models_diff", exist_ok=True)
    os.makedirs(f"{base_path}/neptune", exist_ok=True)
    os.makedirs(f"{openml_path}/cached_lists", exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(openml_path, exist_ok=True)
    if kaggle_cache_path is not None:
        os.makedirs(kaggle_cache_path, exist_ok=True)

    try:
        import openml

        set_openml_config_path(openml_path)
    except ImportError:
        pass


set_cluster_settings()


@contextmanager
def overwrite_cluster_settings(setup: str = "UNITTEST"):
    """
    Context manager to temporarily overwrite the cluster settings.
    This can also be used as decorator.
    :param setup: Setup string from above to use.
    """
    original_setup = os.environ.get("TABPFN_CLUSTER_SETUP")
    os.environ["TABPFN_CLUSTER_SETUP"] = setup
    set_cluster_settings(setup)
    try:
        yield
    finally:
        if original_setup is None:
            os.environ.pop("TABPFN_CLUSTER_SETUP", None)
        else:
            os.environ["TABPFN_CLUSTER_SETUP"] = original_setup
        # Restore the original settings
        set_cluster_settings(original_setup)


def get_neptune_api_token() -> str:
    return os.environ["NEPTUNE_API_TOKEN"]


def get_neptune_project(task_type="multiclass"):
    if task_type == "multiclass":
        return neptune_project_prefix
    else:
        return neptune_project_prefix + f"-{task_type}"
