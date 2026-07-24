"""Run the example files in ``examples/`` and check they work.

Each example is executed as a subprocess with a fixed per-example timeout.
There are two run modes:

* **Smoke** (default): an example that *errors* fails the test; one that simply
  doesn't finish in time *passes* -- we only assert it starts and runs without
  crashing. Examples run with ``TEST_MODE=1`` so those that support it shrink
  their workload. This is the fast, cheap per-PR guard against broken example
  scripts and import errors.
* **Strict** (``EXAMPLE_STRICT=1``): every example must run *to completion*,
  and a timeout is a failure. Examples run at full size (no ``TEST_MODE``) --
  exactly as a user would run them. This is the scheduled GPU full run. The
  switch is an environment variable rather than a pytest CLI option so that
  command-line arguments cannot flip the pass/fail semantics of a run.

An example is **skipped** (not failed) when:

* it needs an optional dependency that isn't installed (the GPL-excluded
  scikit-survival), or
* it is GPU-only and no CUDA device is available (some examples exceed TabPFN's
  CPU sample guard and only run on a GPU), or
* it demonstrates a deprecated module that is scheduled for removal.

Usage:
    uv run --no-sync pytest tests/test_examples.py --run-examples
    EXAMPLE_STRICT=1 uv run --no-sync pytest tests/test_examples.py --run-examples
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Strict mode: run every example at full size to completion; a timeout is a
# failure. Used by the scheduled GPU full run. In the default smoke mode a
# timeout still passes -- we only assert the example starts and runs without
# crashing.
STRICT_MODE = os.environ.get("EXAMPLE_STRICT", "0") == "1"

# Per-example timeout. The smoke budget is short since a timeout passes anyway;
# the strict budget is sized so it only bites on genuine hangs or regressions
# (the slowest full-size example measured ~504s on a Tesla T4, leaving ~45%
# headroom for runner variance).
EXAMPLE_TIMEOUT_SECONDS = 900 if STRICT_MODE else 120

# Directories whose examples need the full TabPFN package (won't work with the
# TabPFN client).
REQUIRES_TABPFN_DIRS = ["embedding/", "bayesian_optimization/"]

# Examples needing a module that is intentionally never installed, so they skip
# (rather than fail) when it is absent. Reserved for the GPL-excluded case:
# scikit-survival is in no extra or group, so survival_example always skips unless
# the user installs it manually. (Other example deps -- shapiq, shap, hyperopt, ... --
# are installed via --all-extras / the "examples" group and are expected to be present.)
REQUIRES_MODULE = {
    "survival_example.py": "sksurv",
}

# Examples that exceed TabPFN's CPU sample guard and only run on a GPU.
# Skipped when no CUDA device is available.
GPU_ONLY = {
    "get_embeddings.py",
}

# Examples for deprecated modules that are scheduled for removal (see the
# deprecation banners in the modules themselves). Not worth CI time to keep
# green -- delete each entry together with its module. The regular unit-test
# suite still covers these modules until they are removed.
DEPRECATED = {
    "phe_example.py": "AutoTabPFN* (post_hoc_ensembles) is deprecated",
    "tuned_tabpfn.py": "TunedTabPFN* (hpo) is deprecated",
}

# Examples known to be broken against current dependencies, with a tracking issue.
# xfailed so the suite stays green while the breakage is documented; when the
# upstream fix lands the example will start passing (reported as XPASS) and the
# entry should be removed.
KNOWN_BROKEN = {
    # TabEBM relies on tabpfn.config.ModelInterfaceConfig / PREPROCESS_TRANSFORMS,
    # removed/renamed in TabPFN's inference_config API -> import error.
    "tabebm_augment_real_world_data.py": (
        "TabEBM is broken with current TabPFN; see "
        "https://github.com/PriorLabs/tabpfn-extensions/issues/225"
    ),
}


def _example_params() -> list:
    """Build the parametrize values, xfailing known-broken examples."""
    params = []
    for example_file in get_example_files():
        marks = []
        reason = KNOWN_BROKEN.get(example_file["name"])
        if reason is not None:
            # A timeout counts as a pass here, so a partially fixed example could
            # XPASS without truly being fixed -- keep xfail non-strict so that
            # only surfaces as XPASS for follow-up rather than failing the suite.
            marks.append(pytest.mark.xfail(reason=reason, strict=False))
        params.append(
            pytest.param(example_file, marks=marks, id=example_file["name"]),
        )
    return params


def get_example_files() -> list[dict]:
    """Discover example files and attach the metadata the runner needs."""
    package_root = Path(__file__).parent.parent
    examples_dir = package_root / "examples"

    files = []
    for file_path in sorted(examples_dir.glob("**/*.py")):
        rel_path = str(file_path.relative_to(package_root))
        name = file_path.name
        files.append(
            {
                "path": file_path,
                "name": name,
                "requires_tabpfn": any(
                    pattern in rel_path for pattern in REQUIRES_TABPFN_DIRS
                ),
                "requires_module": REQUIRES_MODULE.get(name),
                "gpu_only": name in GPU_ONLY,
            },
        )
    return files


@pytest.mark.parametrize("example_file", _example_params())
def test_example(request, example_file):
    """Run a single example file as a subprocess and check the outcome.

    Args:
        request: PyTest request fixture.
        example_file: Dictionary with example file metadata.
    """
    from conftest import HAS_TABPFN, TABPFN_SOURCE

    name = example_file["name"]
    path = example_file["path"]

    if not request.config.getoption("--run-examples"):
        pytest.skip(f"Skipping {name} since --run-examples not set")

    # Deprecated module -> skip (kept visible as a skip until the module and
    # its example are removed)
    deprecation = DEPRECATED.get(name)
    if deprecation is not None:
        pytest.skip(f"Example {name}: {deprecation}")

    # Backend availability
    if example_file["requires_tabpfn"]:
        if not HAS_TABPFN:
            pytest.skip(
                f"Example {name} requires the TabPFN package, which is not installed",
            )
        if TABPFN_SOURCE == "tabpfn_client":
            pytest.skip(
                f"Example {name} requires the TabPFN package, not the client",
            )

    # Optional dependency not installed -> skip (not a failure)
    required_module = example_file["requires_module"]
    if required_module and importlib.util.find_spec(required_module) is None:
        pytest.skip(
            f"Example {name} requires '{required_module}', which is not installed"
        )

    # GPU-only example with no CUDA device -> skip
    if example_file["gpu_only"] and not torch.cuda.is_available():
        pytest.skip(
            f"Example {name} requires a CUDA device (exceeds TabPFN's CPU sample limit)",
        )

    # Examples are top-to-bottom scripts; run each in its own process so a hang
    # can be killed cleanly and state never leaks between examples. The example
    # inherits FAST_TEST_MODE/TABPFN_EXCLUDE_DEVICES from this process.
    env = dict(os.environ)
    if STRICT_MODE:
        # Full size, exactly as a user would run the example. Stripped rather
        # than inherited so a stray TEST_MODE in the environment can't quietly
        # shrink what the strict run verifies.
        env.pop("TEST_MODE", None)
    else:
        # Shrink the workload where the example supports it, so it can
        # complete within the smoke budget.
        env["TEST_MODE"] = "1"
    # Headless plotting: an example calling plt.show() must never block on a
    # GUI window (on a machine with a display, the interactive backend blocks
    # until the window is closed -- fatal in strict mode, where that reads as
    # a timeout).
    env.setdefault("MPLBACKEND", "Agg")

    try:
        proc = subprocess.run(  # noqa: S603 - trusted, repo-local example scripts
            [sys.executable, str(path)],
            cwd=str(path.parent),
            env=env,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if STRICT_MODE:
            pytest.fail(
                f"Example {name} did not complete within "
                f"{EXAMPLE_TIMEOUT_SECONDS}s (strict mode: a timeout is a failure)",
            )
        # Not a failure: the example started and ran without crashing, which is
        # all the smoke gate asserts. Completion is checked by the strict run.
        print(
            f"{name}: ran {EXAMPLE_TIMEOUT_SECONDS}s without error "
            f"(smoke mode; completion not verified)",
        )
        return

    if proc.returncode != 0:
        output = (proc.stdout or b"").decode("utf-8", "replace")
        tail = "\n".join(output.strip().splitlines()[-100:])
        pytest.fail(
            f"Example {name} exited with code {proc.returncode}:\n{tail}",
        )


if __name__ == "__main__":
    pytest.main([__file__])
