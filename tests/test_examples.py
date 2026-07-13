"""Run the example files in ``examples/`` and check they work.

Each example is executed as a subprocess. The outcome is interpreted with one
of two policies, selected per CI layer:

* **Per-PR smoke (default):** each example gets a short timeout. An example that
  *errors* fails the test; one that simply doesn't finish in time *passes* — we
  only assert it starts and runs without crashing. This is a fast, cheap guard
  against broken example scripts and import errors.
* **Weekly full run (``--example-strict``):** a longer timeout, and a normal
  example that doesn't finish is a *failure* — i.e. we assert it actually runs
  to completion. Examples explicitly marked long-running (``LONG_RUNNING``) are
  still allowed to time out even here, since they cannot complete in CI by
  design; we still run them to catch import/startup errors.

Regardless of policy:
* An example that needs an optional dependency which isn't installed is
  **skipped** (not failed).
* A GPU-only example is **skipped** when no CUDA device is available (some
  examples exceed TabPFN's CPU sample guard and only run on a GPU).

Usage:
    # Per-PR smoke (short timeout; not finishing is OK):
    uv run --no-sync pytest tests/test_examples.py --run-examples \
        --example-timeout 120

    # Weekly full run (long timeout; a normal example not finishing fails):
    uv run --no-sync pytest tests/test_examples.py --run-examples \
        --example-strict --example-timeout 600 --example-long-timeout 120
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Enable test mode so examples shrink their workload where they support it.
os.environ["TEST_MODE"] = "1"

# Directories whose examples need the full TabPFN package (won't work with the
# TabPFN client).
REQUIRES_TABPFN_DIRS = ["embedding/"]

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

# Examples that cannot complete within any CI time budget by design (e.g. PHE's
# hard-coded ``max_time``). A timeout is NOT a failure for these, even in strict
# mode; we still run them to catch import/startup errors.
LONG_RUNNING = {
    "phe_example.py",
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
            # Non-strict: in smoke mode a timeout counts as a pass, so a partially
            # fixed example could XPASS without truly being fixed -- don't fail the
            # suite on that, just surface it as XPASS for follow-up.
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
                "long": name in LONG_RUNNING,
            },
        )
    return files


@pytest.mark.example
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

    strict = request.config.getoption("--example-strict")
    if example_file["long"]:
        timeout = request.config.getoption("--example-long-timeout")
    else:
        timeout = request.config.getoption("--example-timeout")

    # Examples are top-to-bottom scripts; run each in its own process so a hang
    # can be killed cleanly and state never leaks between examples. The example
    # inherits TEST_MODE/FAST_TEST_MODE/TABPFN_EXCLUDE_DEVICES from this process.
    env = dict(os.environ)
    env["TEST_MODE"] = "1"

    try:
        proc = subprocess.run(  # noqa: S603 - trusted, repo-local example scripts
            [sys.executable, str(path)],
            cwd=str(path.parent),
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if example_file["long"]:
            print(
                f"{name}: ran {timeout}s without error "
                f"(long-running; not run to completion by design)",
            )
            return
        if strict:
            pytest.fail(f"Example {name} did not finish within {timeout}s")
        print(
            f"{name}: ran {timeout}s without error "
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
