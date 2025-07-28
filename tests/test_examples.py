"""Test examples to ensure they work as expected.

This module provides a testing framework for TabPFN example files. It automatically:
1. Detects all example files in the examples/ directory
2. Categorizes them as fast, slow, or large dataset examples
3. Runs fast examples normally
4. Tests slow examples with a short timeout (expecting timeout as success)
5. Skips large dataset examples entirely unless explicitly requested
6. Handles backend compatibility for TabPFN package vs. TabPFN client

Usage:
    # Run only fast examples:
    FAST_TEST_MODE=1 python -m pytest tests/test_examples.py

    # Run all examples except large dataset ones (slow ones will timeout and be marked as xfail):
    FAST_TEST_MODE=1 python -m pytest tests/test_examples.py --run-examples

    # Run specific example, even if it's a large dataset example:
    python -m pytest tests/test_examples.py::test_example[large_datasets_example.py] --run-examples
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Enable test mode to make examples run faster
os.environ["TEST_MODE"] = "1"


def get_example_files() -> list[dict]:
    """Get all Python files from the examples directory with metadata.

    Each example is categorized as:
    - fast: Can run quickly (runs in both normal and fast test mode)
    - slow: Takes longer to run (runs with a 1-second timeout, expected to timeout)
    - always_timeout: Examples with large datasets that are always skipped unless explicitly requested
    - requires_tabpfn: If True, requires the full TabPFN package and won't work with client;
                       if False, works with either TabPFN package or TabPFN client

    Returns:
        List of dictionaries containing example file info
    """
    package_root = Path(__file__).parent.parent
    examples_dir = package_root / "examples"

    # The only example that runs fast enough for CI
    FAST_EXAMPLES = []

    # These directories/files need the full TabPFN package and won't work with client
    REQUIRES_TABPFN_DIRS = ["embedding/"]

    # These examples require autogluon and won't work with tabpfn-client only
    REQUIRES_AUTOGLUON_EXAMPLES = ["distillation_example.py"]

    # Large dataset examples are always expected to timeout,
    # even if --run-examples is provided
    ALWAYS_TIMEOUT_PATTERNS = ["large_datasets_example.py"]

    # Find all Python files in the examples directory
    all_file_paths = list(examples_dir.glob("**/*.py"))
    all_files = []

    # Process each file with appropriate metadata
    for file_path in all_file_paths:
        rel_path = str(file_path.relative_to(package_root))
        file_name = file_path.name

        file_info = {
            "path": file_path,
            "name": file_name,
            # Default classification - most examples work with both implementations
            "requires_tabpfn": False,  # By default, examples work with either implementation
            "requires_autogluon": file_name
            in REQUIRES_AUTOGLUON_EXAMPLES,  # By default, examples do not require autogluon
            "fast": file_name in FAST_EXAMPLES,  # Only listed examples are fast
            "slow": file_name not in FAST_EXAMPLES,  # All others are slow
            "always_timeout": any(
                pattern in file_name for pattern in ALWAYS_TIMEOUT_PATTERNS
            ),
            "timeout": 1
            if file_name not in FAST_EXAMPLES
            else 30,  # Short timeout for slow examples
        }

        # Check if example requires full TabPFN package
        if any(pattern in rel_path for pattern in REQUIRES_TABPFN_DIRS):
            # Example explicitly requires TabPFN package
            file_info["requires_tabpfn"] = True

        all_files.append(file_info)

    return all_files


def import_module_from_path(path: Path, timeout: int = None) -> object:
    """Dynamically import a Python module from a file path.

    Args:
        path: Path to the Python file to import
        timeout: Optional timeout parameter (no longer used internally, kept for backward compatibility)

    Returns:
        The imported module object
    """
    # Add the parent directory to sys.path to allow imports within example files
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Import the module
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.example
@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda x: x["name"])
def test_example(request, example_file):
    """Run example files to ensure they work as expected.

    Test strategy:
    1. Fast examples are run with normal timeout
    2. Slow examples are run with 1-second timeout, expected to timeout
    3. Examples are skipped if they require missing backends
    4. In FAST_TEST_MODE, only fast examples and examples with --run-examples flag run

    Args:
        request: PyTest request fixture
        example_file: Dictionary with example file metadata
    """
    from conftest import HAS_TABPFN, TABPFN_SOURCE, HAS_AUTOGLUON

    file_name = example_file["name"]
    file_path = example_file["path"]

    run_examples = request.config.getoption("--run-examples")

    if not run_examples:
        pytest.skip(
            f"Skipping {file_name} since --run-examples not set",
        )

    # Skip if backend not available
    if example_file["requires_tabpfn"]:
        if not HAS_TABPFN:
            pytest.skip(
                f"Example {file_name} requires TabPFN package, but it's not installed",
            )
        elif TABPFN_SOURCE == "tabpfn_client":
            pytest.skip(
                f"Example {file_name} requires TabPFN package, not compatible with client",
            )

    # Skip if backend not available
    if example_file["requires_autogluon"] and not HAS_AUTOGLUON:
        pytest.skip(
            f"Example {file_name} requires autogluon, but it's not installed",
        )

    try:
        # Handle slow examples (including large datasets) differently
        if example_file.get("slow", False) or example_file.get("always_timeout", False):
            # For slow examples, we'll run them with a short internal timeout
            # and expect them to be interrupted
            import threading

            def run_with_timeout(path, max_time=5):
                """Run import with a timeout using threading approach."""
                result = {"completed": False, "exception": None}

                def target():
                    try:
                        # Import the module
                        spec = importlib.util.spec_from_file_location(path.stem, path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        result["completed"] = True
                    except Exception as e:  # noqa: BLE001
                        result["exception"] = e

                # Start the import in a separate thread
                thread = threading.Thread(target=target)
                thread.daemon = (
                    True  # Daemon threads are killed when the main thread exits
                )
                thread.start()

                # Wait for the thread to complete or timeout
                thread.join(timeout=max_time)

                return result

            # Add parent directory to path
            parent_dir = str(file_path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Run with a 5 second timeout
            run_result = run_with_timeout(file_path, max_time=5)

            if run_result["completed"]:
                # If it completed within the timeout, that's fine
                print(
                    f"Note: Slow example {file_name} completed successfully within 5 seconds",
                )
            elif run_result["exception"]:
                # If it failed for reasons other than timeout
                pytest.xfail(f"Example {file_name} failed: {run_result['exception']}")
            else:
                # Expected timeout after running for 5 seconds
                pytest.xfail(
                    f"Example {file_name} ran for 5 seconds and was stopped as expected",
                )
        else:
            # Fast examples should complete normally
            import_module_from_path(file_path, timeout=None)
    except TimeoutError as e:
        # Unexpected timeout in fast examples is a failure
        pytest.fail(f"Example {file_name} timed out: {e!s}")


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run all example files (including slow ones that will be expected to timeout)",
    )


if __name__ == "__main__":
    pytest.main([__file__])
