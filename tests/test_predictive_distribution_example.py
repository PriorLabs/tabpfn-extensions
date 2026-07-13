"""Tests for examples/predictive_distribution/predictive_distribution_example.py.

Runs the example with a tiny synthetic dataset so the test stays fast.
The CLI entry-point (``--n-train``, ``--n-test``, ``--n-estimators``,
``--no-show``, ``--out-dir``) is exercised directly by calling ``main()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the example directory to sys.path so its local import
# ``from bar_distribution_plot import plot_bar_distribution`` works.
EXAMPLE_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "predictive_distribution"
)
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _has_local_tabpfn() -> bool:
    """Return True if the full TabPFN package is installed."""
    import importlib.util

    return importlib.util.find_spec("tabpfn") is not None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.local_compatible
@pytest.mark.skipif(
    not _has_local_tabpfn(),
    reason="Requires the full 'tabpfn' package (output_type='full' not available with client).",
)
def test_predictive_distribution_example_runs(tmp_path):
    """main() completes without error on a tiny dataset and saves three PNGs."""
    from predictive_distribution_example import main

    main(
        n_train=60,
        n_test=30,
        n_estimators=1,
        no_show=True,
        out_dir=tmp_path,
    )

    expected_files = [
        "tabpfn_bar_distribution.png",
        "tabpfn_bar_distribution_variants.png",
        "tabpfn_density_slices.png",
    ]
    for fname in expected_files:
        assert (tmp_path / fname).exists(), f"Expected output file not found: {fname}"
