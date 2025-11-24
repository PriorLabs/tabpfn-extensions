"""Checks if the minimum required version of TabPFN is available on PyPI.

This is used by the CI workflow to determine the correct source.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


def main() -> None:
    """Print if the minimum required version of TabPFN is available on PyPI."""
    min_version = _get_min_tabpfn_version()
    if _is_version_on_pypi(min_version):
        print(f"tabpfn=={min_version} is available on PyPI")
        print("USE_GITHUB=false")
    else:
        print(f"tabpfn=={min_version} is not on PyPI yet")
        print("USE_GITHUB=true")


def _get_min_tabpfn_version() -> str:
    pyproject = Path("pyproject.toml").read_text()
    match = re.search(r"tabpfn>=([0-9.]+)", pyproject)
    if not match:
        raise ValueError("Could not find tabpfn version requirement in pyproject.toml")
    return match.group(1)


def _is_version_on_pypi(version: str) -> bool:
    with urllib.request.urlopen(
        "https://pypi.org/pypi/tabpfn/json", timeout=10
    ) as response:
        data = json.loads(response.read())
        return version in data["releases"]


if __name__ == "__main__":
    main()
