"""Tests for tabpfn_extensions.utils."""

from __future__ import annotations

from pytest_mock import MockerFixture

from tabpfn_extensions.utils import infer_device


def test__infer_device__tabpfn_not_installed__returns_fake_device_with_cpu_type(
    mocker: MockerFixture,
) -> None:
    """Test that we get a fake CPU device when tabpfn is not installed.

    Currently our test infrastructure runs the tests for the maximum and minimum
    supported versions of the tabpfn package. This means that the cases where tabpfn is
    installed will be covered by the other tests in this package. However, the case
    where tabpfn is not installed will not be covered, so this is a basic test for that.
    """
    mocker.patch("importlib.util.find_spec", return_value=None)
    assert infer_device(device="auto").type == "cpu"
    assert infer_device(device="cuda").type == "cpu"
    assert infer_device(device="cpu").type == "cpu"
