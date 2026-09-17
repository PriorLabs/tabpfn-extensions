"""Tests for the device helpers of the extension package."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch

from tabpfn_extensions import utils


def _hide_tabpfn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the local tabpfn package look absent, as with the client backend."""
    find_spec = importlib.util.find_spec

    def hidden(name: str, *args: Any) -> Any:
        return None if name == "tabpfn" else find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", hidden)


class TestInferTorchDevice:
    def test__with_tabpfn__is_tabpfns_own_reading(self) -> None:
        pytest.importorskip("tabpfn")
        from tabpfn.utils import infer_devices

        assert utils.infer_torch_device("cpu") == torch.device("cpu")
        assert utils.infer_torch_device("auto") == infer_devices("auto")[0]

    def test__without_tabpfn__reads_torch_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hide_tabpfn(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.delenv("TABPFN_EXCLUDE_DEVICES", raising=False)

        assert utils.infer_torch_device("auto") == torch.device("cuda")
        assert utils.infer_torch_device("cpu") == torch.device("cpu")
        assert utils.infer_torch_device(torch.device("cpu")) == torch.device("cpu")
        assert utils.infer_torch_device(["cpu", "cpu"]) == torch.device("cpu")

        monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "cuda, mps")
        assert utils.infer_torch_device("auto") == torch.device("cpu")


def test__infer_device__without_tabpfn__is_the_cpu_stand_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_tabpfn(monkeypatch)

    assert utils.infer_device("auto") == utils.FakeTorchDevice(type="cpu")
    with pytest.warns(UserWarning, match="does not support GPU"):
        assert utils.infer_device("cuda") == utils.FakeTorchDevice(type="cpu")
