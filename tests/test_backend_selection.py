"""Tests for TabPFN backend selection in tabpfn_extensions.utils.

These tests exercise get_tabpfn_models, the function that decides whether the
local (tabpfn) or the API/client (tabpfn-client) backend is used. Its decision
depends on exactly three module-level globals:

- USE_TABPFN_LOCAL, the boolean derived from the environment variable of the
  same name,
- LocalTabPFNClassifier, the local backend class, or None if tabpfn is not
  importable,
- ClientTabPFNClassifier, the client backend class, or None if tabpfn-client is
  not importable.

Because those are module globals that the function looks up at call time, the
tests simulate every install and configuration combination by monkeypatching
them on the utils module. No real models, no network, and no need for eight
different environments.
"""

from __future__ import annotations

import pytest

from tabpfn_extensions import utils


# Throwaway stand-ins for the four backend classes. The tests only ever check
# which class object came back (its identity), never behaviour, so empty classes
# suffice.
class _FakeLocalClassifier:
    pass


class _FakeLocalRegressor:
    pass


class _FakeClientClassifier:
    pass


class _FakeClientRegressor:
    pass


def _set_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_local: bool,
    local_present: bool,
    client_present: bool,
) -> None:
    """Patch the three globals get_tabpfn_models reads, simulating one scenario.

    A backend that is not installed is represented by its globals being None,
    exactly what the except ImportError fallbacks in utils bind them to.
    """
    monkeypatch.setattr(utils, "USE_TABPFN_LOCAL", use_local)
    monkeypatch.setattr(
        utils, "LocalTabPFNClassifier", _FakeLocalClassifier if local_present else None
    )
    monkeypatch.setattr(
        utils, "LocalTabPFNRegressor", _FakeLocalRegressor if local_present else None
    )
    monkeypatch.setattr(
        utils,
        "ClientTabPFNClassifier",
        _FakeClientClassifier if client_present else None,
    )
    monkeypatch.setattr(
        utils, "ClientTabPFNRegressor", _FakeClientRegressor if client_present else None
    )


# The full decision table. expected is one of "local", "client" or "error".
# Backend selection is symmetric: the flag picks the backend, and if the selected
# backend is missing the function raises ImportError naming it. It never silently
# falls back to the other backend.
_DECISION_TABLE = [
    # use_local, local_present, client_present, expected
    pytest.param(True, True, True, "local", id="row1-local-flag-both-present"),
    pytest.param(True, True, False, "local", id="row2-local-flag-local-only"),
    pytest.param(True, False, True, "error", id="row3-local-flag-local-missing"),
    pytest.param(True, False, False, "error", id="row4-local-flag-none-present"),
    pytest.param(False, True, True, "client", id="row5-client-flag-both-present"),
    pytest.param(False, True, False, "error", id="row6-client-flag-client-missing"),
    pytest.param(False, False, True, "client", id="row7-client-flag-client-only"),
    pytest.param(False, False, False, "error", id="row8-client-flag-none-present"),
]


@pytest.mark.parametrize(
    ("use_local", "local_present", "client_present", "expected"), _DECISION_TABLE
)
def test_get_tabpfn_models_selection(
    monkeypatch: pytest.MonkeyPatch,
    use_local: bool,
    local_present: bool,
    client_present: bool,
    expected: str,
) -> None:
    """get_tabpfn_models returns the selected backend, or raises if it is missing."""
    _set_backends(
        monkeypatch,
        use_local=use_local,
        local_present=local_present,
        client_present=client_present,
    )

    if expected == "error":
        with pytest.raises(ImportError):
            utils.get_tabpfn_models()
        return

    classifier, regressor = utils.get_tabpfn_models()
    if expected == "local":
        assert classifier is _FakeLocalClassifier
        assert regressor is _FakeLocalRegressor
    else:  # "client"
        assert classifier is _FakeClientClassifier
        assert regressor is _FakeClientRegressor
