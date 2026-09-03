#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Tests for `warn_if_no_kv_cache` across the backends.

The endpoint-backed estimators (self-hosted container, SageMaker, Foundry)
reuse a fitted model when built with `use_kv_cache=True`; the managed API
backend has no endpoint-side cache. Stubs stand in for the client estimators,
which are not a test dependency here.
"""

from __future__ import annotations

import warnings

import pytest

from tabpfn_extensions.utils import warn_if_no_kv_cache


def _warnings_from(model) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_no_kv_cache(model, context="SHAP")
    return [str(w.message) for w in caught]


class _Endpoint:
    """Shaped like the endpoint-backed tabpfn-client estimators."""

    __module__ = "tabpfn_client.hosted.estimator"

    def __init__(self, *, use_kv_cache: bool):
        self.use_kv_cache = use_kv_cache


class _Managed:
    """Shaped like the managed tabpfn-client estimator: no endpoint cache."""

    __module__ = "tabpfn_client.estimator"


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_endpoint_backend_with_the_cache_is_silent():
    assert _warnings_from(_Endpoint(use_kv_cache=True)) == []


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_endpoint_backend_without_the_cache_points_at_the_flag():
    (message,) = _warnings_from(_Endpoint(use_kv_cache=False))
    assert "use_kv_cache=True" in message
    # The old advice was to abandon the endpoint for a local install.
    assert "pip install tabpfn" not in message


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_managed_backend_still_recommends_the_local_package():
    (message,) = _warnings_from(_Managed())
    assert "pip install tabpfn" in message


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_local_model_paths_are_unchanged():
    class _Local:
        fit_mode = "fit_preprocessors"

    (message,) = _warnings_from(_Local())
    assert "fit_with_cache" in message

    class _Cached:
        fit_mode = "fit_with_cache"
        executor_ = type("E", (), {"keep_cache_on_device": True})()

    assert _warnings_from(_Cached()) == []


@pytest.mark.local_compatible
@pytest.mark.client_compatible
def test_unknown_estimator_is_left_alone():
    assert _warnings_from(object()) == []
