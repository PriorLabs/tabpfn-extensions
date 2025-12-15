"""Utilities for multi-output learning with TabPFN."""

from __future__ import annotations

from typing import Any, TypeVar

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from .utils import TabPFNClassifier, TabPFNRegressor

_EstimatorT = TypeVar("_EstimatorT")


class _TabPFNMultiOutputMixin:
    """Shared initialisation logic for TabPFN multi-output wrappers."""

    _tabpfn_estimator_cls: type[_EstimatorT]

    def __init__(
        self,
        estimator: _EstimatorT | None = None,
        *,
        n_jobs: int | None = None,
        **tabpfn_params: Any,
    ) -> None:
        if estimator is not None and tabpfn_params:
            msg = "Provide either a custom estimator or tabpfn_params, not both."
            raise ValueError(msg)

        self._estimator_is_default = estimator is None
        self.tabpfn_params = dict(tabpfn_params) if self._estimator_is_default else {}

        if self._estimator_is_default:
            estimator = self._tabpfn_estimator_cls(**tabpfn_params)

        super().__init__(estimator=estimator, n_jobs=n_jobs)

    def get_params(
        self, deep: bool = True
    ) -> dict[str, Any]:  # pragma: no cover - delegating to sklearn
        """Return parameters for this estimator with TabPFN kwargs included."""
        params = super().get_params(deep=deep)
        if getattr(self, "_estimator_is_default", False):
            params.pop("estimator", None)
            params.update(self.tabpfn_params)
        return params

    def set_params(
        self, **params: Any
    ) -> _TabPFNMultiOutputMixin:  # pragma: no cover - delegating to sklearn
        """Update parameters while keeping TabPFN kwargs in sync."""
        if getattr(self, "_estimator_is_default", False):
            tabpfn_updates: dict[str, Any] = {}
            for key in list(params):
                if key in {"estimator", "n_jobs"}:
                    continue
                tabpfn_updates[key] = params.pop(key)

            if tabpfn_updates:
                self.tabpfn_params.update(tabpfn_updates)
                self.estimator = self._tabpfn_estimator_cls(**self.tabpfn_params)

        return super().set_params(**params)


class TabPFNMultiOutputRegressor(_TabPFNMultiOutputMixin, MultiOutputRegressor):
    """A lightweight multi-output wrapper around :class:`TabPFNRegressor`."""

    _tabpfn_estimator_cls = TabPFNRegressor


class TabPFNMultiOutputClassifier(_TabPFNMultiOutputMixin, MultiOutputClassifier):
    """A lightweight multi-output wrapper around :class:`TabPFNClassifier`."""

    _tabpfn_estimator_cls = TabPFNClassifier
