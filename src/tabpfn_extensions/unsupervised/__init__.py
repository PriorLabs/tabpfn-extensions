"""Unsupervised learning module for tabpfn_extensions package."""

#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from .simple_impute import impute_column, simple_impute
from .unsupervised import TabPFNUnsupervisedModel

__all__ = ["TabPFNUnsupervisedModel", "impute_column", "simple_impute"]
