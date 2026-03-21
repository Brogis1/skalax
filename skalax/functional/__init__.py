# SPDX-License-Identifier: MIT
"""Functional module containing the core Skala model implementation."""

from skalax.functional.layers import ScaledSigmoid, Squasher
from skalax.functional.base import (
    enhancement_density_inner_product,
    LDA_PREFACTOR,
)

__all__ = [
    "ScaledSigmoid",
    "Squasher",
    "enhancement_density_inner_product",
    "LDA_PREFACTOR",
]


# Lazy imports for model classes (not yet implemented)
def __getattr__(name):
    if name in ("SkalaFunctional", "NonLocalModel"):
        from skalax.functional import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
