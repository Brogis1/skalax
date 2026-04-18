# SPDX-License-Identifier: MIT
"""Core Skala functional: layers, base utilities, and model."""

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


def __getattr__(name):
    # Defer importing the model (and its e3nn dependency) until needed.
    if name in ("SkalaFunctional", "NonLocalModel"):
        from skalax.functional import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
