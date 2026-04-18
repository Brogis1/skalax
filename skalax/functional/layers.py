# SPDX-License-Identifier: MIT
"""Neural-network layers used by the Skala functional."""

import jax
import jax.numpy as jnp
import equinox as eqx


class Squasher(eqx.Module):
    """Elementwise ``log(|x| + eta)`` squashing for numerical stability."""

    eta: float = eqx.field(static=True)

    def __init__(self, eta: float = 1e-5):
        self.eta = eta

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.log(jnp.abs(x) + self.eta)


class ScaledSigmoid(eqx.Module):
    """Scaled sigmoid: ``scale * sigmoid(x / scale)``."""

    scale: float = eqx.field(static=True)

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.scale * jax.nn.sigmoid(x / self.scale)
