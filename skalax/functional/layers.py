# SPDX-License-Identifier: MIT
"""
Neural network layers for Skala JAX functionals.

This module provides specialized JAX/Equinox layers used in the construction
of neural exchange-correlation functionals, including squashing functions
and scaled activations.
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class Squasher(eqx.Module):
    """
    Elementwise squashing function log(|x| + eta).

    This layer applies a logarithmic squashing to prevent extreme values
    and improve numerical stability in neural functionals.
    """

    eta: float = eqx.field(static=True)
    """Small constant added before taking logarithm for numerical stability."""

    def __init__(self, eta: float = 1e-5):
        """
        Initialize the squasher.

        Parameters
        ----------
        eta : float
            Small constant added before taking logarithm. Default: 1e-5.
        """
        self.eta = eta

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply squashing function log(|x| + eta)."""
        return jnp.log(jnp.abs(x) + self.eta)


class ScaledSigmoid(eqx.Module):
    """
    Sigmoid activation function with scaling.

    Computes: scale * sigmoid(x / scale)
    """

    scale: float = eqx.field(static=True)
    """Scaling factor for the sigmoid."""

    def __init__(self, scale: float = 1.0):
        """
        Initialize scaled sigmoid.

        Parameters
        ----------
        scale : float, optional
            Scaling factor for the sigmoid. Default: 1.0.
        """
        self.scale = scale

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply scaled sigmoid activation."""
        return self.scale * jax.nn.sigmoid(x / self.scale)


class LinearSkip(eqx.Module):
    """
    Linear layer with skip connection, used to initialize close to identity.

    This layer computes: output = input + W @ input + b
    where W is initialized to small values around zero.
    """

    linear: eqx.nn.Linear

    def __init__(self, features: int, *, key: jax.Array):
        """
        Initialize linear skip layer.

        Parameters
        ----------
        features : int
            Number of input/output features (must be equal).
        key : jax.Array
            PRNG key for initialization.
        """
        self.linear = eqx.nn.Linear(features, features, key=key)
        # Note: Weight initialization to small values would be done via
        # a custom initializer or post-initialization modification

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply linear transformation with skip connection."""
        return x + self.linear(x)
