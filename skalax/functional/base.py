# SPDX-License-Identifier: MIT
"""Base utilities for the Skala exchange-correlation functional."""

import jax.numpy as jnp
from jax import Array


# Spin-polarized LDA exchange prefactor.
#
# Spin-agnostic LDA exchange:
#     e_x(rho) = -(3/4) (3/pi)^(1/3) rho^(4/3)
# Spin-polarized form:
#     E_x(rho_a, rho_b) = 0.5 * int [e_x(2 rho_a) + e_x(2 rho_b)] d^3 r
#                       = -2^(1/3) (3/4) (3/pi)^(1/3)
#                         * int [rho_a^(4/3) + rho_b^(4/3)] d^3 r
# The bracketed prefactor evaluates to -0.9305257363491001.
LDA_PREFACTOR = -0.9305257363491001


def enhancement_density_inner_product(
    enhancement_factor: Array,
    density: Array,
) -> Array:
    """Combine a neural enhancement factor with the LDA exchange density.

    Parameters
    ----------
    enhancement_factor : Array
        Shape ``(n, 1)``.
    density : Array
        Spin density, shape ``(2, n)``.

    Returns
    -------
    Array
        XC energy density, shape ``(n,)``.
    """
    density_clipped = jnp.clip(density.astype(jnp.float64), 0, None)
    lda = LDA_PREFACTOR * jnp.power(
        density_clipped, 4.0 / 3.0
    ).sum(axis=0).reshape(-1, 1)

    return (enhancement_factor.astype(lda.dtype) * lda).squeeze(1)
