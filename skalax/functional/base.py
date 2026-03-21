# SPDX-License-Identifier: MIT
"""
Base functions for exchange-correlation functionals in JAX.

This module defines utility functions for implementing exchange-correlation
functionals in Skala JAX.
"""

import jax.numpy as jnp
from jax import Array


# The spin-agnostic version of the lda exchange is
#     E_x(rho) = \int d^3r e_x(rho(r))
#     e_x(rho) = - (3/4) * (3/pi)**(1/3) * rho ** (4/3)
# The spin-polarized version of the lda exchange is
#     E_x(rho_up, rho_down) = 0.5 * \int d^3r e_x(2 rho_up(r)) + e_x(2 rho_down(r))
#                           = 0.5 * 2 ** (4/3) * \int d^3r (e_x(rho_up(r)) + ...
#                           = [ - 2 ** (1/3) * (3/4) * (3/pi)**(1/3) ]
#                              * \int d^3 r rho_up**(4/3) + rho_down**(4/3)
# The prefactor in the squared bracket here is -0.9305257363491001

LDA_PREFACTOR = -0.9305257363491001


def enhancement_density_inner_product(
    enhancement_factor: Array,
    density: Array,
) -> Array:
    """
    Compute the enhancement density as inner product with LDA reference.

    Parameters
    ----------
    enhancement_factor : Array
        Enhancement factor with shape (n, 1).
    density : Array
        Electron density with shape (2, n) for 2 spins and n grid points.

    Returns
    -------
    Array
        Enhanced exchange-correlation density.

    Notes
    -----
    This function computes:
    enhancement_factor * LDA_exchange_density
    where LDA_exchange_density uses the prefactor -0.9305257363491001.
    """
    # Clip density to positive values and compute LDA exchange density
    density_clipped = jnp.clip(density.astype(jnp.float64), 0, None)
    lda = LDA_PREFACTOR * jnp.power(
        density_clipped, 4.0 / 3.0
    ).sum(axis=0).reshape(-1, 1)

    return (enhancement_factor.astype(lda.dtype) * lda).squeeze(1)
