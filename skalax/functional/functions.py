# SPDX-License-Identifier: MIT
"""Distances, radial basis functions, and cutoff envelopes used by Skala."""

import math

import jax
import jax.numpy as jnp
from jax import Array


# Covalent-radius bounds from Pyykko & Atsumi, Chem. Eur. J. 15, 2009, 188-197.
ANGSTROM_TO_BOHR = 1.88973
MIN_COV_RAD = 0.32 * ANGSTROM_TO_BOHR
MAX_COV_RAD = 2.32 * ANGSTROM_TO_BOHR


def vect_cdist(c1: Array, c2: Array) -> tuple[Array, Array]:
    """
    Compute pairwise direction vectors and distances between point sets.

    Parameters
    ----------
    c1 : Array
        First set of coordinates, shape (n, 3).
    c2 : Array
        Second set of coordinates, shape (m, 3).

    Returns
    -------
    tuple[Array, Array]
        - direction: Normalized direction vectors, shape (n, m, 3)
        - dist: Distances, shape (n, m)
    """
    direction = c1[:, None] - c2[None, :]
    # Add a tiny offset so the gradient of sqrt is finite at zero distance.
    dist = jnp.sqrt((direction**2 + 1e-20).sum(-1))
    return direction / dist[:, :, None], dist


def exp_radial_func(dist: Array, num_basis: int, dim: int = 3) -> Array:
    """
    Compute exponential radial basis functions.

    This ensures two standard deviations of the Gaussian kernel would reach
    the desired covalent radius value (95% of the Gaussian mass).

    Parameters
    ----------
    dist : Array
        Distances, shape (...).
    num_basis : int
        Number of basis functions.
    dim : int
        Dimensionality for normalization. Default: 3.

    Returns
    -------
    Array
        Radial basis values, shape (..., num_basis).
    """
    min_std = MIN_COV_RAD / 2
    max_std = MAX_COV_RAD / 2
    s = jnp.linspace(min_std, max_std, num_basis)

    temps = 2 * s**2
    x2 = dist[..., None] ** 2
    emb = (
        jnp.exp(-x2 / temps)
        * 2
        / dim
        * x2
        / temps
        / (math.pi * temps) ** (0.5 * dim)
    )

    return emb


def polynomial_envelope(r: Array, cutoff: float, p: int) -> Array:
    """
    Compute polynomial envelope function.

    This smoothly maps the domain r=[0, cutoff] to the range [1, 0]
    using a polynomial function. Every r >= cutoff is mapped to 0.

    From DimeNet (https://arxiv.org/abs/2003.03123).

    Parameters
    ----------
    r : Array
        Distances.
    cutoff : float
        Cutoff distance.
    p : int
        Polynomial order, must be >= 2.

    Returns
    -------
    Array
        Envelope values.
    """
    r = r / cutoff
    r = jnp.clip(r, 0, 1)
    x = r - 1
    x2 = x * x
    poly = p * (p + 1) * x2 - 2 * p * x + 2
    return jax.nn.relu(1 - 0.5 * jnp.power(r, p) * poly)


def normalization_envelope(r: Array, cutoff: float) -> Array:
    """
    Compute normalization envelope function.

    Parameters
    ----------
    r : Array
        Distances.
    cutoff : float
        Cutoff distance.

    Returns
    -------
    Array
        Normalization weights.
    """
    r = r / cutoff
    r = jnp.clip(r, 0, 1)
    return 1 - jnp.where(r < 0.5, 2 * r**2, -2 * r**2 + 4 * r - 1)


def prepare_features(mol: dict[str, Array]) -> tuple[Array, Array]:
    """
    Prepare features for the Skala model from molecular data.

    Parameters
    ----------
    mol : dict
        Dictionary containing:
        - density: shape (2, num_grid_points)
        - grad: shape (2, 3, num_grid_points)
        - kin: shape (2, num_grid_points)

    Returns
    -------
    tuple[Array, Array]
        - features_ab: Features with original spin ordering
        - features_ba: Features with swapped spin ordering
    """
    x = jnp.concatenate(
        [
            mol["density"].T,
            (mol["grad"] ** 2).sum(1).T,
            mol["kin"].T,
            (mol["grad"].sum(0) ** 2).sum(0).reshape(-1, 1),
        ],
        axis=1,
    )
    x = x.astype(jnp.float64)

    features = jnp.log(jnp.abs(x) + 1e-5)

    features_ab = features
    features_ba = features[:, jnp.array([1, 0, 3, 2, 5, 4, 6])]
    return features_ab, features_ba
