# SPDX-License-Identifier: MIT
"""
Scatter operations for JAX arrays.

This module provides scatter operations similar to pytorch_scatter,
specifically scatter_sum for aggregating values at specified indices.
"""

import jax
import jax.numpy as jnp
from jax import Array


def scatter_sum(
    src: Array,
    index: Array,
    dim: int = -1,
    dim_size: int | None = None,
) -> Array:
    """
    Sum all values from the src array at indices specified in the index array.

    Parameters
    ----------
    src : Array
        Source array containing values to scatter.
    index : Array
        Index array specifying where to scatter values.
    dim : int, optional
        Dimension along which to scatter. Default: -1.
    dim_size : int or None, optional
        Size of the output array along the scatter dimension.

    Returns
    -------
    Array
        Array with scattered and summed values.
    """
    # Handle negative dimension
    if dim < 0:
        dim = src.ndim + dim

    # Broadcast index to match src shape
    index = broadcast(index, src, dim)

    # Determine output size
    if dim_size is None:
        if index.size == 0:
            dim_size = 0
        else:
            dim_size = int(index.max()) + 1

    # Create output shape
    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    # Create zeros output
    out = jnp.zeros(out_shape, dtype=src.dtype)

    # Use segment_sum for simple 1D case, otherwise use at[].add
    if src.ndim == 1:
        return jax.ops.segment_sum(src, index, num_segments=dim_size)
    else:
        # Build slice for scatter operation
        return out.at[_build_scatter_indices(index, dim, src.shape)].add(src)


def _build_scatter_indices(index: Array, dim: int, shape: tuple) -> tuple:
    """Build indices for scatter operation."""
    ndim = len(shape)
    indices = []
    for d in range(ndim):
        if d == dim:
            indices.append(index)
        else:
            # Create range for this dimension and broadcast
            idx = jnp.arange(shape[d])
            # Reshape to broadcast correctly
            reshape_shape = [1] * ndim
            reshape_shape[d] = shape[d]
            idx = idx.reshape(reshape_shape)
            # Broadcast to full shape
            idx = jnp.broadcast_to(idx, shape)
            indices.append(idx)
    return tuple(indices)


def broadcast(src: Array, other: Array, dim: int) -> Array:
    """
    Broadcast src array to match the shape of other array.

    Parameters
    ----------
    src : Array
        Source array to broadcast.
    other : Array
        Target array whose shape to match.
    dim : int
        Dimension along which to perform broadcasting.

    Returns
    -------
    Array
        Broadcasted array with shape matching other.
    """
    if dim < 0:
        dim = other.ndim + dim

    if src.ndim == 1:
        # Add dimensions before dim
        for _ in range(dim):
            src = jnp.expand_dims(src, 0)

    # Add dimensions after to match other's ndim
    while src.ndim < other.ndim:
        src = jnp.expand_dims(src, -1)

    # Broadcast to match other's shape
    src = jnp.broadcast_to(src, other.shape)
    return src
