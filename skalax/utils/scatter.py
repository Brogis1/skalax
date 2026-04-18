# SPDX-License-Identifier: MIT
"""``scatter_sum`` for JAX, matching the ``torch_scatter`` signature."""

import jax
import jax.numpy as jnp
from jax import Array


def scatter_sum(
    src: Array,
    index: Array,
    dim: int = -1,
    dim_size: int | None = None,
) -> Array:
    """Sum values from ``src`` at positions given by ``index``.

    Parameters
    ----------
    src : Array
        Values to scatter.
    index : Array
        Target positions along ``dim`` (broadcastable to ``src``).
    dim : int
        Axis to scatter along. Default: ``-1``.
    dim_size : int, optional
        Output size along ``dim``. If ``None``, inferred as
        ``index.max() + 1`` (requires a non-empty ``index``).

    Returns
    -------
    Array
        ``src`` summed into buckets of size ``dim_size``.
    """
    if dim < 0:
        dim = src.ndim + dim

    index = _broadcast(index, src, dim)

    if dim_size is None:
        dim_size = 0 if index.size == 0 else int(index.max()) + 1

    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    if src.ndim == 1:
        return jax.ops.segment_sum(src, index, num_segments=dim_size)

    out = jnp.zeros(out_shape, dtype=src.dtype)
    return out.at[_build_scatter_indices(index, dim, src.shape)].add(src)


def _build_scatter_indices(
    index: Array, dim: int, shape: tuple,
) -> tuple:
    ndim = len(shape)
    indices = []
    for d in range(ndim):
        if d == dim:
            indices.append(index)
        else:
            reshape_shape = [1] * ndim
            reshape_shape[d] = shape[d]
            idx = jnp.arange(shape[d]).reshape(reshape_shape)
            indices.append(jnp.broadcast_to(idx, shape))
    return tuple(indices)


def _broadcast(src: Array, other: Array, dim: int) -> Array:
    if dim < 0:
        dim = other.ndim + dim

    if src.ndim == 1:
        for _ in range(dim):
            src = jnp.expand_dims(src, 0)

    while src.ndim < other.ndim:
        src = jnp.expand_dims(src, -1)

    return jnp.broadcast_to(src, other.shape)
