# SPDX-License-Identifier: MIT
"""
Skala JAX - JAX implementation of the Skala neural XC functional.

This package provides a JAX/Equinox implementation of the Skala model,
numerically equivalent to the PyTorch version for identical weights.

Example
-------
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
>>>
>>> from skalax import SkalaFunctional, load_weights_from_npz, load_config, get_default_weights_dir
>>>
>>> # Load pretrained weights
>>> weights_dir = get_default_weights_dir()
>>> config = load_config(weights_dir)
>>>
>>> key = jax.random.PRNGKey(0)
>>> model = SkalaFunctional(
...     lmax=config["lmax"],
...     non_local=config["non_local"],
...     non_local_hidden_nf=config["non_local_hidden_nf"],
...     radius_cutoff=config["radius_cutoff"],
...     key=key,
... )
>>> model = load_weights_from_npz(model, weights_dir)
"""

__version__ = "1.0.0"

from skalax.functional.model import SkalaFunctional
from skalax.convert_weights import (
    load_weights_from_npz,
    load_config,
    get_default_weights_dir,
)

__all__ = [
    "SkalaFunctional",
    "load_weights_from_npz",
    "load_config",
    "get_default_weights_dir",
]
