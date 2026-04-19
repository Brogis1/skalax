# SPDX-License-Identifier: MIT
"""skalax: JAX/Equinox port of the Skala neural XC functional.

See the README for usage examples.
"""

__version__ = "1.0.1"

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
