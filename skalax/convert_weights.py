# SPDX-License-Identifier: MIT
"""Load Skala weights (and Wigner-3j buffers) into a JAX model.

Two sources are supported:

* a PyTorch ``nn.Module`` with the reference Skala state-dict layout, and
* a pair of ``.npz`` files (weights + buffers) plus a ``config.json``,
  which is what the bundled pretrained checkpoint uses.

The ``.npz`` path has no PyTorch dependency and is what the package uses
by default via :func:`load_weights_from_npz`.
"""

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import equinox as eqx


# State-dict key layout of the reference Skala PyTorch model. The indices
# ``0, 2, 4, 6`` come from the PyTorch ``Sequential`` where even slots are
# Linear layers and odd slots are activations.
_INPUT_LINEAR_KEYS = [
    ("input_linear1", "input_model.0"),
    ("input_linear2", "input_model.2"),
]
_OUTPUT_LINEAR_KEYS = [
    ("output_linear1", "output_model.0"),
    ("output_linear2", "output_model.2"),
    ("output_linear3", "output_model.4"),
    ("output_linear4", "output_model.6"),
]
_NONLOCAL_LINEAR_KEYS = [
    ("pre_down_linear", "non_local_model.pre_down_layer.0"),
    ("post_up_linear", "non_local_model.post_up_layer.0"),
]


def _torch_to_jax_dict(state_dict: dict) -> dict:
    return {
        k: jnp.array(v.detach().cpu().numpy()) for k, v in state_dict.items()
    }


def _set_linear(model, attr_path, weight, bias):
    def get(m):
        obj = m
        for a in attr_path:
            obj = getattr(obj, a)
        return obj

    layer = get(model)
    new_layer = eqx.tree_at(
        lambda lin: (lin.weight, lin.bias), layer, (weight, bias),
    )
    return eqx.tree_at(get, model, new_layer)


def _apply_weights(model, jax_dict: dict, buffers: dict | None = None):
    """Copy weights (and optionally Wigner-3j buffers) into ``model``."""
    for attr_name, sd_prefix in _INPUT_LINEAR_KEYS + _OUTPUT_LINEAR_KEYS:
        model = _set_linear(
            model,
            (attr_name,),
            jax_dict[f"{sd_prefix}.weight"],
            jax_dict[f"{sd_prefix}.bias"],
        )

    if not (model.non_local and model.non_local_model is not None):
        return model

    for attr_name, sd_prefix in _NONLOCAL_LINEAR_KEYS:
        model = _set_linear(
            model,
            ("non_local_model", attr_name),
            jax_dict[f"{sd_prefix}.weight"],
            jax_dict[f"{sd_prefix}.bias"],
        )

    for tp_name in ("tp_down", "tp_up"):
        sd_prefix = f"non_local_model.{tp_name}."
        new_weights = {
            k.removeprefix(sd_prefix): v
            for k, v in jax_dict.items()
            if k.startswith(sd_prefix + "weight_")
        }
        model = eqx.tree_at(
            lambda m, _n=tp_name: getattr(m.non_local_model, _n).weights,
            model,
            new_weights,
        )

        if buffers is not None:
            new_w3j = {
                k.removeprefix(sd_prefix): v
                for k, v in buffers.items()
                if k.startswith(sd_prefix + "w3j_")
            }
            if new_w3j:
                model = eqx.tree_at(
                    lambda m, _n=tp_name: getattr(m.non_local_model, _n).w3j,
                    model,
                    new_w3j,
                )

    return model


def load_weights_into_model(model, pytorch_state_dict: dict):
    """Load weights (no Wigner-3j buffers) from a PyTorch ``state_dict``."""
    return _apply_weights(model, _torch_to_jax_dict(pytorch_state_dict))


def load_weights_and_buffers_into_model(model, torch_model):
    """Load weights and Wigner-3j buffers from a PyTorch ``nn.Module``.

    Needed for full numerical equivalence with the PyTorch reference.
    """
    jax_dict = _torch_to_jax_dict(torch_model.state_dict())
    buffers = {
        name: jnp.array(buf.detach().cpu().numpy())
        for name, buf in torch_model.named_buffers()
    }
    return _apply_weights(model, jax_dict, buffers=buffers)


def load_weights_from_npz(model, weights_dir: Path | str):
    """Load weights and Wigner-3j buffers from the bundled ``.npz`` files.

    ``weights_dir`` must contain ``skala_weights.npz`` and (when the model
    has a non-local branch) ``skala_buffers.npz``.
    """
    weights_dir = Path(weights_dir)

    weights_file = weights_dir / "skala_weights.npz"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    with np.load(weights_file) as data:
        jax_dict = {k: jnp.array(v) for k, v in data.items()}

    buffers: dict | None = None
    buffers_file = weights_dir / "skala_buffers.npz"
    if buffers_file.exists():
        with np.load(buffers_file) as data:
            buffers = {k: jnp.array(v) for k, v in data.items()}

    return _apply_weights(model, jax_dict, buffers=buffers)


def load_config(weights_dir: Path | str) -> dict:
    """Return the model config saved alongside the weights.

    Falls back to the default Skala hyperparameters if ``config.json`` is
    absent.
    """
    config_file = Path(weights_dir) / "config.json"
    if not config_file.exists():
        return {
            "lmax": 3,
            "non_local": True,
            "non_local_hidden_nf": 16,
            "radius_cutoff": 5.0,
        }
    with open(config_file) as f:
        return json.load(f)


def get_default_weights_dir() -> Path:
    """Return the filesystem path to the bundled pretrained weights."""
    from importlib.resources import files

    # ``files("skalax") / "weights"`` is a ``Traversable``. For installed
    # wheels on the filesystem this is a real path; zipped installs are
    # not supported because the loaders read files through ``pathlib``.
    return Path(str(files("skalax") / "weights"))
