# SPDX-License-Identifier: MIT
"""
Weight conversion utilities for PyTorch to JAX.

Converts PyTorch state_dict to JAX pytree format compatible with Equinox.
Supports loading from PyTorch models or from local .npz files.
"""

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import equinox as eqx


def convert_pytorch_state_dict(state_dict: dict) -> dict:
    """
    Convert PyTorch state_dict to JAX-compatible dictionary.

    Parameters
    ----------
    state_dict : dict
        PyTorch state_dict with tensor values.

    Returns
    -------
    dict
        Dictionary with JAX arrays.
    """
    jax_dict = {}
    for key, tensor in state_dict.items():
        # Convert to numpy then to JAX
        np_array = tensor.detach().cpu().numpy()
        jax_dict[key] = jnp.array(np_array)
    return jax_dict


def extract_buffers(torch_model) -> dict:
    """
    Extract all buffers (including Wigner 3j) from PyTorch model.

    Parameters
    ----------
    torch_model : nn.Module
        PyTorch model.

    Returns
    -------
    dict
        Dictionary with buffer names and JAX arrays.
    """
    buffers = {}
    for name, buf in torch_model.named_buffers():
        buffers[name] = jnp.array(buf.detach().cpu().numpy())
    return buffers


def load_weights_into_model(model, pytorch_state_dict: dict):
    """
    Load PyTorch weights into a JAX/Equinox SkalaFunctional model.

    Parameters
    ----------
    model : SkalaFunctional
        JAX model to load weights into.
    pytorch_state_dict : dict
        PyTorch state_dict.

    Returns
    -------
    SkalaFunctional
        Model with loaded weights.
    """
    jax_dict = convert_pytorch_state_dict(pytorch_state_dict)

    # Helper to update a linear layer
    def update_linear(layer, weight_key, bias_key):
        weight = jax_dict[weight_key]
        bias = jax_dict[bias_key]
        return eqx.tree_at(
            lambda lin: (lin.weight, lin.bias),
            layer,
            (weight, bias),
        )

    # Update input model
    model = eqx.tree_at(
        lambda m: m.input_linear1,
        model,
        update_linear(
            model.input_linear1,
            "input_model.0.weight",
            "input_model.0.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.input_linear2,
        model,
        update_linear(
            model.input_linear2,
            "input_model.2.weight",
            "input_model.2.bias",
        ),
    )

    # Update output model
    model = eqx.tree_at(
        lambda m: m.output_linear1,
        model,
        update_linear(
            model.output_linear1,
            "output_model.0.weight",
            "output_model.0.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.output_linear2,
        model,
        update_linear(
            model.output_linear2,
            "output_model.2.weight",
            "output_model.2.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.output_linear3,
        model,
        update_linear(
            model.output_linear3,
            "output_model.4.weight",
            "output_model.4.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.output_linear4,
        model,
        update_linear(
            model.output_linear4,
            "output_model.6.weight",
            "output_model.6.bias",
        ),
    )

    # Update non-local model if present
    if model.non_local and model.non_local_model is not None:
        # Pre-down layer
        model = eqx.tree_at(
            lambda m: m.non_local_model.pre_down_linear,
            model,
            update_linear(
                model.non_local_model.pre_down_linear,
                "non_local_model.pre_down_layer.0.weight",
                "non_local_model.pre_down_layer.0.bias",
            ),
        )

        # Post-up layer
        model = eqx.tree_at(
            lambda m: m.non_local_model.post_up_linear,
            model,
            update_linear(
                model.non_local_model.post_up_linear,
                "non_local_model.post_up_layer.0.weight",
                "non_local_model.post_up_layer.0.bias",
            ),
        )

        # TensorProduct weights for tp_down
        tp_down_weights = {}
        for key, val in jax_dict.items():
            if key.startswith("non_local_model.tp_down.weight_"):
                weight_key = key.replace("non_local_model.tp_down.", "")
                tp_down_weights[weight_key] = val

        model = eqx.tree_at(
            lambda m: m.non_local_model.tp_down.weights,
            model,
            tp_down_weights,
        )

        # TensorProduct weights for tp_up
        tp_up_weights = {}
        for key, val in jax_dict.items():
            if key.startswith("non_local_model.tp_up.weight_"):
                weight_key = key.replace("non_local_model.tp_up.", "")
                tp_up_weights[weight_key] = val

        model = eqx.tree_at(
            lambda m: m.non_local_model.tp_up.weights,
            model,
            tp_up_weights,
        )

    return model


def load_weights_and_buffers_into_model(model, torch_model):
    """
    Load PyTorch weights AND buffers (Wigner 3j) into JAX model.

    This is the complete loading function that ensures numerical equivalence.

    Parameters
    ----------
    model : SkalaFunctional
        JAX model to load weights into.
    torch_model : nn.Module
        PyTorch model to extract weights and buffers from.

    Returns
    -------
    SkalaFunctional
        Model with loaded weights and buffers.
    """
    # First load the regular weights
    state_dict = {k: v for k, v in torch_model.state_dict().items()}
    model = load_weights_into_model(model, state_dict)

    # Now load the Wigner 3j buffers for non-local model
    if model.non_local and model.non_local_model is not None:
        # Extract buffers from PyTorch model
        buffers = {}
        for name, buf in torch_model.named_buffers():
            buffers[name] = jnp.array(buf.detach().cpu().numpy())

        # Load tp_down w3j buffers
        tp_down_w3j = {}
        for key, val in buffers.items():
            if key.startswith("non_local_model.tp_down.w3j_"):
                w3j_key = key.replace("non_local_model.tp_down.", "")
                tp_down_w3j[w3j_key] = val

        if tp_down_w3j:
            model = eqx.tree_at(
                lambda m: m.non_local_model.tp_down.w3j,
                model,
                tp_down_w3j,
            )

        # Load tp_up w3j buffers
        tp_up_w3j = {}
        for key, val in buffers.items():
            if key.startswith("non_local_model.tp_up.w3j_"):
                w3j_key = key.replace("non_local_model.tp_up.", "")
                tp_up_w3j[w3j_key] = val

        if tp_up_w3j:
            model = eqx.tree_at(
                lambda m: m.non_local_model.tp_up.w3j,
                model,
                tp_up_w3j,
            )

    return model


def load_weights_from_npz(model, weights_dir: Path | str):
    """
    Load weights from local .npz files (no PyTorch or HuggingFace needed).

    Parameters
    ----------
    model : SkalaFunctional
        JAX model to load weights into.
    weights_dir : Path or str
        Directory containing skala_weights.npz, skala_buffers.npz, and config.json

    Returns
    -------
    SkalaFunctional
        Model with loaded weights and buffers.

    Example
    -------
    >>> from skalax.functional.model import SkalaFunctional
    >>> from skalax.convert_weights import load_weights_from_npz
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> model = SkalaFunctional(lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key)
    >>> model = load_weights_from_npz(model, "skalax/weights")
    """
    weights_dir = Path(weights_dir)

    # Load weights
    weights_file = weights_dir / "skala_weights.npz"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    weights_data = np.load(weights_file)
    jax_dict = {k: jnp.array(v) for k, v in weights_data.items()}

    # Helper to update a linear layer
    def update_linear(layer, weight_key, bias_key):
        weight = jax_dict[weight_key]
        bias = jax_dict[bias_key]
        return eqx.tree_at(
            lambda lin: (lin.weight, lin.bias),
            layer,
            (weight, bias),
        )

    # Update input model
    model = eqx.tree_at(
        lambda m: m.input_linear1,
        model,
        update_linear(
            model.input_linear1,
            "input_model.0.weight",
            "input_model.0.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.input_linear2,
        model,
        update_linear(
            model.input_linear2,
            "input_model.2.weight",
            "input_model.2.bias",
        ),
    )

    # Update output model
    model = eqx.tree_at(
        lambda m: m.output_linear1,
        model,
        update_linear(
            model.output_linear1,
            "output_model.0.weight",
            "output_model.0.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.output_linear2,
        model,
        update_linear(
            model.output_linear2,
            "output_model.2.weight",
            "output_model.2.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.output_linear3,
        model,
        update_linear(
            model.output_linear3,
            "output_model.4.weight",
            "output_model.4.bias",
        ),
    )
    model = eqx.tree_at(
        lambda m: m.output_linear4,
        model,
        update_linear(
            model.output_linear4,
            "output_model.6.weight",
            "output_model.6.bias",
        ),
    )

    # Update non-local model if present
    if model.non_local and model.non_local_model is not None:
        # Pre-down layer
        model = eqx.tree_at(
            lambda m: m.non_local_model.pre_down_linear,
            model,
            update_linear(
                model.non_local_model.pre_down_linear,
                "non_local_model.pre_down_layer.0.weight",
                "non_local_model.pre_down_layer.0.bias",
            ),
        )

        # Post-up layer
        model = eqx.tree_at(
            lambda m: m.non_local_model.post_up_linear,
            model,
            update_linear(
                model.non_local_model.post_up_linear,
                "non_local_model.post_up_layer.0.weight",
                "non_local_model.post_up_layer.0.bias",
            ),
        )

        # TensorProduct weights for tp_down
        tp_down_weights = {}
        for key in jax_dict.keys():
            if key.startswith("non_local_model.tp_down.weight_"):
                weight_key = key.replace("non_local_model.tp_down.", "")
                tp_down_weights[weight_key] = jax_dict[key]

        model = eqx.tree_at(
            lambda m: m.non_local_model.tp_down.weights,
            model,
            tp_down_weights,
        )

        # TensorProduct weights for tp_up
        tp_up_weights = {}
        for key in jax_dict.keys():
            if key.startswith("non_local_model.tp_up.weight_"):
                weight_key = key.replace("non_local_model.tp_up.", "")
                tp_up_weights[weight_key] = jax_dict[key]

        model = eqx.tree_at(
            lambda m: m.non_local_model.tp_up.weights,
            model,
            tp_up_weights,
        )

        # Load buffers (Wigner 3j coefficients)
        buffers_file = weights_dir / "skala_buffers.npz"
        if buffers_file.exists():
            buffers_data = np.load(buffers_file)
            buffers = {k: jnp.array(v) for k, v in buffers_data.items()}

            # Load tp_down w3j buffers
            tp_down_w3j = {}
            for key, val in buffers.items():
                if key.startswith("non_local_model.tp_down.w3j_"):
                    w3j_key = key.replace("non_local_model.tp_down.", "")
                    tp_down_w3j[w3j_key] = val

            if tp_down_w3j:
                model = eqx.tree_at(
                    lambda m: m.non_local_model.tp_down.w3j,
                    model,
                    tp_down_w3j,
                )

            # Load tp_up w3j buffers
            tp_up_w3j = {}
            for key, val in buffers.items():
                if key.startswith("non_local_model.tp_up.w3j_"):
                    w3j_key = key.replace("non_local_model.tp_up.", "")
                    tp_up_w3j[w3j_key] = val

            if tp_up_w3j:
                model = eqx.tree_at(
                    lambda m: m.non_local_model.tp_up.w3j,
                    model,
                    tp_up_w3j,
                )

    return model


def load_config(weights_dir: Path | str) -> dict:
    """
    Load model configuration from weights directory.

    Parameters
    ----------
    weights_dir : Path or str
        Directory containing config.json

    Returns
    -------
    dict
        Model configuration.
    """
    weights_dir = Path(weights_dir)
    config_file = weights_dir / "config.json"

    if not config_file.exists():
        # Return default config
        return {
            "lmax": 3,
            "non_local": True,
            "non_local_hidden_nf": 16,
            "radius_cutoff": 5.0,
        }

    with open(config_file) as f:
        return json.load(f)


def get_default_weights_dir():
    """
    Get the path to bundled pretrained weights.

    This function returns the path to the weights directory that is bundled
    with the skala-jax package. These weights are the official Skala pretrained
    weights converted to float64 format.

    Returns
    -------
    Traversable
        Path-like object to the weights directory bundled with the package.

    Example
    -------
    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>>
    >>> from skalax import SkalaFunctional, load_weights_from_npz, load_config, get_default_weights_dir
    >>>
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
    from importlib.resources import files

    return files("skalax") / "weights"
