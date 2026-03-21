# SPDX-License-Identifier: MIT
"""
Tests using actual Skala weights loaded from HuggingFace.
"""

import numpy as np
import pytest
import torch

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# PyTorch model loading
from skala.functional import load_functional
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional

# JAX model
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import (
    load_weights_into_model,
    load_weights_and_buffers_into_model,
)


@pytest.fixture(scope="module")
def loaded_torch_model():
    """Load the pretrained Skala model."""
    return load_functional("skala")


@pytest.fixture(scope="module")
def clean_state_dict(loaded_torch_model):
    """Get clean state dict from loaded model."""
    state_dict = loaded_torch_model.state_dict()
    return {
        k.replace("_traced_model.", ""): v
        for k, v in state_dict.items()
    }


@pytest.fixture(scope="module")
def torch_model_from_weights(clean_state_dict):
    """Create fresh PyTorch model with loaded weights."""
    model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    model.load_state_dict(clean_state_dict, strict=True)
    return model


@pytest.fixture(scope="module")
def jax_model_with_weights(clean_state_dict):
    """Create JAX model with loaded weights."""
    key = jax.random.PRNGKey(0)
    jax_model = JaxSkalaFunctional(
        lmax=3,
        non_local=True,
        non_local_hidden_nf=16,
        radius_cutoff=5.0,
        key=key,
    )
    return load_weights_into_model(jax_model, clean_state_dict)


class TestWithLoadedWeights:
    """Tests using actual loaded weights."""

    def test_input_model_with_loaded_weights(
        self, torch_model_from_weights, jax_model_with_weights
    ):
        """Test input model with loaded weights."""
        np.random.seed(42)

        # Create test input
        x_np = np.random.randn(10, 7).astype(np.float64)

        # PyTorch forward
        x_torch = torch.from_numpy(x_np).double()
        torch_model_from_weights.double()
        with torch.no_grad():
            y_torch = torch_model_from_weights.input_model(x_torch).numpy()

        # JAX forward
        x_jax = jnp.array(x_np)
        y_jax = np.array(jax_model_with_weights._input_model(x_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_output_model_with_loaded_weights(
        self, torch_model_from_weights, jax_model_with_weights
    ):
        """Test output model with loaded weights."""
        np.random.seed(42)

        # Create test input (256 + 16 features for non_local=True)
        x_np = np.random.randn(10, 272).astype(np.float64)

        # PyTorch forward
        x_torch = torch.from_numpy(x_np).double()
        torch_model_from_weights.double()
        with torch.no_grad():
            y_torch = torch_model_from_weights.output_model(x_torch).numpy()

        # JAX forward
        x_jax = jnp.array(x_np)
        y_jax = np.array(jax_model_with_weights._output_model(x_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_get_exc_no_nonlocal_with_loaded_weights(self, clean_state_dict):
        """Test get_exc without non-local using loaded weights."""
        np.random.seed(42)

        # Create PyTorch model without non-local
        torch_model = TorchSkalaFunctional(lmax=3, non_local=False)
        # Load only the compatible weights
        torch_model.input_model[0].weight.data = clean_state_dict[
            "input_model.0.weight"
        ]
        torch_model.input_model[0].bias.data = clean_state_dict[
            "input_model.0.bias"
        ]
        torch_model.input_model[2].weight.data = clean_state_dict[
            "input_model.2.weight"
        ]
        torch_model.input_model[2].bias.data = clean_state_dict[
            "input_model.2.bias"
        ]

        # Create JAX model
        key = jax.random.PRNGKey(0)
        jax_model = JaxSkalaFunctional(lmax=3, non_local=False, key=key)
        jax_model = load_weights_into_model(
            jax_model, torch_model.state_dict()
        )

        # Create test molecular features
        n_points = 20
        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3),
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(2, 3),
        }

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64))
            for k, v in mol_np.items()
        }
        mol_jax = {
            k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()
        }

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


class TestFullModelWithHuggingFaceWeights:
    """Tests for full model with HuggingFace pretrained weights."""

    def test_all_weights_loaded_correctly(self, clean_state_dict):
        """Verify all 24 weights from HuggingFace are loaded correctly."""
        # Create fresh PyTorch model
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
        )
        torch_model.load_state_dict(clean_state_dict, strict=True)
        torch_model.double()

        # Create JAX model
        key = jax.random.PRNGKey(0)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
        )
        jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

        # Verify input model weights
        np.testing.assert_allclose(
            clean_state_dict["input_model.0.weight"].numpy(),
            np.array(jax_model.input_linear1.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["input_model.0.bias"].numpy(),
            np.array(jax_model.input_linear1.bias),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["input_model.2.weight"].numpy(),
            np.array(jax_model.input_linear2.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["input_model.2.bias"].numpy(),
            np.array(jax_model.input_linear2.bias),
            atol=1e-10,
        )

        # Verify non-local pre-down weights
        np.testing.assert_allclose(
            clean_state_dict["non_local_model.pre_down_layer.0.weight"].numpy(),
            np.array(jax_model.non_local_model.pre_down_linear.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["non_local_model.pre_down_layer.0.bias"].numpy(),
            np.array(jax_model.non_local_model.pre_down_linear.bias),
            atol=1e-10,
        )

        # Verify tp_down weights
        for key in ["weight_0_0_0", "weight_0_1_1", "weight_0_2_2", "weight_0_3_3"]:
            np.testing.assert_allclose(
                clean_state_dict[f"non_local_model.tp_down.{key}"].numpy(),
                np.array(jax_model.non_local_model.tp_down.weights[key]),
                atol=1e-10,
            )

        # Verify tp_up weights
        for key in ["weight_0_0_0", "weight_1_1_0", "weight_2_2_0", "weight_3_3_0"]:
            np.testing.assert_allclose(
                clean_state_dict[f"non_local_model.tp_up.{key}"].numpy(),
                np.array(jax_model.non_local_model.tp_up.weights[key]),
                atol=1e-10,
            )

        # Verify non-local post-up weights
        np.testing.assert_allclose(
            clean_state_dict["non_local_model.post_up_layer.0.weight"].numpy(),
            np.array(jax_model.non_local_model.post_up_linear.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["non_local_model.post_up_layer.0.bias"].numpy(),
            np.array(jax_model.non_local_model.post_up_linear.bias),
            atol=1e-10,
        )

        # Verify output model weights
        np.testing.assert_allclose(
            clean_state_dict["output_model.0.weight"].numpy(),
            np.array(jax_model.output_linear1.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.0.bias"].numpy(),
            np.array(jax_model.output_linear1.bias),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.2.weight"].numpy(),
            np.array(jax_model.output_linear2.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.2.bias"].numpy(),
            np.array(jax_model.output_linear2.bias),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.4.weight"].numpy(),
            np.array(jax_model.output_linear3.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.4.bias"].numpy(),
            np.array(jax_model.output_linear3.bias),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.6.weight"].numpy(),
            np.array(jax_model.output_linear4.weight),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            clean_state_dict["output_model.6.bias"].numpy(),
            np.array(jax_model.output_linear4.bias),
            atol=1e-10,
        )

    def test_get_exc_with_pretrained_weights(self, clean_state_dict):
        """Test get_exc with full pretrained model including non-local."""
        np.random.seed(42)

        # Create fresh PyTorch model
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
        )
        torch_model.load_state_dict(clean_state_dict, strict=True)
        torch_model.double()

        # Create JAX model
        key = jax.random.PRNGKey(0)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
        )
        jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

        # Create test molecular features
        n_points = 50
        n_atoms = 5
        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3) * 3,
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(n_atoms, 3) * 2,
        }

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64)) for k, v in mol_np.items()
        }
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_get_exc_density_with_pretrained_weights(self, clean_state_dict):
        """Test get_exc_density with full pretrained model including non-local."""
        np.random.seed(42)

        # Create fresh PyTorch model
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
        )
        torch_model.load_state_dict(clean_state_dict, strict=True)
        torch_model.double()

        # Create JAX model
        key = jax.random.PRNGKey(0)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
        )
        jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

        # Create test molecular features
        n_points = 50
        n_atoms = 5
        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3) * 3,
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(n_atoms, 3) * 2,
        }

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64)) for k, v in mol_np.items()
        }
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc_density(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc_density(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
