# SPDX-License-Identifier: MIT
"""
Tests for NonLocalModel and TensorProduct components.
"""

import numpy as np
import pytest
import torch

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# PyTorch model
from skala.functional.model import (
    SkalaFunctional as TorchSkalaFunctional,
    TensorProduct as TorchTensorProduct,
    NonLocalModel as TorchNonLocalModel,
)
from e3nn import o3

# JAX model
from skalax.functional.model import (
    SkalaFunctional as JaxSkalaFunctional,
    TensorProduct as JaxTensorProduct,
    NonLocalModel as JaxNonLocalModel,
)
from skalax.convert_weights import load_weights_and_buffers_into_model
import e3nn_jax as e3nn


class TestTensorProductEquivalence:
    """Test TensorProduct matches between PyTorch and JAX."""

    def test_tensor_product_with_loaded_weights(self):
        """Test TensorProduct with weights loaded from PyTorch."""
        np.random.seed(42)

        # Create PyTorch TensorProduct
        torch_irreps_in1 = o3.Irreps("16x0e")
        torch_irreps_in2 = o3.Irreps.spherical_harmonics(3, p=1)
        torch_irreps_out = o3.Irreps("+".join([f"16x{i}e" for i in range(4)]))

        torch_tp = TorchTensorProduct(
            torch_irreps_in1, torch_irreps_in2, torch_irreps_out
        )
        torch_tp.double()  # Convert to double precision

        # Create JAX TensorProduct
        jax_irreps_in1 = e3nn.Irreps("16x0e")
        jax_irreps_in2 = e3nn.Irreps.spherical_harmonics(3, p=1)
        jax_irreps_out = e3nn.Irreps("+".join([f"16x{i}e" for i in range(4)]))

        key = jax.random.PRNGKey(0)
        jax_tp = JaxTensorProduct(
            jax_irreps_in1, jax_irreps_in2, jax_irreps_out, key=key
        )

        # Load PyTorch weights into JAX model
        # Get weights
        jax_weights = {}
        for name, param in torch_tp.named_parameters():
            jax_weights[name] = jnp.array(param.detach().numpy())

        # Get buffers (w3j)
        jax_w3j = {}
        for name, buf in torch_tp.named_buffers():
            jax_w3j[name] = jnp.array(buf.detach().numpy())

        # Update JAX model
        import equinox as eqx
        jax_tp = eqx.tree_at(lambda tp: tp.weights, jax_tp, jax_weights)
        jax_tp = eqx.tree_at(lambda tp: tp.w3j, jax_tp, jax_w3j)

        # Create test input
        batch_size = 10
        x1_np = np.random.randn(batch_size, 16).astype(np.float64)
        x2_np = np.random.randn(batch_size, 16).astype(np.float64)

        # PyTorch forward
        x1_torch = torch.from_numpy(x1_np)
        x2_torch = torch.from_numpy(x2_np)
        with torch.no_grad():
            y_torch = torch_tp(x1_torch, x2_torch).numpy()

        # JAX forward
        x1_jax = jnp.array(x1_np)
        x2_jax = jnp.array(x2_np)
        y_jax = np.array(jax_tp(x1_jax, x2_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


class TestNonLocalModelEquivalence:
    """Test NonLocalModel matches between PyTorch and JAX."""

    def test_nonlocal_model_with_loaded_weights(self):
        """Test NonLocalModel with weights loaded from PyTorch."""
        np.random.seed(42)

        # Create PyTorch NonLocalModel
        torch_nl = TorchNonLocalModel(
            input_nf=256,
            hidden_nf=16,
            lmax=3,
            radius_cutoff=5.0,
        )
        torch_nl.double()

        # Create JAX NonLocalModel
        key = jax.random.PRNGKey(0)
        jax_nl = JaxNonLocalModel(
            input_nf=256,
            hidden_nf=16,
            lmax=3,
            radius_cutoff=5.0,
            key=key,
        )

        # Load PyTorch weights into JAX model
        import equinox as eqx

        # Pre-down layer
        pre_w = jnp.array(torch_nl.pre_down_layer[0].weight.detach().numpy())
        pre_b = jnp.array(torch_nl.pre_down_layer[0].bias.detach().numpy())
        jax_nl = eqx.tree_at(
            lambda m: (m.pre_down_linear.weight, m.pre_down_linear.bias),
            jax_nl,
            (pre_w, pre_b),
        )

        # Post-up layer
        post_w = jnp.array(torch_nl.post_up_layer[0].weight.detach().numpy())
        post_b = jnp.array(torch_nl.post_up_layer[0].bias.detach().numpy())
        jax_nl = eqx.tree_at(
            lambda m: (m.post_up_linear.weight, m.post_up_linear.bias),
            jax_nl,
            (post_w, post_b),
        )

        # TensorProduct tp_down weights and w3j
        tp_down_weights = {}
        tp_down_w3j = {}
        for name, param in torch_nl.tp_down.named_parameters():
            tp_down_weights[name] = jnp.array(param.detach().numpy())
        for name, buf in torch_nl.tp_down.named_buffers():
            tp_down_w3j[name] = jnp.array(buf.detach().numpy())

        jax_nl = eqx.tree_at(lambda m: m.tp_down.weights, jax_nl, tp_down_weights)
        jax_nl = eqx.tree_at(lambda m: m.tp_down.w3j, jax_nl, tp_down_w3j)

        # TensorProduct tp_up weights and w3j
        tp_up_weights = {}
        tp_up_w3j = {}
        for name, param in torch_nl.tp_up.named_parameters():
            tp_up_weights[name] = jnp.array(param.detach().numpy())
        for name, buf in torch_nl.tp_up.named_buffers():
            tp_up_w3j[name] = jnp.array(buf.detach().numpy())

        jax_nl = eqx.tree_at(lambda m: m.tp_up.weights, jax_nl, tp_up_weights)
        jax_nl = eqx.tree_at(lambda m: m.tp_up.w3j, jax_nl, tp_up_w3j)

        # Create test input
        num_fine = 20
        num_coarse = 3
        h_np = np.random.randn(num_fine, 256).astype(np.float64)
        grid_coords_np = np.random.randn(num_fine, 3).astype(np.float64)
        coarse_coords_np = np.random.randn(num_coarse, 3).astype(np.float64)
        grid_weights_np = np.abs(np.random.randn(num_fine)).astype(np.float64) + 0.1

        # PyTorch forward
        h_torch = torch.from_numpy(h_np)
        grid_coords_torch = torch.from_numpy(grid_coords_np)
        coarse_coords_torch = torch.from_numpy(coarse_coords_np)
        grid_weights_torch = torch.from_numpy(grid_weights_np)

        with torch.no_grad():
            y_torch = torch_nl(
                h_torch, grid_coords_torch, coarse_coords_torch, grid_weights_torch
            ).numpy()

        # JAX forward
        h_jax = jnp.array(h_np)
        grid_coords_jax = jnp.array(grid_coords_np)
        coarse_coords_jax = jnp.array(coarse_coords_np)
        grid_weights_jax = jnp.array(grid_weights_np)

        y_jax = np.array(
            jax_nl(h_jax, grid_coords_jax, coarse_coords_jax, grid_weights_jax)
        )

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


class TestFullModelWithNonLocal:
    """Test full model with non-local component."""

    def test_get_exc_with_nonlocal(self):
        """Test get_exc with non-local model."""
        np.random.seed(42)

        # Create PyTorch model
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
        )

        # Create JAX model
        key = jax.random.PRNGKey(0)
        jax_model = JaxSkalaFunctional(
            lmax=3,
            non_local=True,
            non_local_hidden_nf=16,
            radius_cutoff=5.0,
            key=key,
        )

        # Load weights and buffers
        jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

        # Create test molecular features
        n_points = 30
        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3),
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(3, 3),
        }

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64)) for k, v in mol_np.items()
        }
        mol_jax = {
            k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()
        }

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_get_exc_density_with_nonlocal(self):
        """Test get_exc_density with non-local model."""
        np.random.seed(42)

        # Create PyTorch model
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
        )

        # Create JAX model
        key = jax.random.PRNGKey(0)
        jax_model = JaxSkalaFunctional(
            lmax=3,
            non_local=True,
            non_local_hidden_nf=16,
            radius_cutoff=5.0,
            key=key,
        )

        # Load weights and buffers
        jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

        # Create test molecular features
        n_points = 30
        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3),
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(3, 3),
        }

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64)) for k, v in mol_np.items()
        }
        mol_jax = {
            k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()
        }

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc_density(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc_density(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
