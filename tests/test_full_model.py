# SPDX-License-Identifier: MIT
"""
Tests for full model comparison between PyTorch and JAX.
"""

import numpy as np
import pytest
import torch

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# PyTorch model
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional

# JAX model
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import load_weights_into_model


class TestTensorProduct:
    """Test TensorProduct matches between PyTorch and JAX."""

    def test_tensor_product_output_shape(self):
        """Test TensorProduct output shape matches."""
        import e3nn_jax as e3nn
        from skalax.functional.model import TensorProduct

        key = jax.random.PRNGKey(0)
        irreps_in1 = e3nn.Irreps("16x0e")
        irreps_in2 = e3nn.Irreps.spherical_harmonics(3, p=1)
        irreps_out = e3nn.Irreps("+".join([f"16x{i}e" for i in range(4)]))

        tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, key=key)

        x1 = jnp.ones((5, 16))
        x2 = jnp.ones((5, 16))
        out = tp(x1, x2)

        assert out.shape == (5, 256)


class TestModelInstantiation:
    """Test model can be instantiated."""

    def test_jax_model_instantiation(self):
        """Test JAX model can be created."""
        key = jax.random.PRNGKey(42)
        model = JaxSkalaFunctional(
            lmax=3, non_local=True, radius_cutoff=5.0, key=key
        )
        assert model.non_local is True
        assert model.num_feats == 256

    def test_jax_model_no_nonlocal(self):
        """Test JAX model without non-local."""
        key = jax.random.PRNGKey(42)
        model = JaxSkalaFunctional(
            lmax=3, non_local=False, key=key
        )
        assert model.non_local is False
        assert model.non_local_model is None


class TestWeightLoading:
    """Test weight loading from PyTorch to JAX."""

    def test_load_weights_input_model(self):
        """Test loading weights into input model."""
        # Create PyTorch model
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=True, radius_cutoff=5.0
        )
        torch_state = torch_model.state_dict()

        # Create JAX model
        key = jax.random.PRNGKey(42)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=True, radius_cutoff=5.0, key=key
        )

        # Load weights
        jax_model = load_weights_into_model(jax_model, torch_state)

        # Check input weights match
        torch_w = torch_state["input_model.0.weight"].numpy()
        jax_w = np.array(jax_model.input_linear1.weight)
        np.testing.assert_allclose(torch_w, jax_w, atol=1e-6)


class TestInputModel:
    """Test input model component."""

    def test_input_model_equivalence(self):
        """Test input model produces same output."""
        np.random.seed(42)

        # Create models
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=False, radius_cutoff=5.0
        )

        key = jax.random.PRNGKey(42)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=False, radius_cutoff=5.0, key=key
        )

        # Load PyTorch weights into JAX model
        jax_model = load_weights_into_model(jax_model, torch_model.state_dict())

        # Create test input
        x_np = np.random.randn(10, 7).astype(np.float32)

        # PyTorch forward
        x_torch = torch.from_numpy(x_np)
        with torch.no_grad():
            y_torch = torch_model.input_model(x_torch).numpy()

        # JAX forward
        x_jax = jnp.array(x_np)
        y_jax = np.array(jax_model._input_model(x_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


class TestOutputModel:
    """Test output model component."""

    def test_output_model_equivalence(self):
        """Test output model produces same output."""
        np.random.seed(42)

        # Create models (without non-local for simplicity)
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=False, radius_cutoff=5.0
        )

        key = jax.random.PRNGKey(42)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=False, radius_cutoff=5.0, key=key
        )

        # Load PyTorch weights into JAX model
        jax_model = load_weights_into_model(jax_model, torch_model.state_dict())

        # Create test input (256 features for non-local=False)
        x_np = np.random.randn(10, 256).astype(np.float32)

        # PyTorch forward
        x_torch = torch.from_numpy(x_np)
        with torch.no_grad():
            y_torch = torch_model.output_model(x_torch).numpy()

        # JAX forward
        x_jax = jnp.array(x_np)
        y_jax = np.array(jax_model._output_model(x_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


class TestFullModelNoNonLocal:
    """Test full model without non-local component."""

    def test_get_exc_density_no_nonlocal(self):
        """Test get_exc_density without non-local."""
        np.random.seed(42)

        # Create models
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=False
        )

        key = jax.random.PRNGKey(42)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=False, key=key
        )

        # Load weights
        jax_model = load_weights_into_model(jax_model, torch_model.state_dict())

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

        # Convert to appropriate types
        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64))
            for k, v in mol_np.items()
        }
        mol_jax = {
            k: jnp.array(v.astype(np.float64))
            for k, v in mol_np.items()
        }

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc_density(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc_density(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_get_exc_no_nonlocal(self):
        """Test get_exc without non-local."""
        np.random.seed(42)

        # Create models
        torch_model = TorchSkalaFunctional(
            lmax=3, non_local=False
        )

        key = jax.random.PRNGKey(42)
        jax_model = JaxSkalaFunctional(
            lmax=3, non_local=False, key=key
        )

        # Load weights
        jax_model = load_weights_into_model(jax_model, torch_model.state_dict())

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

        # Convert to appropriate types
        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64))
            for k, v in mol_np.items()
        }
        mol_jax = {
            k: jnp.array(v.astype(np.float64))
            for k, v in mol_np.items()
        }

        # Forward pass
        with torch.no_grad():
            y_torch = torch_model.get_exc(mol_torch).numpy()

        y_jax = np.array(jax_model.get_exc(mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
