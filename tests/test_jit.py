# SPDX-License-Identifier: MIT
"""JIT compilation tests for ``SkalaFunctional`` (including the non-local branch).

Checks that ``eqx.filter_jit`` compiles the model, that JIT output matches
eager evaluation, that ``jax.grad`` composes with JIT, and that numerical
equivalence to the PyTorch reference is preserved after compilation.
"""

import numpy as np
import pytest
import torch

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx

from skala.functional.model import SkalaFunctional as TorchSkalaFunctional

from skalax.functional.model import (
    SkalaFunctional as JaxSkalaFunctional,
    NonLocalModel as JaxNonLocalModel,
)
from skalax.convert_weights import load_weights_and_buffers_into_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@eqx.filter_jit
def jit_get_exc(model, mol):
    return model.get_exc(mol)


@eqx.filter_jit
def jit_get_exc_density(model, mol):
    return model.get_exc_density(mol)


@eqx.filter_jit
def jit_grad_get_exc(model, mol):
    return jax.grad(model.get_exc)(mol)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_mol():
    """Generate a random mol dictionary for testing."""
    np.random.seed(42)
    n_points = 20
    n_atoms = 3
    mol_np = {
        "density": np.abs(np.random.randn(2, n_points)) + 0.1,
        "grad": np.random.randn(2, 3, n_points),
        "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
        "grid_coords": np.random.randn(n_points, 3),
        "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
        "coarse_0_atomic_coords": np.random.randn(n_atoms, 3),
    }
    return mol_np


@pytest.fixture
def torch_jax_models():
    """Create PyTorch and JAX models with identical weights."""
    torch_model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    torch_model.double()

    key = jax.random.PRNGKey(0)
    jax_model = JaxSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
    )
    jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)
    return torch_model, jax_model


@pytest.fixture
def nonlocal_model_with_weights():
    """Create a NonLocalModel with PyTorch-transferred weights."""
    from skala.functional.model import NonLocalModel as TorchNonLocalModel

    torch_nl = TorchNonLocalModel(
        input_nf=256, hidden_nf=16, lmax=3, radius_cutoff=5.0
    )
    torch_nl.double()

    key = jax.random.PRNGKey(0)
    jax_nl = JaxNonLocalModel(
        input_nf=256, hidden_nf=16, lmax=3, radius_cutoff=5.0, key=key
    )

    # Transfer weights
    pre_w = jnp.array(torch_nl.pre_down_layer[0].weight.detach().numpy())
    pre_b = jnp.array(torch_nl.pre_down_layer[0].bias.detach().numpy())
    jax_nl = eqx.tree_at(
        lambda m: (m.pre_down_linear.weight, m.pre_down_linear.bias),
        jax_nl, (pre_w, pre_b),
    )

    post_w = jnp.array(torch_nl.post_up_layer[0].weight.detach().numpy())
    post_b = jnp.array(torch_nl.post_up_layer[0].bias.detach().numpy())
    jax_nl = eqx.tree_at(
        lambda m: (m.post_up_linear.weight, m.post_up_linear.bias),
        jax_nl, (post_w, post_b),
    )

    for attr, torch_tp in [("tp_down", torch_nl.tp_down), ("tp_up", torch_nl.tp_up)]:
        weights = {n: jnp.array(p.detach().numpy()) for n, p in torch_tp.named_parameters()}
        w3j = {n: jnp.array(b.detach().numpy()) for n, b in torch_tp.named_buffers()}
        jax_nl = eqx.tree_at(lambda m, a=attr: getattr(m, a).weights, jax_nl, weights)
        jax_nl = eqx.tree_at(lambda m, a=attr: getattr(m, a).w3j, jax_nl, w3j)

    return torch_nl, jax_nl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJITCompilation:
    """Test that eqx.filter_jit compiles the full model without error."""

    def test_jit_get_exc(self, torch_jax_models, random_mol):
        """eqx.filter_jit(model.get_exc) compiles and runs."""
        _, jax_model = torch_jax_models
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        result = jit_get_exc(jax_model, mol_jax)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_jit_get_exc_density(self, torch_jax_models, random_mol):
        """eqx.filter_jit(model.get_exc_density) compiles and runs."""
        _, jax_model = torch_jax_models
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        result = jit_get_exc_density(jax_model, mol_jax)
        n_points = random_mol["density"].shape[1]
        assert result.shape == (n_points,)
        assert jnp.all(jnp.isfinite(result))


class TestJITMatchesEager:
    """Test that JIT output matches eager execution."""

    def test_get_exc_jit_vs_eager(self, torch_jax_models, random_mol):
        """JIT'd get_exc matches eager get_exc."""
        _, jax_model = torch_jax_models
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        eager_result = jax_model.get_exc(mol_jax)
        jit_result = jit_get_exc(jax_model, mol_jax)

        np.testing.assert_allclose(
            float(eager_result), float(jit_result), atol=1e-10
        )

    def test_get_exc_density_jit_vs_eager(self, torch_jax_models, random_mol):
        """JIT'd get_exc_density matches eager get_exc_density."""
        _, jax_model = torch_jax_models
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        eager_result = np.array(jax_model.get_exc_density(mol_jax))
        jit_result = np.array(jit_get_exc_density(jax_model, mol_jax))

        np.testing.assert_allclose(eager_result, jit_result, atol=1e-10)


class TestJITMatchesForwardEager:
    """``NonLocalModel.__call__`` must match ``forward_eager`` (reference)."""

    def test_nonlocal_call_vs_forward_eager(self, nonlocal_model_with_weights):
        np.random.seed(42)
        _, jax_nl = nonlocal_model_with_weights

        num_fine, num_coarse = 20, 3
        h = jnp.array(np.random.randn(num_fine, 256).astype(np.float64))
        grid_coords = jnp.array(
            np.random.randn(num_fine, 3).astype(np.float64)
        )
        coarse_coords = jnp.array(
            np.random.randn(num_coarse, 3).astype(np.float64)
        )
        grid_weights = jnp.array(
            (np.abs(np.random.randn(num_fine)) + 0.1).astype(np.float64)
        )

        result_call = np.array(
            jax_nl(h, grid_coords, coarse_coords, grid_weights)
        )
        result_eager = np.array(
            jax_nl.forward_eager(
                h, grid_coords, coarse_coords, grid_weights,
            )
        )

        np.testing.assert_allclose(result_call, result_eager, atol=1e-10)


class TestJITGrad:
    """Test that JIT + jax.grad works."""

    def test_jit_grad_get_exc(self, torch_jax_models, random_mol):
        """eqx.filter_jit(jax.grad(model.get_exc)) produces finite gradients."""
        _, jax_model = torch_jax_models
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        grads = jit_grad_get_exc(jax_model, mol_jax)

        for key in ["density", "grad", "kin", "grid_weights"]:
            assert jnp.all(jnp.isfinite(grads[key])), f"Non-finite gradient for {key}"

    def test_jit_grad_matches_eager_grad(self, torch_jax_models, random_mol):
        """JIT'd gradients match eager gradients."""
        _, jax_model = torch_jax_models
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        eager_grads = jax.grad(jax_model.get_exc)(mol_jax)
        jit_grads = jit_grad_get_exc(jax_model, mol_jax)

        for key in ["density", "grad", "kin", "grid_weights"]:
            np.testing.assert_allclose(
                np.array(eager_grads[key]),
                np.array(jit_grads[key]),
                atol=1e-10,
                err_msg=f"Gradient mismatch for {key}",
            )


class TestJITPreservesPyTorchEquivalence:
    """Test that JIT'd model still matches PyTorch."""

    def test_get_exc_jit_matches_pytorch(self, torch_jax_models, random_mol):
        """JIT'd JAX get_exc matches PyTorch get_exc."""
        torch_model, jax_model = torch_jax_models

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64)) for k, v in random_mol.items()
        }
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        with torch.no_grad():
            y_torch = torch_model.get_exc(mol_torch).numpy()

        y_jax = float(jit_get_exc(jax_model, mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_get_exc_density_jit_matches_pytorch(self, torch_jax_models, random_mol):
        """JIT'd JAX get_exc_density matches PyTorch get_exc_density."""
        torch_model, jax_model = torch_jax_models

        mol_torch = {
            k: torch.from_numpy(v.astype(np.float64)) for k, v in random_mol.items()
        }
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in random_mol.items()}

        with torch.no_grad():
            y_torch = torch_model.get_exc_density(mol_torch).numpy()

        y_jax = np.array(jit_get_exc_density(jax_model, mol_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    def test_nonlocal_jit_matches_pytorch(self, nonlocal_model_with_weights):
        """JIT'd NonLocalModel matches PyTorch NonLocalModel."""
        np.random.seed(42)
        torch_nl, jax_nl = nonlocal_model_with_weights

        num_fine, num_coarse = 20, 3
        h_np = np.random.randn(num_fine, 256).astype(np.float64)
        grid_coords_np = np.random.randn(num_fine, 3).astype(np.float64)
        coarse_coords_np = np.random.randn(num_coarse, 3).astype(np.float64)
        grid_weights_np = (np.abs(np.random.randn(num_fine)) + 0.1).astype(np.float64)

        with torch.no_grad():
            y_torch = torch_nl(
                torch.from_numpy(h_np),
                torch.from_numpy(grid_coords_np),
                torch.from_numpy(coarse_coords_np),
                torch.from_numpy(grid_weights_np),
            ).numpy()

        @eqx.filter_jit
        def jit_nl(model, h, gc, cc, gw):
            return model(h, gc, cc, gw)

        y_jax = np.array(jit_nl(
            jax_nl,
            jnp.array(h_np),
            jnp.array(grid_coords_np),
            jnp.array(coarse_coords_np),
            jnp.array(grid_weights_np),
        ))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)


class TestJITWithPartialCutoff:
    """Test JIT with cases where some pairs are outside the cutoff."""

    def test_sparse_radius_mask(self, torch_jax_models):
        """JIT works when only some pairs are within cutoff."""
        _, jax_model = torch_jax_models

        np.random.seed(123)
        n_points = 15
        # Spread points far apart so some fall outside radius_cutoff=5.0
        mol_jax = {
            "density": jnp.array(np.abs(np.random.randn(2, n_points)) + 0.1),
            "grad": jnp.array(np.random.randn(2, 3, n_points)),
            "kin": jnp.array(np.abs(np.random.randn(2, n_points)) + 0.1),
            "grid_coords": jnp.array(np.random.randn(n_points, 3) * 5.0),
            "grid_weights": jnp.array(np.abs(np.random.randn(n_points)) + 0.1),
            "coarse_0_atomic_coords": jnp.array(np.random.randn(3, 3) * 5.0),
        }

        result = jit_get_exc(jax_model, mol_jax)
        assert jnp.isfinite(result)

        grads = jit_grad_get_exc(jax_model, mol_jax)
        for key in ["density", "grad", "kin", "grid_weights"]:
            assert jnp.all(jnp.isfinite(grads[key]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
