# SPDX-License-Identifier: MIT
"""
Tests for PySCF integration with JAX Skala model.

These tests verify that the JAX model produces consistent results
when used with PySCF for DFT calculations.
"""

import numpy as np
import pytest
import torch

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from pyscf import gto, dft

# PyTorch model loading
from skala.functional import load_functional
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional

# JAX model
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import load_weights_and_buffers_into_model
from skalax.pyscf import JaxSkalaKS
from skalax.pyscf.numint import JaxExcFunction
from skalax.pyscf.features import generate_features
from skalax.pyscf.backend import from_numpy_or_cupy
from skalax.convert_weights import load_weights_from_npz, load_config, get_default_weights_dir


@pytest.fixture(scope="module")
def torch_model():
    """Load PyTorch model with HuggingFace weights."""
    func_torch = load_functional("skala")
    state_dict = {
        k.replace("_traced_model.", ""): v
        for k, v in func_torch.state_dict().items()
    }
    model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    model.load_state_dict(state_dict, strict=True)
    model.double()
    return model


@pytest.fixture(scope="module")
def jax_model(torch_model):
    """Create JAX model with loaded weights."""
    key = jax.random.PRNGKey(0)
    model = JaxSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
    )
    return load_weights_and_buffers_into_model(model, torch_model)


@pytest.fixture(scope="module")
def water_molecule():
    """Create water molecule for testing."""
    return gto.M(
        atom="""
        O 0.000000 0.000000 0.000000
        H 0.757000 0.586000 0.000000
        H -0.757000 0.586000 0.000000
        """,
        basis="sto-3g",
        verbose=0
    )


class TestWeightDtypes:
    """Tests for weight dtype consistency."""

    def test_huggingface_weights_are_float32(self):
        """Verify HuggingFace weights are stored as float32."""
        func_torch = load_functional("skala")
        state_dict = {
            k.replace("_traced_model.", ""): v
            for k, v in func_torch.state_dict().items()
        }
        for k, v in state_dict.items():
            assert v.dtype == torch.float32, f"{k} should be float32"

    def test_torch_model_double_converts_to_float64(self, torch_model):
        """Verify .double() converts PyTorch model to float64."""
        for name, param in torch_model.named_parameters():
            assert param.dtype == torch.float64, f"{name} should be float64"

    def test_jax_model_weights_are_float64(self, jax_model):
        """Verify JAX model weights are float64."""
        assert jax_model.input_linear1.weight.dtype == jnp.float64
        assert jax_model.input_linear1.bias.dtype == jnp.float64
        assert jax_model.output_linear1.weight.dtype == jnp.float64


class TestDirectExcComparison:
    """Tests for direct E_xc comparison without PySCF."""

    def test_exc_matches_with_random_input(self, torch_model, jax_model):
        """Test E_xc matches between PyTorch and JAX with random input."""
        np.random.seed(42)
        n_points = 50
        n_atoms = 3

        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3) * 2,
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(n_atoms, 3) * 2,
        }

        mol_torch = {k: torch.from_numpy(v.astype(np.float64)) for k, v in mol_np.items()}
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}

        with torch.no_grad():
            exc_torch = torch_model.get_exc(mol_torch).numpy()

        exc_jax = np.array(jax_model.get_exc(mol_jax))

        # Should match within ~1e-7 (floating-point ordering differences)
        np.testing.assert_allclose(exc_torch, exc_jax, atol=1e-6)


class TestPySCFFeatureGeneration:
    """Tests for PySCF feature generation."""

    def test_feature_dtypes_are_float64(self, water_molecule):
        """Verify generated features are float64."""
        grids = dft.Grids(water_molecule)
        grids.level = 3
        grids.build()

        dm = dft.RKS(water_molecule).get_init_guess()
        dm_tensor = from_numpy_or_cupy(dm, dtype=torch.float64)

        features = {"density", "grad", "kin", "grid_coords", "grid_weights", "coarse_0_atomic_coords"}
        mol_features = generate_features(
            water_molecule, dm_tensor, grids, features, gpu=False
        )

        for k, v in mol_features.items():
            assert v.dtype == torch.float64, f"{k} should be float64"

    def test_exc_from_pyscf_features_matches(self, torch_model, jax_model, water_molecule):
        """Test E_xc matches when using PySCF-generated features."""
        grids = dft.Grids(water_molecule)
        grids.level = 3
        grids.build()

        dm = dft.RKS(water_molecule).get_init_guess()
        dm_tensor = from_numpy_or_cupy(dm, dtype=torch.float64)

        features = {"density", "grad", "kin", "grid_coords", "grid_weights", "coarse_0_atomic_coords"}
        mol_features = generate_features(
            water_molecule, dm_tensor, grids, features, gpu=False
        )

        # PyTorch
        with torch.no_grad():
            exc_torch = torch_model.get_exc({k: v for k, v in mol_features.items()}).item()

        # JAX
        mol_features_jax = {k: jnp.array(v.detach().numpy()) for k, v in mol_features.items()}
        exc_jax = float(jax_model.get_exc(mol_features_jax))

        # Should match very closely (~1e-9)
        np.testing.assert_allclose(exc_torch, exc_jax, atol=1e-8)


class TestGradientComputation:
    """Tests for gradient computation consistency."""

    def test_gradients_match_between_torch_and_jax(self, torch_model, jax_model):
        """Test gradients dE/d(features) match between PyTorch and JAX."""
        np.random.seed(42)
        n_points = 100
        n_atoms = 3

        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3) * 2,
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(n_atoms, 3) * 2,
        }

        # PyTorch gradients
        mol_torch = {k: torch.from_numpy(v.astype(np.float64)).requires_grad_(True) for k, v in mol_np.items()}
        exc_torch = torch_model.get_exc(mol_torch)
        grads_torch = torch.autograd.grad(exc_torch, list(mol_torch.values()), torch.ones_like(exc_torch))
        grads_torch_dict = {k: g.detach().numpy() for k, g in zip(mol_torch.keys(), grads_torch)}

        # JAX gradients
        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}
        grads_jax = jax.grad(lambda m: jax_model.get_exc(m))(mol_jax)

        # Compare gradients (should match within ~1e-6)
        for key in ["density", "grad", "kin", "grid_weights"]:
            g_torch = grads_torch_dict[key]
            g_jax = np.array(grads_jax[key])
            rel_diff = np.linalg.norm(g_torch - g_jax) / (np.linalg.norm(g_torch) + 1e-10)
            assert rel_diff < 1e-5, f"Gradient for {key} differs: rel_diff={rel_diff}"

    def test_numerical_gradient_matches_analytical(self, jax_model):
        """Test JAX analytical gradient matches numerical gradient."""
        np.random.seed(42)
        n_points = 50
        n_atoms = 3

        mol_np = {
            "density": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grad": np.random.randn(2, 3, n_points),
            "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
            "grid_coords": np.random.randn(n_points, 3) * 2,
            "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
            "coarse_0_atomic_coords": np.random.randn(n_atoms, 3) * 2,
        }

        mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}

        # Analytical gradient
        grads_jax = jax.grad(lambda m: jax_model.get_exc(m))(mol_jax)
        analytical_grad = float(grads_jax["density"][0, 0])

        # Numerical gradient
        eps = 1e-6
        mol_plus = {k: v.copy() for k, v in mol_jax.items()}
        mol_minus = {k: v.copy() for k, v in mol_jax.items()}
        mol_plus["density"] = mol_plus["density"].at[0, 0].add(eps)
        mol_minus["density"] = mol_minus["density"].at[0, 0].add(-eps)

        exc_plus = float(jax_model.get_exc(mol_plus))
        exc_minus = float(jax_model.get_exc(mol_minus))
        numerical_grad = (exc_plus - exc_minus) / (2 * eps)

        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-6)


class TestJaxExcFunction:
    """Tests for the JaxExcFunction autograd bridge."""

    def test_jax_exc_function_forward(self, jax_model, water_molecule):
        """Test JaxExcFunction forward pass."""
        grids = dft.Grids(water_molecule)
        grids.level = 3
        grids.build()

        dm = dft.RKS(water_molecule).get_init_guess()
        dm_tensor = from_numpy_or_cupy(dm, dtype=torch.float64)

        features = {"density", "grad", "kin", "grid_coords", "grid_weights", "coarse_0_atomic_coords"}
        mol_features = generate_features(
            water_molecule, dm_tensor, grids, features, gpu=False
        )

        keys = tuple(mol_features.keys())
        values = tuple(mol_features[k] for k in keys)

        exc = JaxExcFunction.apply(*values, keys, jax_model)
        assert exc.dtype == torch.float64
        assert exc.shape == ()
        assert exc.item() < 0  # E_xc should be negative

    def test_jax_exc_function_backward(self, jax_model, water_molecule):
        """Test JaxExcFunction backward pass produces valid gradients."""
        grids = dft.Grids(water_molecule)
        grids.level = 3
        grids.build()

        dm = dft.RKS(water_molecule).get_init_guess()
        dm_tensor = from_numpy_or_cupy(dm, dtype=torch.float64).requires_grad_()

        features = {"density", "grad", "kin", "grid_coords", "grid_weights", "coarse_0_atomic_coords"}
        mol_features = generate_features(
            water_molecule, dm_tensor, grids, features, gpu=False
        )

        keys = tuple(mol_features.keys())
        values = tuple(mol_features[k] for k in keys)

        exc = JaxExcFunction.apply(*values, keys, jax_model)
        vxc = torch.autograd.grad(exc, dm_tensor, torch.ones_like(exc))[0]

        assert vxc.shape == dm_tensor.shape
        assert vxc.dtype == torch.float64
        assert torch.isfinite(vxc).all()


class TestSCFConvergence:
    """Tests for SCF convergence with JAX model."""

    def test_scf_converges(self, jax_model, water_molecule):
        """Test that SCF converges with JAX model."""
        ks = JaxSkalaKS(water_molecule, xc=jax_model)
        ks.verbose = 0
        ks.conv_tol = 1e-9
        energy = ks.kernel()

        assert ks.converged
        assert energy < 0  # Energy should be negative
        assert energy > -100  # Sanity check

    def test_vxc_matches_at_same_dm(self, torch_model, jax_model, water_molecule):
        """Test V_xc matches between PyTorch and JAX at the same density matrix."""
        from skala.pyscf import SkalaKS

        # Run PyTorch to get converged DM
        ks_torch = SkalaKS(water_molecule, xc="skala")
        ks_torch.verbose = 0
        ks_torch.kernel()
        dm_torch = ks_torch.make_rdm1()

        # Setup JAX KS
        ks_jax = JaxSkalaKS(water_molecule, xc=jax_model)
        ks_jax.verbose = 0
        ks_jax.grids.build()

        # Compare E_xc at the same DM
        n_torch, exc_torch, vxc_torch = ks_torch._numint.nr_rks(
            water_molecule, ks_torch.grids, None, dm_torch
        )
        n_jax, exc_jax, vxc_jax = ks_jax._numint.nr_rks(
            water_molecule, ks_jax.grids, None, dm_torch
        )

        # E_xc should match very closely
        np.testing.assert_allclose(exc_torch, exc_jax, atol=1e-8)

        # V_xc should match closely
        np.testing.assert_allclose(vxc_torch, vxc_jax, atol=1e-7)

    def test_energy_difference_is_small(self, jax_model, water_molecule):
        """Test that energy difference between PyTorch and JAX is small."""
        from skala.pyscf import SkalaKS

        # PyTorch
        ks_torch = SkalaKS(water_molecule, xc="skala")
        ks_torch.verbose = 0
        energy_torch = ks_torch.kernel()

        # JAX
        ks_jax = JaxSkalaKS(water_molecule, xc=jax_model)
        ks_jax.verbose = 0
        energy_jax = ks_jax.kernel()

        # Difference should be < 1 mHa
        diff_ha = abs(energy_torch - energy_jax)
        diff_kcal = diff_ha * 627.5

        assert diff_ha < 0.001, (
            f"Energy difference {diff_ha:.6f} Ha ({diff_kcal:.4f} kcal/mol) "
            "is too large"
        )


class TestLocalWeightLoading:
    """Tests for loading weights from local .npz files."""

    def test_load_config(self):
        """Test loading config from local files."""
        weights_dir = get_default_weights_dir()
        config = load_config(weights_dir)

        assert config["lmax"] == 3
        assert config["non_local"] is True
        assert config["non_local_hidden_nf"] == 16
        assert config["radius_cutoff"] == 5.0

    def test_load_weights_from_npz(self):
        """Test loading weights from local .npz files."""
        weights_dir = get_default_weights_dir()
        config = load_config(weights_dir)

        key = jax.random.PRNGKey(0)
        model = JaxSkalaFunctional(
            lmax=config["lmax"],
            non_local=config["non_local"],
            non_local_hidden_nf=config["non_local_hidden_nf"],
            radius_cutoff=config["radius_cutoff"],
            key=key,
        )
        model = load_weights_from_npz(model, weights_dir)

        # Verify model works
        np.random.seed(42)
        n_points = 20
        mol = {
            "density": jnp.array(np.abs(np.random.randn(2, n_points)) + 0.1),
            "grad": jnp.array(np.random.randn(2, 3, n_points)),
            "kin": jnp.array(np.abs(np.random.randn(2, n_points)) + 0.1),
            "grid_coords": jnp.array(np.random.randn(n_points, 3)),
            "grid_weights": jnp.array(np.abs(np.random.randn(n_points)) + 0.1),
            "coarse_0_atomic_coords": jnp.array(np.random.randn(3, 3)),
        }

        exc = model.get_exc(mol)
        assert exc < 0  # E_xc should be negative
        assert jnp.isfinite(exc)

    def test_local_weights_match_huggingface(self, torch_model, jax_model):
        """Test that local weights produce same results as HuggingFace."""
        weights_dir = get_default_weights_dir()
        config = load_config(weights_dir)

        key = jax.random.PRNGKey(0)
        local_model = JaxSkalaFunctional(
            lmax=config["lmax"],
            non_local=config["non_local"],
            non_local_hidden_nf=config["non_local_hidden_nf"],
            radius_cutoff=config["radius_cutoff"],
            key=key,
        )
        local_model = load_weights_from_npz(local_model, weights_dir)

        # Compare with HuggingFace-loaded model
        np.random.seed(42)
        n_points = 50
        mol = {
            "density": jnp.array(np.abs(np.random.randn(2, n_points)) + 0.1),
            "grad": jnp.array(np.random.randn(2, 3, n_points)),
            "kin": jnp.array(np.abs(np.random.randn(2, n_points)) + 0.1),
            "grid_coords": jnp.array(np.random.randn(n_points, 3) * 2),
            "grid_weights": jnp.array(np.abs(np.random.randn(n_points)) + 0.1),
            "coarse_0_atomic_coords": jnp.array(np.random.randn(3, 3) * 2),
        }

        exc_local = float(local_model.get_exc(mol))
        exc_hf = float(jax_model.get_exc(mol))

        # Should be identical (same weights, just loaded differently)
        np.testing.assert_allclose(exc_local, exc_hf, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
