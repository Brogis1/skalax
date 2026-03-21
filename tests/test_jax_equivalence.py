# SPDX-License-Identifier: MIT
"""
Tests to verify JAX implementation matches PyTorch implementation.
"""

import numpy as np
import pytest
import torch

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# PyTorch implementations
from skala.functional.layers import ScaledSigmoid as TorchScaledSigmoid
from skala.functional.layers import Squasher as TorchSquasher
from skala.functional.base import (
    enhancement_density_inner_product as torch_enhancement,
)
from skala.functional.base import LDA_PREFACTOR as TORCH_LDA_PREFACTOR
from skala.utils.scatter import scatter_sum as torch_scatter_sum
from skala.functional.model import (
    vect_cdist as torch_vect_cdist,
    exp_radial_func as torch_exp_radial_func,
    polynomial_envelope as torch_polynomial_envelope,
    normalization_envelope as torch_normalization_envelope,
    _prepare_features as torch_prepare_features,
)

# JAX implementations
from skalax.functional.layers import ScaledSigmoid as JaxScaledSigmoid
from skalax.functional.layers import Squasher as JaxSquasher
from skalax.functional.base import (
    enhancement_density_inner_product as jax_enhancement,
)
from skalax.functional.base import LDA_PREFACTOR as JAX_LDA_PREFACTOR
from skalax.utils.scatter import scatter_sum as jax_scatter_sum
from skalax.functional.functions import (
    vect_cdist as jax_vect_cdist,
    exp_radial_func as jax_exp_radial_func,
    polynomial_envelope as jax_polynomial_envelope,
    normalization_envelope as jax_normalization_envelope,
    prepare_features as jax_prepare_features,
)


class TestLayers:
    """Test custom layers match between PyTorch and JAX."""

    def test_lda_prefactor_matches(self):
        """Verify LDA prefactor constant is identical."""
        assert TORCH_LDA_PREFACTOR == JAX_LDA_PREFACTOR

    @pytest.mark.parametrize("scale", [1.0, 2.0, 0.5])
    def test_scaled_sigmoid(self, scale):
        """Test ScaledSigmoid matches between PyTorch and JAX."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float64)

        # PyTorch
        torch_layer = TorchScaledSigmoid(scale=scale)
        x_torch = torch.from_numpy(x_np)
        y_torch = torch_layer(x_torch).numpy()

        # JAX
        jax_layer = JaxScaledSigmoid(scale=scale)
        x_jax = jnp.array(x_np)
        y_jax = np.array(jax_layer(x_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)

    @pytest.mark.parametrize("eta", [1e-5, 1e-3, 1e-7])
    def test_squasher(self, eta):
        """Test Squasher matches between PyTorch and JAX."""
        np.random.seed(42)
        x_np = np.random.randn(100).astype(np.float64)

        # PyTorch
        torch_layer = TorchSquasher(eta=eta)
        x_torch = torch.from_numpy(x_np)
        y_torch = torch_layer(x_torch).numpy()

        # JAX
        jax_layer = JaxSquasher(eta=eta)
        x_jax = jnp.array(x_np)
        y_jax = np.array(jax_layer(x_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)


class TestBase:
    """Test base utility functions."""

    def test_enhancement_density_inner_product(self):
        """Test enhancement_density_inner_product matches."""
        np.random.seed(42)

        n_points = 50
        enhancement_factor = np.random.randn(n_points, 1).astype(np.float64)
        density = np.abs(np.random.randn(2, n_points).astype(np.float64)) + 0.1

        # PyTorch
        ef_torch = torch.from_numpy(enhancement_factor)
        density_torch = torch.from_numpy(density)
        y_torch = torch_enhancement(ef_torch, density_torch).numpy()

        # JAX
        ef_jax = jnp.array(enhancement_factor)
        density_jax = jnp.array(density)
        y_jax = np.array(jax_enhancement(ef_jax, density_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)

    def test_enhancement_density_with_negative_density(self):
        """Test enhancement handles negative density (should be clipped)."""
        np.random.seed(42)

        n_points = 50
        enhancement_factor = np.random.randn(n_points, 1).astype(np.float64)
        density = np.random.randn(2, n_points).astype(np.float64)

        # PyTorch
        ef_torch = torch.from_numpy(enhancement_factor)
        density_torch = torch.from_numpy(density)
        y_torch = torch_enhancement(ef_torch, density_torch).numpy()

        # JAX
        ef_jax = jnp.array(enhancement_factor)
        density_jax = jnp.array(density)
        y_jax = np.array(jax_enhancement(ef_jax, density_jax))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)


class TestScatter:
    """Test scatter operations."""

    def test_scatter_sum_1d(self):
        """Test scatter_sum for 1D arrays."""
        np.random.seed(42)

        src = np.random.randn(20).astype(np.float64)
        index = np.array([0, 1, 0, 2, 1, 0, 2, 3, 3, 1,
                         0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int64)
        dim_size = 5

        # PyTorch
        src_torch = torch.from_numpy(src)
        index_torch = torch.from_numpy(index)
        y_torch = torch_scatter_sum(
            src_torch, index_torch, dim=0, dim_size=dim_size
        ).numpy()

        # JAX
        src_jax = jnp.array(src)
        index_jax = jnp.array(index)
        y_jax = np.array(jax_scatter_sum(
            src_jax, index_jax, dim=0, dim_size=dim_size
        ))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)

    def test_scatter_sum_2d(self):
        """Test scatter_sum for 2D arrays."""
        np.random.seed(42)

        src = np.random.randn(10, 8).astype(np.float64)
        index = np.array([0, 1, 0, 2, 1, 0, 2, 3, 3, 1], dtype=np.int64)
        dim_size = 5

        # PyTorch
        src_torch = torch.from_numpy(src)
        index_torch = torch.from_numpy(index)
        y_torch = torch_scatter_sum(
            src_torch, index_torch, dim=0, dim_size=dim_size
        ).numpy()

        # JAX
        src_jax = jnp.array(src)
        index_jax = jnp.array(index)
        y_jax = np.array(jax_scatter_sum(
            src_jax, index_jax, dim=0, dim_size=dim_size
        ))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)


class TestFunctions:
    """Test utility functions match between PyTorch and JAX."""

    def test_vect_cdist(self):
        """Test vect_cdist matches."""
        np.random.seed(42)
        c1 = np.random.randn(10, 3).astype(np.float64)
        c2 = np.random.randn(5, 3).astype(np.float64)

        # PyTorch
        c1_torch = torch.from_numpy(c1)
        c2_torch = torch.from_numpy(c2)
        dir_torch, dist_torch = torch_vect_cdist(c1_torch, c2_torch)
        dir_torch = dir_torch.numpy()
        dist_torch = dist_torch.numpy()

        # JAX
        c1_jax = jnp.array(c1)
        c2_jax = jnp.array(c2)
        dir_jax, dist_jax = jax_vect_cdist(c1_jax, c2_jax)
        dir_jax = np.array(dir_jax)
        dist_jax = np.array(dist_jax)

        np.testing.assert_allclose(dir_torch, dir_jax, atol=1e-10)
        np.testing.assert_allclose(dist_torch, dist_jax, atol=1e-10)

    @pytest.mark.parametrize("num_basis", [8, 16, 32])
    def test_exp_radial_func(self, num_basis):
        """Test exp_radial_func matches."""
        np.random.seed(42)
        dist = np.abs(np.random.randn(50).astype(np.float64)) + 0.1

        # PyTorch
        dist_torch = torch.from_numpy(dist)
        y_torch = torch_exp_radial_func(dist_torch, num_basis).numpy()

        # JAX
        dist_jax = jnp.array(dist)
        y_jax = np.array(jax_exp_radial_func(dist_jax, num_basis))

        # Tolerance of 1e-5 as agreed (differences ~1e-7 due to linspace impl)
        np.testing.assert_allclose(y_torch, y_jax, atol=1e-5)

    @pytest.mark.parametrize("cutoff", [3.0, 5.0, 10.0])
    @pytest.mark.parametrize("p", [2, 4, 8])
    def test_polynomial_envelope(self, cutoff, p):
        """Test polynomial_envelope matches."""
        np.random.seed(42)
        r = np.abs(np.random.randn(50).astype(np.float64)) * cutoff

        # PyTorch
        r_torch = torch.from_numpy(r)
        y_torch = torch_polynomial_envelope(r_torch, cutoff, p).numpy()

        # JAX
        r_jax = jnp.array(r)
        y_jax = np.array(jax_polynomial_envelope(r_jax, cutoff, p))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)

    @pytest.mark.parametrize("cutoff", [3.0, 5.0, 10.0])
    def test_normalization_envelope(self, cutoff):
        """Test normalization_envelope matches."""
        np.random.seed(42)
        r = np.abs(np.random.randn(50).astype(np.float64)) * cutoff

        # PyTorch
        r_torch = torch.from_numpy(r)
        y_torch = torch_normalization_envelope(r_torch, cutoff).numpy()

        # JAX
        r_jax = jnp.array(r)
        y_jax = np.array(jax_normalization_envelope(r_jax, cutoff))

        np.testing.assert_allclose(y_torch, y_jax, atol=1e-10)

    def test_prepare_features(self):
        """Test prepare_features matches."""
        np.random.seed(42)
        n_points = 50

        mol_np = {
            "density": np.abs(np.random.randn(2, n_points).astype(np.float64)),
            "grad": np.random.randn(2, 3, n_points).astype(np.float64),
            "kin": np.abs(np.random.randn(2, n_points).astype(np.float64)),
        }

        # PyTorch
        mol_torch = {
            k: torch.from_numpy(v) for k, v in mol_np.items()
        }
        feat_ab_torch, feat_ba_torch = torch_prepare_features(mol_torch)
        feat_ab_torch = feat_ab_torch.numpy()
        feat_ba_torch = feat_ba_torch.numpy()

        # JAX
        mol_jax = {
            k: jnp.array(v) for k, v in mol_np.items()
        }
        feat_ab_jax, feat_ba_jax = jax_prepare_features(mol_jax)
        feat_ab_jax = np.array(feat_ab_jax)
        feat_ba_jax = np.array(feat_ba_jax)

        np.testing.assert_allclose(feat_ab_torch, feat_ab_jax, atol=1e-10)
        np.testing.assert_allclose(feat_ba_torch, feat_ba_jax, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
