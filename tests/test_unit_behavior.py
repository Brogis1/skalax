# SPDX-License-Identifier: MIT
"""Unit-level behavior checks for skalax (independent of PyTorch).

These tests exercise shapes, boundary values, invariants, and API
contracts of the individual building blocks. Equivalence with the
PyTorch reference is covered in ``test_jax_equivalence.py``; this
file focuses on "does the function behave as advertised" rather than
on chemical accuracy.
"""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import e3nn_jax as e3nn  # noqa: E402

from skalax import (  # noqa: E402
    SkalaFunctional,
    get_default_weights_dir,
    load_config,
    load_weights_from_npz,
)
from skalax.functional.base import (  # noqa: E402
    LDA_PREFACTOR,
    enhancement_density_inner_product,
)
from skalax.functional.functions import (  # noqa: E402
    ANGSTROM_TO_BOHR,
    exp_radial_func,
    normalization_envelope,
    polynomial_envelope,
    prepare_features,
    vect_cdist,
)
from skalax.functional.layers import ScaledSigmoid, Squasher  # noqa: E402
from skalax.functional.model import NonLocalModel, TensorProduct  # noqa: E402
from skalax.utils.scatter import scatter_sum  # noqa: E402


# ---------------------------------------------------------------------------
# Layers: ScaledSigmoid, Squasher
# ---------------------------------------------------------------------------

class TestScaledSigmoid:
    def test_zero_maps_to_half_scale(self):
        # sigmoid(0) = 0.5, so scale * sigmoid(0 / scale) = scale / 2.
        for scale in (0.5, 1.0, 2.0, 5.0):
            layer = ScaledSigmoid(scale=scale)
            y = float(layer(jnp.array(0.0)))
            assert y == pytest.approx(scale / 2)

    def test_output_is_bounded_by_scale(self):
        layer = ScaledSigmoid(scale=2.0)
        x = jnp.linspace(-50.0, 50.0, 201)
        y = np.asarray(layer(x))
        assert np.all((y >= 0.0) & (y <= 2.0))

    def test_saturates_at_scale_for_large_input(self):
        layer = ScaledSigmoid(scale=3.0)
        y_pos = float(layer(jnp.array(1e3)))
        y_neg = float(layer(jnp.array(-1e3)))
        assert y_pos == pytest.approx(3.0, abs=1e-6)
        assert y_neg == pytest.approx(0.0, abs=1e-6)

    def test_preserves_input_shape(self):
        layer = ScaledSigmoid(scale=1.0)
        x = jnp.ones((4, 5, 6))
        assert layer(x).shape == (4, 5, 6)


class TestSquasher:
    def test_zero_input_yields_log_eta(self):
        for eta in (1e-5, 1e-3, 1.0):
            layer = Squasher(eta=eta)
            y = float(layer(jnp.array(0.0)))
            assert y == pytest.approx(np.log(eta))

    def test_symmetric_in_sign(self):
        layer = Squasher(eta=1e-5)
        x = jnp.linspace(-10.0, 10.0, 21)
        y_pos = np.asarray(layer(x))
        y_neg = np.asarray(layer(-x))
        np.testing.assert_allclose(y_pos, y_neg)

    def test_monotone_in_abs(self):
        layer = Squasher(eta=1e-5)
        x = jnp.array([0.0, 0.1, 1.0, 10.0, 100.0])
        y = np.asarray(layer(x))
        assert np.all(np.diff(y) > 0)


# ---------------------------------------------------------------------------
# Utility functions: vect_cdist, radial basis, envelopes, features
# ---------------------------------------------------------------------------

class TestVectCDist:
    def test_shapes(self):
        c1 = jnp.zeros((7, 3))
        c2 = jnp.zeros((4, 3))
        direction, dist = vect_cdist(c1, c2)
        assert direction.shape == (7, 4, 3)
        assert dist.shape == (7, 4)

    def test_directions_are_unit_norm(self):
        rng = np.random.default_rng(0)
        c1 = jnp.asarray(rng.normal(size=(5, 3)))
        c2 = jnp.asarray(rng.normal(size=(6, 3)) + 3.0)
        direction, _ = vect_cdist(c1, c2)
        norms = np.linalg.norm(np.asarray(direction), axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_distances_match_euclidean(self):
        c1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        c2 = jnp.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        _, dist = vect_cdist(c1, c2)
        expected = np.array(
            [
                [0.0, 2.0],
                [1.0, np.sqrt(5.0)],
            ]
        )
        # The `+1e-20` offset introduces a negligible bias at exact zero.
        np.testing.assert_allclose(np.asarray(dist), expected, atol=1e-9)

    def test_gradient_finite_at_zero_distance(self):
        # vect_cdist adds 1e-20 inside sqrt so the gradient is well-defined
        # even when two points coincide. Regression guard.
        def f(p):
            _, d = vect_cdist(p[None, :], jnp.zeros((1, 3)))
            return d.sum()

        g = jax.grad(f)(jnp.zeros(3))
        assert np.all(np.isfinite(np.asarray(g)))


class TestExpRadialFunc:
    @pytest.mark.parametrize("num_basis", [4, 8, 16])
    def test_shape(self, num_basis):
        dist = jnp.array([0.0, 0.5, 1.5, 4.0])
        out = exp_radial_func(dist, num_basis)
        assert out.shape == (4, num_basis)

    def test_non_negative(self):
        dist = jnp.linspace(0.0, 10.0, 50)
        out = np.asarray(exp_radial_func(dist, num_basis=8))
        assert np.all(out >= 0)

    def test_decays_for_large_distance(self):
        near = np.asarray(exp_radial_func(jnp.array([0.5]), num_basis=8))
        far = np.asarray(exp_radial_func(jnp.array([50.0]), num_basis=8))
        assert near.sum() > far.sum()
        assert far.sum() == pytest.approx(0.0, abs=1e-10)

    def test_angstrom_to_bohr_is_self_consistent(self):
        assert ANGSTROM_TO_BOHR == pytest.approx(1.88973, rel=1e-4)


class TestPolynomialEnvelope:
    @pytest.mark.parametrize("cutoff", [1.0, 3.0, 5.0])
    @pytest.mark.parametrize("p", [2, 4, 8])
    def test_boundaries(self, cutoff, p):
        # At r=0 the envelope is 1; at r>=cutoff it is 0.
        at_zero = float(polynomial_envelope(jnp.array(0.0), cutoff, p))
        at_cutoff = float(polynomial_envelope(jnp.array(cutoff), cutoff, p))
        beyond = float(
            polynomial_envelope(jnp.array(2 * cutoff), cutoff, p),
        )
        assert at_zero == pytest.approx(1.0)
        assert at_cutoff == pytest.approx(0.0, abs=1e-12)
        assert beyond == pytest.approx(0.0, abs=1e-12)

    def test_monotone_decreasing_in_range(self):
        r = jnp.linspace(0.0, 5.0, 101)
        y = np.asarray(polynomial_envelope(r, cutoff=5.0, p=8))
        assert np.all(np.diff(y) <= 1e-12)

    def test_stays_in_unit_interval(self):
        r = jnp.linspace(0.0, 10.0, 500)
        y = np.asarray(polynomial_envelope(r, cutoff=5.0, p=6))
        assert np.all((y >= -1e-12) & (y <= 1.0 + 1e-12))


class TestNormalizationEnvelope:
    def test_endpoints(self):
        assert float(
            normalization_envelope(jnp.array(0.0), cutoff=5.0),
        ) == pytest.approx(1.0)
        assert float(
            normalization_envelope(jnp.array(5.0), cutoff=5.0),
        ) == pytest.approx(0.0, abs=1e-12)

    def test_clipped_beyond_cutoff(self):
        y = float(
            normalization_envelope(jnp.array(12.0), cutoff=5.0),
        )
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_shape_preserved(self):
        r = jnp.linspace(0.0, 10.0, 37)
        y = normalization_envelope(r, cutoff=5.0)
        assert y.shape == (37,)


class TestPrepareFeatures:
    def _make_mol(self, n_points=11, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "density": jnp.asarray(
                rng.uniform(0.01, 1.0, size=(2, n_points)),
            ),
            "grad": jnp.asarray(rng.normal(size=(2, 3, n_points))),
            "kin": jnp.asarray(
                rng.uniform(0.0, 1.0, size=(2, n_points)),
            ),
        }

    def test_output_shapes(self):
        mol = self._make_mol(n_points=13)
        ab, ba = prepare_features(mol)
        # Seven scalar features per grid point.
        assert ab.shape == (13, 7)
        assert ba.shape == (13, 7)

    def test_spin_swap_relationship(self):
        # The 7 features are ordered so columns (0,1), (2,3), (4,5) form
        # spin pairs and column 6 is the α+β cross term. prepare_features
        # returns an (ab) ordering and its spin-swapped (ba) view.
        mol = self._make_mol()
        ab, ba = prepare_features(mol)
        ab_np = np.asarray(ab)
        ba_np = np.asarray(ba)

        np.testing.assert_allclose(ba_np[:, 0], ab_np[:, 1])
        np.testing.assert_allclose(ba_np[:, 1], ab_np[:, 0])
        np.testing.assert_allclose(ba_np[:, 2], ab_np[:, 3])
        np.testing.assert_allclose(ba_np[:, 3], ab_np[:, 2])
        np.testing.assert_allclose(ba_np[:, 4], ab_np[:, 5])
        np.testing.assert_allclose(ba_np[:, 5], ab_np[:, 4])
        # Cross term is symmetric under the swap.
        np.testing.assert_allclose(ba_np[:, 6], ab_np[:, 6])

    def test_dtype_is_float64_when_x64_enabled(self):
        mol = self._make_mol()
        ab, _ = prepare_features(mol)
        assert ab.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Base: enhancement_density_inner_product
# ---------------------------------------------------------------------------

class TestEnhancementDensityInnerProduct:
    def test_shape(self):
        n = 17
        enhancement = jnp.ones((n, 1))
        density = jnp.ones((2, n)) * 0.3
        out = enhancement_density_inner_product(enhancement, density)
        assert out.shape == (n,)

    def test_matches_lda_for_unit_enhancement(self):
        # With enhancement_factor == 1, the result is the plain spin-
        # polarized LDA exchange density.
        n = 5
        density = jnp.ones((2, n)) * 0.25
        enhancement = jnp.ones((n, 1))
        out = np.asarray(
            enhancement_density_inner_product(enhancement, density),
        )
        expected = LDA_PREFACTOR * (2 * 0.25 ** (4 / 3)) * np.ones(n)
        np.testing.assert_allclose(out, expected)

    def test_zero_density_gives_zero(self):
        n = 4
        enhancement = jnp.ones((n, 1)) * 7.0
        density = jnp.zeros((2, n))
        out = np.asarray(
            enhancement_density_inner_product(enhancement, density),
        )
        np.testing.assert_allclose(out, np.zeros(n))

    def test_negative_density_is_clipped(self):
        # Density below zero (e.g. numerical noise on small grids) must
        # not produce complex or NaN output.
        n = 3
        enhancement = jnp.ones((n, 1))
        density = -jnp.ones((2, n))
        out = np.asarray(
            enhancement_density_inner_product(enhancement, density),
        )
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, np.zeros(n))


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

class TestScatterSum:
    def test_1d_explicit_dim_size(self):
        src = jnp.array([1.0, 2.0, 3.0, 4.0])
        index = jnp.array([0, 0, 1, 2])
        out = np.asarray(scatter_sum(src, index, dim=0, dim_size=3))
        np.testing.assert_allclose(out, [3.0, 3.0, 4.0])

    def test_1d_infers_dim_size(self):
        src = jnp.array([1.0, 2.0, 3.0])
        index = jnp.array([0, 2, 2])
        out = np.asarray(scatter_sum(src, index, dim=0))
        np.testing.assert_allclose(out, [1.0, 0.0, 5.0])

    def test_2d_scatter_along_rows(self):
        src = jnp.arange(12, dtype=jnp.float64).reshape(4, 3)
        index = jnp.array([0, 1, 0, 1])
        out = np.asarray(scatter_sum(src, index, dim=0, dim_size=2))
        expected = np.zeros((2, 3))
        expected[0] = np.arange(0, 3) + np.arange(6, 9)
        expected[1] = np.arange(3, 6) + np.arange(9, 12)
        np.testing.assert_allclose(out, expected)

    def test_negative_dim_indexes_from_end(self):
        src = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = jnp.array([0, 1, 0])
        out_neg = np.asarray(scatter_sum(src, index, dim=-1, dim_size=2))
        out_pos = np.asarray(scatter_sum(src, index, dim=1, dim_size=2))
        np.testing.assert_allclose(out_neg, out_pos)

    def test_dtype_preserved(self):
        src = jnp.array([1.0, 2.0], dtype=jnp.float32)
        index = jnp.array([0, 1])
        out = scatter_sum(src, index, dim=0, dim_size=2)
        assert out.dtype == jnp.float32


# ---------------------------------------------------------------------------
# TensorProduct: init reproducibility and shape
# ---------------------------------------------------------------------------

class TestTensorProductInit:
    def _make_tp(self, key):
        irreps_in1 = e3nn.Irreps("4x0e + 2x1e")
        irreps_in2 = e3nn.Irreps.spherical_harmonics(1, p=1)
        irreps_out = e3nn.Irreps("4x0e + 2x1e")
        return TensorProduct(
            irreps_in1, irreps_in2, irreps_out, key=key,
        )

    def test_init_is_deterministic_from_key(self):
        # Regression guard: an earlier version seeded weights from
        # ``hash(str)``, so the init varied per process. ``jax.random``
        # must make it fully reproducible from the input key.
        key = jax.random.PRNGKey(123)
        tp1 = self._make_tp(key)
        tp2 = self._make_tp(key)
        for name in tp1.weights:
            np.testing.assert_array_equal(
                np.asarray(tp1.weights[name]),
                np.asarray(tp2.weights[name]),
            )

    def test_different_keys_give_different_weights(self):
        tp1 = self._make_tp(jax.random.PRNGKey(0))
        tp2 = self._make_tp(jax.random.PRNGKey(1))
        any_different = any(
            not np.allclose(
                np.asarray(tp1.weights[k]),
                np.asarray(tp2.weights[k]),
            )
            for k in tp1.weights
        )
        assert any_different

    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        tp = self._make_tp(key)
        x1 = jnp.zeros((5, tp.irreps_in1.dim))
        x2 = jnp.zeros((5, tp.irreps_in2.dim))
        y = tp(x1, x2)
        assert y.shape == (5, tp.irreps_out.dim)


# ---------------------------------------------------------------------------
# SkalaFunctional: dtype, no-nonlocal path, output shapes
# ---------------------------------------------------------------------------

class TestSkalaFunctionalApi:
    def _mol(self, n_points=32, n_atoms=2, seed=0):
        rng = np.random.default_rng(seed)
        return {
            "density": jnp.asarray(
                rng.uniform(0.01, 0.5, size=(2, n_points)),
            ),
            "grad": jnp.asarray(
                rng.normal(size=(2, 3, n_points)) * 0.1,
            ),
            "kin": jnp.asarray(
                rng.uniform(0.0, 0.3, size=(2, n_points)),
            ),
            "grid_coords": jnp.asarray(
                rng.uniform(-3.0, 3.0, size=(n_points, 3)),
            ),
            "grid_weights": jnp.asarray(
                rng.uniform(0.01, 0.1, size=n_points),
            ),
            "coarse_0_atomic_coords": jnp.asarray(
                rng.uniform(-1.0, 1.0, size=(n_atoms, 3)),
            ),
        }

    def test_dtype_property_reflects_input_layer(self):
        model = SkalaFunctional(
            lmax=2,
            non_local=False,
            key=jax.random.PRNGKey(0),
        )
        assert model.dtype == model.input_linear1.weight.dtype

    def test_no_nonlocal_branch_is_absent(self):
        model = SkalaFunctional(
            lmax=2,
            non_local=False,
            key=jax.random.PRNGKey(0),
        )
        assert model.non_local_model is None
        assert model.num_non_local_contributions == 0

    def test_get_exc_density_shape(self):
        model = SkalaFunctional(
            lmax=2,
            non_local=False,
            key=jax.random.PRNGKey(0),
        )
        mol = self._mol(n_points=21)
        out = model.get_exc_density(mol)
        assert out.shape == (21,)

    def test_get_exc_is_scalar(self):
        model = SkalaFunctional(
            lmax=2,
            non_local=False,
            key=jax.random.PRNGKey(0),
        )
        mol = self._mol(n_points=17)
        out = model.get_exc(mol)
        assert out.shape == ()
        assert np.isfinite(float(out))

    def test_get_exc_is_integral_of_density(self):
        # By construction, ``get_exc == sum(exc_density * grid_weights)``.
        model = SkalaFunctional(
            lmax=2,
            non_local=False,
            key=jax.random.PRNGKey(0),
        )
        mol = self._mol(n_points=23)
        exc = float(model.get_exc(mol))
        exc_density = np.asarray(model.get_exc_density(mol))
        weights = np.asarray(mol["grid_weights"])
        np.testing.assert_allclose(
            exc, (exc_density * weights).sum(), rtol=1e-12,
        )


class TestNonLocalPaddedVsEager:
    """The JIT-safe padded-sparse path must match the eager path."""

    def test_padded_matches_eager_on_small_grid(self):
        key = jax.random.PRNGKey(7)
        nl = NonLocalModel(
            input_nf=32,
            hidden_nf=8,
            lmax=2,
            radius_cutoff=3.0,
            key=key,
        )

        rng = np.random.default_rng(0)
        num_fine, num_coarse = 12, 3
        h = jnp.asarray(rng.normal(size=(num_fine, 32)))
        grid_coords = jnp.asarray(rng.normal(size=(num_fine, 3)))
        coarse_coords = jnp.asarray(rng.normal(size=(num_coarse, 3)))
        grid_weights = jnp.asarray(
            np.abs(rng.normal(size=num_fine)) + 0.1,
        )

        out_padded = np.asarray(
            nl(h, grid_coords, coarse_coords, grid_weights),
        )
        out_eager = np.asarray(
            nl.forward_eager(
                h, grid_coords, coarse_coords, grid_weights,
            ),
        )
        np.testing.assert_allclose(out_padded, out_eager, atol=1e-10)


# ---------------------------------------------------------------------------
# Weight loading: load_config default, get_default_weights_dir, npz round-trip
# ---------------------------------------------------------------------------

class TestWeightLoading:
    def test_default_weights_dir_exists(self):
        weights_dir = get_default_weights_dir()
        assert Path(weights_dir).is_dir()
        assert (Path(weights_dir) / "skala_weights.npz").is_file()

    def test_load_config_returns_default_when_missing(self, tmp_path):
        cfg = load_config(tmp_path)
        assert cfg == {
            "lmax": 3,
            "non_local": True,
            "non_local_hidden_nf": 16,
            "radius_cutoff": 5.0,
        }

    def test_load_config_reads_json(self, tmp_path):
        custom = {
            "lmax": 2,
            "non_local": False,
            "non_local_hidden_nf": 8,
            "radius_cutoff": 2.5,
        }
        (tmp_path / "config.json").write_text(json.dumps(custom))
        assert load_config(tmp_path) == custom

    def test_load_weights_from_npz_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_weights_from_npz(object(), tmp_path)

    def test_loaded_model_is_reproducible_under_get_exc(self):
        # Two independently loaded models with the bundled weights must
        # produce identical energies on the same input.
        weights_dir = get_default_weights_dir()
        cfg = load_config(weights_dir)

        def _build():
            m = SkalaFunctional(
                lmax=cfg["lmax"],
                non_local=cfg["non_local"],
                non_local_hidden_nf=cfg["non_local_hidden_nf"],
                radius_cutoff=cfg["radius_cutoff"],
                key=jax.random.PRNGKey(0),
            )
            return load_weights_from_npz(m, weights_dir)

        m1 = _build()
        m2 = _build()

        rng = np.random.default_rng(42)
        mol = {
            "density": jnp.asarray(
                rng.uniform(0.01, 0.5, size=(2, 40)),
            ),
            "grad": jnp.asarray(rng.normal(size=(2, 3, 40)) * 0.1),
            "kin": jnp.asarray(
                rng.uniform(0.0, 0.3, size=(2, 40)),
            ),
            "grid_coords": jnp.asarray(
                rng.uniform(-2.0, 2.0, size=(40, 3)),
            ),
            "grid_weights": jnp.asarray(
                rng.uniform(0.01, 0.05, size=40),
            ),
            "coarse_0_atomic_coords": jnp.asarray(
                rng.uniform(-1.0, 1.0, size=(3, 3)),
            ),
        }
        e1 = float(m1.get_exc(mol))
        e2 = float(m2.get_exc(mol))
        assert e1 == e2


# ---------------------------------------------------------------------------
# PySCF factory dispatcher
# ---------------------------------------------------------------------------

class TestSkalaFunctionalWithNonLocal:
    """Random-weight non-local path runs and returns sensible shapes."""

    def _mol(self, n_points=24, n_atoms=3, seed=1):
        rng = np.random.default_rng(seed)
        return {
            "density": jnp.asarray(
                rng.uniform(0.01, 0.3, size=(2, n_points)),
            ),
            "grad": jnp.asarray(
                rng.normal(size=(2, 3, n_points)) * 0.1,
            ),
            "kin": jnp.asarray(
                rng.uniform(0.0, 0.2, size=(2, n_points)),
            ),
            "grid_coords": jnp.asarray(
                rng.uniform(-2.0, 2.0, size=(n_points, 3)),
            ),
            "grid_weights": jnp.asarray(
                rng.uniform(0.01, 0.1, size=n_points),
            ),
            "coarse_0_atomic_coords": jnp.asarray(
                rng.uniform(-1.0, 1.0, size=(n_atoms, 3)),
            ),
        }

    def test_nonlocal_random_weights_runs(self):
        model = SkalaFunctional(
            lmax=2,
            non_local=True,
            non_local_hidden_nf=8,
            radius_cutoff=3.0,
            key=jax.random.PRNGKey(0),
        )
        mol = self._mol(n_points=20, n_atoms=2)
        exc = float(model.get_exc(mol))
        assert np.isfinite(exc)

    def test_grad_get_exc_is_finite_per_feature(self):
        model = SkalaFunctional(
            lmax=2,
            non_local=False,
            key=jax.random.PRNGKey(0),
        )
        mol = self._mol(n_points=15)
        grads = jax.grad(model.get_exc)(mol)
        # Only float inputs carry meaningful gradients.
        for k in ("density", "grad", "kin", "grid_weights"):
            arr = np.asarray(grads[k])
            assert arr.shape == np.asarray(mol[k]).shape
            assert np.all(np.isfinite(arr)), f"non-finite grad for {k}"


class TestLdaPrefactorValue:
    def test_matches_closed_form(self):
        expected = -(2 ** (1 / 3)) * (3 / 4) * (3 / np.pi) ** (1 / 3)
        assert LDA_PREFACTOR == pytest.approx(expected, rel=1e-12)


class TestJaxSkalaKSDispatcher:
    def test_dispatch_by_spin(self):
        pytest.importorskip("pyscf")
        from pyscf import gto
        from skalax.pyscf import JaxSkalaKS, JaxSkalaRKS, JaxSkalaUKS

        weights_dir = get_default_weights_dir()
        cfg = load_config(weights_dir)
        model = SkalaFunctional(
            lmax=cfg["lmax"],
            non_local=cfg["non_local"],
            non_local_hidden_nf=cfg["non_local_hidden_nf"],
            radius_cutoff=cfg["radius_cutoff"],
            key=jax.random.PRNGKey(0),
        )
        model = load_weights_from_npz(model, weights_dir)

        mol_rks = gto.M(atom="H 0 0 0; H 0 0 1.0", basis="sto-3g", spin=0)
        mol_uks = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)

        assert isinstance(JaxSkalaKS(mol_rks, xc=model), JaxSkalaRKS)
        assert isinstance(JaxSkalaKS(mol_uks, xc=model), JaxSkalaUKS)
