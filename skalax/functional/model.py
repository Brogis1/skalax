# SPDX-License-Identifier: MIT
"""
Main model implementation for Skala JAX.

Contains SkalaFunctional, NonLocalModel, and TensorProduct classes.
"""

from typing import Optional
import math

import jax
import jax.numpy as jnp
import equinox as eqx
import e3nn_jax as e3nn

from skalax.functional.layers import ScaledSigmoid
from skalax.functional.base import enhancement_density_inner_product
from skalax.functional.functions import (
    vect_cdist,
    exp_radial_func,
    polynomial_envelope,
    normalization_envelope,
    prepare_features,
)
from skalax.utils.scatter import scatter_sum


class TensorProduct(eqx.Module):
    """
    Equivariant tensor product with learnable weights.

    Implements the same tensor product as the PyTorch e3nn version,
    with weights for each valid (i_1, i_2, i_out) instruction.
    """

    irreps_in1: e3nn.Irreps = eqx.field(static=True)
    irreps_in2: e3nn.Irreps = eqx.field(static=True)
    irreps_out: e3nn.Irreps = eqx.field(static=True)
    instructions: list = eqx.field(static=True)
    slices: list = eqx.field(static=True)

    # Learnable weights stored as a dict
    weights: dict

    # Wigner 3j coefficients (not trainable, but not static to avoid warnings)
    w3j: dict

    def __init__(
        self,
        irreps_in1: e3nn.Irreps,
        irreps_in2: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        *,
        key: jax.Array,
    ):
        """
        Initialize tensor product.

        Parameters
        ----------
        irreps_in1 : Irreps
            First input irreps.
        irreps_in2 : Irreps
            Second input irreps (typically spherical harmonics).
        irreps_out : Irreps
            Output irreps.
        key : jax.Array
            PRNG key for weight initialization.
        """
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        # Build instructions: (i_1, i_2, i_out) where ir_out in ir_1 * ir_2
        instructions = []
        for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
            for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                    if ir_out in ir_1 * ir_2:
                        instructions.append((i_1, i_2, i_out))
        self.instructions = instructions

        # Compute slices for each irrep
        def compute_slices(irreps):
            slices = []
            start = 0
            for mul, ir in irreps:
                end = start + mul * ir.dim
                slices.append((start, end))
                start = end
            return slices

        self.slices = [
            compute_slices(irreps_in1),
            compute_slices(irreps_in2),
            compute_slices(irreps_out),
        ]

        # Initialize weights
        weights = {}
        w3j = {}
        for i_1, i_2, i_out in instructions:
            mul_1, ir_1 = irreps_in1[i_1]
            mul_2, ir_2 = irreps_in2[i_2]
            mul_out, ir_out = irreps_out[i_out]

            # Weight shape: (mul_1, mul_2, mul_out)
            key, subkey = jax.random.split(key)
            w = jax.random.normal(subkey, (mul_1, mul_2, mul_out))
            weights[f"weight_{i_1}_{i_2}_{i_out}"] = w

            # Wigner 3j coefficient
            w3j_val = e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)
            # Permute from (j, i, k) to (k, i, j) to match PyTorch o3.wigner_3j
            w3j_val = jnp.transpose(w3j_val, (2, 0, 1))
            w3j[f"w3j_{i_1}_{i_2}_{i_out}"] = w3j_val

        self.weights = weights
        self.w3j = w3j

        # Reset parameters to match PyTorch initialization
        self.weights = self._reset_parameters(self.weights)

    def _reset_parameters(self, weights: dict) -> dict:
        """Initialize weights with proper scaling."""
        new_weights = {}
        for ins in self.instructions:
            i_1, i_2, i_out = ins

            # Count number of paths feeding into this output
            num_in = sum(
                self.irreps_in1[ins_[0]].mul * self.irreps_in2[ins_[1]].mul
                for ins_ in self.instructions
                if ins_[2] == ins[2]
            )
            num_out = self.irreps_out[ins[2]].mul
            x = (6 / (num_in + num_out)) ** 0.5

            key_name = f"weight_{i_1}_{i_2}_{i_out}"
            shape = weights[key_name].shape
            new_weights[key_name] = jax.random.uniform(
                jax.random.PRNGKey(hash(key_name) % (2**31)),
                shape,
                minval=-x,
                maxval=x,
            )

        return new_weights

    def __call__(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        """
        Apply tensor product.

        Parameters
        ----------
        x1 : Array
            First input, shape (m, irreps_in1.dim).
        x2 : Array
            Second input, shape (m, irreps_in2.dim).

        Returns
        -------
        Array
            Output, shape (m, irreps_out.dim).
        """
        m = x1.shape[0]
        outs = []

        for i_1, i_2, i_out in self.instructions:
            mul_1, ir_1 = self.irreps_in1[i_1]
            mul_2, ir_2 = self.irreps_in2[i_2]
            mul_out, ir_out = self.irreps_out[i_out]

            l1, l2, l3 = ir_1.l, ir_2.l, ir_out.l

            # Slice inputs
            s1_start, s1_end = self.slices[0][i_1]
            s2_start, s2_end = self.slices[1][i_2]
            x1_i = x1[:, s1_start:s1_end]
            x2_i = x2[:, s2_start:s2_end]

            w = self.weights[f"weight_{i_1}_{i_2}_{i_out}"]
            w3j = self.w3j[f"w3j_{i_1}_{i_2}_{i_out}"]

            # Compute tensor product based on l values
            if (l1, l2, l3) == (0, 0, 0):
                # Scalar x Scalar -> Scalar
                out = jnp.einsum("mu,uvw,mv->mw", x1_i, w, x2_i)
            elif l1 == 0:
                # Scalar x Vector -> Vector
                x2_view = x2_i.reshape(m, mul_2, ir_2.dim)
                out = jnp.einsum(
                    "mu,uvw,mvj->mwj", x1_i, w, x2_view
                ).reshape(m, mul_out * ir_out.dim)
                out = out / math.sqrt(ir_out.dim)
            elif l2 == 0:
                # Vector x Scalar -> Vector
                x1_view = x1_i.reshape(m, mul_1, ir_1.dim)
                out = jnp.einsum(
                    "mui,uvw,mv->mwi", x1_view, w, x2_i
                ).reshape(m, mul_out * ir_out.dim)
                out = out / math.sqrt(ir_out.dim)
            elif l3 == 0:
                # Vector x Vector -> Scalar
                x1_view = x1_i.reshape(m, mul_1, ir_1.dim)
                x2_view = x2_i.reshape(m, mul_2, ir_2.dim)
                out = jnp.einsum(
                    "mui,uvw,mvi->mw", x1_view, w, x2_view
                )
                out = out / math.sqrt(ir_1.dim)
            else:
                # General case with Wigner 3j
                x1_view = x1_i.reshape(m, mul_1, ir_1.dim)
                x2_view = x2_i.reshape(m, mul_2, ir_2.dim)
                out = jnp.einsum(
                    "mui,uvw,mvj,kij->mwk", x1_view, w, x2_view, w3j
                ).reshape(m, mul_out * ir_out.dim)

            outs.append((i_out, out))

        # Sum outputs for each i_out
        result_parts = []
        for i_out, (mul, ir) in enumerate(self.irreps_out):
            if mul > 0:
                parts = [out for idx, out in outs if idx == i_out]
                if parts:
                    result_parts.append(sum(parts))

        if len(result_parts) > 1:
            return jnp.concatenate(result_parts, axis=-1)
        else:
            return result_parts[0]


class NonLocalModel(eqx.Module):
    """
    Non-local model for message passing between grid points.

    Uses spherical harmonics and tensor products for equivariant processing.
    """

    input_nf: int = eqx.field(static=True)
    hidden_nf: int = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    radius_cutoff: float = eqx.field(static=True)

    in_irreps: e3nn.Irreps = eqx.field(static=True)
    out_irreps: e3nn.Irreps = eqx.field(static=True)
    hidden_irreps: e3nn.Irreps = eqx.field(static=True)
    sph_irreps: e3nn.Irreps = eqx.field(static=True)

    pre_down_linear: eqx.nn.Linear
    post_up_linear: eqx.nn.Linear
    tp_down: TensorProduct
    tp_up: TensorProduct

    def __init__(
        self,
        input_nf: int,
        hidden_nf: int,
        lmax: int,
        radius_cutoff: float = float("inf"),
        *,
        key: jax.Array,
    ):
        """
        Initialize non-local model.

        Parameters
        ----------
        input_nf : int
            Number of input features.
        hidden_nf : int
            Number of hidden features.
        lmax : int
            Maximum angular momentum.
        radius_cutoff : float
            Cutoff radius for interactions.
        key : jax.Array
            PRNG key.
        """
        self.input_nf = input_nf
        self.hidden_nf = hidden_nf
        self.lmax = lmax
        self.radius_cutoff = radius_cutoff

        # Irreps
        self.in_irreps = e3nn.Irreps(f"{hidden_nf}x0e")
        self.out_irreps = e3nn.Irreps(f"{hidden_nf}x0e")
        self.hidden_irreps = e3nn.Irreps(
            "+".join([f"{hidden_nf}x{i}e" for i in range(lmax + 1)])
        )
        self.sph_irreps = e3nn.Irreps.spherical_harmonics(lmax, p=1)

        # Split keys
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        # Pre-down layer
        self.pre_down_linear = eqx.nn.Linear(input_nf, hidden_nf, key=k1)

        # Tensor products
        self.tp_down = TensorProduct(
            self.in_irreps,
            self.sph_irreps,
            self.hidden_irreps,
            key=k2,
        )
        self.tp_up = TensorProduct(
            self.hidden_irreps,
            self.sph_irreps,
            self.out_irreps,
            key=k3,
        )

        # Post-up layer
        self.post_up_linear = eqx.nn.Linear(hidden_nf, hidden_nf, key=k4)

    def __call__(
        self,
        h: jax.Array,
        grid_coords: jax.Array,
        coarse_coords: jax.Array,
        grid_weights: jax.Array,
    ) -> jax.Array:
        """
        Forward pass (JIT-compatible).

        Uses a padded-sparse approach: edge indices are allocated at the
        static upper-bound size ``num_fine * num_coarse`` and unused
        (padding) entries are zeroed out via an edge mask.  All tensor
        shapes are therefore statically known, making this method safe
        to use inside ``jax.jit``.

        Parameters
        ----------
        h : Array
            Input features, shape (num_fine, input_nf).
        grid_coords : Array
            Fine grid coordinates, shape (num_fine, 3).
        coarse_coords : Array
            Coarse grid coordinates, shape (num_coarse, 3).
        grid_weights : Array
            Grid weights, shape (num_fine,).

        Returns
        -------
        Array
            Output features, shape (num_fine, hidden_nf).
        """
        # Pre-down layer (batched matmul)
        h = jax.nn.silu(
            h @ self.pre_down_linear.weight.T + self.pre_down_linear.bias
        )

        # Compute distances and directions
        directions, distances = vect_cdist(grid_coords, coarse_coords)

        # Envelope for normalization
        if self.radius_cutoff != float("inf"):
            up_weight = normalization_envelope(distances, self.radius_cutoff)
        else:
            up_weight = jnp.ones_like(distances)

        # --- Padded-sparse edge construction (JIT-safe) ---
        num_fine, num_coarse = distances.shape
        max_edges = num_fine * num_coarse
        radius_mask = distances <= self.radius_cutoff

        # Static-size edge list; padding entries filled with index 0
        # jnp.argwhere places real entries first, padding at the end.
        edge_indices = jnp.argwhere(radius_mask, size=max_edges, fill_value=0)
        edge_fine_idx = edge_indices[:, 0]
        edge_coarse_idx = edge_indices[:, 1]

        # 1 for real edges, 0 for padding.
        # Cannot use radius_mask[edge_fine_idx, edge_coarse_idx] because the
        # fill index (0,0) may itself be within the cutoff radius.  Instead,
        # rely on argwhere's ordering guarantee: the first n_real entries are
        # real, the rest are padding.
        n_real = radius_mask.sum()
        edge_mask = (jnp.arange(max_edges) < n_real).astype(h.dtype)

        # Gather using static-shape indices
        edge_directions = directions[edge_fine_idx, edge_coarse_idx]
        edge_distances = distances[edge_fine_idx, edge_coarse_idx]
        up_weight_edges = up_weight[edge_fine_idx, edge_coarse_idx] * edge_mask

        # Radial features (mask out padding)
        edge_dist_ft = exp_radial_func(edge_distances, self.hidden_nf)
        edge_dist_ft = edge_dist_ft * edge_mask[:, None]

        # Envelope for smoothness
        if self.radius_cutoff != float("inf"):
            envelope = polynomial_envelope(edge_distances, self.radius_cutoff, 8)
            edge_dist_ft = edge_dist_ft * envelope[:, None]

        # Spherical harmonics
        edge_direction_ft = e3nn.spherical_harmonics(
            self.sph_irreps,
            edge_directions,
            normalize=False,
            normalization="norm",
        ).array
        edge_direction_ft = edge_direction_ft * edge_mask[:, None]

        # Process fine -> coarse
        edge_h = h[edge_fine_idx]
        down = self.tp_down(edge_h, edge_direction_ft)
        down = self._mul_repeat(edge_dist_ft, down, self.hidden_irreps)

        # Scatter to coarse points
        down_weighted = (
            down.astype(jnp.float64)
            * grid_weights[edge_fine_idx, None].astype(jnp.float64)
        )
        h_coarse = scatter_sum(
            down_weighted,
            edge_coarse_idx,
            dim=0,
            dim_size=coarse_coords.shape[0],
        ).astype(h.dtype)

        # Process coarse -> fine
        edge_coarse_ft = h_coarse[edge_coarse_idx]
        up = self.tp_up(edge_coarse_ft, edge_direction_ft)

        # Compute normalization
        denom = scatter_sum(
            up_weight_edges,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )[edge_fine_idx]
        up_weight_normalized = up_weight_edges / (denom + 0.1)
        up = self._mul_repeat(
            edge_dist_ft * up_weight_normalized[:, None], up, self.out_irreps
        )

        # Scatter back to fine points
        h_fine = scatter_sum(
            up,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )

        # Post-up layer (batched matmul)
        h_fine = jax.nn.silu(
            h_fine @ self.post_up_linear.weight.T + self.post_up_linear.bias
        )

        return h_fine

    def _forward_eager(
        self,
        h: jax.Array,
        grid_coords: jax.Array,
        coarse_coords: jax.Array,
        grid_weights: jax.Array,
    ) -> jax.Array:
        """
        Forward pass (original eager implementation, not JIT-compatible).

        Uses boolean indexing and data-dependent shapes.  Kept for
        reference and testing against the JIT-compatible ``__call__``.

        Parameters
        ----------
        h : Array
            Input features, shape (num_fine, input_nf).
        grid_coords : Array
            Fine grid coordinates, shape (num_fine, 3).
        coarse_coords : Array
            Coarse grid coordinates, shape (num_coarse, 3).
        grid_weights : Array
            Grid weights, shape (num_fine,).

        Returns
        -------
        Array
            Output features, shape (num_fine, hidden_nf).
        """
        # Pre-down layer (batched matmul)
        h = jax.nn.silu(
            h @ self.pre_down_linear.weight.T + self.pre_down_linear.bias
        )

        # Compute distances and directions
        directions, distances = vect_cdist(grid_coords, coarse_coords)

        # Envelope for normalization
        if self.radius_cutoff != float("inf"):
            up_weight = normalization_envelope(distances, self.radius_cutoff)
        else:
            up_weight = jnp.ones_like(distances)

        # Find edges within radius cutoff
        radius_mask = distances <= self.radius_cutoff
        edge_directions = directions[radius_mask]
        edge_distances = distances[radius_mask]
        up_weight_edges = up_weight[radius_mask]

        # Get edge indices
        edge_indices = jnp.argwhere(radius_mask, size=radius_mask.sum())
        edge_fine_idx = edge_indices[:, 0]
        edge_coarse_idx = edge_indices[:, 1]

        # Radial features
        edge_dist_ft = exp_radial_func(edge_distances, self.hidden_nf)

        # Envelope for smoothness
        if self.radius_cutoff != float("inf"):
            envelope = polynomial_envelope(edge_distances, self.radius_cutoff, 8)
            edge_dist_ft = edge_dist_ft * envelope[:, None]

        # Spherical harmonics
        edge_direction_ft = e3nn.spherical_harmonics(
            self.sph_irreps,
            edge_directions,
            normalize=False,
            normalization="norm",
        ).array

        # Process fine -> coarse
        edge_h = h[edge_fine_idx]
        down = self.tp_down(edge_h, edge_direction_ft)
        down = self._mul_repeat(edge_dist_ft, down, self.hidden_irreps)

        # Scatter to coarse points
        down_weighted = (
            down.astype(jnp.float64)
            * grid_weights[edge_fine_idx, None].astype(jnp.float64)
        )
        h_coarse = scatter_sum(
            down_weighted,
            edge_coarse_idx,
            dim=0,
            dim_size=coarse_coords.shape[0],
        ).astype(h.dtype)

        # Process coarse -> fine
        edge_coarse_ft = h_coarse[edge_coarse_idx]
        up = self.tp_up(edge_coarse_ft, edge_direction_ft)

        # Compute normalization
        denom = scatter_sum(
            up_weight_edges,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )[edge_fine_idx]
        up_weight_normalized = up_weight_edges / (denom + 0.1)
        up = self._mul_repeat(
            edge_dist_ft * up_weight_normalized[:, None], up, self.out_irreps
        )

        # Scatter back to fine points
        h_fine = scatter_sum(
            up,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )

        # Post-up layer (batched matmul)
        h_fine = jax.nn.silu(
            h_fine @ self.post_up_linear.weight.T + self.post_up_linear.bias
        )

        return h_fine

    @staticmethod
    def _mul_repeat(
        mul_by: jax.Array, edge_attrs: jax.Array, irreps: e3nn.Irreps
    ) -> jax.Array:
        """Multiply edge attributes by channels per irrep."""
        mul_by_shape = mul_by.shape[:-1]
        parts = []

        start = 0
        for mul, ir in irreps:
            end = start + mul * ir.dim
            chunk = edge_attrs[..., start:end]
            chunk_view = chunk.reshape(*mul_by_shape, mul, ir.dim)
            product = mul_by[..., None] * chunk_view
            result = product.reshape(*mul_by_shape, -1)
            parts.append(result)
            start = end

        return jnp.concatenate(parts, axis=-1)


class SkalaFunctional(eqx.Module):
    """
    Skala neural exchange-correlation functional.

    JAX/Equinox implementation equivalent to the PyTorch version.
    """

    num_scalar_features: int = eqx.field(static=True)
    num_feats: int = eqx.field(static=True)
    non_local: bool = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    num_non_local_contributions: int = eqx.field(static=True)

    # Input model layers
    input_linear1: eqx.nn.Linear
    input_linear2: eqx.nn.Linear

    # Non-local model (optional)
    non_local_model: Optional[NonLocalModel]

    # Output model layers
    output_linear1: eqx.nn.Linear
    output_linear2: eqx.nn.Linear
    output_linear3: eqx.nn.Linear
    output_linear4: eqx.nn.Linear
    output_activation: ScaledSigmoid

    def __init__(
        self,
        lmax: int = 3,
        non_local: bool = True,
        non_local_hidden_nf: int = 16,
        radius_cutoff: float = float("inf"),
        *,
        key: jax.Array,
    ):
        """
        Initialize Skala functional.

        Parameters
        ----------
        lmax : int
            Maximum angular momentum for spherical harmonics.
        non_local : bool
            Whether to include non-local contributions.
        non_local_hidden_nf : int
            Number of hidden features in non-local model.
        radius_cutoff : float
            Cutoff radius for non-local interactions.
        key : jax.Array
            PRNG key.
        """
        self.num_scalar_features = 7
        self.non_local = non_local
        self.lmax = lmax
        self.num_feats = 256

        if non_local:
            self.num_non_local_contributions = non_local_hidden_nf
        else:
            self.num_non_local_contributions = 0

        # Split keys
        keys = jax.random.split(key, 7)

        # Input model
        self.input_linear1 = eqx.nn.Linear(
            self.num_scalar_features, self.num_feats, key=keys[0]
        )
        self.input_linear2 = eqx.nn.Linear(
            self.num_feats, self.num_feats, key=keys[1]
        )

        # Non-local model
        if non_local:
            self.non_local_model = NonLocalModel(
                input_nf=self.num_feats,
                hidden_nf=non_local_hidden_nf,
                lmax=lmax,
                radius_cutoff=radius_cutoff,
                key=keys[2],
            )
        else:
            self.non_local_model = None

        # Output model
        output_in = self.num_feats + self.num_non_local_contributions
        self.output_linear1 = eqx.nn.Linear(
            output_in, self.num_feats, key=keys[3]
        )
        self.output_linear2 = eqx.nn.Linear(
            self.num_feats, self.num_feats, key=keys[4]
        )
        self.output_linear3 = eqx.nn.Linear(
            self.num_feats, self.num_feats, key=keys[5]
        )
        self.output_linear4 = eqx.nn.Linear(self.num_feats, 1, key=keys[6])
        self.output_activation = ScaledSigmoid(scale=2.0)

    def _input_model(self, x: jax.Array) -> jax.Array:
        """Apply input model (batched)."""
        # Equinox Linear expects unbatched input, so we use manual matmul
        x = jax.nn.silu(x @ self.input_linear1.weight.T + self.input_linear1.bias)
        x = jax.nn.silu(x @ self.input_linear2.weight.T + self.input_linear2.bias)
        return x

    def _output_model(self, x: jax.Array) -> jax.Array:
        """Apply output model (batched)."""
        x = jax.nn.silu(x @ self.output_linear1.weight.T + self.output_linear1.bias)
        x = jax.nn.silu(x @ self.output_linear2.weight.T + self.output_linear2.bias)
        x = jax.nn.silu(x @ self.output_linear3.weight.T + self.output_linear3.bias)
        x = x @ self.output_linear4.weight.T + self.output_linear4.bias
        x = self.output_activation(x)
        return x

    def get_exc_density(self, mol: dict[str, jax.Array]) -> jax.Array:
        """
        Compute exchange-correlation energy density.

        Parameters
        ----------
        mol : dict
            Dictionary containing molecular features.

        Returns
        -------
        Array
            XC energy density, shape (num_grid_points,).
        """
        grid_coords = mol["grid_coords"]
        grid_weights = mol["grid_weights"]
        coarse_coords = mol["coarse_0_atomic_coords"]
        features_ab, features_ba = prepare_features(mol)

        # Learned symmetrized features
        spin_feats = jnp.concatenate([features_ab, features_ba], axis=0)
        spin_feats = spin_feats.astype(self.dtype)
        spin_feats = self._input_model(spin_feats)

        # Average AB and BA
        ab, ba = jnp.split(spin_feats, 2, axis=0)
        features = (ab + ba) / 2

        # Non-local model
        if self.non_local and self.non_local_model is not None:
            h_grid_non_local = self.non_local_model(
                features,
                grid_coords,
                coarse_coords,
                grid_weights,
            )
            # Multiply by exp(-density)
            density_sum = mol["density"].sum(0).reshape(-1, 1)
            h_grid_non_local = h_grid_non_local * jnp.exp(-density_sum).astype(
                self.dtype
            )
            features = jnp.concatenate([features, h_grid_non_local], axis=-1)

        enhancement_factor = self._output_model(features)
        return enhancement_density_inner_product(
            enhancement_factor=enhancement_factor, density=mol["density"]
        )

    def get_exc(self, mol: dict[str, jax.Array]) -> jax.Array:
        """
        Compute total exchange-correlation energy.

        Parameters
        ----------
        mol : dict
            Dictionary containing molecular features.

        Returns
        -------
        Array
            Total XC energy (scalar).
        """
        exc_density = self.get_exc_density(mol).astype(jnp.float64)
        grid_weights = mol["grid_weights"].astype(jnp.float64)
        return (exc_density * grid_weights).sum()

    @property
    def dtype(self) -> jnp.dtype:
        """Get model dtype from first layer weight."""
        return self.input_linear1.weight.dtype
