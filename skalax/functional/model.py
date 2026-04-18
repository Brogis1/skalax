# SPDX-License-Identifier: MIT
"""Skala functional model: ``SkalaFunctional`` and friends."""

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
    """Equivariant tensor product with learnable weights per valid path.

    Mirrors the PyTorch ``e3nn`` ``FullyConnectedTensorProduct`` used by Skala:
    one weight tensor of shape ``(mul_1, mul_2, mul_out)`` for each allowed
    ``(i_1, i_2, i_out)`` instruction where ``ir_out in ir_1 * ir_2``.
    """

    irreps_in1: e3nn.Irreps = eqx.field(static=True)
    irreps_in2: e3nn.Irreps = eqx.field(static=True)
    irreps_out: e3nn.Irreps = eqx.field(static=True)
    instructions: list = eqx.field(static=True)
    slices: list = eqx.field(static=True)

    weights: dict
    # Wigner 3j coefficients. Not trainable, but kept as a pytree leaf (not
    # static) so loaded buffers can replace them via ``eqx.tree_at``.
    w3j: dict

    def __init__(
        self,
        irreps_in1: e3nn.Irreps,
        irreps_in2: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        *,
        key: jax.Array,
    ):
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        instructions = []
        for i_1, (_, ir_1) in enumerate(irreps_in1):
            for i_2, (_, ir_2) in enumerate(irreps_in2):
                for i_out, (_, ir_out) in enumerate(irreps_out):
                    if ir_out in ir_1 * ir_2:
                        instructions.append((i_1, i_2, i_out))
        self.instructions = instructions

        def compute_slices(irreps):
            out = []
            start = 0
            for mul, ir in irreps:
                end = start + mul * ir.dim
                out.append((start, end))
                start = end
            return out

        self.slices = [
            compute_slices(irreps_in1),
            compute_slices(irreps_in2),
            compute_slices(irreps_out),
        ]

        # One subkey per instruction so init is deterministic from ``key``.
        subkeys = jax.random.split(key, len(instructions))

        weights = {}
        w3j = {}
        for subkey, (i_1, i_2, i_out) in zip(
            subkeys, instructions, strict=True,
        ):
            mul_1, ir_1 = irreps_in1[i_1]
            mul_2, ir_2 = irreps_in2[i_2]
            mul_out, ir_out = irreps_out[i_out]

            # He-like scaling: bound ~ sqrt(6 / (fan_in + fan_out)) summed
            # over all paths feeding the same output irrep.
            num_in = sum(
                self.irreps_in1[j[0]].mul * self.irreps_in2[j[1]].mul
                for j in instructions
                if j[2] == i_out
            )
            num_out = self.irreps_out[i_out].mul
            bound = (6 / (num_in + num_out)) ** 0.5

            weights[f"weight_{i_1}_{i_2}_{i_out}"] = jax.random.uniform(
                subkey,
                (mul_1, mul_2, mul_out),
                minval=-bound,
                maxval=bound,
            )

            w3j_val = e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)
            # Permute (j, i, k) -> (k, i, j) to match ``torch e3nn``'s
            # ``o3.wigner_3j`` convention.
            w3j[f"w3j_{i_1}_{i_2}_{i_out}"] = jnp.transpose(
                w3j_val, (2, 0, 1)
            )

        self.weights = weights
        self.w3j = w3j

    def __call__(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        """Contract ``x1`` and ``x2`` over all valid paths.

        Parameters
        ----------
        x1 : Array
            Shape ``(m, irreps_in1.dim)``.
        x2 : Array
            Shape ``(m, irreps_in2.dim)``.

        Returns
        -------
        Array
            Shape ``(m, irreps_out.dim)``.
        """
        m = x1.shape[0]
        outs = []

        for i_1, i_2, i_out in self.instructions:
            mul_1, ir_1 = self.irreps_in1[i_1]
            mul_2, ir_2 = self.irreps_in2[i_2]
            mul_out, ir_out = self.irreps_out[i_out]

            l1, l2, l3 = ir_1.l, ir_2.l, ir_out.l

            s1_start, s1_end = self.slices[0][i_1]
            s2_start, s2_end = self.slices[1][i_2]
            x1_i = x1[:, s1_start:s1_end]
            x2_i = x2[:, s2_start:s2_end]

            w = self.weights[f"weight_{i_1}_{i_2}_{i_out}"]
            w3j = self.w3j[f"w3j_{i_1}_{i_2}_{i_out}"]

            # Specialize by angular momentum to avoid materializing w3j when
            # possible. Normalizations match the PyTorch reference.
            if (l1, l2, l3) == (0, 0, 0):
                out = jnp.einsum("mu,uvw,mv->mw", x1_i, w, x2_i)
            elif l1 == 0:
                x2_view = x2_i.reshape(m, mul_2, ir_2.dim)
                out = jnp.einsum(
                    "mu,uvw,mvj->mwj", x1_i, w, x2_view
                ).reshape(m, mul_out * ir_out.dim)
                out = out / math.sqrt(ir_out.dim)
            elif l2 == 0:
                x1_view = x1_i.reshape(m, mul_1, ir_1.dim)
                out = jnp.einsum(
                    "mui,uvw,mv->mwi", x1_view, w, x2_i
                ).reshape(m, mul_out * ir_out.dim)
                out = out / math.sqrt(ir_out.dim)
            elif l3 == 0:
                x1_view = x1_i.reshape(m, mul_1, ir_1.dim)
                x2_view = x2_i.reshape(m, mul_2, ir_2.dim)
                out = jnp.einsum(
                    "mui,uvw,mvi->mw", x1_view, w, x2_view
                )
                out = out / math.sqrt(ir_1.dim)
            else:
                x1_view = x1_i.reshape(m, mul_1, ir_1.dim)
                x2_view = x2_i.reshape(m, mul_2, ir_2.dim)
                out = jnp.einsum(
                    "mui,uvw,mvj,kij->mwk", x1_view, w, x2_view, w3j
                ).reshape(m, mul_out * ir_out.dim)

            outs.append((i_out, out))

        result_parts = []
        for i_out, (mul, _) in enumerate(self.irreps_out):
            if mul > 0:
                parts = [out for idx, out in outs if idx == i_out]
                if parts:
                    result_parts.append(sum(parts))

        if len(result_parts) > 1:
            return jnp.concatenate(result_parts, axis=-1)
        return result_parts[0]


class NonLocalModel(eqx.Module):
    """Equivariant message passing between fine grid points and atomic centers.

    The fine grid sends messages (``tp_down``) to coarse atomic centers, which
    broadcast back (``tp_up``) to the fine grid. Both steps use spherical
    harmonics up to ``lmax`` and an exponential radial basis. A
    ``radius_cutoff`` limits the neighbor set; edges beyond it are dropped.
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
        self.input_nf = input_nf
        self.hidden_nf = hidden_nf
        self.lmax = lmax
        self.radius_cutoff = radius_cutoff

        self.in_irreps = e3nn.Irreps(f"{hidden_nf}x0e")
        self.out_irreps = e3nn.Irreps(f"{hidden_nf}x0e")
        self.hidden_irreps = e3nn.Irreps(
            "+".join(f"{hidden_nf}x{ell}e" for ell in range(lmax + 1))
        )
        self.sph_irreps = e3nn.Irreps.spherical_harmonics(lmax, p=1)

        k_pre, k_tp_down, k_tp_up, k_post = jax.random.split(key, 4)
        self.pre_down_linear = eqx.nn.Linear(
            input_nf, hidden_nf, key=k_pre,
        )
        self.tp_down = TensorProduct(
            self.in_irreps,
            self.sph_irreps,
            self.hidden_irreps,
            key=k_tp_down,
        )
        self.tp_up = TensorProduct(
            self.hidden_irreps,
            self.sph_irreps,
            self.out_irreps,
            key=k_tp_up,
        )
        self.post_up_linear = eqx.nn.Linear(
            hidden_nf, hidden_nf, key=k_post,
        )

    def __call__(
        self,
        h: jax.Array,
        grid_coords: jax.Array,
        coarse_coords: jax.Array,
        grid_weights: jax.Array,
    ) -> jax.Array:
        """Forward pass (JIT-safe via padded-sparse edges).

        The edge list is allocated at the static upper bound
        ``num_fine * num_coarse`` and padding entries are zeroed out via an
        edge mask, so all shapes are known at trace time.

        Parameters
        ----------
        h : Array
            Fine-grid features, shape ``(num_fine, input_nf)``.
        grid_coords : Array
            Fine-grid coordinates, shape ``(num_fine, 3)``.
        coarse_coords : Array
            Coarse-grid (atomic) coordinates, shape ``(num_coarse, 3)``.
        grid_weights : Array
            Fine-grid integration weights, shape ``(num_fine,)``.

        Returns
        -------
        Array
            Updated fine-grid features, shape ``(num_fine, hidden_nf)``.
        """
        h = jax.nn.silu(
            h @ self.pre_down_linear.weight.T + self.pre_down_linear.bias
        )

        directions, distances = vect_cdist(grid_coords, coarse_coords)

        if self.radius_cutoff != float("inf"):
            up_weight = normalization_envelope(distances, self.radius_cutoff)
        else:
            up_weight = jnp.ones_like(distances)

        num_fine, num_coarse = distances.shape
        max_edges = num_fine * num_coarse
        radius_mask = distances <= self.radius_cutoff

        # argwhere places real entries first and pads with (0, 0); a plain
        # mask lookup on (0, 0) would be ambiguous if that pair is also a
        # real edge, so we build the mask from the known count of real edges.
        edge_indices = jnp.argwhere(
            radius_mask, size=max_edges, fill_value=0,
        )
        edge_fine_idx = edge_indices[:, 0]
        edge_coarse_idx = edge_indices[:, 1]

        n_real = radius_mask.sum()
        edge_mask = (jnp.arange(max_edges) < n_real).astype(h.dtype)

        edge_directions = directions[edge_fine_idx, edge_coarse_idx]
        edge_distances = distances[edge_fine_idx, edge_coarse_idx]
        up_weight_edges = (
            up_weight[edge_fine_idx, edge_coarse_idx] * edge_mask
        )

        edge_dist_ft = exp_radial_func(edge_distances, self.hidden_nf)
        edge_dist_ft = edge_dist_ft * edge_mask[:, None]

        if self.radius_cutoff != float("inf"):
            envelope = polynomial_envelope(
                edge_distances, self.radius_cutoff, 8,
            )
            edge_dist_ft = edge_dist_ft * envelope[:, None]

        edge_direction_ft = e3nn.spherical_harmonics(
            self.sph_irreps,
            edge_directions,
            normalize=False,
            normalization="norm",
        ).array
        edge_direction_ft = edge_direction_ft * edge_mask[:, None]

        # Fine -> coarse.
        edge_h = h[edge_fine_idx]
        down = self.tp_down(edge_h, edge_direction_ft)
        down = self._mul_repeat(edge_dist_ft, down, self.hidden_irreps)

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

        # Coarse -> fine.
        edge_coarse_ft = h_coarse[edge_coarse_idx]
        up = self.tp_up(edge_coarse_ft, edge_direction_ft)

        denom = scatter_sum(
            up_weight_edges,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )[edge_fine_idx]
        # 0.1 regularizer keeps the division well-conditioned where the
        # envelope is nearly zero; matches the PyTorch reference.
        up_weight_normalized = up_weight_edges / (denom + 0.1)
        up = self._mul_repeat(
            edge_dist_ft * up_weight_normalized[:, None],
            up,
            self.out_irreps,
        )

        h_fine = scatter_sum(
            up,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )

        h_fine = jax.nn.silu(
            h_fine @ self.post_up_linear.weight.T + self.post_up_linear.bias
        )

        return h_fine

    def forward_eager(
        self,
        h: jax.Array,
        grid_coords: jax.Array,
        coarse_coords: jax.Array,
        grid_weights: jax.Array,
    ) -> jax.Array:
        """Reference forward pass using data-dependent shapes.

        This path uses boolean indexing (``distances <= radius_cutoff``) and
        is therefore **not** ``jax.jit``-compatible. It's kept as a readable
        reference and a debugging aid: the JIT-safe ``__call__`` above should
        reproduce its output exactly.
        """
        h = jax.nn.silu(
            h @ self.pre_down_linear.weight.T + self.pre_down_linear.bias
        )

        directions, distances = vect_cdist(grid_coords, coarse_coords)

        if self.radius_cutoff != float("inf"):
            up_weight = normalization_envelope(distances, self.radius_cutoff)
        else:
            up_weight = jnp.ones_like(distances)

        radius_mask = distances <= self.radius_cutoff
        edge_directions = directions[radius_mask]
        edge_distances = distances[radius_mask]
        up_weight_edges = up_weight[radius_mask]

        edge_indices = jnp.argwhere(radius_mask, size=radius_mask.sum())
        edge_fine_idx = edge_indices[:, 0]
        edge_coarse_idx = edge_indices[:, 1]

        edge_dist_ft = exp_radial_func(edge_distances, self.hidden_nf)

        if self.radius_cutoff != float("inf"):
            envelope = polynomial_envelope(
                edge_distances, self.radius_cutoff, 8,
            )
            edge_dist_ft = edge_dist_ft * envelope[:, None]

        edge_direction_ft = e3nn.spherical_harmonics(
            self.sph_irreps,
            edge_directions,
            normalize=False,
            normalization="norm",
        ).array

        edge_h = h[edge_fine_idx]
        down = self.tp_down(edge_h, edge_direction_ft)
        down = self._mul_repeat(edge_dist_ft, down, self.hidden_irreps)

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

        edge_coarse_ft = h_coarse[edge_coarse_idx]
        up = self.tp_up(edge_coarse_ft, edge_direction_ft)

        denom = scatter_sum(
            up_weight_edges,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )[edge_fine_idx]
        up_weight_normalized = up_weight_edges / (denom + 0.1)
        up = self._mul_repeat(
            edge_dist_ft * up_weight_normalized[:, None],
            up,
            self.out_irreps,
        )

        h_fine = scatter_sum(
            up,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.shape[0],
        )

        h_fine = jax.nn.silu(
            h_fine @ self.post_up_linear.weight.T + self.post_up_linear.bias
        )

        return h_fine

    @staticmethod
    def _mul_repeat(
        mul_by: jax.Array,
        edge_attrs: jax.Array,
        irreps: e3nn.Irreps,
    ) -> jax.Array:
        """Broadcast-multiply per-irrep scalars into an irreps tensor."""
        mul_by_shape = mul_by.shape[:-1]
        parts = []

        start = 0
        for mul, ir in irreps:
            end = start + mul * ir.dim
            chunk = edge_attrs[..., start:end]
            chunk_view = chunk.reshape(*mul_by_shape, mul, ir.dim)
            product = mul_by[..., None] * chunk_view
            parts.append(product.reshape(*mul_by_shape, -1))
            start = end

        return jnp.concatenate(parts, axis=-1)


class SkalaFunctional(eqx.Module):
    """JAX/Equinox implementation of the Skala neural XC functional.

    Given a dictionary of molecular features on a DFT integration grid,
    ``get_exc_density`` returns a per-point exchange-correlation energy
    density (in Hartree) and ``get_exc`` integrates it against the grid
    weights to give the total XC energy.

    With weights loaded from the bundled PyTorch checkpoint, this model
    reproduces the PyTorch Skala reference to machine precision on a
    fixed feature dictionary (see the ``Numerical Equivalence`` section
    of the README). Through ``skalax.pyscf`` it can also drive PySCF DFT
    calculations end-to-end; some divergence from the PyTorch Skala PySCF
    integration is expected there because the feature pipeline round-trips
    through a custom autograd bridge.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum for the non-local spherical-harmonic basis.
    non_local : bool
        Enable the equivariant non-local branch.
    non_local_hidden_nf : int
        Width of the non-local message-passing layers.
    radius_cutoff : float
        Cutoff distance (Bohr) for fine-to-coarse neighbor edges.
    key : jax.Array
        PRNG key. The loaded weights replace the initialized values, so the
        key only matters when training from scratch.

    Notes
    -----
    The input ``mol`` dictionary must contain keys ``density`` (shape
    ``(2, n_points)``), ``grad`` (``(2, 3, n_points)``), ``kin``
    (``(2, n_points)``), ``grid_coords`` (``(n_points, 3)``),
    ``grid_weights`` (``(n_points,)``) and ``coarse_0_atomic_coords``
    (``(n_atoms, 3)``). Coordinates are in Bohr.
    """

    num_scalar_features: int = eqx.field(static=True)
    num_feats: int = eqx.field(static=True)
    non_local: bool = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    num_non_local_contributions: int = eqx.field(static=True)

    input_linear1: eqx.nn.Linear
    input_linear2: eqx.nn.Linear

    non_local_model: NonLocalModel | None

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
        self.num_scalar_features = 7
        self.non_local = non_local
        self.lmax = lmax
        self.num_feats = 256
        self.num_non_local_contributions = (
            non_local_hidden_nf if non_local else 0
        )

        keys = jax.random.split(key, 7)

        self.input_linear1 = eqx.nn.Linear(
            self.num_scalar_features, self.num_feats, key=keys[0],
        )
        self.input_linear2 = eqx.nn.Linear(
            self.num_feats, self.num_feats, key=keys[1],
        )

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

        output_in = self.num_feats + self.num_non_local_contributions
        self.output_linear1 = eqx.nn.Linear(
            output_in, self.num_feats, key=keys[3],
        )
        self.output_linear2 = eqx.nn.Linear(
            self.num_feats, self.num_feats, key=keys[4],
        )
        self.output_linear3 = eqx.nn.Linear(
            self.num_feats, self.num_feats, key=keys[5],
        )
        self.output_linear4 = eqx.nn.Linear(
            self.num_feats, 1, key=keys[6],
        )
        self.output_activation = ScaledSigmoid(scale=2.0)

    def _input_model(self, x: jax.Array) -> jax.Array:
        # Manual matmul so the MLP runs on stacked grid points without
        # per-point vmap over eqx.nn.Linear.
        x = jax.nn.silu(
            x @ self.input_linear1.weight.T + self.input_linear1.bias
        )
        x = jax.nn.silu(
            x @ self.input_linear2.weight.T + self.input_linear2.bias
        )
        return x

    def _output_model(self, x: jax.Array) -> jax.Array:
        x = jax.nn.silu(
            x @ self.output_linear1.weight.T + self.output_linear1.bias
        )
        x = jax.nn.silu(
            x @ self.output_linear2.weight.T + self.output_linear2.bias
        )
        x = jax.nn.silu(
            x @ self.output_linear3.weight.T + self.output_linear3.bias
        )
        x = x @ self.output_linear4.weight.T + self.output_linear4.bias
        return self.output_activation(x)

    def get_exc_density(
        self, mol: dict[str, jax.Array],
    ) -> jax.Array:
        """Per-point XC energy density, shape ``(n_points,)``.

        See the class docstring for the expected keys of ``mol``.
        """
        grid_coords = mol["grid_coords"]
        grid_weights = mol["grid_weights"]
        coarse_coords = mol["coarse_0_atomic_coords"]
        features_ab, features_ba = prepare_features(mol)

        # Symmetrize over the two spin orderings.
        spin_feats = jnp.concatenate([features_ab, features_ba], axis=0)
        spin_feats = spin_feats.astype(self.dtype)
        spin_feats = self._input_model(spin_feats)
        ab, ba = jnp.split(spin_feats, 2, axis=0)
        features = (ab + ba) / 2

        if self.non_local and self.non_local_model is not None:
            h_nl = self.non_local_model(
                features, grid_coords, coarse_coords, grid_weights,
            )
            # Damp the non-local contribution in high-density regions.
            density_sum = mol["density"].sum(0).reshape(-1, 1)
            h_nl = h_nl * jnp.exp(-density_sum).astype(self.dtype)
            features = jnp.concatenate([features, h_nl], axis=-1)

        enhancement_factor = self._output_model(features)
        return enhancement_density_inner_product(
            enhancement_factor=enhancement_factor,
            density=mol["density"],
        )

    def get_exc(self, mol: dict[str, jax.Array]) -> jax.Array:
        """Total XC energy (scalar, Hartree)."""
        exc_density = self.get_exc_density(mol).astype(jnp.float64)
        grid_weights = mol["grid_weights"].astype(jnp.float64)
        return (exc_density * grid_weights).sum()

    @property
    def dtype(self) -> jnp.dtype:
        return self.input_linear1.weight.dtype
