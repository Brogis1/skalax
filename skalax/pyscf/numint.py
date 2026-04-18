# SPDX-License-Identifier: MIT
"""Pluggable ``NumInt`` that routes XC evaluation through the JAX model.

Architecture note: :func:`skalax.pyscf.features.generate_features` is still
a PyTorch function. Each SCF iteration therefore round-trips
numpy to PyTorch to features to PyTorch to numpy to JAX to model to numpy to
PyTorch (autograd). A pure-JAX feature generator would remove the bridge
but is a sizeable refactor; this module is the thin adapter that makes
the current hybrid work.
"""

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import torch
from torch import Tensor
from torch.autograd import Function
from pyscf import dft, gto

from skalax.pyscf.backend import (
    KS,
    Array,
    Grid,
    from_numpy_or_cupy,
    to_numpy,
)
from skalax.pyscf.features import generate_features


@eqx.filter_jit
def _jit_get_exc(model, features_jax):
    return model.get_exc(features_jax)


@eqx.filter_jit
def _jit_grad_get_exc(model, features_jax):
    return jax.grad(model.get_exc)(features_jax)


class JaxExcFunction(Function):
    """PyTorch autograd bridge to the JAX model.

    Forward converts PyTorch feature tensors to JAX arrays and evaluates
    ``E_xc``. Backward calls ``jax.grad`` to get dE_xc / d(features) and
    hands the gradients back as PyTorch tensors, so the downstream chain
    rule through the feature pipeline proceeds in PyTorch as usual.
    """

    @staticmethod
    def forward(ctx, *args):
        # args = (*feature_tensors, keys_tuple, jax_func)
        jax_func = args[-1]
        keys = args[-2]
        feature_tensors = args[:-2]

        features_jax = {
            key: jnp.asarray(val.detach().numpy())
            for key, val in zip(keys, feature_tensors, strict=True)
        }

        E_xc = _jit_get_exc(jax_func, features_jax)

        ctx.features_jax = features_jax
        ctx.jax_func = jax_func
        ctx.keys = keys
        ctx.dtype = feature_tensors[0].dtype
        ctx.device = feature_tensors[0].device

        # .copy() gives a writable buffer for ``torch.from_numpy``.
        return torch.from_numpy(
            np.asarray(E_xc).copy()
        ).to(dtype=ctx.dtype, device=ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        grads_jax = _jit_grad_get_exc(ctx.jax_func, ctx.features_jax)

        grad_scale = grad_output.item()
        grads_torch = []
        for key in ctx.keys:
            if key in grads_jax and grads_jax[key] is not None:
                grad_np = np.asarray(grads_jax[key])
                if not grad_np.flags.writeable:
                    grad_np = grad_np.copy()
                grad_tensor = torch.from_numpy(grad_np).to(
                    dtype=ctx.dtype, device=ctx.device,
                )
                grads_torch.append(grad_tensor * grad_scale)
            else:
                grads_torch.append(None)

        return (*grads_torch, None, None)


class JaxSkalaNumInt:
    """Drop-in replacement for :class:`pyscf.dft.numint.NumInt`.

    Produces ``E_xc`` and ``V_xc`` from a JAX :class:`SkalaFunctional`,
    including the response function needed for stability and TDDFT.
    """

    def __init__(self, functional, chunk_size: int | None = None):
        self.func = functional
        self.chunk_size = chunk_size
        self.features = {
            "density",
            "grad",
            "kin",
            "grid_coords",
            "grid_weights",
            "coarse_0_atomic_coords",
        }

    def from_backend(
        self,
        x: Array,
        device: torch.device | None = None,
        transpose: bool = False,
    ) -> Tensor:
        return from_numpy_or_cupy(x, device=device, transpose=transpose)

    def to_backend(self, x: Tensor | list[Tensor]) -> Array | list[Array]:
        if isinstance(x, list):
            return [self.to_backend(y) for y in x]
        return to_numpy(x)

    def __call__(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: Tensor,
        second_order: bool = False,
        max_memory: int = 2000,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(N, E_xc, V_xc)`` for a density matrix ``dm``.

        ``xc_code`` is ignored: the XC is defined by ``self.func``.
        """
        dm = dm.requires_grad_()

        mol_features = generate_features(
            mol,
            dm,
            grids,
            self.features,
            chunk_size=self.chunk_size,
            max_memory=max_memory,
            gpu=False,
        )

        keys = tuple(mol_features.keys())
        values = tuple(mol_features[k] for k in keys)

        E_xc = JaxExcFunction.apply(*values, keys, self.func)

        (V_xc,) = torch.autograd.grad(
            E_xc,
            dm,
            torch.ones_like(E_xc),
            retain_graph=second_order,
            create_graph=second_order,
        )

        rho = mol_features["density"]
        grid_weights = mol_features["grid_weights"]
        N = (rho * grid_weights).sum(dim=-1)

        return N, E_xc, V_xc

    def get_rho(
        self,
        mol: gto.Mole,
        dm: Array,
        grids: Grid,
        max_memory: int = 2000,
        verbose: int = 0,
    ) -> Array:
        """Total electron density on the grid."""
        mol_features = generate_features(
            mol,
            self.from_backend(dm),
            grids,
            features={"density"},
            chunk_size=self.chunk_size,
            max_memory=max_memory,
            gpu=False,
        )
        return self.to_backend(mol_features["density"].sum(0))

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Array,
        max_memory: int = 2000,
    ) -> tuple[float, float, Array]:
        """Closed-shell XC evaluation."""
        assert len(dm.shape) == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, self.from_backend(dm),
            max_memory=max_memory,
        )
        return N.sum().item(), E_xc.item(), self.to_backend(V_xc)

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Array,
        max_memory: int = 2000,
    ) -> tuple[Array, float, Array]:
        """Open-shell XC evaluation. ``dm`` has shape ``(2, nao, nao)``."""
        assert len(dm.shape) == 3 and dm.shape[0] == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, self.from_backend(dm),
            max_memory=max_memory,
        )
        return self.to_backend(N), E_xc.item(), self.to_backend(V_xc)

    def rsh_and_hybrid_coeff(self) -> tuple[float, float, float]:
        # Skala is a pure functional, no exact exchange.
        return 0, 0, 0

    class libxc:
        """Stub libxc interface for PySCF compatibility."""

        __version__ = None
        __reference__ = None

        @staticmethod
        def is_hybrid_xc(xc: str) -> bool:
            return False

        @staticmethod
        def is_nlc(xc: str) -> bool:
            return False

    def gen_response(
        self,
        mo_coeff: Array | None,
        mo_occ: Array | None,
        *,
        ks: KS,
        **kwargs: Any,
    ):
        """Hessian-vector product for TDDFT / stability analysis."""
        assert mo_coeff is not None
        assert mo_occ is not None

        # Skala is a meta-GGA: only the Hermitian, closed-shell-adjacent
        # response path is supported.
        if kwargs:
            if "hermi" in kwargs:
                assert kwargs["hermi"] == 1
            if "singlet" in kwargs:
                assert kwargs["singlet"] is None
            if "with_j" in kwargs:
                assert kwargs["with_j"]

        dm0 = self.from_backend(ks.make_rdm1(mo_coeff, mo_occ))
        # Cache V_xc so the HVP only pays one JAX autograd per SCF step.
        V_xc = self(ks.mol, ks.grids, None, dm0, second_order=True)[2]

        def hessian_vector_product(dm1: Array) -> Array:
            v1 = self.to_backend(
                torch.autograd.grad(
                    V_xc, dm0, self.from_backend(dm1), retain_graph=True,
                )[0]
            )
            vj = ks.get_j(ks.mol, dm1, hermi=1)

            if ks.mol.spin == 0:
                v1 += vj
            else:
                v1 += vj[0] + vj[1]

            return v1

        return hessian_vector_product
