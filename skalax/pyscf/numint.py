# SPDX-License-Identifier: MIT
# JAX-based NumInt for PySCF integration

# TODO:
# Architecture note: The real bottleneck is that generate_features() is still a PyTorch function. Each SCF iteration goes: PySCF numpy → PyTorch tensor → compute density features → PyTorch tensor → numpy → JAX array → run model → JAX result → numpy → PyTorch tensor → autograd. A pure-JAX feature generator would eliminate the PyTorch↔JAX bridge entirely, but that's a larger refactor.

from collections.abc import Callable
from typing import Any, Protocol

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


# ---------------------------------------------------------------------------
# JIT-compiled wrappers for the JAX model (cached after first SCF iteration)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _jit_get_exc(model, features_jax):
    """JIT-compiled forward pass: compute E_xc."""
    return model.get_exc(features_jax)


@eqx.filter_jit
def _jit_grad_get_exc(model, features_jax):
    """JIT-compiled backward pass: compute dE_xc/d(features)."""
    return jax.grad(model.get_exc)(features_jax)


class LibXCSpec(Protocol):
    __version__: str | None
    __references__: str | None

    @staticmethod
    def is_hybrid_xc(xc: str) -> bool: ...

    @staticmethod
    def is_nlc(xc: str) -> bool: ...


class JaxExcFunction(Function):
    """Custom autograd function to bridge PyTorch features → JAX model → PyTorch gradients.

    This function enables the chain rule to flow through JAX's autodiff:
    - Forward: Converts PyTorch tensors to JAX arrays, computes E_xc using JAX model
    - Backward: Uses JAX's grad to compute dE_xc/d(features), converts back to PyTorch
    """

    @staticmethod
    def forward(ctx, *args):
        """
        Forward pass: compute E_xc using JAX model.

        args = (*feature_tensors, keys_tuple, jax_func)
        """
        # Last two args are keys and func
        jax_func = args[-1]
        keys = args[-2]
        feature_tensors = args[:-2]

        # Convert PyTorch tensors to JAX arrays via numpy (zero-copy for CPU tensors)
        features_jax = {}
        for key, val in zip(keys, feature_tensors):
            features_jax[key] = jnp.asarray(val.detach().numpy())

        # Compute E_xc using JIT-compiled JAX model
        E_xc = _jit_get_exc(jax_func, features_jax)

        # Save for backward pass
        ctx.features_jax = features_jax
        ctx.jax_func = jax_func
        ctx.keys = keys
        ctx.dtype = feature_tensors[0].dtype
        ctx.device = feature_tensors[0].device

        # .copy() ensures the numpy scalar is writable for PyTorch
        return torch.from_numpy(
            np.asarray(E_xc).copy()
        ).to(dtype=ctx.dtype, device=ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients using JAX autodiff."""

        # Compute gradients w.r.t. each feature using JIT-compiled JAX
        grads_jax = _jit_grad_get_exc(ctx.jax_func, ctx.features_jax)

        # Convert JAX gradients back to PyTorch tensors via numpy (minimal copy)
        grads_torch = []
        grad_scale = grad_output.item()
        for key in ctx.keys:
            if key in grads_jax and grads_jax[key] is not None:
                grad_np = np.asarray(grads_jax[key])
                if not grad_np.flags.writeable:
                    grad_np = grad_np.copy()
                grad_tensor = torch.from_numpy(grad_np).to(
                    dtype=ctx.dtype, device=ctx.device
                )
                grads_torch.append(grad_tensor * grad_scale)
            else:
                grads_torch.append(None)

        # Return: gradients for features, None for keys and func
        return (*grads_torch, None, None)


class JaxSkalaNumInt:
    """PySCF-compatible NumInt using JAX Skala model.

    This class provides the same interface as `pyscf.dft.numint.NumInt` but uses
    the JAX implementation of the Skala functional for exchange-correlation energy
    and potential computation.

    Example
    -------
    >>> from skalax.pyscf import JaxSkalaKS
    >>> from skalax.functional.model import SkalaFunctional
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> jax_func = SkalaFunctional(lmax=3, non_local=True, key=key)
    >>> # Load weights...
    >>> ks = JaxSkalaKS(mol, xc=jax_func)
    >>> energy = ks.kernel()
    """

    def __init__(self, functional, chunk_size: int | None = None):
        """Initialize the JAX NumInt.

        Parameters
        ----------
        functional : SkalaFunctional
            The JAX Skala functional (with loaded weights).
        chunk_size : int, optional
            Chunk size for grid batching. If None, determined automatically.
        """
        self.func = functional  # JAX SkalaFunctional
        self.chunk_size = chunk_size
        # Features needed by the Skala model
        self.features = {
            "density", "grad", "kin",
            "grid_coords", "grid_weights", "coarse_0_atomic_coords"
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
        """Compute exchange-correlation energy and potential.

        Parameters
        ----------
        mol : gto.Mole
            PySCF molecule object.
        grids : dft.Grids
            Integration grid.
        xc_code : str or None
            XC code (ignored, uses JAX functional).
        dm : Tensor
            Density matrix as PyTorch tensor.
        second_order : bool
            Whether to retain graph for second-order derivatives.
        max_memory : int
            Maximum memory in MB for chunked evaluation.

        Returns
        -------
        N : Tensor
            Number of electrons per spin channel.
        E_xc : Tensor
            Exchange-correlation energy (scalar).
        V_xc : Tensor
            Exchange-correlation potential (same shape as dm).
        """
        dm = dm.requires_grad_()

        # Generate features using PyTorch code (well-tested, handles chunking)
        mol_features = generate_features(
            mol,
            dm,
            grids,
            self.features,
            chunk_size=self.chunk_size,
            max_memory=max_memory,
            gpu=False,  # JAX model runs on CPU for now
        )

        # Prepare for custom autograd function
        keys = tuple(mol_features.keys())
        values = tuple(mol_features[k] for k in keys)

        # Compute E_xc through JAX (with autograd bridge)
        E_xc = JaxExcFunction.apply(*values, keys, self.func)

        # Compute V_xc via PyTorch autograd (chain rule through features)
        (V_xc,) = torch.autograd.grad(
            E_xc,
            dm,
            torch.ones_like(E_xc),
            retain_graph=second_order,
            create_graph=second_order,
        )

        # Compute N (number of electrons)
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
        """Compute electron density on the grid.

        Parameters
        ----------
        mol : gto.Mole
            PySCF molecule object.
        dm : Array
            Density matrix.
        grids : Grid
            Integration grid.
        max_memory : int
            Maximum memory in MB.
        verbose : int
            Verbosity level (unused).

        Returns
        -------
        rho : Array
            Electron density on the grid.
        """
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
        """Restricted Kohn-Sham method, applicable when both spin-densities are equal.

        Parameters
        ----------
        mol : gto.Mole
            PySCF molecule object.
        grids : Grid
            Integration grid.
        xc_code : str or None
            XC code (ignored).
        dm : Array
            Density matrix (2D array for RKS).
        max_memory : int
            Maximum memory in MB.

        Returns
        -------
        N : float
            Total number of electrons.
        E_xc : float
            Exchange-correlation energy.
        V_xc : Array
            Exchange-correlation potential matrix.
        """
        assert len(dm.shape) == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, self.from_backend(dm), max_memory=max_memory
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
        """Unrestricted Kohn-Sham method, spin densities can be different.

        Parameters
        ----------
        mol : gto.Mole
            PySCF molecule object.
        grids : Grid
            Integration grid.
        xc_code : str or None
            XC code (ignored).
        dm : Array
            Density matrices (3D array: [2, nao, nao] for alpha/beta).
        max_memory : int
            Maximum memory in MB.

        Returns
        -------
        N : Array
            Number of electrons per spin channel.
        E_xc : float
            Exchange-correlation energy.
        V_xc : Array
            Exchange-correlation potential matrices.
        """
        assert len(dm.shape) == 3 and dm.shape[0] == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, self.from_backend(dm), max_memory=max_memory
        )
        return self.to_backend(N), E_xc.item(), self.to_backend(V_xc)

    def rsh_and_hybrid_coeff(self) -> tuple[float, float, float]:
        """Return range-separation and hybrid coefficients.

        Skala is a pure functional (no exact exchange), so returns zeros.
        """
        return 0, 0, 0

    class libxc:
        """Mock libxc interface for PySCF compatibility."""
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
    ) -> Callable[[Array], Array]:
        """Generate the response function for TDDFT/stability analysis.

        Parameters
        ----------
        mo_coeff : Array
            Molecular orbital coefficients.
        mo_occ : Array
            Molecular orbital occupations.
        ks : KS
            Kohn-Sham object.
        **kwargs : dict
            Additional arguments (hermi, singlet, with_j).

        Returns
        -------
        hessian_vector_product : Callable
            Function that computes Hessian-vector products.
        """
        assert mo_coeff is not None
        assert mo_occ is not None

        # Validate kwargs for meta-GGA
        if kwargs is not None:
            if "hermi" in kwargs:
                assert kwargs["hermi"] == 1
            if "singlet" in kwargs:
                assert kwargs["singlet"] is None
            if "with_j" in kwargs:
                assert kwargs["with_j"]

        dm0 = self.from_backend(ks.make_rdm1(mo_coeff, mo_occ))
        # Cache V_xc to save a forward pass in each iteration
        V_xc = self(ks.mol, ks.grids, None, dm0, second_order=True)[2]

        def hessian_vector_product(dm1: Array) -> Array:
            v1 = self.to_backend(
                torch.autograd.grad(
                    V_xc, dm0, self.from_backend(dm1), retain_graph=True
                )[0]
            )
            vj = ks.get_j(ks.mol, dm1, hermi=1)

            if ks.mol.spin == 0:
                v1 += vj
            else:
                v1 += vj[0] + vj[1]

            return v1

        return hessian_vector_product
