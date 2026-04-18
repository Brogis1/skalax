# SPDX-License-Identifier: MIT
"""PySCF integration for the JAX Skala functional."""

try:
    import pyscf  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PySCF is not installed. Install it with `pip install pyscf`.",
    ) from e

from pyscf import gto

from skalax.pyscf.dft import JaxSkalaRKS, JaxSkalaUKS

__all__ = ["JaxSkalaKS", "JaxSkalaRKS", "JaxSkalaUKS"]


def JaxSkalaKS(mol: gto.Mole, xc):
    """Kohn-Sham calculator using the JAX Skala functional.

    Picks ``JaxSkalaRKS`` for closed-shell molecules (``mol.spin == 0``)
    and ``JaxSkalaUKS`` otherwise.
    """
    if mol.spin == 0:
        return JaxSkalaRKS(mol, xc)
    return JaxSkalaUKS(mol, xc)
