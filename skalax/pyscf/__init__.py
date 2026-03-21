# SPDX-License-Identifier: MIT

"""
PySCF integration for JAX Skala functional.

This module provides seamless integration between the JAX implementation of
Skala exchange-correlation functionals and the PySCF quantum chemistry package,
enabling DFT calculations with neural network-based functionals.
"""

try:
    import pyscf  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PySCF is not installed. Please install it with `pip install pyscf` or `conda install pyscf`."
    ) from e

from pyscf import gto

from skalax.pyscf.dft import JaxSkalaRKS, JaxSkalaUKS


def JaxSkalaKS(mol: gto.Mole, xc):
    """Create a Kohn-Sham calculator using the JAX Skala functional.

    Automatically selects between restricted (RKS) and unrestricted (UKS)
    Kohn-Sham based on the molecule's spin state.

    Parameters
    ----------
    mol : gto.Mole
        The PySCF molecule object.
    xc : SkalaFunctional
        The JAX Skala functional (with loaded weights).

    Returns
    -------
    JaxSkalaRKS or JaxSkalaUKS
        The Kohn-Sham calculator object.

    Example
    -------
    >>> from pyscf import gto
    >>> from skalax.pyscf import JaxSkalaKS
    >>> from skalax.functional.model import SkalaFunctional
    >>> from skalax.convert_weights import load_weights_and_buffers_into_model
    >>> import jax
    >>>
    >>> # Setup molecule
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp", verbose=0)
    >>>
    >>> # Create and load JAX model
    >>> key = jax.random.PRNGKey(0)
    >>> jax_func = SkalaFunctional(lmax=3, non_local=True, key=key)
    >>> # ... load weights from PyTorch model ...
    >>>
    >>> # Run calculation
    >>> ks = JaxSkalaKS(mol, xc=jax_func)
    >>> energy = ks.kernel()
    """
    if mol.spin == 0:
        return JaxSkalaRKS(mol, xc)
    else:
        return JaxSkalaUKS(mol, xc)


__all__ = ["JaxSkalaKS", "JaxSkalaRKS", "JaxSkalaUKS"]
