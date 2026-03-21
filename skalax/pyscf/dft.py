# SPDX-License-Identifier: MIT
# JAX-based DFT classes for PySCF integration

from pyscf import dft, gto

from skalax.pyscf.numint import JaxSkalaNumInt


class JaxSkalaRKS(dft.rks.RKS):
    """Restricted Kohn-Sham DFT with JAX Skala functional.

    This class extends PySCF's RKS to use the JAX implementation of the
    Skala exchange-correlation functional.

    Example
    -------
    >>> from pyscf import gto
    >>> from skalax.pyscf import JaxSkalaRKS
    >>> from skalax.functional.model import SkalaFunctional
    >>> import jax
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
    >>> key = jax.random.PRNGKey(0)
    >>> jax_func = SkalaFunctional(lmax=3, non_local=True, key=key)
    >>> # Load weights...
    >>> ks = JaxSkalaRKS(mol, jax_func)
    >>> energy = ks.kernel()
    """

    def __init__(self, mol: gto.Mole, xc):
        """Initialize restricted Kohn-Sham with JAX Skala functional.

        Parameters
        ----------
        mol : gto.Mole
            PySCF molecule object.
        xc : SkalaFunctional
            JAX Skala functional (with loaded weights).
        """
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = JaxSkalaNumInt(xc)
        self.with_dftd3 = None  # D3 dispersion not supported yet

    def energy_nuc(self):
        """Return nuclear repulsion energy as float for compatibility."""
        return float(super().energy_nuc())


class JaxSkalaUKS(dft.uks.UKS):
    """Unrestricted Kohn-Sham DFT with JAX Skala functional.

    This class extends PySCF's UKS to use the JAX implementation of the
    Skala exchange-correlation functional for open-shell systems.

    Example
    -------
    >>> from pyscf import gto
    >>> from skalax.pyscf import JaxSkalaUKS
    >>> from skalax.functional.model import SkalaFunctional
    >>> import jax
    >>>
    >>> mol = gto.M(atom="O", basis="def2-svp", spin=2)
    >>> key = jax.random.PRNGKey(0)
    >>> jax_func = SkalaFunctional(lmax=3, non_local=True, key=key)
    >>> # Load weights...
    >>> ks = JaxSkalaUKS(mol, jax_func)
    >>> energy = ks.kernel()
    """

    def __init__(self, mol: gto.Mole, xc):
        """Initialize unrestricted Kohn-Sham with JAX Skala functional.

        Parameters
        ----------
        mol : gto.Mole
            PySCF molecule object.
        xc : SkalaFunctional
            JAX Skala functional (with loaded weights).
        """
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = JaxSkalaNumInt(xc)
        self.with_dftd3 = None  # D3 dispersion not supported yet

    def energy_nuc(self):
        """Return nuclear repulsion energy as float for compatibility."""
        return float(super().energy_nuc())
