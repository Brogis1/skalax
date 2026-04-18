# SPDX-License-Identifier: MIT
"""Kohn-Sham DFT classes that drop the JAX Skala functional into PySCF."""

from pyscf import dft, gto

from skalax.pyscf.numint import JaxSkalaNumInt


class JaxSkalaRKS(dft.rks.RKS):
    """Restricted Kohn-Sham using the JAX Skala functional.

    ``xc`` is a :class:`~skalax.SkalaFunctional` with weights already
    loaded. D3 dispersion is not wired up yet.
    """

    def __init__(self, mol: gto.Mole, xc):
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = JaxSkalaNumInt(xc)
        self.with_dftd3 = None

    def energy_nuc(self):
        return float(super().energy_nuc())


class JaxSkalaUKS(dft.uks.UKS):
    """Unrestricted Kohn-Sham using the JAX Skala functional.

    ``xc`` is a :class:`~skalax.SkalaFunctional` with weights already
    loaded. D3 dispersion is not wired up yet.
    """

    def __init__(self, mol: gto.Mole, xc):
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = JaxSkalaNumInt(xc)
        self.with_dftd3 = None

    def energy_nuc(self):
        return float(super().energy_nuc())
