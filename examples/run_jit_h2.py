# SPDX-License-Identifier: MIT
"""
JIT-compiled H2 with Skala JAX — per-iteration timing.

The JaxSkalaKS driver automatically JIT-compiles the JAX model's forward
and backward passes (via eqx.filter_jit in numint.py).  The first SCF
iteration triggers XLA tracing + compilation; subsequent iterations
within the same geometry reuse the compiled code.

This script:
  1. Runs one SCF and times every iteration to show the JIT speedup.
  2. Computes an H2 dissociation curve comparing Skala-JAX to FCI.

Usage:
    python examples/run_jit_h2.py
"""
# ruff: noqa: E402

import time

import numpy as np

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, fci

from skalax import (
    SkalaFunctional,
    load_weights_from_npz,
    load_config,
    get_default_weights_dir,
)
from skalax.pyscf import JaxSkalaKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_h2_mol(distance: float, basis: str = "sto-3g"):
    """Build a PySCF H2 molecule at a given bond length (Angstrom)."""
    return gto.M(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis=basis,
        spin=0,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("JIT-Compiled H2 with Skala JAX")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/3] Loading pretrained model...")
    weights_dir = get_default_weights_dir()
    config = load_config(weights_dir)

    key = jax.random.PRNGKey(0)
    model = SkalaFunctional(
        lmax=config["lmax"],
        non_local=config["non_local"],
        non_local_hidden_nf=config["non_local_hidden_nf"],
        radius_cutoff=config["radius_cutoff"],
        key=key,
    )
    model = load_weights_from_npz(model, weights_dir)
    print(f"       Config: lmax={config['lmax']}, non_local={config['non_local']}")

    # ------------------------------------------------------------------
    # 2. Per-iteration timing (single geometry)
    # ------------------------------------------------------------------
    print("\n[2/3] SCF iteration timing (H2 at 0.74 A)")
    print("       First iteration compiles XLA; rest reuse the cache.\n")

    mol = make_h2_mol(0.74)
    ks = JaxSkalaKS(mol, xc=model)

    # PySCF calls callback(locals()) after every SCF iteration.
    iter_times = []
    iter_start = [None]  # mutable container for closure

    def _timer_callback(_envs):
        now = time.perf_counter()
        if iter_start[0] is not None:
            iter_times.append(now - iter_start[0])
        iter_start[0] = now

    # Capture time just before kernel() starts
    iter_start[0] = time.perf_counter()
    ks.callback = _timer_callback
    ks.kernel()

    print(f"  {'Iter':<6} {'Time (s)':<10}")
    print(f"  {'-' * 16}")
    for i, dt in enumerate(iter_times):
        tag = " <-- JIT compile" if i == 0 else ""
        print(f"  {i+1:<6} {dt:<10.3f}{tag}")

    if len(iter_times) > 1:
        first = iter_times[0]
        avg_rest = np.mean(iter_times[1:])
        print(f"\n  First iteration:      {first:.3f} s")
        print(f"  Subsequent (avg):     {avg_rest:.3f} s")
        print(f"  Speedup:              {first / avg_rest:.1f}x")

    # ------------------------------------------------------------------
    # 3. H2 dissociation curve
    # ------------------------------------------------------------------
    print("\n[3/3] H2 dissociation curve (Skala-JAX vs FCI)")

    basis = "sto-3g"
    distances = np.linspace(0.5, 4.0, 20)

    energies_skala = []
    energies_fci = []
    energies_hf = []

    header = f"  {'Dist(A)':<9} {'HF':<13} {'FCI':<13} {'Skala-JAX':<13}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for d in distances:
        mol = make_h2_mol(d, basis)

        # HF + FCI reference
        mf = scf.RHF(mol)
        e_hf = mf.kernel()
        e_fci, _ = fci.FCI(mf).kernel(verbose=0)

        # Skala JAX via PySCF KS driver
        ks = JaxSkalaKS(mol, xc=model)
        e_skala = ks.kernel()

        energies_hf.append(e_hf)
        energies_fci.append(e_fci)
        energies_skala.append(e_skala)

        print(f"  {d:<9.3f} {e_hf:<13.6f} {e_fci:<13.6f} {e_skala:<13.6f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nSummary")
    print("-" * 65)

    fci_arr = np.array(energies_fci)
    skala_arr = np.array(energies_skala)
    mae = np.mean(np.abs(fci_arr - skala_arr))
    max_err = np.max(np.abs(fci_arr - skala_arr))

    print(f"  Skala-JAX vs FCI  —  MAE: {mae:.6f} Ha, Max: {max_err:.6f} Ha")
    print("=" * 65)


if __name__ == "__main__":
    main()
