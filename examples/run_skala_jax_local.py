# SPDX-License-Identifier: MIT
"""
Run PySCF DFT calculation with JAX Skala functional using local weights.

This script demonstrates how to use the JAX implementation without
needing the skala package or internet access to HuggingFace.

Run save_weights.py first to download weights from HuggingFace.
"""
# ruff: noqa: E402

import sys
from pathlib import Path

# Add dev folder to path for skalax imports
dev_path = Path(__file__).parent
sys.path.insert(0, str(dev_path))

# Enable JAX 64-bit mode for numerical precision
import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto

# JAX model - no PyTorch/skala imports needed!
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import load_weights_from_npz, load_config
from skalax.pyscf import JaxSkalaKS


def main():
    print("=" * 50)
    print("JAX Skala PySCF Integration (Local Weights)")
    print("=" * 50)

    # Load weights from local files
    weights_dir = dev_path / "skalax" / "weights"
    print(f"\n[1/3] Loading weights from {weights_dir}...")

    config = load_config(weights_dir)
    key = jax.random.PRNGKey(0)
    jax_model = JaxSkalaFunctional(
        lmax=config["lmax"],
        non_local=config["non_local"],
        non_local_hidden_nf=config["non_local_hidden_nf"],
        radius_cutoff=config["radius_cutoff"],
        key=key,
    )
    jax_model = load_weights_from_npz(jax_model, weights_dir)

    # Setup molecule (Water)
    print("[2/3] Setting up molecule...")
    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.000000
        H 0.757000 0.586000 0.000000
        H -0.757000 0.586000 0.000000
        """,
        basis="sto-3g",
        verbose=4
    )

    # Run JAX calculation
    print("[3/3] Running Skala with JAX model...")
    ks = JaxSkalaKS(mol, xc=jax_model)
    energy = ks.kernel()

    print("\n" + "=" * 50)
    print(f"Total Energy (JAX Skala): {energy:.8f} Ha")
    print("=" * 50)

    return energy


if __name__ == "__main__":
    main()
