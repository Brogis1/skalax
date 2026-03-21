# SPDX-License-Identifier: MIT
"""
Run PySCF DFT calculation with JAX Skala functional.

This script demonstrates how to use the JAX implementation of the Skala
functional with PySCF for quantum chemistry calculations.
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

# PyTorch model loading (for weights)
from skala.functional import load_functional
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional

# JAX model
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import load_weights_and_buffers_into_model
from skalax.pyscf import JaxSkalaKS


def main():
    print("=" * 50)
    print("JAX Skala PySCF Integration Demo")
    print("=" * 50)

    # Load weights from HuggingFace
    print("\n[1/4] Loading weights from HuggingFace...")
    func_torch = load_functional("skala")
    state_dict = {
        k.replace("_traced_model.", ""): v
        for k, v in func_torch.state_dict().items()
    }

    # Create fresh PyTorch model and load weights
    torch_model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.double()

    # Create JAX model and load weights
    print("[2/4] Creating JAX model and loading weights...")
    key = jax.random.PRNGKey(0)
    jax_model = JaxSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
    )
    jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

    # Setup molecule (Water)
    print("[3/4] Setting up molecule...")
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
    print("[4/4] Running Skala with JAX model...")
    ks = JaxSkalaKS(mol, xc=jax_model)
    energy = ks.kernel()

    print("\n" + "=" * 50)
    print(f"Total Energy (JAX Skala): {energy:.8f} Ha")
    print("=" * 50)

    return energy


if __name__ == "__main__":
    main()
