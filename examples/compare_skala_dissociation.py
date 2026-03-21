
# SPDX-License-Identifier: MIT
"""
Compare H2 dissociation profile between Skala (Torch), Skala (JAX), and FCI.
"""
# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add dev folder to path for skalax imports
dev_path = Path(__file__).parent
sys.path.insert(0, str(dev_path))

# Enable JAX 64-bit mode
import jax
try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

from pyscf import gto, scf, fci

# --- Skala Torch Imports ---
from skala.functional import load_functional
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional
from skala.pyscf import SkalaKS as TorchSkalaKS

# --- Skala JAX Imports ---
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import load_weights_and_buffers_into_model
from skalax.pyscf import JaxSkalaKS


def setup_models():
    """Load both Torch and JAX models with the same weights."""
    print("Loading functional weights...")

    # 1. Load Traced Functional (for Torch usage)
    # This is the "official" way Skala is usually used
    torch_traced_func = load_functional("skala")

    # 2. Extract weights to build pure Python Torch model
    state_dict = {
        k.replace("_traced_model.", ""): v
        for k, v in torch_traced_func.state_dict().items()
    }

    # 3. Create pure Torch model (needed for weight conversion source)
    torch_model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    torch_model.load_state_dict(state_dict, strict=True)
    torch_model.double()

    # 4. Create JAX model and transfer weights
    key = jax.random.PRNGKey(0)
    jax_model = JaxSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
    )
    jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

    return torch_traced_func, jax_model


def main():
    print("=" * 60)
    print("H2 Dissociation Profile Comparison: Torch vs JAX vs FCI")
    print("=" * 60)

    # Setup models
    torch_func, jax_func = setup_models()

    # Define distances
    distances = np.linspace(0.5, 4.0, 30)
    basis = "sto-3g"

    # Results containers
    results = {
        "dist": distances,
        "fci": [],
        "torch": [],
        "jax": [],
        "hf": []
    }

    print(f"\n{'Dist(A)':<8} {'FCI':<12} {'Skala(Torch)':<14} {'Skala(JAX)':<14} {'Diff(T-J)':<12}")
    print("-" * 65)

    for d in distances:
        try:
            mol = gto.M(
                atom=f"H 0 0 0; H 0 0 {d}",
                basis=basis,
                spin=0,
                verbose=0
            )

            # 1. Hartree-Fock & FCI
            mf_hf = scf.RHF(mol)
            e_hf = mf_hf.kernel()

            myfci = fci.FCI(mf_hf)
            e_fci, _ = myfci.kernel(verbose=0)

            # 2. Skala Torch
            # Note: Using the traced functional as is standard usage
            ks_torch = TorchSkalaKS(mol, xc=torch_func)
            e_torch = ks_torch.kernel()

            # 3. Skala JAX
            ks_jax = JaxSkalaKS(mol, xc=jax_func)
            e_jax = ks_jax.kernel()

            # Store
            results["fci"].append(e_fci)
            results["torch"].append(e_torch)
            results["jax"].append(e_jax)
            results["hf"].append(e_hf)

            diff = abs(e_torch - e_jax)
            print(f"{d:<8.2f} {e_fci:<12.6f} {e_torch:<14.6f} {e_jax:<14.6f} {diff:<12.2e}")

        except Exception as e:
            print(f"Error at {d}: {e}")
            results["fci"].append(np.nan)
            results["torch"].append(np.nan)
            results["jax"].append(np.nan)
            results["hf"].append(np.nan)

    # Save data
    np.savetxt("dev/compare_dissociation_data.txt",
               np.column_stack((results["dist"], results["fci"], results["torch"], results["jax"], results["hf"])),
               header="Dist(A) FCI(Ha) Skala_Torch(Ha) Skala_JAX(Ha) HF(Ha)",
               fmt="%15.8f")

    print("\nData saved to dev/compare_dissociation_data.txt")

    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(distances, results["fci"], 'k-', label='FCI', linewidth=2)
        plt.plot(distances, results["torch"], 'r--', label='Skala (Torch)', linewidth=2)
        plt.plot(distances, results["jax"], 'g:', label='Skala (JAX)', linewidth=2)
        plt.plot(distances, results["hf"], 'b-.', label='HF', alpha=0.5)

        plt.xlabel('Bond Length ($\AA$)')
        plt.ylabel('Energy (Hartree)')
        plt.title(f'H2 Dissociation: Skala Torch vs JAX ({basis})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(min(results["fci"]) - 0.1, -0.6)  # Focus on relevant range

        plt.savefig("dev/compare_dissociation_plot.png", dpi=300)
        print("Plot saved to dev/compare_dissociation_plot.png")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
