
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
from skala.pyscf import SkalaKS


def main():
    print("Calculating H2 dissociation profile...")

    # Define bond distances (in Angstroms)
    distances = np.linspace(0.5, 4.0, 40)

    skala_energies = []
    fci_energies = []
    hf_energies = []

    print(f"{'Distance (A)':<15} {'FCI (Ha)':<15} {'Skala (Ha)':<15} {'HF (Ha)':<15}")
    print("-" * 60)

    basis = "sto-3g"

    # Pre-load Skala functional outside the loop
    # This avoids reloading/checking for weights on every iteration
    from skala.functional import load_functional
    print("Pre-loading Skala functional...")
    xc_functional = load_functional("skala")

    for d in distances:
        try:
            # Define molecule
            mol = gto.M(
                atom=f"H 0 0 0; H 0 0 {d}",
                basis=basis,
                spin=0,
                verbose=0
            )

            # Hartree-Fock Calculation (Reference for FCI)
            mf_hf = scf.RHF(mol)
            e_hf = mf_hf.kernel()
            hf_energies.append(e_hf)

            # FCI Calculation
            # For H2, this provides the exact solution within the basis set
            myfci = fci.FCI(mf_hf)
            e_fci, _ = myfci.kernel(verbose=0)
            fci_energies.append(e_fci)

            # Skala DFT Calculation
            # Pass the pre-loaded functional object instead of the string "skala"
            ks_skala = SkalaKS(mol, xc=xc_functional)
            e_skala = ks_skala.kernel()
            skala_energies.append(e_skala)

            print(f"{d:<15.2f} {e_fci:<15.6f} {e_skala:<15.6f} {e_hf:<15.6f}")

        except Exception as e:
            print(f"Error at distance {d}: {e}")

    # Save results to a file
    np.savetxt("h2_dissociation_data.txt",
               np.column_stack((distances, fci_energies, skala_energies, hf_energies)),
               header="Distance(A) FCI(Ha) Skala(Ha) HF(Ha)",
               fmt="%15.6f")
    print("\nData saved to h2_dissociation_data.txt")

    # Plot results
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(distances, fci_energies, 'k-', label='FCI', linewidth=2)
        plt.plot(distances, skala_energies, 'r--', label='Skala', linewidth=2)
        plt.plot(distances, hf_energies, 'b:', label='Hartree-Fock', linewidth=1.5)

        plt.xlabel('Bond Length ($\AA$)')
        plt.ylabel('Energy (Hartree)')
        plt.title(f'H2 Dissociation Profile ({basis})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.ylim(min(fci_energies) - 0.1, max(hf_energies[:5]) + 0.1)  # Focus on the binding region

        plot_filename = "h2_dissociation_plot.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved to {plot_filename}")

    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    main()
