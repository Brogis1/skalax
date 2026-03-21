
from pyscf import gto
from skala.pyscf import SkalaKS


def main():
    print("Setting up molecule...")
    # Define a simple molecule (Water)
    mol = gto.M(
        atom="""
        O 0.000000 0.000000 0.000000
        H 0.757000 0.586000 0.000000
        H -0.757000 0.586000 0.000000
        """,
        basis="sto-3g",  # def2-tzvp
        verbose=4
    )

    print("Initializing Skala KS calculation...")
    # Create the Kohn-Sham solver with Skala functional
    ks = SkalaKS(mol, xc="skala")

    # Run the calculation
    print("Running kernel...")
    energy = ks.kernel()

    print("-" * 30)
    print(f"Total Energy: {energy:.8f} Ha")
    print("-" * 30)


if __name__ == "__main__":
    main()
