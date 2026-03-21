# SPDX-License-Identifier: MIT
"""
Example: Load Skala JAX model and run a forward pass.

This script demonstrates:
1. Loading weights from bundled package data
2. Creating input features with correct shapes
3. Running forward pass to compute E_xc

Input shapes:
    density:                 (2, n_points)    - spin densities [alpha, beta]
    grad:                    (2, 3, n_points) - density gradients [spin, xyz, points]
    kin:                     (2, n_points)    - kinetic energy densities
    grid_coords:             (n_points, 3)    - grid point coordinates in Bohr
    grid_weights:            (n_points,)      - integration weights
    coarse_0_atomic_coords:  (n_atoms, 3)     - atomic positions in Bohr

Output:
    E_xc:        scalar      - exchange-correlation energy in Hartree
    exc_density: (n_points,) - exchange-correlation energy density per grid point
"""

# Enable JAX 64-bit mode
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from skalax import (
    SkalaFunctional,
    load_weights_from_npz,
    load_config,
    get_default_weights_dir,
)


def main():
    print("=" * 70)
    print("Skala JAX Model - Forward Pass Example")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load model with pretrained weights
    # =========================================================================
    print("\n[1] Loading model from bundled weights...")

    weights_dir = get_default_weights_dir()
    config = load_config(weights_dir)

    print(f"    Config: lmax={config['lmax']}, non_local={config['non_local']}")

    key = jax.random.PRNGKey(0)
    model = SkalaFunctional(
        lmax=config["lmax"],
        non_local=config["non_local"],
        non_local_hidden_nf=config["non_local_hidden_nf"],
        radius_cutoff=config["radius_cutoff"],
        key=key,
    )
    model = load_weights_from_npz(model, weights_dir)
    print("    Model loaded successfully!")

    # =========================================================================
    # Step 2: Create input features with correct shapes
    # =========================================================================
    print("\n[2] Creating input features...")

    # Example dimensions
    n_points = 100  # Number of grid points
    n_atoms = 3     # Number of atoms (e.g., water: O, H, H)

    # Use random data for demonstration (in real use, these come from DFT code)
    np.random.seed(42)

    # Input feature dictionary
    mol_features = {
        # Electron density for alpha and beta spin
        # Shape: (2, n_points) where 2 = [alpha, beta]
        # Values should be positive (density >= 0)
        "density": jnp.array(np.abs(np.random.randn(2, n_points)) * 0.5 + 0.1),

        # Gradient of electron density
        # Shape: (2, 3, n_points) where 2 = spin, 3 = [x, y, z]
        "grad": jnp.array(np.random.randn(2, 3, n_points) * 0.1),

        # Kinetic energy density (tau)
        # Shape: (2, n_points)
        # Values should be positive
        "kin": jnp.array(np.abs(np.random.randn(2, n_points)) * 0.3 + 0.05),

        # Grid point coordinates in Bohr
        # Shape: (n_points, 3)
        "grid_coords": jnp.array(np.random.randn(n_points, 3) * 2.0),

        # Integration weights for numerical integration
        # Shape: (n_points,)
        # Values should be positive
        "grid_weights": jnp.array(np.abs(np.random.randn(n_points)) * 0.1 + 0.01),

        # Atomic coordinates in Bohr (coarse grid for non-local)
        # Shape: (n_atoms, 3)
        "coarse_0_atomic_coords": jnp.array([
            [0.0, 0.0, 0.0],      # O atom
            [1.43, 1.11, 0.0],    # H atom
            [-1.43, 1.11, 0.0],   # H atom
        ]),
    }

    print("\n    Input shapes:")
    for key, val in mol_features.items():
        print(f"      {key:30s}: {str(val.shape):20s} dtype={val.dtype}")

    # =========================================================================
    # Step 3: Run forward pass
    # =========================================================================
    print("\n[3] Running forward pass...")

    # Compute total exchange-correlation energy
    E_xc = model.get_exc(mol_features)
    print(f"\n    E_xc = {float(E_xc):.10f} Hartree")
    print(f"    E_xc shape: {E_xc.shape} (scalar)")

    # Compute exchange-correlation energy density per grid point
    exc_density = model.get_exc_density(mol_features)
    print(f"\n    exc_density shape: {exc_density.shape}")
    print(f"    exc_density dtype: {exc_density.dtype}")
    print(f"    exc_density range: [{float(exc_density.min()):.6f}, "
          f"{float(exc_density.max()):.6f}]")

    # Verify: E_xc = sum(exc_density * grid_weights)
    E_xc_from_density = jnp.sum(exc_density * mol_features["grid_weights"])
    print(f"\n    Verification: sum(exc_density * weights) = "
          f"{float(E_xc_from_density):.10f}")
    print(f"    Matches E_xc: {jnp.allclose(E_xc, E_xc_from_density)}")

    # =========================================================================
    # Step 4: Compute gradients with JAX autodiff
    # =========================================================================
    print("\n[4] Computing gradients with JAX autodiff...")

    # Gradient of E_xc with respect to density
    grad_fn = jax.grad(lambda mol: model.get_exc(mol))
    grads = grad_fn(mol_features)

    print("\n    Gradient shapes (dE_xc/d_feature):")
    for key, val in grads.items():
        if val is not None:
            print(f"      d(E_xc)/d({key:25s}): {str(val.shape):20s}")

    print("\n" + "=" * 70)
    print("Forward pass complete!")
    print("=" * 70)

    # =========================================================================
    # Summary of shapes
    # =========================================================================
    print("""
Input/Output Shape Summary
==========================

INPUTS (mol_features dict):
  density                  : (2, n_points)      # [alpha, beta] spin densities
  grad                     : (2, 3, n_points)   # density gradients [spin, xyz, points]
  kin                      : (2, n_points)      # kinetic energy densities
  grid_coords              : (n_points, 3)      # grid coordinates [points, xyz]
  grid_weights             : (n_points,)        # integration weights
  coarse_0_atomic_coords   : (n_atoms, 3)       # atomic positions [atoms, xyz]

OUTPUTS:
  model.get_exc(mol)       : ()                 # scalar E_xc in Hartree
  model.get_exc_density(mol): (n_points,)       # energy density per grid point

GRADIENTS (via jax.grad):
  dE_xc/d(density)         : (2, n_points)
  dE_xc/d(grad)            : (2, 3, n_points)
  dE_xc/d(kin)             : (2, n_points)
  dE_xc/d(grid_coords)     : (n_points, 3)
  dE_xc/d(grid_weights)    : (n_points,)

Units:
  - Coordinates: Bohr (atomic units)
  - Energy: Hartree
  - Density: electrons/Bohr^3
""")


if __name__ == "__main__":
    main()
