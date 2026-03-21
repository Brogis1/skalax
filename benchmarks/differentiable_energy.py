#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark 6: Differentiable DFT — JAX-unique dE_xc/dR demonstration.

Shows that JAX autodiff can compute analytical gradients of the
exchange-correlation energy with respect to nuclear coordinates,
verified against finite differences.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import argparse
import numpy as np
import jax.numpy as jnp
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_jax_model,
    make_pyscf_mol,
    format_table,
    print_header,
    print_pass,
    print_fail,
    print_info,
    setup_plot_style,
    save_benchmark_data,
    load_benchmark_data,
)


def generate_realistic_features(atom_str: str = "H 0 0 0; H 0 0 0.74", basis: str = "sto-3g"):
    """Generate realistic DFT features from PySCF for a given molecule."""
    from pyscf import dft, gto
    from skalax.pyscf.features import generate_features
    from skalax.pyscf.backend import from_numpy_or_cupy

    mol = gto.M(atom=atom_str, basis=basis, verbose=0)
    grids = dft.Grids(mol)
    grids.level = 3
    grids.build()

    dm = dft.RKS(mol).get_init_guess()
    dm_tensor = from_numpy_or_cupy(dm, dtype=torch.float64)

    features = {"density", "grad", "kin", "grid_coords", "grid_weights", "coarse_0_atomic_coords"}
    mol_features = generate_features(mol, dm_tensor, grids, features, gpu=False)

    # Convert to JAX
    mol_jax = {}
    for k, v in mol_features.items():
        if isinstance(v, torch.Tensor):
            mol_jax[k] = jnp.array(v.detach().numpy())
        else:
            mol_jax[k] = jnp.array(np.array(v))

    return mol_jax


def collect_data():
    """Compute all gradient data and return serializable dict."""
    print_info("Loading JAX model...")
    model = load_jax_model()

    print_info("Generating realistic H2 features from PySCF...")
    mol_jax = generate_realistic_features()

    print_info("Computing feature-level gradients via jax.grad...")
    grads = jax.grad(lambda m: model.get_exc(m))(mol_jax)
    grads = {k: np.array(v) for k, v in grads.items()}

    # Feature gradient norms
    grad_norms = {}
    for key, g in grads.items():
        grad_norms[key] = {
            "norm": float(np.linalg.norm(g)),
            "finite": bool(np.all(np.isfinite(g))),
        }

    # Coordinate gradient
    coord_grad = grads["coarse_0_atomic_coords"]
    n_atoms = coord_grad.shape[0]

    # Full FD verification
    eps = 1e-5
    fd_rows = []
    for atom_idx in range(n_atoms):
        for coord_idx, coord_name in enumerate(["x", "y", "z"]):
            mol_p = {k: jnp.array(v) for k, v in mol_jax.items()}
            mol_m = {k: jnp.array(v) for k, v in mol_jax.items()}
            key = "coarse_0_atomic_coords"
            mol_p[key] = mol_p[key].at[atom_idx, coord_idx].add(eps)
            mol_m[key] = mol_m[key].at[atom_idx, coord_idx].add(-eps)

            ep = float(model.get_exc(mol_p))
            em = float(model.get_exc(mol_m))
            fd = (ep - em) / (2 * eps)
            an = float(coord_grad[atom_idx, coord_idx])
            err = abs(fd - an)
            rel = err / (abs(an) + 1e-30)

            if abs(an) < 1e-10:
                ok = err < 1e-8
            else:
                ok = rel < 1e-3

            fd_rows.append([
                f"Atom {atom_idx}/{coord_name}",
                f"{an:.6e}", f"{fd:.6e}", f"{rel:.3e}",
                "PASS" if ok else "FAIL",
            ])

    return {
        "grad_norms": grad_norms,
        "fd_rows": fd_rows,
        "features": list(grads.keys()),
        "feature_norms": [
            float(np.linalg.norm(grads[k])) for k in grads
        ],
    }


def print_report(data):
    """Print pass/fail from data dict."""
    all_finite = all(
        v["finite"] for v in data["grad_norms"].values()
    )
    fd_all_pass = all(r[4] == "PASS" for r in data["fd_rows"])

    # Grad norms table
    rows = []
    for key, info in data["grad_norms"].items():
        rows.append([
            key, f"{info['norm']:.6e}",
            "Yes" if info["finite"] else "NO",
        ])
    print(format_table(["Feature", "Grad Norm", "All Finite"], rows))

    # FD table
    print(format_table(
        ["Coordinate", "Analytical", "Finite Diff",
         "Rel Error", "Status"],
        data["fd_rows"],
    ))

    all_pass = all_finite and fd_all_pass
    print()
    if all_pass:
        print_pass("All gradients finite, FD verification passed")
    else:
        if not all_finite:
            print_fail("Some gradients contain non-finite values")
        if not fd_all_pass:
            print_fail("Finite-difference verification failed")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data")
    args = parser.parse_args()

    print_header("Benchmark 6: Differentiable Energy (JAX-unique dE_xc/dR)")

    if args.replot:
        data = load_benchmark_data("differentiable_energy")
        if data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        data = collect_data()
        save_benchmark_data("differentiable_energy", data)

    all_pass = print_report(data)

    _make_plot_from_data(data)
    return 0 if all_pass else 1


def _make_plot_from_data(data):
    """Generate differentiable energy plots from saved data."""
    plt = setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: gradient norms by feature
    features = data["features"]
    norms = data["feature_norms"]

    ax = axes[0]
    ax.barh(range(len(features)), norms,
            color="#2196F3", edgecolor="white")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("||dE_xc / d(feature)||  (L2 norm)")
    ax.set_title("Feature Gradient Norms via jax.grad",
                 fontweight="bold")
    ax.set_xscale("log")

    # Right: analytical vs FD for nuclear coords
    ax = axes[1]
    fd_rows = data["fd_rows"]
    n_coords = len(fd_rows)
    labels = [r[0] for r in fd_rows]
    analytical = [float(r[1]) for r in fd_rows]
    fd_vals = [float(r[2]) for r in fd_rows]

    x = np.arange(n_coords)
    w = 0.35
    ax.bar(x - w / 2, analytical, w,
           label="Analytical (jax.grad)", color="#2196F3")
    ax.bar(x + w / 2, fd_vals, w,
           label="Finite difference (\u0394E/\u0394R)",
           color="#FF9800", alpha=0.8)
    ax.set_ylabel("dE_xc / dR (Ha / Bohr)")
    ax.set_title("Nuclear Gradient: Analytical vs Finite Diff",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "differentiable_energy.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
