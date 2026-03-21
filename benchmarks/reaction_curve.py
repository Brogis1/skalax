#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark 5: H2 dissociation curve consistency between PyTorch and JAX.

Computes the potential energy surface along the H-H bond stretch and
verifies that both implementations produce matching energies.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import numpy as np

import argparse

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_torch_model,
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

THRESHOLD_HA = 0.001  # 1 mHa
N_POINTS = 15
D_MIN = 0.5
D_MAX = 3.0


def run_scf_at_distance(d, jax_model, torch_func):
    """Run SCF at a given H-H distance for both frameworks."""
    from skalax.pyscf import JaxSkalaKS
    from skala.pyscf import SkalaKS

    atom_str = f"H 0 0 0; H 0 0 {d:.6f}"
    mol = make_pyscf_mol(atom_str, basis="sto-3g")

    # JAX
    ks_jax = JaxSkalaKS(mol, xc=jax_model)
    ks_jax.verbose = 0
    ks_jax.conv_tol = 1e-9
    e_jax = float(ks_jax.kernel())

    # PyTorch
    ks_torch = SkalaKS(mol, xc=torch_func)
    ks_torch.verbose = 0
    ks_torch.conv_tol = 1e-9
    e_torch = float(ks_torch.kernel())

    return e_jax, e_torch


def collect_data():
    """Run H2 dissociation curve and return data dict."""
    print_info("Loading functionals...")
    _, torch_func = load_torch_model()
    jax_model = load_jax_model()

    distances = np.linspace(D_MIN, D_MAX, N_POINTS)
    e_jax_list, e_torch_list, diffs = [], [], []

    print_info(f"Computing {N_POINTS} points from {D_MIN} to {D_MAX} A...")
    for i, d in enumerate(distances):
        e_jax, e_torch = run_scf_at_distance(d, jax_model, torch_func)
        diff = abs(e_jax - e_torch)
        e_jax_list.append(e_jax)
        e_torch_list.append(e_torch)
        diffs.append(diff)
        sys.stdout.write(
            f"\r  Point {i+1}/{N_POINTS}: "
            f"d={d:.3f} A, diff={diff:.2e} Ha"
        )
        sys.stdout.flush()
    print()

    return {
        "distances": distances.tolist(),
        "e_torch": e_torch_list,
        "e_jax": e_jax_list,
        "diffs": diffs,
    }


def print_report(data):
    """Print pass/fail from data dict."""
    rows = []
    for i, d in enumerate(data["distances"]):
        rows.append([
            f"{d:.3f}",
            f"{data['e_torch'][i]:.8f}",
            f"{data['e_jax'][i]:.8f}",
            f"{data['diffs'][i]:.2e}",
        ])
    print(format_table(
        ["d (A)", "E_torch (Ha)", "E_jax (Ha)", "Diff (Ha)"], rows
    ))

    mae = np.mean(data["diffs"])
    max_err = np.max(data["diffs"])
    all_pass = max_err < THRESHOLD_HA

    print(f"\n  MAE:       {mae:.2e} Ha")
    print(f"  Max Error: {max_err:.2e} Ha")
    print()
    if all_pass:
        print_pass(f"All point-by-point diffs < {THRESHOLD_HA} Ha")
    else:
        print_fail(f"Max error {max_err:.2e} exceeds {THRESHOLD_HA} Ha")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data")
    args = parser.parse_args()

    print_header("Benchmark 5: H2 Dissociation Curve")

    if args.replot:
        data = load_benchmark_data("reaction_curve")
        if data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        data = collect_data()
        save_benchmark_data("reaction_curve", data)

    all_pass = print_report(data)
    _make_plot(
        data["distances"], data["e_torch"], data["e_jax"], data["diffs"]
    )
    return 0 if all_pass else 1


def _make_plot(distances, e_torch, e_jax, diffs):
    """Generate H2 dissociation curve plot."""
    plt = setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Top: energy curves
    ax = axes[0]
    ax.plot(distances, e_torch, "o-", color="#2196F3",
            label="PyTorch Skala (PySCF KS-DFT)", markersize=5)
    ax.plot(distances, e_jax, "s--", color="#FF9800",
            label="JAX Skala (PySCF KS-DFT)", markersize=5, alpha=0.8)
    ax.set_ylabel("Total Energy (Hartree)")
    ax.set_title("H$_2$ Dissociation Curve: PyTorch vs JAX", fontweight="bold")
    ax.legend(fontsize=10)

    # Bottom: residuals
    ax = axes[1]
    ax.bar(distances, diffs, width=(distances[1] - distances[0]) * 0.6,
           color="#4CAF50", edgecolor="white", label="|E_torch - E_jax|")
    ax.axhline(y=THRESHOLD_HA, color="#D32F2F", linestyle="--", linewidth=1.5,
               label=f"Pass threshold ({THRESHOLD_HA * 1000:.0f} mHa)")
    ax.set_ylabel("|E_torch - E_jax| (Hartree)")
    ax.set_xlabel("H-H Distance (\u00c5)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reaction_curve.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
