#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CH4 symmetric stretch: PyTorch vs JAX energy curves + NPE analysis.

Stretches all C-H bonds simultaneously (symmetric breathing mode) and
compares total energies from PyTorch and JAX Skala SCF calculations.
Reports the non-parallelity error (NPE) between the two curves.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import argparse
import numpy as np

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

# Tetrahedral CH4 unit vectors (C at origin, H along cube diagonals)
_H_DIRS = np.array([
    [1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
], dtype=float) / np.sqrt(3.0)

# Equilibrium C-H distance in Angstrom
R_EQ = 1.089
N_POINTS = 12
SCALE_MIN = 0.85   # 85% of equilibrium
SCALE_MAX = 2.0    # 200% of equilibrium


def ch4_atom_str(scale: float) -> str:
    """Build PySCF atom string for CH4 at given scale factor."""
    r = R_EQ * scale
    coords = _H_DIRS * r
    parts = ["C 0.0 0.0 0.0"]
    for i in range(4):
        parts.append(
            f"H {coords[i, 0]:.8f} {coords[i, 1]:.8f} {coords[i, 2]:.8f}"
        )
    return "; ".join(parts)


def run_scf(atom_str, functional, framework):
    """Run SCF and return (energy, converged)."""
    mol = make_pyscf_mol(atom_str, basis="sto-3g")
    if framework == "jax":
        from skalax.pyscf import JaxSkalaKS
        ks = JaxSkalaKS(mol, xc=functional)
    else:
        from skala.pyscf import SkalaKS
        ks = SkalaKS(mol, xc=functional)
    ks.verbose = 0
    ks.conv_tol = 1e-9
    energy = ks.kernel()
    return float(energy), bool(ks.converged)


def collect_data():
    """Compute CH4 symmetric stretch curves for both frameworks."""
    print_info("Loading PyTorch functional...")
    _, torch_func = load_torch_model()

    print_info("Loading JAX model...")
    jax_model = load_jax_model()

    scales = np.linspace(SCALE_MIN, SCALE_MAX, N_POINTS)
    distances = (scales * R_EQ).tolist()
    e_torch, e_jax, diffs = [], [], []
    conv_torch_list, conv_jax_list = [], []

    print_info(
        f"Computing {N_POINTS} points, "
        f"C-H = {SCALE_MIN * R_EQ:.3f} to {SCALE_MAX * R_EQ:.3f} A..."
    )
    for i, s in enumerate(scales):
        print(f"Scale {i + 1}/{N_POINTS}: Computing SCF...  ", end="  \r")
        atom_str = ch4_atom_str(s)
        print("\nRunning torch SCF...")
        et, ct = run_scf(atom_str, torch_func, "torch")
        print("\nRunning jax SCF...")
        ej, cj = run_scf(atom_str, jax_model, "jax")
        diff = abs(et - ej)
        e_torch.append(et)
        e_jax.append(ej)
        diffs.append(diff)
        conv_torch_list.append(ct)
        conv_jax_list.append(cj)
        sys.stdout.write(
            f"\r  Point {i + 1}/{N_POINTS}: "
            f"r_CH={s * R_EQ:.3f} A, diff={diff:.2e} Ha"
        )
        sys.stdout.flush()
    print()

    return {
        "distances": distances,
        "scales": scales.tolist(),
        "e_torch": e_torch,
        "e_jax": e_jax,
        "diffs": diffs,
        "conv_torch": conv_torch_list,
        "conv_jax": conv_jax_list,
    }


def compute_npe(data):
    """Compute non-parallelity error (NPE) between two curves.

    NPE = max(delta_E) - min(delta_E) where delta_E = E_jax - E_torch.
    This measures how non-parallel the two curves are.
    """
    delta = np.array(data["e_jax"]) - np.array(data["e_torch"])
    npe = float(np.max(delta) - np.min(delta))
    return delta.tolist(), npe


def print_report(data):
    """Print detailed report with NPE."""
    delta, npe = compute_npe(data)

    rows = []
    for i, d in enumerate(data["distances"]):
        rows.append([
            f"{d:.3f}",
            f"{data['e_torch'][i]:.8f}",
            f"{data['e_jax'][i]:.8f}",
            f"{delta[i]:+.6f}",
            f"{data['diffs'][i]:.2e}",
        ])
    print(format_table(
        ["r_CH (A)", "E_torch (Ha)", "E_jax (Ha)",
         "E_jax-E_torch (Ha)", "|diff| (Ha)"],
        rows,
    ))

    mae = np.mean(data["diffs"])
    max_err = np.max(data["diffs"])

    print(f"\n  MAE:                    {mae:.4f} Ha  ({mae * 627.509:.2f} kcal/mol)")
    print(f"  Max |E_torch - E_jax|:  {max_err:.4f} Ha  ({max_err * 627.509:.2f} kcal/mol)")
    print(f"  NPE (non-parallelity):  {npe:.4f} Ha  ({npe * 627.509:.2f} kcal/mol)")
    print(f"  NPE:                    {npe * 1000:.2f} mHa")
    print()

    # NPE < 1 kcal/mol is a reasonable threshold for functional comparison
    npe_kcal = npe * 627.509
    if npe_kcal < 1.0:
        print_pass(f"NPE = {npe_kcal:.2f} kcal/mol (< 1 kcal/mol)")
    else:
        print_fail(f"NPE = {npe_kcal:.2f} kcal/mol (>= 1 kcal/mol)")

    return npe_kcal < 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data")
    args = parser.parse_args()

    print_header("CH4 Symmetric Stretch: PyTorch vs JAX (NPE Analysis)")

    if args.replot:
        data = load_benchmark_data("ch4_stretch")
        if data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        data = collect_data()
        save_benchmark_data("ch4_stretch", data)

    all_pass = print_report(data)
    _make_plot(data)
    return 0 if all_pass else 1


def _make_plot(data):
    """Generate CH4 stretch plot with energy curves and NPE panel."""
    plt = setup_plot_style()
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 10),
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
        sharex=True,
    )

    distances = data["distances"]
    e_torch = np.array(data["e_torch"])
    e_jax = np.array(data["e_jax"])
    delta, npe = compute_npe(data)
    delta = np.array(delta)
    diffs = np.array(data["diffs"])

    # --- Panel 1: Energy curves ---
    ax = axes[0]
    ax.plot(distances, e_torch, "o-", color="#2196F3",
            label="PyTorch Skala", markersize=5, linewidth=2)
    ax.plot(distances, e_jax, "s--", color="#FF9800",
            label="JAX Skala", markersize=5, linewidth=2, alpha=0.85)
    ax.set_ylabel("Total Energy (Hartree)")
    ax.set_title(
        "CH$_4$ Symmetric Stretch: PyTorch vs JAX",
        fontweight="bold", fontsize=14,
    )
    ax.legend(fontsize=11)

    # --- Panel 2: Signed difference (E_jax - E_torch) ---
    ax = axes[1]
    ax.plot(distances, delta * 1000, "D-", color="#7B1FA2",
            markersize=4, linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    # shade the NPE band
    ax.axhspan(
        np.min(delta) * 1000, np.max(delta) * 1000,
        alpha=0.15, color="#7B1FA2",
        label=f"NPE = {npe * 1000:.2f} mHa ({npe * 627.509:.2f} kcal/mol)",
    )
    ax.set_ylabel("E$_{JAX}$ \u2013 E$_{Torch}$ (mHa)")
    ax.set_title("Signed Difference (non-parallelity)", fontweight="bold")
    ax.legend(fontsize=10)

    # --- Panel 3: Absolute difference ---
    ax = axes[2]
    w = (distances[1] - distances[0]) * 0.6 if len(distances) > 1 else 0.05
    colors = ["#4CAF50" if d < 0.002 else "#F44336" for d in diffs]
    ax.bar(distances, diffs * 1000, width=w,
           color=colors, edgecolor="white",
           label="|E$_{Torch}$ \u2013 E$_{JAX}$|")
    ax.axhline(y=2.0, color="#D32F2F", linestyle="--", linewidth=1.5,
               label="2 mHa threshold")
    ax.set_ylabel("|E$_{Torch}$ \u2013 E$_{JAX}$| (mHa)")
    ax.set_xlabel("C\u2013H Distance (\u00c5)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ch4_stretch.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
