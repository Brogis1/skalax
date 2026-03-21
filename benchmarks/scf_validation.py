#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark 4: SCF convergence validation inside PySCF Kohn-Sham loops.

Verifies that PyTorch and JAX models produce consistent total energies
when integrated into a self-consistent field (SCF) calculation.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
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
    print_warn,
    setup_plot_style,
    save_benchmark_data,
    load_benchmark_data,
)

HA_TO_KCAL = 627.509

MOLECULES = {
    "H2": {
        "atom": "H 0 0 0; H 0 0 0.74",
        "basis": "sto-3g",
        "spin": 0,
    },
    "H2O": {
        "atom": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
        "basis": "sto-3g",
        "spin": 0,
    },
    "CH4": {
        "atom": "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629",
        "basis": "sto-3g",
        "spin": 0,
    },
}

THRESHOLD_HA = 0.002  # 2 mHa


def run_scf(mol_pyscf, functional, framework="jax"):
    """Run SCF and return (energy, converged)."""
    if framework == "jax":
        from skalax.pyscf import JaxSkalaKS
        ks = JaxSkalaKS(mol_pyscf, xc=functional)
    else:
        from skala.pyscf import SkalaKS
        ks = SkalaKS(mol_pyscf, xc=functional)

    ks.verbose = 0
    ks.conv_tol = 1e-9
    energy = ks.kernel()
    return float(energy), bool(ks.converged)


def collect_data(quick=False):
    """Run SCF for all molecules and return data dict."""
    print_info("Loading PyTorch functional...")
    _, torch_func = load_torch_model()

    print_info("Loading JAX model...")
    jax_model = load_jax_model()

    mol_names = ["H2"] if quick else list(MOLECULES.keys())

    plot_data = {
        "molecules": [], "torch_e": [], "jax_e": [], "diff_ha": [],
    }
    for name in mol_names:
        spec = MOLECULES[name]
        print_info(f"Running SCF for {name}...")
        mol_pyscf = make_pyscf_mol(
            spec["atom"], spec["basis"], spec["spin"]
        )
        e_torch, _ = run_scf(mol_pyscf, torch_func, "torch")
        e_jax, _ = run_scf(mol_pyscf, jax_model, "jax")
        plot_data["molecules"].append(name)
        plot_data["torch_e"].append(e_torch)
        plot_data["jax_e"].append(e_jax)
        plot_data["diff_ha"].append(abs(e_torch - e_jax))

    return plot_data


def print_report(data):
    """Print pass/fail from data dict."""
    all_pass = True
    rows = []
    for i, name in enumerate(data["molecules"]):
        d = data["diff_ha"][i]
        status = "PASS" if d < THRESHOLD_HA else "FAIL"
        if d >= THRESHOLD_HA:
            all_pass = False
        rows.append([
            name,
            f"{data['torch_e'][i]:.8f}",
            f"{data['jax_e'][i]:.8f}",
            f"{d:.2e}",
            f"{d * HA_TO_KCAL:.4f}",
            status,
        ])
    headers = [
        "Molecule", "E_torch (Ha)", "E_jax (Ha)",
        "Diff (Ha)", "Diff (kcal/mol)", "Status",
    ]
    print(format_table(headers, rows))
    print()
    if all_pass:
        print_pass(
            f"All energy diffs < {THRESHOLD_HA} Ha "
            f"({THRESHOLD_HA * HA_TO_KCAL:.2f} kcal/mol)"
        )
    else:
        print_fail(f"Some energy diffs exceed {THRESHOLD_HA} Ha")
    return all_pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="H2 only")
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data")
    args = parser.parse_args()

    print_header("Benchmark 4: SCF Convergence Validation")

    if args.replot:
        plot_data = load_benchmark_data("scf_validation")
        if plot_data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        plot_data = collect_data(quick=args.quick)
        save_benchmark_data("scf_validation", plot_data)

    all_pass = print_report(plot_data)
    _make_plot(plot_data)
    return 0 if all_pass else 1


def _make_plot(data):
    """Generate SCF validation bar chart."""
    plt = setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2, 1]})

    molecules = data["molecules"]
    x = np.arange(len(molecules))
    w = 0.35

    # Left: absolute energies
    ax = axes[0]
    ax.bar(x - w / 2, data["torch_e"], w,
           label="PyTorch Skala (PySCF SCF)", color="#2196F3")
    ax.bar(x + w / 2, data["jax_e"], w,
           label="JAX Skala (PySCF SCF)", color="#FF9800")
    ax.set_ylabel("Total SCF Energy (Hartree)")
    ax.set_title("Converged SCF Energies", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, fontsize=12)
    ax.legend(fontsize=10, loc="lower left")

    # Right: differences
    ax = axes[1]
    from matplotlib.patches import Patch
    colors = ["#4CAF50" if d < THRESHOLD_HA else "#F44336" for d in data["diff_ha"]]
    ax.bar(x, data["diff_ha"], color=colors, edgecolor="white")
    ax.axhline(y=THRESHOLD_HA, color="#D32F2F", linestyle="--", linewidth=1.5)

    legend_elements = [
        Patch(facecolor="#4CAF50", label="Below threshold (PASS)"),
        Patch(facecolor="#F44336", label="Above threshold (FAIL)"),
        plt.Line2D([0], [0], color="#D32F2F", linestyle="--", linewidth=1.5,
                    label=f"{THRESHOLD_HA * 1000:.0f} mHa threshold"),
    ]
    ax.set_yscale("log")
    ax.set_ylabel("|E_torch - E_jax| (Hartree)")
    ax.set_title("Energy Difference", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, fontsize=12)
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scf_validation.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
