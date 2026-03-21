#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark 1: Forward pass equivalence between PyTorch and JAX.

Verifies that PyTorch and JAX produce identical E_xc and exc_density
for both the weight-transfer path and the NPZ path.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import numpy as np
import torch

import argparse

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_torch_model,
    load_jax_model,
    load_jax_model_from_torch,
    generate_mol,
    to_torch,
    to_jax,
    format_table,
    print_header,
    print_pass,
    print_fail,
    print_info,
    setup_plot_style,
    save_benchmark_data,
    load_benchmark_data,
)

THRESHOLD_DENSITY = 5e-6   # per-point max absolute error for get_exc_density
THRESHOLD_REL = 1e-6       # relative error for get_exc (scalar)

CONFIGS = [
    (50, 3),
    (200, 5),
    (1000, 10),
]


def run_forward_comparison(torch_model, jax_model, mol_np, label=""):
    """Run forward pass on both models and compute error metrics."""
    mol_torch = to_torch(mol_np)
    mol_jax = to_jax(mol_np)

    with torch.no_grad():
        exc_torch = torch_model.get_exc(mol_torch).numpy()
        exc_density_torch = torch_model.get_exc_density(mol_torch).numpy()

    exc_jax = np.array(jax_model.get_exc(mol_jax))
    exc_density_jax = np.array(jax_model.get_exc_density(mol_jax))

    # get_exc metrics
    exc_mae = float(np.abs(exc_torch - exc_jax))
    exc_max = float(np.abs(exc_torch - exc_jax))
    exc_rel = float(exc_mae / (np.abs(exc_torch) + 1e-30))

    # get_exc_density metrics
    diff = np.abs(exc_density_torch - exc_density_jax)
    dens_mae = float(diff.mean())
    dens_max = float(diff.max())
    dens_rel = float(diff.sum() / (np.abs(exc_density_torch).sum() + 1e-30))

    return {
        "exc_mae": exc_mae,
        "exc_max": exc_max,
        "exc_rel": exc_rel,
        "dens_mae": dens_mae,
        "dens_max": dens_max,
        "dens_rel": dens_rel,
    }


def collect_data():
    """Run all forward comparisons and return plot data dict."""
    print_info("Loading PyTorch model...")
    torch_model, _ = load_torch_model()

    print_info("Loading JAX model via weight transfer...")
    jax_transfer = load_jax_model_from_torch(torch_model)

    print_info("Loading JAX model from NPZ...")
    jax_npz = load_jax_model()

    plot_data = {"configs": [], "transfer": {"exc": [], "dens": []}, "npz": {"exc": [], "dens": []}}

    for n_points, n_atoms in CONFIGS:
        config_label = f"{n_points}pt/{n_atoms}atom"
        mol_np = generate_mol(n_points, n_atoms)
        plot_data["configs"].append(config_label)

        r = run_forward_comparison(torch_model, jax_transfer, mol_np, "transfer")
        plot_data["transfer"]["exc"].append(r["exc_rel"])
        plot_data["transfer"]["dens"].append(r["dens_max"])

        r2 = run_forward_comparison(torch_model, jax_npz, mol_np, "npz")
        plot_data["npz"]["exc"].append(r2["exc_rel"])
        plot_data["npz"]["dens"].append(r2["dens_max"])

    return plot_data


def print_report(plot_data):
    """Print pass/fail report from collected data."""
    all_pass = True
    for i, cfg in enumerate(plot_data["configs"]):
        for path_name, path_data in [("transfer", plot_data["transfer"]), ("npz", plot_data["npz"])]:
            if path_data["exc"][i] > THRESHOLD_REL or path_data["dens"][i] > THRESHOLD_DENSITY:
                all_pass = False
    if all_pass:
        print_pass(f"All relative errors < {THRESHOLD_REL:.0e}, density max errors < {THRESHOLD_DENSITY:.0e}")
    else:
        print_fail("Some errors exceed thresholds")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data only")
    args = parser.parse_args()

    print_header("Benchmark 1: Forward Pass Equivalence (PyTorch vs JAX)")

    if args.replot:
        plot_data = load_benchmark_data("forward_equivalence")
        if plot_data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        plot_data = collect_data()
        save_benchmark_data("forward_equivalence", plot_data)

    all_pass = print_report(plot_data)
    _make_plot(plot_data)
    return 0 if all_pass else 1


def _make_plot(data):
    """Generate forward equivalence bar chart."""
    plt = setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    configs = data["configs"]
    x = np.arange(len(configs))
    w = 0.35

    for ax, key, title, ylabel in [
        (axes[0], "exc", "Total XC Energy (get_exc)", "Relative Error"),
        (axes[1], "dens", "Per-Point XC Density (get_exc_density)", "Max Absolute Error"),
    ]:
        ax.bar(x - w / 2, data["transfer"][key], w,
               label="PyTorch \u2192 JAX (direct weight transfer)", color="#2196F3")
        ax.bar(x + w / 2, data["npz"][key], w,
               label="JAX from NPZ (bundled weights)", color="#FF9800")
        thresh = THRESHOLD_DENSITY if key == "dens" else THRESHOLD_REL
        ax.axhline(y=thresh, color="#D32F2F", linestyle="--", linewidth=1.5,
                    label=f"Pass threshold ({thresh:.0e})")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Forward Pass Equivalence: PyTorch vs JAX", fontweight="bold", fontsize=15)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "forward_equivalence.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
