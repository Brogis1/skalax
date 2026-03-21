#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark 2: Gradient equivalence between PyTorch autograd and JAX autodiff.

Compares torch.autograd.grad vs jax.grad for dE_xc/d(features),
plus a finite-difference self-check for the JAX gradients.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import numpy as np
import jax.numpy as jnp
import torch

import argparse

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_torch_model,
    load_jax_model_from_torch,
    generate_mol,
    to_torch,
    to_jax,
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

THRESHOLD = 1e-5
COORD_THRESHOLD = 1e-3  # coordinate grads can be noisier


def collect_data():
    """Compute all gradient comparisons and return data dict."""
    print_info("Loading PyTorch model...")
    torch_model, _ = load_torch_model()

    print_info("Loading JAX model (weight transfer)...")
    jax_model = load_jax_model_from_torch(torch_model)

    mol_np = generate_mol(n_points=100, n_atoms=5)

    print_info("Computing PyTorch gradients...")
    mol_torch = to_torch(mol_np, requires_grad=True)
    exc_torch = torch_model.get_exc(mol_torch)
    grads_torch = torch.autograd.grad(
        exc_torch, list(mol_torch.values()), torch.ones_like(exc_torch)
    )
    grads_torch = {k: g.detach().numpy() for k, g in zip(mol_torch.keys(), grads_torch)}

    print_info("Computing JAX gradients...")
    mol_jax = to_jax(mol_np)
    grads_jax = jax.grad(lambda m: jax_model.get_exc(m))(mol_jax)
    grads_jax = {k: np.array(v) for k, v in grads_jax.items()}

    feature_keys = ["density", "grad", "kin", "grid_weights", "grid_coords", "coarse_0_atomic_coords"]
    plot_data = {"features": [], "l2_err": [], "rel_l2": [], "max_err": []}

    for key in feature_keys:
        gt, gj = grads_torch[key], grads_jax[key]
        l2_err = float(np.linalg.norm(gt - gj))
        rel_l2 = float(l2_err / (np.linalg.norm(gt) + 1e-30))
        max_err = float(np.abs(gt - gj).max())
        plot_data["features"].append(key)
        plot_data["l2_err"].append(l2_err)
        plot_data["rel_l2"].append(rel_l2)
        plot_data["max_err"].append(max_err)

    # Finite-difference self-check
    print_info("Finite-difference verification (density[0,0])...")
    eps = 1e-5
    mol_plus = {k: jnp.array(v.copy()) for k, v in mol_np.items()}
    mol_minus = {k: jnp.array(v.copy()) for k, v in mol_np.items()}
    mol_plus["density"] = mol_plus["density"].at[0, 0].add(eps)
    mol_minus["density"] = mol_minus["density"].at[0, 0].add(-eps)

    exc_plus = float(jax_model.get_exc(mol_plus))
    exc_minus = float(jax_model.get_exc(mol_minus))
    fd_grad = (exc_plus - exc_minus) / (2 * eps)
    analytical_grad = float(grads_jax["density"][0, 0])
    fd_err = abs(fd_grad - analytical_grad)
    fd_rel = fd_err / (abs(analytical_grad) + 1e-30)

    plot_data["fd"] = {
        "analytical": analytical_grad,
        "fd": fd_grad,
        "abs_err": fd_err,
        "rel_err": fd_rel,
    }
    return plot_data


def print_report(data):
    """Print pass/fail report from data dict."""
    all_pass = True
    rows = []
    for i, key in enumerate(data["features"]):
        threshold = COORD_THRESHOLD if "coord" in key else THRESHOLD
        status = "PASS" if data["rel_l2"][i] < threshold else "FAIL"
        if data["rel_l2"][i] >= threshold:
            all_pass = False
        rows.append([key, f"{data['l2_err'][i]:.3e}", f"{data['rel_l2'][i]:.3e}",
                      f"{data['max_err'][i]:.3e}", status])
    print(format_table(["Feature", "L2 Error", "Rel L2", "Max Abs Error", "Status"], rows))

    fd = data["fd"]
    print(f"\n    FD check: analytical={fd['analytical']:.6e}  fd={fd['fd']:.6e}  rel_err={fd['rel_err']:.3e}")
    if fd["rel_err"] >= 1e-4:
        print_fail("Finite-difference check failed")
        all_pass = False
    else:
        print_pass("Finite-difference check passed")

    print()
    if all_pass:
        print_pass("All gradient comparisons within tolerance")
    else:
        print_fail("Some gradient comparisons exceed tolerance")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data only")
    args = parser.parse_args()

    print_header("Benchmark 2: Gradient Equivalence (PyTorch vs JAX)")

    if args.replot:
        plot_data = load_benchmark_data("gradient_equivalence")
        if plot_data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        plot_data = collect_data()
        save_benchmark_data("gradient_equivalence", plot_data)

    all_pass = print_report(plot_data)
    _make_plot(plot_data)
    return 0 if all_pass else 1


def _make_plot(data):
    """Generate gradient comparison bar chart."""
    plt = setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    features = data["features"]
    x = np.arange(len(features))

    # Separate density/local features vs coordinate features
    colors = []
    for feat in features:
        if "coord" in feat:
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")
    bars = ax.bar(x, data["rel_l2"], color=colors, edgecolor="white", linewidth=0.5)

    # Color bars that exceed threshold
    for i, (bar, feat) in enumerate(zip(bars, features)):
        threshold = COORD_THRESHOLD if "coord" in feat else THRESHOLD
        if data["rel_l2"][i] > threshold:
            bar.set_color("#F44336")

    ax.axhline(y=THRESHOLD, color="#D32F2F", linestyle="--", linewidth=1.5,
               label=f"Feature gradient pass/fail ({THRESHOLD:.0e})")
    ax.axhline(y=COORD_THRESHOLD, color="#E65100", linestyle=":", linewidth=1.5,
               label=f"Coordinate gradient pass/fail ({COORD_THRESHOLD:.0e})")

    # Manual legend entries for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Density/local features"),
        Patch(facecolor="#FF9800", label="Coordinate features"),
        plt.Line2D([0], [0], color="#D32F2F", linestyle="--", linewidth=1.5,
                    label=f"Feature threshold ({THRESHOLD:.0e})"),
        plt.Line2D([0], [0], color="#E65100", linestyle=":", linewidth=1.5,
                    label=f"Coord threshold ({COORD_THRESHOLD:.0e})"),
    ]

    ax.set_yscale("log")
    ax.set_ylabel("Relative L2 Error (torch.autograd vs jax.grad)")
    ax.set_title("Gradient Equivalence: PyTorch vs JAX", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=25, ha="right")
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gradient_equivalence.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
