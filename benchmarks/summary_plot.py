#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Generate a single combined summary plot for LinkedIn / README.

Collects data from all benchmarks, saves to JSON, and renders a 6-panel figure.
If the JSON data file already exists, skips computation and just re-renders.

Usage:
    python summary_plot.py           # collect + plot
    python summary_plot.py --replot  # plot from saved data only
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import json
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_torch_model,
    load_jax_model,
    load_jax_model_from_torch,
    generate_mol,
    to_torch,
    to_jax,
    make_pyscf_mol,
    print_header,
    print_info,
    setup_plot_style,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "summary_data.json")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_forward_data(torch_model, jax_model):
    import torch as th
    configs = [(50, 3), (200, 5), (1000, 10)]
    rel_errors, labels = [], []
    for n_pts, n_at in configs:
        mol_np = generate_mol(n_pts, n_at)
        with th.no_grad():
            e_t = torch_model.get_exc(to_torch(mol_np)).numpy()
        e_j = np.array(jax_model.get_exc(to_jax(mol_np)))
        rel_errors.append(float(abs(e_t - e_j) / (abs(e_t) + 1e-30)))
        labels.append(f"{n_pts}pts")
    return labels, rel_errors


def collect_gradient_data(torch_model, jax_model):
    import torch as th
    mol_np = generate_mol(100, 5)
    mol_t = to_torch(mol_np, requires_grad=True)
    exc = torch_model.get_exc(mol_t)
    grads_t = th.autograd.grad(exc, list(mol_t.values()), th.ones_like(exc))
    grads_t = {k: g.detach().numpy() for k, g in zip(mol_t.keys(), grads_t)}
    mol_j = to_jax(mol_np)
    grads_j = jax.grad(lambda m: jax_model.get_exc(m))(mol_j)
    grads_j = {k: np.array(v) for k, v in grads_j.items()}
    features = ["density", "grad", "kin", "grid_weights"]
    rel_errors = []
    for f in features:
        err = np.linalg.norm(grads_t[f] - grads_j[f])
        rel_errors.append(float(err / (np.linalg.norm(grads_t[f]) + 1e-30)))
    return features, rel_errors


def collect_performance_data(torch_model, jax_model):
    import torch as th
    import equinox as eqx
    jax_jit_model = load_jax_model()

    @eqx.filter_jit
    def jit_fwd(model, mol):
        return model.get_exc(mol)

    grid_sizes = [32, 256, 1024]
    results = {"PyTorch forward": [], "JAX eager": [], "JAX JIT-compiled": []}
    for n in grid_sizes:
        mol_np = generate_mol(n, 3)
        mol_t, mol_j = to_torch(mol_np), to_jax(mol_np)
        mol_j_jit = to_jax(mol_np)
        # Warmup
        with th.no_grad():
            torch_model.get_exc(mol_t)
        jax_model.get_exc(mol_j).block_until_ready()
        for _ in range(3):
            jit_fwd(jax_jit_model, mol_j_jit).block_until_ready()
        reps = 3
        t0 = time.perf_counter()
        for _ in range(reps):
            with th.no_grad():
                torch_model.get_exc(mol_t)
        results["PyTorch forward"].append((time.perf_counter() - t0) / reps * 1000)
        t0 = time.perf_counter()
        for _ in range(reps):
            jax_model.get_exc(mol_j).block_until_ready()
        results["JAX eager"].append((time.perf_counter() - t0) / reps * 1000)
        t0 = time.perf_counter()
        for _ in range(reps):
            jit_fwd(jax_jit_model, mol_j_jit).block_until_ready()
        results["JAX JIT-compiled"].append((time.perf_counter() - t0) / reps * 1000)
    return grid_sizes, results


def collect_scf_data(jax_model, torch_func):
    from skalax.pyscf import JaxSkalaKS
    from skala.pyscf import SkalaKS
    molecules = {
        "H2": "H 0 0 0; H 0 0 0.74",
        "H2O": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    }
    names, diffs = [], []
    for name, atom in molecules.items():
        mol = make_pyscf_mol(atom, "sto-3g")
        ks_j = JaxSkalaKS(mol, xc=jax_model)
        ks_j.verbose, ks_j.conv_tol = 0, 1e-9
        e_j = float(ks_j.kernel())
        ks_t = SkalaKS(mol, xc=torch_func)
        ks_t.verbose, ks_t.conv_tol = 0, 1e-9
        e_t = float(ks_t.kernel())
        names.append(name)
        diffs.append(abs(e_j - e_t))
    return names, diffs


def collect_dissociation_data(jax_model, torch_func):
    from skalax.pyscf import JaxSkalaKS
    from skala.pyscf import SkalaKS
    distances = np.linspace(0.5, 3.0, 10)
    e_jax, e_torch = [], []
    for d in distances:
        mol = make_pyscf_mol(f"H 0 0 0; H 0 0 {d:.6f}", "sto-3g")
        ks_j = JaxSkalaKS(mol, xc=jax_model)
        ks_j.verbose, ks_j.conv_tol = 0, 1e-9
        e_jax.append(float(ks_j.kernel()))
        ks_t = SkalaKS(mol, xc=torch_func)
        ks_t.verbose, ks_t.conv_tol = 0, 1e-9
        e_torch.append(float(ks_t.kernel()))
    return distances.tolist(), e_torch, e_jax


def collect_all():
    """Run all data collection and return a JSON-serializable dict."""
    print_info("Loading models...")
    torch_model, torch_func = load_torch_model()
    jax_model = load_jax_model()
    jax_transfer = load_jax_model_from_torch(torch_model)

    print_info("Collecting forward equivalence...")
    fwd_labels, fwd_errors = collect_forward_data(torch_model, jax_transfer)

    print_info("Collecting gradient equivalence...")
    grad_features, grad_errors = collect_gradient_data(torch_model, jax_transfer)

    print_info("Collecting performance...")
    perf_sizes, perf_results = collect_performance_data(torch_model, jax_transfer)

    print_info("Collecting SCF validation...")
    scf_names, scf_diffs = collect_scf_data(jax_model, torch_func)

    print_info("Collecting H2 dissociation...")
    diss_d, diss_torch, diss_jax = collect_dissociation_data(jax_model, torch_func)

    return {
        "forward": {"labels": fwd_labels, "rel_errors": fwd_errors},
        "gradient": {"features": grad_features, "rel_errors": grad_errors},
        "performance": {"grid_sizes": perf_sizes, "results": perf_results},
        "scf": {"molecules": scf_names, "diffs": scf_diffs},
        "dissociation": {"distances": diss_d, "e_torch": diss_torch, "e_jax": diss_jax},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_summary_plot(data):
    """Render the 6-panel summary figure from collected data."""
    plt = setup_plot_style()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

    # ---- Panel A: Forward Equivalence ----
    ax = fig.add_subplot(gs[0, 0])
    fwd = data["forward"]
    x = np.arange(len(fwd["labels"]))
    ax.bar(x, fwd["rel_errors"], color="#2196F3", edgecolor="white",
           label="PyTorch vs JAX relative error")
    ax.axhline(y=1e-6, color="#D32F2F", linestyle="--", linewidth=1.5,
               label="Pass threshold (1e-6)")
    ax.set_yscale("log")
    ax.set_ylabel("Relative Error")
    ax.set_title("A. Forward Equivalence", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(fwd["labels"])
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel B: Gradient Equivalence ----
    ax = fig.add_subplot(gs[0, 1])
    grad = data["gradient"]
    x = np.arange(len(grad["features"]))
    ax.bar(x, grad["rel_errors"], color="#4CAF50", edgecolor="white",
           label="torch.autograd vs jax.grad")
    ax.axhline(y=1e-5, color="#D32F2F", linestyle="--", linewidth=1.5,
               label="Pass threshold (1e-5)")
    ax.set_yscale("log")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("B. Gradient Equivalence", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(grad["features"], rotation=15, fontsize=9)
    ax.legend(fontsize=8)

    # ---- Panel C: Performance ----
    ax = fig.add_subplot(gs[0, 2])
    perf = data["performance"]
    x = np.arange(len(perf["grid_sizes"]))
    w = 0.25
    colors_perf = ["#42A5F5", "#FFA726", "#66BB6A"]
    for i, (name, vals) in enumerate(perf["results"].items()):
        ax.bar(x + i * w - w, vals, w, label=name,
               color=colors_perf[i], edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, lower is better)")
    ax.set_title("C. Performance", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(perf["grid_sizes"])
    ax.set_xlabel("Grid points")
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel D: SCF Convergence ----
    ax = fig.add_subplot(gs[1, 0])
    scf = data["scf"]
    x = np.arange(len(scf["molecules"]))
    colors_scf = ["#4CAF50" if d < 0.002 else "#F44336" for d in scf["diffs"]]
    ax.bar(x, scf["diffs"], color=colors_scf, edgecolor="white",
           label="|E_torch - E_jax|")
    ax.axhline(y=0.002, color="#D32F2F", linestyle="--", linewidth=1.5,
               label="Pass threshold (2 mHa)")
    ax.set_yscale("log")
    ax.set_ylabel("|E_torch - E_jax| (Ha)")
    ax.set_title("D. SCF Validation", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scf["molecules"], fontsize=11)
    ax.legend(fontsize=8)

    # ---- Panel E: H2 Dissociation Curve ----
    ax = fig.add_subplot(gs[1, 1])
    diss = data["dissociation"]
    ax.plot(diss["distances"], diss["e_torch"], "o-", color="#2196F3",
            label="PyTorch Skala", markersize=4, linewidth=1.5)
    ax.plot(diss["distances"], diss["e_jax"], "s--", color="#FF9800",
            label="JAX Skala", markersize=4, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("H-H Distance (\u00c5)")
    ax.set_ylabel("Total Energy (Ha)")
    ax.set_title("E. H$_2$ Dissociation", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)

    # ---- Panel F: Dissociation Residuals ----
    ax = fig.add_subplot(gs[1, 2])
    diss_diffs = np.abs(np.array(diss["e_jax"]) - np.array(diss["e_torch"]))
    ax.bar(range(len(diss["distances"])), diss_diffs,
           color="#9C27B0", edgecolor="white", label="|E_torch - E_jax|")
    ax.axhline(y=0.001, color="#D32F2F", linestyle="--", linewidth=1.5,
               label="1 mHa threshold")
    ax.set_yscale("log")
    ax.set_ylabel("|E_torch - E_jax| (Ha)")
    ax.set_xlabel("Geometry index")
    ax.set_title("F. Curve Residuals", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Skala-JAX: Comprehensive Benchmark Suite",
        fontsize=16, fontweight="bold", y=0.98,
    )

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    for name, dpi in [("summary_linkedin.png", 200), ("summary_readme.png", 150)]:
        path = os.path.join(out_dir, name)
        plt.savefig(path, dpi=dpi)
        print_info(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip data collection; render from saved JSON")
    args = parser.parse_args()

    print_header("Summary Plot: Skala-JAX Benchmark Suite")

    if args.replot:
        if not os.path.exists(DATA_FILE):
            print(f"ERROR: {DATA_FILE} not found. Run without --replot first.")
            return 1
        print_info(f"Loading saved data from {DATA_FILE}")
        with open(DATA_FILE) as f:
            data = json.load(f)
    else:
        data = collect_all()
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print_info(f"Data saved to {DATA_FILE}")

    make_summary_plot(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
