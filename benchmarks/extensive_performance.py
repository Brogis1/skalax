#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark: Extensive performance sweep across a wide range of grid sizes.

Covers grid sizes from 32 to 2^15 (32768) points.
PyTorch uses torch.compile for a fair compiled-to-compiled comparison.
JAX JIT timings are steady-state (post-compilation).

Operations timed:
  - PyTorch forward (eager)
  - PyTorch forward (compiled via torch.compile)
  - PyTorch forward + backward (eager)
  - PyTorch forward + backward (compiled via torch.compile)
  - JAX forward (eager)
  - JAX forward (JIT, steady-state)
  - JAX forward + grad (eager)
  - JAX forward + grad (JIT, steady-state)
"""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import argparse
import time
import numpy as np
import equinox as eqx
import torch
sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_torch_model,
    load_jax_model_from_torch,
    generate_mol,
    to_torch,
    to_jax,
    format_table,
    print_header,
    print_info,
    setup_plot_style,
    save_benchmark_data,
    load_benchmark_data,
)

# Grid sizes: powers of 2 from 32 to 32768 (2^15)
GRID_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
N_ATOMS = 3
N_REPEATS = 5
N_WARMUP = 3

OP_NAMES = [
    "Torch fwd",
    "Torch compiled fwd",
    "Torch fwd+bwd",
    "Torch compiled fwd+bwd",
    "JAX fwd (eager)",
    "JAX JIT fwd",
    "JAX fwd+grad",
    "JAX JIT fwd+grad",
]


def time_fn(fn, n_repeats: int, warmup: int = 2) -> tuple[float, float]:
    """Time a function, returning (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def collect_data(grid_sizes=None, quick=False):
    """Run all timing measurements and return data dict."""
    if quick:
        sizes = [32, 256, 2048]
        n_rep = 2
    else:
        sizes = grid_sizes if grid_sizes is not None else GRID_SIZES
        n_rep = N_REPEATS

    print_info("Loading PyTorch model...")
    torch_model, _ = load_torch_model()

    print_info("Compiling PyTorch model (torch.compile)...")
    torch_compiled = torch.compile(torch_model)

    print_info("Loading JAX model...")
    jax_model = load_jax_model_from_torch(torch_model)
    jax_jit_model = load_jax_model_from_torch(torch_model)

    @eqx.filter_jit
    def jit_get_exc(model, mol):
        return model.get_exc(mol)

    @eqx.filter_jit
    def jit_grad_exc(model, mol):
        return jax.grad(lambda m: model.get_exc(m))(mol)

    plot_data = {"grid_sizes": [], "operations": {}}
    for name in OP_NAMES:
        plot_data["operations"][name] = {"mean": [], "std": []}

    for n_points in sizes:
        print_info(f"Grid size: {n_points} points...")
        mol_np = generate_mol(n_points, N_ATOMS)
        mol_torch = to_torch(mol_np)
        mol_jax = to_jax(mol_np)
        mol_jax_jit = to_jax(mol_np)
        plot_data["grid_sizes"].append(n_points)

        # --- PyTorch untraced forward ---
        def torch_fwd():
            with torch.no_grad():
                torch_model.get_exc(mol_torch)

        m, s = time_fn(torch_fwd, n_rep)
        plot_data["operations"]["Torch fwd"]["mean"].append(m)
        plot_data["operations"]["Torch fwd"]["std"].append(s)

        # --- PyTorch compiled forward (torch.compile) ---
        def torch_compiled_fwd():
            with torch.no_grad():
                torch_compiled.get_exc(mol_torch)

        m, s = time_fn(torch_compiled_fwd, n_rep)
        plot_data["operations"]["Torch compiled fwd"]["mean"].append(m)
        plot_data["operations"]["Torch compiled fwd"]["std"].append(s)

        # --- PyTorch untraced forward + backward ---
        def torch_fwd_bwd():
            mol_g = to_torch(mol_np, requires_grad=True)
            exc = torch_model.get_exc(mol_g)
            torch.autograd.grad(exc, list(mol_g.values()), torch.ones_like(exc))

        m, s = time_fn(torch_fwd_bwd, n_rep)
        plot_data["operations"]["Torch fwd+bwd"]["mean"].append(m)
        plot_data["operations"]["Torch fwd+bwd"]["std"].append(s)

        # --- PyTorch compiled forward + backward ---
        def torch_compiled_fwd_bwd():
            mol_g = to_torch(mol_np, requires_grad=True)
            exc = torch_compiled.get_exc(mol_g)
            torch.autograd.grad(exc, list(mol_g.values()), torch.ones_like(exc))

        m, s = time_fn(torch_compiled_fwd_bwd, n_rep)
        plot_data["operations"]["Torch compiled fwd+bwd"]["mean"].append(m)
        plot_data["operations"]["Torch compiled fwd+bwd"]["std"].append(s)

        # --- JAX eager forward ---
        def jax_fwd():
            jax_model.get_exc(mol_jax).block_until_ready()

        m, s = time_fn(jax_fwd, n_rep)
        plot_data["operations"]["JAX fwd (eager)"]["mean"].append(m)
        plot_data["operations"]["JAX fwd (eager)"]["std"].append(s)

        # --- JAX JIT forward (compile then time steady-state) ---
        for _ in range(N_WARMUP):
            jit_get_exc(jax_jit_model, mol_jax_jit).block_until_ready()

        def jax_jit_fwd():
            jit_get_exc(jax_jit_model, mol_jax_jit).block_until_ready()

        m, s = time_fn(jax_jit_fwd, n_rep, warmup=0)
        plot_data["operations"]["JAX JIT fwd"]["mean"].append(m)
        plot_data["operations"]["JAX JIT fwd"]["std"].append(s)

        # --- JAX eager forward + grad ---
        def jax_fwd_grad():
            jax.grad(lambda m: jax_model.get_exc(m))(mol_jax)["density"].block_until_ready()

        m, s = time_fn(jax_fwd_grad, n_rep)
        plot_data["operations"]["JAX fwd+grad"]["mean"].append(m)
        plot_data["operations"]["JAX fwd+grad"]["std"].append(s)

        # --- JAX JIT forward + grad ---
        for _ in range(N_WARMUP):
            jit_grad_exc(jax_jit_model, mol_jax_jit)["density"].block_until_ready()

        def jax_jit_fwd_grad():
            jit_grad_exc(jax_jit_model, mol_jax_jit)["density"].block_until_ready()

        m, s = time_fn(jax_jit_fwd_grad, n_rep, warmup=0)
        plot_data["operations"]["JAX JIT fwd+grad"]["mean"].append(m)
        plot_data["operations"]["JAX JIT fwd+grad"]["std"].append(s)

    return plot_data


def main():
    parser = argparse.ArgumentParser(description="Extensive performance benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (3 sizes, 2 repeats)")
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data only")
    args = parser.parse_args()

    print_header("Extensive Performance Benchmark: PyTorch vs JAX")

    if args.replot:
        plot_data = load_benchmark_data("extensive_performance")
        if plot_data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        plot_data = collect_data(quick=args.quick)
        save_benchmark_data("extensive_performance", plot_data)

    # Print table
    rows = []
    for i, n in enumerate(plot_data["grid_sizes"]):
        for op_name, vals in plot_data["operations"].items():
            rows.append([
                n, op_name,
                f"{vals['mean'][i]:.2f}",
                f"{vals['std'][i]:.2f}",
            ])
    print(format_table(["Grid Size", "Operation", "Mean (ms)", "Std (ms)"], rows))

    _make_plot(plot_data)
    return 0


def _make_plot(data):
    """Generate two-panel performance plot: forward-only and forward+grad."""
    plt = setup_plot_style()
    fig, (ax_fwd, ax_grad) = plt.subplots(1, 2, figsize=(16, 6))

    grid_sizes = data["grid_sizes"]
    x = np.array(grid_sizes)

    # Colors: Torch eager (light), Torch compiled (dark), JAX eager (orange), JAX JIT (green)
    style = {
        "Torch fwd":              dict(color="#42A5F5", ls="-",  marker="o", label="PyTorch (eager)"),
        "Torch compiled fwd":     dict(color="#1565C0", ls="--", marker="s", label="PyTorch (compiled)"),
        "Torch fwd+bwd":          dict(color="#42A5F5", ls="-",  marker="o", label="PyTorch (eager)"),
        "Torch compiled fwd+bwd": dict(color="#1565C0", ls="--", marker="s", label="PyTorch (compiled)"),
        "JAX fwd (eager)":        dict(color="#FFA726", ls="-",  marker="^", label="JAX (eager)"),
        "JAX JIT fwd":            dict(color="#2E7D32", ls="--", marker="D", label="JAX (JIT, steady-state)"),
        "JAX fwd+grad":           dict(color="#FFA726", ls="-",  marker="^", label="JAX (eager)"),
        "JAX JIT fwd+grad":       dict(color="#2E7D32", ls="--", marker="D", label="JAX (JIT, steady-state)"),
    }

    # Forward-only panel
    for key in ["Torch fwd", "Torch compiled fwd", "JAX fwd (eager)", "JAX JIT fwd"]:
        vals = data["operations"][key]
        means = np.array(vals["mean"])
        stds = np.array(vals["std"])
        s = style[key]
        ax_fwd.plot(x, means, color=s["color"], ls=s["ls"], marker=s["marker"],
                    label=s["label"], linewidth=2, markersize=5)
        ax_fwd.fill_between(x, means - stds, means + stds, color=s["color"], alpha=0.15)

    ax_fwd.set_xscale("log", base=2)
    ax_fwd.set_yscale("log")
    ax_fwd.set_xlabel("Grid Size (number of integration points)")
    ax_fwd.set_ylabel("Wall-Clock Time (ms, lower is better)")
    ax_fwd.set_title("Forward Pass", fontweight="bold")
    ax_fwd.set_xticks(x)
    ax_fwd.set_xticklabels([str(n) for n in grid_sizes], rotation=45, ha="right", fontsize=8)
    # Deduplicate legend entries
    handles, labels = ax_fwd.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax_fwd.legend(seen.values(), seen.keys(), fontsize=9)

    # Forward + grad/backward panel
    for key in ["Torch fwd+bwd", "Torch compiled fwd+bwd", "JAX fwd+grad", "JAX JIT fwd+grad"]:
        vals = data["operations"][key]
        means = np.array(vals["mean"])
        stds = np.array(vals["std"])
        s = style[key]
        ax_grad.plot(x, means, color=s["color"], ls=s["ls"], marker=s["marker"],
                     label=s["label"], linewidth=2, markersize=5)
        ax_grad.fill_between(x, means - stds, means + stds, color=s["color"], alpha=0.15)

    ax_grad.set_xscale("log", base=2)
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("Grid Size (number of integration points)")
    ax_grad.set_ylabel("Wall-Clock Time (ms, lower is better)")
    ax_grad.set_title("Forward + Gradient/Backward Pass", fontweight="bold")
    ax_grad.set_xticks(x)
    ax_grad.set_xticklabels([str(n) for n in grid_sizes], rotation=45, ha="right", fontsize=8)
    handles, labels = ax_grad.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax_grad.legend(seen.values(), seen.keys(), fontsize=9)

    fig.suptitle(
        "Performance: PyTorch vs JAX \u2014 radius_cutoff=5.0, CPU, steady-state after compilation",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "extensive_performance.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
