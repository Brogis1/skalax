#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Benchmark 3: Performance comparison — PyTorch vs JAX (eager + compiled/JIT).

Measures runtime for forward and backward passes across grid sizes.
PyTorch uses torch.compile; JAX uses eqx.filter_jit.
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


def time_fn(fn, n_repeats: int, warmup: int = 2) -> tuple[float, float]:
    """Time a function, returning (mean_ms, std_ms)."""
    # warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def collect_data(quick=False):
    """Run all timing measurements and return data dict."""
    if quick:
        grid_sizes = [32, 256]
        n_repeats = 2
    else:
        grid_sizes = [32, 128, 512, 2048]
        n_repeats = 5

    n_atoms = 3

    print_info("Loading PyTorch model (eager)...")
    torch_model, _ = load_torch_model()

    # Compile torch_model.get_exc with torch.compile for optimised execution.
    print_info("Compiling PyTorch model (torch.compile)...")
    torch_compiled = torch.compile(torch_model)

    print_info("Loading JAX model (with non-local)...")
    jax_model = load_jax_model_from_torch(torch_model)

    print_info("Loading JAX model for JIT...")
    jax_jit_model = load_jax_model_from_torch(torch_model)

    @eqx.filter_jit
    def jit_get_exc(model, mol):
        return model.get_exc(mol)

    @eqx.filter_jit
    def jit_grad_exc(model, mol):
        return jax.grad(lambda m: model.get_exc(m))(mol)

    op_names = [
        "Torch fwd",
        "Torch compiled fwd",
        "Torch fwd+bwd",
        "Torch compiled fwd+bwd",
        "JAX fwd (eager)",
        "JAX JIT fwd",
        "JAX fwd+grad",
        "JAX JIT fwd+grad",
    ]
    plot_data = {"grid_sizes": [], "operations": {}}
    for name in op_names:
        plot_data["operations"][name] = {"mean": [], "std": []}

    for n_points in grid_sizes:
        print_info(f"Grid size: {n_points} points...")
        mol_np = generate_mol(n_points, n_atoms)
        mol_torch = to_torch(mol_np)
        mol_jax = to_jax(mol_np)
        mol_jax_jit = to_jax(mol_np)
        plot_data["grid_sizes"].append(n_points)

        def torch_fwd():
            with torch.no_grad():
                torch_model.get_exc(mol_torch)

        m, s = time_fn(torch_fwd, n_repeats)
        plot_data["operations"]["Torch fwd"]["mean"].append(m)
        plot_data["operations"]["Torch fwd"]["std"].append(s)

        def torch_compiled_fwd():
            with torch.no_grad():
                torch_compiled.get_exc(mol_torch)

        m, s = time_fn(torch_compiled_fwd, n_repeats)
        plot_data["operations"]["Torch compiled fwd"]["mean"].append(m)
        plot_data["operations"]["Torch compiled fwd"]["std"].append(s)

        def torch_fwd_bwd():
            mol_g = to_torch(mol_np, requires_grad=True)
            exc = torch_model.get_exc(mol_g)
            torch.autograd.grad(
                exc, list(mol_g.values()), torch.ones_like(exc)
            )

        m, s = time_fn(torch_fwd_bwd, n_repeats)
        plot_data["operations"]["Torch fwd+bwd"]["mean"].append(m)
        plot_data["operations"]["Torch fwd+bwd"]["std"].append(s)

        def torch_compiled_fwd_bwd():
            mol_g = to_torch(mol_np, requires_grad=True)
            exc = torch_compiled.get_exc(mol_g)
            torch.autograd.grad(
                exc, list(mol_g.values()), torch.ones_like(exc)
            )

        m, s = time_fn(torch_compiled_fwd_bwd, n_repeats)
        plot_data["operations"]["Torch compiled fwd+bwd"]["mean"].append(m)
        plot_data["operations"]["Torch compiled fwd+bwd"]["std"].append(s)

        def jax_fwd():
            jax_model.get_exc(mol_jax).block_until_ready()

        m, s = time_fn(jax_fwd, n_repeats)
        plot_data["operations"]["JAX fwd (eager)"]["mean"].append(m)
        plot_data["operations"]["JAX fwd (eager)"]["std"].append(s)

        def jax_jit_fwd():
            jit_get_exc(jax_jit_model, mol_jax_jit).block_until_ready()

        for _ in range(3):
            jax_jit_fwd()

        m, s = time_fn(jax_jit_fwd, n_repeats, warmup=0)
        plot_data["operations"]["JAX JIT fwd"]["mean"].append(m)
        plot_data["operations"]["JAX JIT fwd"]["std"].append(s)

        def jax_fwd_grad():
            jax.grad(lambda m: jax_model.get_exc(m))(
                mol_jax
            )["density"].block_until_ready()

        m, s = time_fn(jax_fwd_grad, n_repeats)
        plot_data["operations"]["JAX fwd+grad"]["mean"].append(m)
        plot_data["operations"]["JAX fwd+grad"]["std"].append(s)

        def jax_jit_fwd_grad():
            jit_grad_exc(jax_jit_model, mol_jax_jit)["density"].block_until_ready()

        for _ in range(3):
            jax_jit_fwd_grad()

        m, s = time_fn(jax_jit_fwd_grad, n_repeats, warmup=0)
        plot_data["operations"]["JAX JIT fwd+grad"]["mean"].append(m)
        plot_data["operations"]["JAX JIT fwd+grad"]["std"].append(s)

    return plot_data


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer sizes/repeats)")
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation; plot from saved data only")
    args = parser.parse_args()

    print_header("Benchmark 3: Performance (PyTorch vs JAX)")

    if args.replot:
        plot_data = load_benchmark_data("performance")
        if plot_data is None:
            print("ERROR: No saved data. Run without --replot first.")
            return 1
        print_info("Loaded saved data.")
    else:
        plot_data = collect_data(quick=args.quick)
        save_benchmark_data("performance", plot_data)

    # Print table
    rows = []
    for i, n in enumerate(plot_data["grid_sizes"]):
        for op_name, vals in plot_data["operations"].items():
            rows.append([
                n, op_name,
                f"{vals['mean'][i]:.2f}",
                f"{vals['std'][i]:.2f}",
            ])
    print(format_table(
        ["Grid Size", "Operation", "Mean (ms)", "Std (ms)"], rows
    ))

    _make_plot(plot_data)
    return 0


def _make_plot(data):
    """Generate performance comparison plot."""
    plt = setup_plot_style()
    fig, ax = plt.subplots(figsize=(13, 5.5))

    grid_sizes = data["grid_sizes"]
    x = np.arange(len(grid_sizes))

    ops = [
        ("Torch fwd",              "PyTorch forward",                "#42A5F5"),
        ("Torch compiled fwd",     "PyTorch compiled forward",       "#1565C0"),
        ("Torch fwd+bwd",          "PyTorch forward + backward",     "#FFA726"),
        ("Torch compiled fwd+bwd", "PyTorch compiled fwd + backward","#FF6F00"),
        ("JAX fwd (eager)",        "JAX forward (eager)",            "#AB47BC"),
        ("JAX JIT fwd",            "JAX forward (JIT)",              "#66BB6A"),
        ("JAX fwd+grad",           "JAX forward + grad (eager)",     "#E65100"),
        ("JAX JIT fwd+grad",       "JAX forward + grad (JIT)",       "#2E7D32"),
    ]
    w = 0.8 / len(ops)
    for i, (key, label, color) in enumerate(ops):
        vals = data["operations"][key]
        ax.bar(
            x + i * w - (len(ops) - 1) * w / 2,
            vals["mean"], w, yerr=vals["std"],
            label=label, color=color,
            edgecolor="white", linewidth=0.5, capsize=3,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Wall-Clock Time (ms, lower is better)")
    ax.set_xlabel("Grid Size (number of integration points)")
    ax.set_title("Performance: PyTorch vs JAX  [radius_cutoff=5.0, steady-state]",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(grid_sizes)
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "performance.png")
    plt.savefig(path)
    plt.close()
    print_info(f"Plot saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
