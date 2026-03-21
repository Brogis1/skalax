# SPDX-License-Identifier: MIT
"""Shared utilities for skala-jax benchmarks."""

import jax

jax.config.update("jax_enable_x64", True)

import sys
import os
import json
import time
import numpy as np
import jax.numpy as jnp
import torch

from skalax import (
    SkalaFunctional,
    load_weights_from_npz,
    load_config,
    get_default_weights_dir,
)
from skalax.convert_weights import load_weights_and_buffers_into_model

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_jax_model(non_local: bool = True) -> SkalaFunctional:
    """Load JAX model from bundled NPZ weights."""
    weights_dir = get_default_weights_dir()
    config = load_config(weights_dir)
    key = jax.random.PRNGKey(0)
    model = SkalaFunctional(
        lmax=config["lmax"],
        non_local=non_local if non_local is not None else config["non_local"],
        non_local_hidden_nf=config["non_local_hidden_nf"],
        radius_cutoff=config["radius_cutoff"],
        key=key,
    )
    model = load_weights_from_npz(model, weights_dir)
    return model


def load_torch_model():
    """Load PyTorch Skala model with float64 weights.

    Returns (torch_model, traced_functional):
        torch_model: SkalaFunctional with state dict loaded, in float64
        traced_functional: the traced functional from load_functional (for PySCF)
    """
    from skala.functional import load_functional
    from skala.functional.model import SkalaFunctional as TorchSkalaFunctional

    func_torch = load_functional("skala")
    state_dict = {
        k.replace("_traced_model.", ""): v
        for k, v in func_torch.state_dict().items()
    }
    model = TorchSkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
    )
    model.load_state_dict(state_dict, strict=True)
    model.double()
    model.eval()
    return model, func_torch


def load_jax_model_from_torch(torch_model) -> SkalaFunctional:
    """Create JAX model with weights transferred from PyTorch model (exact)."""
    key = jax.random.PRNGKey(0)
    jax_model = SkalaFunctional(
        lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
    )
    jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)
    return jax_model


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_mol(n_points: int, n_atoms: int, seed: int = 42) -> dict:
    """Generate a molecular feature dict with physically reasonable random values.

    Returns numpy arrays (float64).
    """
    rng = np.random.RandomState(seed)
    return {
        "density": np.abs(rng.randn(2, n_points)) + 0.1,
        "grad": rng.randn(2, 3, n_points),
        "kin": np.abs(rng.randn(2, n_points)) + 0.1,
        "grid_coords": rng.randn(n_points, 3) * 2,
        "grid_weights": np.abs(rng.randn(n_points)) + 0.1,
        "coarse_0_atomic_coords": rng.randn(n_atoms, 3) * 2,
    }


def to_torch(mol_np: dict, requires_grad: bool = False) -> dict:
    """Convert numpy mol dict to PyTorch float64 tensors."""
    return {
        k: torch.from_numpy(v.astype(np.float64)).requires_grad_(requires_grad)
        for k, v in mol_np.items()
    }


def to_jax(mol_np: dict) -> dict:
    """Convert numpy mol dict to JAX float64 arrays."""
    return {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}


# ---------------------------------------------------------------------------
# PySCF helpers
# ---------------------------------------------------------------------------

def make_pyscf_mol(atom_str: str, basis: str = "sto-3g", spin: int = 0):
    """Create a PySCF Mole object."""
    from pyscf import gto
    return gto.M(atom=atom_str, basis=basis, spin=spin, verbose=0)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def format_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None) -> str:
    """Format a simple ASCII table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for row in rows:
                if i < len(row):
                    w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    lines = [sep]

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 20
            parts.append(f" {str(cell):<{w-1}}")
        return "|" + "|".join(parts) + "|"

    lines.append(fmt_row(headers))
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def print_header(title: str):
    """Print a formatted section header."""
    width = max(60, len(title) + 4)
    print(f"\n{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}\n")


def print_pass(msg: str):
    print(f"  {GREEN}PASS{RESET}  {msg}")


def print_fail(msg: str):
    print(f"  {RED}FAIL{RESET}  {msg}")


def print_info(msg: str):
    print(f"  {CYAN}INFO{RESET}  {msg}")


def print_warn(msg: str):
    print(f"  {YELLOW}WARN{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def save_benchmark_data(name: str, data: dict):
    """Save benchmark data to JSON file in data/ directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print_info(f"Data saved to {path}")


def load_benchmark_data(name: str) -> dict | None:
    """Load benchmark data from JSON. Returns None if not found."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    return plt
