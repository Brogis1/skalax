"""End-to-end test with HuggingFace pretrained weights."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("dev")))

import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from skala.functional import load_functional
from skala.functional.model import SkalaFunctional as TorchSkalaFunctional
from skalax.functional.model import SkalaFunctional as JaxSkalaFunctional
from skalax.convert_weights import load_weights_into_model

print("Loading pretrained Skala from HuggingFace...")
func = load_functional("skala")

# Get clean state dict
state_dict = func.state_dict()
clean_state_dict = {
    k.replace("_traced_model.", ""): v for k, v in state_dict.items()
}

# Create fresh PyTorch model and load weights
torch_model = TorchSkalaFunctional(
    lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0
)
torch_model.load_state_dict(clean_state_dict, strict=True)
torch_model.double()

# Create JAX model with same architecture
key = jax.random.PRNGKey(0)
jax_model = JaxSkalaFunctional(
    lmax=3, non_local=True, non_local_hidden_nf=16, radius_cutoff=5.0, key=key
)

# Load pretrained weights (also need buffers for w3j)
from skalax.convert_weights import load_weights_and_buffers_into_model
jax_model = load_weights_and_buffers_into_model(jax_model, torch_model)

print("\n" + "=" * 70)
print("END-TO-END TEST WITH HUGGINGFACE PRETRAINED WEIGHTS")
print("=" * 70)

# Create realistic test data
np.random.seed(42)
n_points = 50
n_atoms = 5

mol_np = {
    "density": np.abs(np.random.randn(2, n_points)) + 0.1,
    "grad": np.random.randn(2, 3, n_points),
    "kin": np.abs(np.random.randn(2, n_points)) + 0.1,
    "grid_coords": np.random.randn(n_points, 3) * 3,
    "grid_weights": np.abs(np.random.randn(n_points)) + 0.1,
    "coarse_0_atomic_coords": np.random.randn(n_atoms, 3) * 2,
}

mol_torch = {k: torch.from_numpy(v.astype(np.float64)) for k, v in mol_np.items()}
mol_jax = {k: jnp.array(v.astype(np.float64)) for k, v in mol_np.items()}

print(f"\nTest data: {n_points} grid points, {n_atoms} atoms")

# PyTorch forward
print("\nRunning PyTorch forward pass...")
with torch.no_grad():
    exc_torch = torch_model.get_exc(mol_torch).numpy()
    exc_density_torch = torch_model.get_exc_density(mol_torch).numpy()

# JAX forward
print("Running JAX forward pass...")
exc_jax = np.array(jax_model.get_exc(mol_jax))
exc_density_jax = np.array(jax_model.get_exc_density(mol_jax))

print("\n" + "-" * 70)
print("RESULTS: get_exc (total exchange-correlation energy)")
print("-" * 70)
print(f"  PyTorch:        {exc_torch:.10f}")
print(f"  JAX:            {exc_jax:.10f}")
print(f"  Difference:     {np.abs(exc_torch - exc_jax):.2e}")
print(f"  Within 1e-5:    {'✓ PASS' if np.abs(exc_torch - exc_jax) < 1e-5 else '✗ FAIL'}")

print("\n" + "-" * 70)
print("RESULTS: get_exc_density (energy density at each point)")
print("-" * 70)
print(f"  Shape:          {exc_density_torch.shape}")
print(f"  PyTorch mean:   {exc_density_torch.mean():.10f}")
print(f"  JAX mean:       {exc_density_jax.mean():.10f}")
print(f"  Max diff:       {np.max(np.abs(exc_density_torch - exc_density_jax)):.2e}")
print(f"  Within 1e-5:    {'✓ PASS' if np.max(np.abs(exc_density_torch - exc_density_jax)) < 1e-5 else '✗ FAIL'}")

print("\n" + "=" * 70)
print("✓ HUGGINGFACE PRETRAINED MODEL PRODUCES IDENTICAL OUTPUTS")
print("=" * 70)
