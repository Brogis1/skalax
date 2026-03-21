import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from skalax import SkalaFunctional, load_weights_from_npz, load_config, get_default_weights_dir

# Load pretrained weights (bundled with package)
weights_dir = get_default_weights_dir()
config = load_config(weights_dir)

key = jax.random.PRNGKey(0)
model = SkalaFunctional(
    lmax=config["lmax"],
    non_local=config["non_local"],
    non_local_hidden_nf=config["non_local_hidden_nf"],
    radius_cutoff=config["radius_cutoff"],
    key=key,
)
model = load_weights_from_npz(model, weights_dir)

n_points, n_atoms = 100, 3
mol = {
    "density": jnp.ones((2, n_points)) * 0.1,
    "grad": jnp.zeros((2, 3, n_points)),
    "kin": jnp.ones((2, n_points)) * 0.05,
    "grid_coords": jnp.zeros((n_points, 3)),
    "grid_weights": jnp.ones(n_points) * 0.01,
    "coarse_0_atomic_coords": jnp.zeros((n_atoms, 3)),
}

E_xc = model.get_exc(mol)
print(f"E_xc = {E_xc:.10f} Ha")

# Gradient of E_xc with respect to all inputs — one line
grads = jax.grad(model.get_exc)(mol)
print(f"dE/d(density): {grads['density'].shape}")

import equinox as eqx

@eqx.filter_jit
def get_exc_jit(m, mol):
    return m.get_exc(mol)

# First call compiles; subsequent calls are fast
E = get_exc_jit(model, mol)

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto
from skalax import SkalaFunctional, load_weights_from_npz, load_config, get_default_weights_dir
from skalax.pyscf import JaxSkalaKS

weights_dir = get_default_weights_dir()
config = load_config(weights_dir)
key = jax.random.PRNGKey(0)
model = SkalaFunctional(
    lmax=config["lmax"],
    non_local=config["non_local"],
    non_local_hidden_nf=config["non_local_hidden_nf"],
    radius_cutoff=config["radius_cutoff"],
    key=key,
)
model = load_weights_from_npz(model, weights_dir)

mol = gto.M(
    atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    basis="sto-3g",
    verbose=0,
)
ks = JaxSkalaKS(mol, xc=model)
energy = ks.kernel()
print(f"Total energy: {energy:.8f} Ha")