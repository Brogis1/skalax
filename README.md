# skalax

<img src="https://raw.githubusercontent.com/Brogis1/skalax/main/img/logo-skala.png" alt="Skalax Logo" width="400">

JAX/Equinox implementation of the [Skala](https://github.com/microsoft/skala) neural exchange-correlation functional for density functional theory (DFT) calculations.

## Overview

skalax is a pure JAX port of the Skala neural XC functional. It achieves **good numerical equivalence** with the PyTorch reference (~1 kcal/mol NPE for the tested systems) and unlocks JAX-native capabilities: `jax.grad`, `jax.jit`, `jax.vmap`, and full XLA compilation.

The idea is to enable the use and finetuning of this functional in JAX-based DFT codes.

> [!WARNING]
> Work in progress — so far tested on CPU only. No guarantees are provided. Bug reports are very welcome!


### Performance

JAX JIT compilation (XLA) delivers significant speedups over PyTorch eager — and beats PyTorch `torch.jit.trace` at large grid sizes. All variants use `radius_cutoff=5.0` and are benchmarked in steady state (post-compilation, CPU):

![Performance benchmark](https://raw.githubusercontent.com/Brogis1/skalax/main/benchmarks/plots/extensive_performance.png)

> **Left:** Forward pass only. **Right:** Forward + backward.
> JAX JIT forward is ~1.4× faster than PyTorch traced at 32k grid points;
> JAX JIT fwd+grad is ~1.6× faster than PyTorch traced fwd+backward.

## Installation

Requires `gfortran` and `cmake` (for `dftd3`/`pyscf` via [skala](https://github.com/microsoft/skala)).

```bash
pip install skalax
```

### GPU Support

```bash
pip install skalax
pip install --upgrade "jax[cuda12]"
```

### Development (no Fortran compiler needed)

```bash
git clone https://github.com/Brogis1/skalax
cd skalax
pip install -e .[dev]
pip install --no-deps microsoft-skala
```

## Quick Start

### Basic Usage

```python
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
```

### JAX Autodiff

```python
# Gradient of E_xc with respect to all inputs — one line
grads = jax.grad(model.get_exc)(mol)
print(f"dE/d(density): {grads['density'].shape}")
```

### JIT Compilation

```python
import equinox as eqx

@eqx.filter_jit
def get_exc_jit(m, mol):
    return m.get_exc(mol)

# First call compiles; subsequent calls are fast
E = get_exc_jit(model, mol)
```

### PySCF Integration

```python
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
```

## Input/Output Specification

### Input Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `density` | `(2, n_points)` | Spin densities [α, β] |
| `grad` | `(2, 3, n_points)` | Density gradients [spin, xyz, points] |
| `kin` | `(2, n_points)` | Kinetic energy densities |
| `grid_coords` | `(n_points, 3)` | Grid coordinates (Bohr) |
| `grid_weights` | `(n_points,)` | Integration weights |
| `coarse_0_atomic_coords` | `(n_atoms, 3)` | Atomic positions (Bohr) |

### Outputs

| Method | Shape | Description |
|--------|-------|-------------|
| `model.get_exc(mol)` | `()` | Scalar E_xc (Hartree) |
| `model.get_exc_density(mol)` | `(n_points,)` | Energy density per grid point |

## Model Architecture

```
SkalaFunctional (~276,000 parameters)
├── input_model: Linear(7→256) → SiLU → Linear(256→256) → SiLU
├── non_local_model: NonLocalModel (equivariant message passing)
│   ├── pre_down_linear: Linear(256→16) → SiLU
│   ├── tp_down: TensorProduct (e3nn, lmax=3)
│   ├── tp_up: TensorProduct (e3nn, lmax=3)
│   └── post_up_linear: Linear(16→16) → SiLU
└── output_model:
    ├── Linear(272→256) → SiLU
    ├── Linear(256→256) → SiLU × 2
    ├── Linear(256→1)
    └── ScaledSigmoid(scale=2.0)
```

## Numerical Equivalence

The JAX implementation is numerically equivalent to the PyTorch reference:

| Test | Max \|ΔE\| |
|------|------------|
| `get_exc` (local only) | 0.00e+00 Ha |
| `get_exc` (with non-local) | 1.14e-13 Ha |
| `get_exc_density` | 1.17e-13 Ha |


## Benchmarks

We provide a few important tests to check the correctness of the implementation.

### Forward pass equivalence

![Forward pass equivalence: relative error on total XC energy and max absolute error on per-point XC density, both well below threshold across system sizes](https://raw.githubusercontent.com/Brogis1/skalax/main/benchmarks/plots/forward_equivalence.png)


### Reaction curves

The non-parallelity error (NPE) is crucial for correcness of the prediction (here we compare the JAX implementation with the PyTorch reference implementation).

![CH4 symmetric stretch: total energy vs C–H distance (PyTorch vs JAX) and non-parallelity error](https://raw.githubusercontent.com/Brogis1/skalax/main/benchmarks/plots/ch4_stretch.png)

![H2 dissociation curve: total energy vs H–H distance (PyTorch vs JAX) and absolute energy difference](https://raw.githubusercontent.com/Brogis1/skalax/main/benchmarks/plots/reaction_curve.png)

## Dependencies

- `jax >= 0.4.0`, `jaxlib >= 0.4.0`
- `equinox >= 0.11.0`
- `e3nn-jax >= 0.20.0`
- `numpy >= 1.21.0`
- `skala >= 1.0.0` (PyTorch reference; brings in `pyscf`, `dftd3`, `torch`)

For a lightweight install without `skala` (no Fortran needed), use `pip install --no-deps` and install JAX dependencies manually — see [Development](#development-no-fortran-compiler-needed).

## Tests

```bash
pytest tests/ -v
```

## License

MIT License — see [LICENSE.txt](LICENSE.txt).

The pretrained weights bundled with this package are derived from the Skala model
originally released by Microsoft Corporation under the MIT License
([github.com/microsoft/skala](https://github.com/microsoft/skala)).

## Citation

If you use skalax, please cite the Skala paper:

```bibtex
@misc{luise2025skala,
  title={Accurate and scalable exchange-correlation with deep learning},
  author={Giulia Luise and Chin-Wei Huang and Thijs Vogels and Derk P. Kooi
          and Sebastian Ehlert and Stephanie Lanius and Klaas J. H. Giesbertz
          and Amir Karton and Deniz Gunceler and Megan Stanley
          and Wessel P. Bruinsma and Lin Huang and Xinran Wei
          and Jose Garrido Torres and Abylay Katbashev and Rodrigo Chavez Zavaleta
          and B{\'a}lint M{\'a}t{\'e} and Roberto Sordillo and Yingrong Chen
          and David B. Williams-Young and Christopher M. Bishop
          and Jan Hermann and Rianne van den Berg and Paola Gori-Giorgi},
  year={2025},
  eprint={2506.14665},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2506.14665}
}
```

If you additionally use this JAX implementation, please also cite:

```bibtex
@software{sokolov2025skalax,
  author  = {Sokolov, Igor O.},
  title   = {skalax: {JAX} implementation of the {Skala} neural exchange-correlation functional},
  year    = {2025},
  url     = {https://github.com/Brogis1/skalax},
  license = {MIT}
}
```

## Related Projects

- [Skala (PyTorch)](https://github.com/microsoft/skala) — original PyTorch implementation
- [e3nn-jax](https://github.com/e3nn/e3nn-jax) — equivariant neural networks for JAX
- [Equinox](https://github.com/patrick-kidger/equinox) — JAX neural network library
- [PySCF](https://github.com/pyscf/pyscf) — quantum chemistry in Python
