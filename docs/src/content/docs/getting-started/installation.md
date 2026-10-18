---
title: Installation
description: Install freeofcharge and its max-models dependencies.
---

freeofcharge needs Python 3.10 or newer. Install it from GitHub with pip:

```bash
pip install "freeofcharge @ git+https://github.com/max-models/freeofcharge"
```

or, for development, clone the repository and install it with [uv](https://docs.astral.sh/uv/)
(`make install` runs `uv sync --extra dev` and installs the pre-commit hooks) or pip:

```bash
git clone https://github.com/max-models/freeofcharge
cd freeofcharge
pip install -e ".[dev]"
```

## Dependencies

Besides NumPy, SciPy and xarray, freeofcharge is built on the max-models building blocks,
which pip installs with it:

| Package                                                     | Used for                                                       | Installed from        |
| ----------------------------------------------------------- | -------------------------------------------------------------- | --------------------- |
| [cunumpy](https://github.com/max-models/cunumpy)            | arrays on NumPy or CuPy                                        | PyPI                  |
| [maybempi](https://github.com/max-models/maybempi)          | MPI when launched with `mpiexec`, a serial stand-in otherwise | PyPI                  |
| [plasmacoll](https://github.com/max-models/plasmacoll)      | the markers (`ParticleArrays`), random streams and constants  | GitHub, `devel`       |
| [simgeometry](https://github.com/max-models/simgeometry)    | the grid and the boundary and wall polygons                   | GitHub, `devel`       |

plasmacoll and simgeometry come from their `devel` branches on GitHub, so pip needs `git`.

## GPUs and MPI

The arrays live on the cunumpy backend. To run on a GPU, install CuPy and select its
backend before importing freeofcharge:

```bash
export CUNUMPY_BACKEND=cupy
```

With MPI (install `mpi4py`), pass a communicator as `comm=` to `NeutralBeam.gaussian`,
`deposit` and `slow_down`: each rank draws and traces its own markers, and the profiles are
summed over the ranks once.

## Optional extras

| Extra  | Installs                                                         |
| ------ | ---------------------------------------------------------------- |
| `test` | `pytest` and `pytest-cov`                                        |
| `docs` | the notebook runner, griffe, matplotlib and plasma-plots          |
| `dev`  | ruff, pyright, pre-commit and JupyterLab, plus `test` and `docs` |

```bash
pip install -e ".[docs]"   # to run the tutorials and examples
```
