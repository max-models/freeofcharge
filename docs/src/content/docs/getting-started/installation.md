---
title: Installation
description: Create a Python environment and install freeofcharge.
---

Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

Install the package and its requirements with pip:

```bash
pip install -e .
```

It depends on the max-models building blocks: cunumpy (NumPy or CuPy arrays), simgeometry
(polygons and grids), plasmacoll (markers and constants) and maybempi (MPI when it is there).
For CuPy, install it and set `CUNUMPY_BACKEND=cupy`.

## Optional extras

| Extra  | Installs                                                |
| ------ | ------------------------------------------------------- |
| `test` | `pytest` and `coverage`                                 |
| `docs` | the notebook runner, griffe and plasma-plots for the examples |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```
