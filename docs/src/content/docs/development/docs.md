---
title: Building the docs
description: How this documentation site is built from Markdown pages, Jupyter notebooks and docstrings.
---

The site in `docs/` is built with [Astro](https://astro.build/) and
[Starlight](https://starlight.astro.build/). It has three kinds of pages:

- **Hand-written pages** in `docs/src/content/docs/`, as Markdown or MDX.
- **Examples**, converted from the Jupyter notebooks in `tutorials/` by `starlight-notebooks`
  (the `section` setting in `docs/package.json` publishes them under `examples/`). Each
  notebook is executed and its outputs (text, figures, tables) are written into an MDX page.
  The figures are drawn with [plasma-plots](https://max-models.github.io/plasma-plots/) and
  its Plotly backend (`plasma_plots.set_backend("plotly")` in the first cell), so the page
  embeds them interactively, in the site's light and dark colours. One figure per plot: a plot
  that is the last expression of a cell is shown, an earlier one with `.show()`.
- **Example scripts**, a "Scripts" group of the Examples, built from the Python scripts in
  `examples/` by `docs/tools/examples.py`. Each page shows the module docstring, the complete
  source file, and its figures, converted to web figures (`*.plotly.json`, with a picture for
  each in light and dark mode) and embedded with the theme's `PlotlyFigure`.
- **API reference**, generated from the docstrings by
  [starlight-pydocs](https://ewels.github.io/starlight-pydocs/), one page per module.

Guides, when a project has them, are hand-written pages that explain the model or a concept;
anything runnable belongs in the Examples.

The configuration (`defineDocs` in `docs/astro.config.mjs`), the look, the header, the notebook
tool and the build workflow come from the shared
[starlight-theme](https://github.com/max-models/starlight-theme) of the max-models projects;
change them there, and update the version in `docs/package.json` and
`.github/workflows/docs.yml` to pick up a new release.

## Requirements

- Node 22 or newer.
- The package installed with the `docs` extra: `pip install -e ".[docs]"`. The pictures of the
  example scripts' figures need kaleido's browser (`plotly_get_chrome` if none is installed);
  without one the figures are still embedded, only without a picture while they load.

## Commands

```bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py and publish their figures
make docs-dev         # live preview at http://localhost:4321/freeofcharge/
make docs-build       # the static site in docs/dist, as in CI
make docs-preview     # serve docs/dist
make docs-clean       # remove generated pages and build output
```

`make docs-notebooks` only reruns a notebook whose cells changed since the last run; pass
`FORCE=1` to rerun all of them.

## Writing example notebooks

Put a Jupyter notebook in `tutorials/`. Examples are authored only as `.ipynb` files. Its first `# Heading` becomes the page title and the first
paragraph its description. Cell tags control the output:

| Tag                | Effect                                   |
| ------------------ | ---------------------------------------- |
| `remove-cell`      | the cell is left out of the page         |
| `hide-input`       | the code is hidden, the outputs are kept |
| `fold-input`       | the code is in a closed "Show code" box  |
| `remove-output`    | the outputs are left out                 |
| `raises-exception` | an error output is shown, not a failure  |

Math in `$...$` and `$$...$$` is rendered with KaTeX. Write formulas and symbols as math,
not as code: `$\rho = \sqrt{\psi_n}$`, not `` `rho = sqrt(psi_n)` ``; code font is for what a
reader types (function and parameter names).

## Formulas in figures

The notebooks call `plasma_plots.set_math("latex")` after `set_backend("plotly")`, and the
example scripts' figures are converted the same way: labels written with Matplotlib's mathtext
(`ax.set_xlabel("$R$ [m]")`, `long_name=r"$\rho_\mathrm{pol}$"`) are typeset as LaTeX by
MathJax on the page, like the KaTeX math of the text. A label is typeset as one formula, so
keep a label's math in `$...$` and its words outside. Units stay plain strings (`"W/m^3"`); the
labels show their exponents as superscripts. The pictures of the example scripts' figures are
drawn by kaleido, which loads MathJax from a CDN.

## Writing example scripts

Put a Python script in `examples/` with a module docstring. Its first line becomes the page
title and the rest becomes the description. Draw the figures with plasma-plots (the
`.plasma.plot` accessors, or a `plasma_plots.figure` block for a drawing of your own), one figure
per plot rather than a grid of panels, which is too crowded on the page and on a phone, and leave
them open at the end of the script: the docs builder converts every open figure to a web figure
and embeds it. The complete script is shown on the page.
