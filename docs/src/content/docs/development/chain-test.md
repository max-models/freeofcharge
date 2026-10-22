---
title: Chain test
description: The injector chain run end to end against the releases and the devel branches of the packages it joins.
---

The injector chain (`freeofcharge.chain`, `examples/injector_chain.py`) joins several
max-models packages: the source is a wellbalanced global model, the slit beamlet uses
tree-d-3d and simgeometry, the gas neutralizer plasmacoll's charge states, and the plasma
freeofcharge itself. The chain test, `src/freeofcharge/tests/integration`, runs it end to
end at a small size (two RF powers, three perveances, 200 beam markers; a few seconds), so
that a release of one package that no longer fits the others shows up.

## What it checks

- **The stages run**: source, slit optics, gas neutralizer, duct and plasma.
- **The interfaces between them**:
  - species named by formula (`"e"`, `"D2"`, `"D"`, `"D2+"` in wellbalanced) and the
    charge states in plasmacoll's order (`"-"`, `"0"`, `"+"`, the neutral share second);
  - units: temperatures in eV, not K; the chain's Bohm flux from `T_e` in eV against
    wellbalanced's D2+ wall loss; marker energies in eV and weights in atoms/s that carry
    the beam power;
  - shapes: one divergence and transmitted share per perveance, `(N, 3)` positions and
    velocities and `(N,)` weights of plasmacoll's `ParticleArrays`, one value per shell;
  - the source's perveance inside the range the optics scan covers.
- **The results add up**: positive currents and powers, shares in [0, 1], the electron
  power balance of the source, and the power bookkeeping from the accelerator to the
  heating (delivered below accelerated, absorbed plus shine-through equal to injected,
  heating equal to absorbed, the wall hits carrying the shine-through).

When a stage raises, the test fails with a `StageError` that names the stage, its package
and the installed version, for example
`stage 3 gas neutralizer (plasmacoll 0.3.0) raised TypeError: ...`; every interface
assertion starts with the two stages it joins (`wellbalanced -> source: ...`). The
workflow lists the installed versions and a table of the tests with these messages in the
run's summary.

Run it locally with

```bash
uv run pytest src/freeofcharge/tests/integration -v
```

## The workflow

The `Chain Test` workflow (`.github/workflows/chain_test.yml`) runs the test twice:

- **pypi**: the latest PyPI releases of plasmacoll, wellbalanced, simgeometry, cunumpy and
  the others, unpinned (no `uv.lock`, `UV_NO_SOURCES=true`). tree-d-3d has no PyPI release,
  so the slit optics stage is skipped and the beam line takes a fixed divergence of
  5 mrad. picnic is not part of the chain.
- **devel**: the devel branches from `[tool.uv.sources]` at their latest commits
  (`uv sync --extra test --upgrade`).

It runs every Monday at 05:00 UTC, on pull requests that change `src/freeofcharge`,
`pyproject.toml` or the workflow, by hand (Actions → Chain Test → Run workflow, or
`gh workflow run chain_test.yml --repo max-models/freeofcharge`), and when another package
sends a `package-released` event.

### Triggering it from a release

Each package's release workflow needs one step after "Publish to PyPI", and a token that
may dispatch events to freeofcharge: a fine-grained personal access token with access to
`max-models/freeofcharge` and the **Contents: Read and write** permission, stored as the
secret `CHAIN_TEST_TOKEN` in the package's repository (or once as an organization secret).

```yaml
      - name: Run the freeofcharge chain test
        if: steps.pypi-check.outputs.published != 'true'
        env:
          GH_TOKEN: ${{ secrets.CHAIN_TEST_TOKEN }}
          VERSION: ${{ steps.version.outputs.version }}
        run: |
          gh api repos/max-models/freeofcharge/dispatches \
            -f event_type=package-released \
            -f "client_payload[package]=${{ github.event.repository.name }}" \
            -f "client_payload[version]=$VERSION"
```

The step ids (`version`, `pypi-check`) are those of the max-models release template
(plasmacoll, wellbalanced, simgeometry, mpiarray, maybempi, tree-d-3d); in cunumpy's
`publish.yml` and scope-profiler's `publish-pypi.yml` take the version from their own
steps. The run waits up to ten minutes for the version to appear on PyPI before it
installs the releases.
