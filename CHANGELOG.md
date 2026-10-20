# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `freeofcharge.chain` and the `chain` extra (wellbalanced, tree-d-3d): a D- source from a
  global model, slit extraction optics with a divergence V-curve, the gas neutralizer
  (checked against its closed form) and the duct transmission; the example
  `injector_chain.py` scans RF power and neutralizer line density end to end.
- simgeometry comes from PyPI (`simgeometry>=0.1.0`).

- `Equilibrium`: flux on an (R, Z) grid from G-EQDSK files or the Solov'ev solution, field,
  flux surfaces and shells, boundary and wall as simgeometry polygons; G-EQDSK reader and writer.
- `NeutralBeam`: beam markers in a plasmacoll `ParticleArrays`, Gaussian beams with core and
  halo divergence, `tangential_injection`.
- `SuzukiStopping`: Suzuki et al. (1998) beam-stopping cross sections, checked against ASCOT5.
- `deposit`: straight-line attenuation, birth profiles, sampled births, shine-through and wall hits.
- `slow_down`: Stix slowing down per flux surface, heating, fast-ion pressure, pitch relaxation
  and beam-driven current.
- Documentation: a landing page with the max-models injector chain, installation with the
  max-models dependencies, a quickstart with its output, and five tutorials (equilibria and
  flux coordinates, beam markers, stopping and shine-through, slowing down and current
  drive, an injector design scan) that assert closed forms, conservation laws, symmetries
  and trends, so the tutorial CI catches regressions.

### Fixed

- CI: simgeometry is installed from GitHub (`devel`), as it is not on PyPI, and `uv.lock`
  is up to date.
- The docs workflow smoke-tests pages that exist, not the template's.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
