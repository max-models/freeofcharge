# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `Equilibrium`: flux on an (R, Z) grid from G-EQDSK files or the Solov'ev solution, field,
  flux surfaces and shells, boundary and wall as simgeometry polygons; G-EQDSK reader and writer.
- `NeutralBeam`: beam markers in a plasmacoll `ParticleArrays`, Gaussian beams with core and
  halo divergence, `tangential_injection`.
- `SuzukiStopping`: Suzuki et al. (1998) beam-stopping cross sections, checked against ASCOT5.
- `deposit`: straight-line attenuation, birth profiles, sampled births, shine-through and wall hits.
- `slow_down`: Stix slowing down per flux surface, heating, fast-ion pressure, pitch relaxation
  and beam-driven current.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
