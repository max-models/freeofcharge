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
- Documentation: a landing page with the max-models injector chain, installation with the
  max-models dependencies, a quickstart with its output, and five tutorials (equilibria and
  flux coordinates, beam markers, stopping and shine-through, slowing down and current
  drive, an injector design scan) that assert closed forms, conservation laws, symmetries
  and trends, so the tutorial CI catches regressions.

### Changed

- `deposit` is about 5 times faster: the boundary and wall tests of `Equilibrium` use the
  star shape of the polygons about the magnetic axis (one edge per point instead of all of
  them), with ray casting kept for polygons that are not star-shaped. Results are identical.
- simgeometry comes from PyPI (`simgeometry>=0.1.0`).

### Fixed

- CI: simgeometry is installed from GitHub (`devel`), as it was not on PyPI, and `uv.lock`
  is up to date.
- The docs workflow smoke-tests pages that exist, not the template's.
- `SuzukiStopping` is continuous at 100 keV/amu: the published low- and high-energy fits
  differ there by a few per cent (their coefficients agree with ASCOT5 and STELLOPT), and are
  now blended in `log sigma` over 100/1.1 to 110 keV/amu; `blend=1.0` gives the fits as
  published.
- Docstrings no longer refer to a planning file that is not in the repository.

## [0.1.0] - 2026-10-05

### Added

- Package layout with a console entry point and pytest tests.
- Astro + Starlight documentation site with executed tutorials and a generated API reference.
- GitHub Actions for tests, ruff, pyright, tutorials, documentation and PyPI publishing.
