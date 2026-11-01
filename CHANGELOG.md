# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- `slow_down`: each species has its own Coulomb logarithm (`coulomb_logarithms`, as ASCOT5
  evaluates it), so the ion drag and the pitch-angle scattering grow by about a third
  against the shared electron logarithm; the slowing down is integrated numerically in speed
  (`slowing_down_moments`, Simpson's rule in `ln v`). On the ASCOT5 benchmark the fast-ion
  content goes from 14% above ASCOT5 (without orbits) to 4% below, within the 4% by which
  ASCOT5's moments overshoot its birth power.
- `slow_down`: the current is that of the orbits (`orbit_average=True`, new module
  `freeofcharge.trapping`): trapped ions carry none, passing ones their transit-averaged
  pitch, and pitch-angle scattering is averaged over the orbit, its odd eigenmodes decaying
  faster than Gaffey's single exponential. The ASCOT5 benchmark's fast-ion current goes from
  2.66 to 1.18 times ASCOT5's. The rest is most likely the orbit width
  (not yet shown).
- `slow_down`: the fast ions can end at `max(thermal_factor T_i, minimum_energy)`, the rest
  of their energy reported as `SlowingDown.thermalised_power`; the ASCOT5 benchmark uses
  ASCOT5's end conditions (2 T_i, 1 keV).
- The trapped-electron correction `G` is Sauter's `L31` with collisions from the surfaces'
  trapped fraction, safety factor and collisionality (`sauter_trapped_electron_correction`,
  Lin-Liu and Hinton 1997, Honda et al. 2012). The old fit is
  `start_cordey_trapped_electron_correction`, now attributed to Mikkelsen and Singer's fit
  of Start and Cordey's values; the hook takes `(surfaces, plasma)`.
- `FluxSurfaces`: the field strength and flux-surface average weights on each shell centre,
  the safety factor, the major radius and the trapped fraction; `Deposition.birth_field`.
- The ITER-sized quickstart drives 0.59 MA instead of 0.97 MA.

### Added

- `PolarFEECEquilibrium` (`freeofcharge.polarfeec_equilibrium`): GVEC and DESC equilibria
  through PolarFEEC.jl and its Python wrapper `polarfeec` (installed from a PolarFEEC.jl
  checkout, not yet an extra), which inverts the maps robustly across the axis; `rho_at`,
  `field_at` and the flux surfaces agree with the GVEC adapter on its test equilibrium.
- `ElectronCRStopping` (the `cr` extra): the electron-impact share of the beam stopping from
  OpenCR's collisional-radiative ionization coefficient, a check of Suzuki's fit.
- `freeofcharge.ascot5` and `benchmarks/ascot5`: the ASCOT5 benchmark. The equilibrium,
  plasma, wall and beam in a5py's input format (`B_2DS`, `plasma_1D`, `wall_2D`, one BBNBI5
  beamlet per marker), a benchmark case of ASDEX Upgrade's size with a 60 keV deuterium beam,
  a script that runs BBNBI5 and ASCOT5 on it, their results, and the comparison as the
  example `ascot5_benchmark.py` and the documentation page "ASCOT5 benchmark".
- `freeofcharge.chain` and the `chain` extra (wellbalanced, tree-d-3d): a D- source from a
  global model, slit extraction optics with a divergence V-curve, the gas neutralizer
  (checked against its closed form) and the duct transmission; the example
  `injector_chain.py` scans RF power and neutralizer line density end to end.
- simgeometry comes from PyPI (`simgeometry>=0.1.0`).
- The chain test (`tests/integration`): the injector chain end to end at a small size,
  checking the units, shapes and species names between the stages and the power
  bookkeeping, with errors that name the stage and package. The `Chain Test` workflow runs
  it weekly, by hand, on `package-released` events from the other packages' releases, and
  against both the latest PyPI releases and the devel branches.
- `Source.state`: the wellbalanced steady state behind `deuterium_source`.

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

- The max-models dependencies come from their devel branches (`[tool.uv.sources]`) on
  `devel`, and from PyPI (`UV_NO_SOURCES`) for pull requests to and pushes on `main`.
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
