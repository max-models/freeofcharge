---
title: ASCOT5 benchmark
description: The same equilibrium, plasma and beam in freeofcharge and in ASCOT5 (BBNBI5 and ascot5_main), and how far they agree.
---

[ASCOT5](https://github.com/ascot4fusion/ascot5) (LGPL-3.0) is the reference for the beam in the
plasma: BBNBI5 for the deposition and shine-through, `ascot5_main` (guiding centres with Monte
Carlo Coulomb collisions) for the slowing down. `freeofcharge.ascot5` writes freeofcharge's
equilibrium, plasma, wall and beam as a5py inputs, and `benchmarks/ascot5/run_ascot5.py` runs both
ASCOT5 programs on them and stores the results in `benchmarks/ascot5/ascot5_results.json`. The
example [ASCOT5 benchmark](/freeofcharge/full-examples/ascot5-benchmark/) runs freeofcharge on
the same case and plots both.

## The case

`freeofcharge.ascot5.benchmark_case()`:

| | |
|---|---|
| Equilibrium | Solov'ev, R0 = 1.65 m, outboard minor radius 0.5 m, elongation 1.6, 2.5 T, q = 1 on the axis; 129 × 129 grid; wall on psi_n = 1.3 |
| Plasma | D with 1% C (Z_eff = 1.3); n_e = 5e19 m^-3, T_e = T_i = 2.5 keV in the centre, parabolic (exponents 0.5 and 1.5, 5% at the edge) |
| Beam | 2.5 MW of 60 keV D, tangent to R = 1.0 m from R = 3.5 m, along the field; 10 cm wide, 10 mrad divergence; 2000 markers |
| Shells | 20 equal shells of rho = sqrt(psi_n) |

What ASCOT5 is given:

- `B_2DS`: freeofcharge's psi on its grid and B_phi = F/R; ASCOT5 takes B_R and B_Z from psi with
  the same signs. psi0 is moved 1e-6 of the flux difference below the axis value, as ASCOT5 stops
  markers where its spline of psi dips below psi0.
- `plasma_1D`: the profiles on freeofcharge's grid, with two points outside rho = 1 at 1 m^-3 (ASCOT5
  stops markers beyond its grid; freeofcharge's densities are zero there). The ion masses are the mass
  numbers, as in freeofcharge.
- `wall_2D`: the wall polygon.
- `nbi`: one injector with one beamlet per freeofcharge marker and no divergence, so BBNBI5 flies
  along the freeofcharge lines (its divergence is sqrt(2) times a standard deviation).

BBNBI5 injects 50000 markers at each density of a scan (1e19, 2e19, 5e19 m^-3). Of the ions born
at 5e19, 1000 drawn at random are slowed down twice, until 2 T_i or 1 keV: following their orbits
(adaptive step, wall and 120 s CPU time per marker as end conditions), and without orbits, the
markers staying at their birth points and only colliding (fixed step of 5 µs), which is
freeofcharge's local model. With orbits, 20 of the 1000 markers reached the CPU-time limit before
thermalising; none was lost to the wall. The moments come from a5py's `getdist_moments` on a rho-theta
distribution (8 poloidal bins).

## Results

ASCOT5 at commit `24be846`, built with Homebrew GCC 16 and HDF5 1.14 on macOS, 2 OpenMP threads.
Ratios are freeofcharge over ASCOT5.

| Quantity | freeofcharge | ASCOT5 | ratio | ASCOT5 without orbits | ratio |
|---|---|---|---|---|---|
| Shine-through, n_e0 = 1e19 m^-3 | 23.03% | 23.10 ± 0.19% | 0.997 | | |
| Shine-through, 2e19 | 5.19% | 5.13 ± 0.10% | 1.012 | | |
| Shine-through, 5e19 | 0.085% | 0.094 ± 0.014% | 0.91 | | |
| Ions born [1/s] | 2.598e20 | 2.598e20 | 1.000 | | |
| Electron heating [MW] | 1.216 | 1.227 | 0.991 | 1.171 | 1.039 |
| Ion heating [MW] | 1.282 | 1.232 | 1.040 | 1.304 | 0.983 |
| Fast ions | 1.29e19 | 1.09e19 | 1.18 | 1.13e19 | 1.14 |
| Fast-ion energy [kJ] | 61.0 | 54.9 | 1.11 | 57.2 | 1.07 |
| Fast-ion current [kA] | 156 | 59 | 2.66 | 136 | 1.15 |
| Driven current, with shielding [kA] | 109 | 37 | 2.93 | 93 | 1.18 |

The errors are the binomial standard errors of BBNBI5's 50000 markers. Shell by shell, the
deposited power of the two codes differs by at most 4% of its peak (BBNBI5's noise); the slowing-down
profiles of ASCOT5, from 1000 markers in 20 shells, carry a noise of about 10% per shell.

### What agrees

- **Deposition and shine-through.** Both codes use Suzuki's beam-stopping fit (freeofcharge's
  `SuzukiStopping` equals ASCOT5's `suzuki_sigmav` to 1e-10) along the same lines, so they agree
  within BBNBI5's Monte Carlo error at all three densities, and the birth profiles agree shell by
  shell.
- **Heating.** The totals and the split between electrons and ions agree to 4% with both ASCOT5
  runs. ASCOT5 stops at 2 T_i and hands the rest of the energy (118 kW, 5% of the beam) to the
  thermal plasma; freeofcharge slows down to rest, mostly heating the ions in that last part. The
  heating moments of ASCOT5 plus that thermalised energy add up to 103% of the birth power, a
  measure of the accuracy of the moments.

### What does not, and why

- **Current, with orbits.** freeofcharge's fast-ion current is 2.7 times ASCOT5's when ASCOT5 follows
  the orbits, but agrees with ASCOT5 without orbits to 15%. The beam is born with a mean pitch of
  0.55; by the zero-orbit-width criterion (pitch squared below 1 - B/B_max on the birth surface) 42% of
  the ions are born on trapped orbits, which carry almost no parallel current once their bounce
  motion is averaged, and the passing ones are spread over a wider radius (the ASCOT5 current is
  flat inside rho = 0.2). freeofcharge keeps each ion's local pitch on its birth surface, so it
  overestimates the current of a beam injected this perpendicularly. An orbit average of the pitch
  (step 5's first-orbit correction, or level 3) is needed for the current.
- **Fast-ion content (14% to 18%).** Without orbits the difference stays, so it is in the collision
  model, not the orbits. freeofcharge uses one Coulomb logarithm, the electrons' (NRL, 16.0 in the
  centre), for the electron and the ion drag; ASCOT5 evaluates it per species and velocity (about 17
  for the electrons and 20 for the deuterons at the beam speed, estimated from its formula). A
  larger ion logarithm raises the critical energy by about 15% and with the shorter slowing-down time
  shortens the time to thermalise by about 16% at the centre, as observed; it also moves heating to
  the ions, which the 2 T_i cut-off of ASCOT5 hides in the totals. Separate Coulomb logarithms for
  electrons and ions are the next change to test.
- **Central fast-ion density with orbits.** ASCOT5's density is flat inside rho = 0.2 where
  freeofcharge's peaks: the orbit widths of 60 keV deuterons (a few cm) are not small next to the
  central shells.

## Reproducing the ASCOT5 side

ASCOT5 is a benchmark, never a dependency of freeofcharge. On macOS with Homebrew (`brew install
gcc hdf5`):

```bash
git clone https://github.com/ascot4fusion/ascot5.git && cd ascot5/src
# run make inside src/: the top-level make writes gitver.h twice in parallel
make -j3 bbnbi5 ascot5_main CC=gcc-16 FLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib"
make -j3 libascot.so CC=gcc-16 FLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib -shared"
mkdir -p ../build && mv bbnbi5 ascot5_main libascot.so ../build/
```

Homebrew's `h5cc` wraps Apple's clang, which has no OpenMP; GCC with the HDF5 flags works. On
Linux, `make bbnbi5 ascot5_main libascot` with `h5cc` as in ASCOT5's documentation.

a5py and freeofcharge in one environment; a5py needs NumPy below 2.3 (`int()` of size-1 arrays)
and unyt 3.0 (recursion in unit conversions with unyt 3.2):

```bash
python -m venv a5py-venv && . a5py-venv/bin/activate
pip install -e ./ascot5 -e /path/to/freeofcharge "numpy<2.3" "unyt==3.0.3"
OMP_NUM_THREADS=2 python benchmarks/ascot5/run_ascot5.py --ascot5 /path/to/ascot5/build \
    --work /tmp/ascot5-benchmark --slowing-down-markers 1000
```

On a laptop with 2 threads, BBNBI5 takes 1 to 3 minutes per density, the slowing down with orbits
about 25 minutes, without orbits 10 seconds. BBNBI5 seeds its random numbers from the clock, so the
shine-through changes from run to run within its error. `--reuse` keeps the BBNBI5 runs of `--work`.
Then

```bash
python examples/ascot5_benchmark.py
```

prints both tables and plots the profiles.

## References

- E. Hirvijoki et al., ASCOT: Solving the kinetic equation of minority particle species in tokamak
  plasmas, Comput. Phys. Commun. 185, 1310 (2014).
- J. Varje et al., High-performance orbit-following code ASCOT5 for Monte Carlo simulations in
  fusion plasmas, arXiv:1908.02482 (2019).
- S. Suzuki et al., Attenuation of high-energy neutral hydrogen beams in high-density plasmas,
  Plasma Phys. Control. Fusion 40, 2097 (1998).
