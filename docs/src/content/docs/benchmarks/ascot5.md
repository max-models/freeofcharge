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
markers staying at their birth points and only colliding (fixed step of 5 µs). freeofcharge
slows its ions down to the same end, `max(2 T_i, 1 keV)` (the case's `thermal_factor` and
`minimum_energy`), and gives two currents: that of the orbits (`orbit_average=True`, the
default), compared with the run with orbits, and that of the local pitch, compared with the run
without. With orbits, 20 of the 1000 markers reached the CPU-time limit before
thermalising; none was lost to the wall. The moments come from a5py's `getdist_moments` on a rho-theta
distribution (8 poloidal bins).

## Results

ASCOT5 at commit `24be846`, built with Homebrew GCC 16 and HDF5 1.14 on macOS, 2 OpenMP threads.
Ratios are freeofcharge over ASCOT5. The first freeofcharge column is the version of the first
benchmark (one Coulomb logarithm, slowing down to rest, the local pitch); the second is the
present one.

| Quantity | freeofcharge before | freeofcharge | ASCOT5 | ratio |
|---|---|---|---|---|
| Shine-through, n_e0 = 1e19 m^-3 | 23.03% | 23.03% | 23.10 ± 0.19% | 0.997 |
| Shine-through, 2e19 | 5.19% | 5.19% | 5.13 ± 0.10% | 1.012 |
| Shine-through, 5e19 | 0.085% | 0.085% | 0.094 ± 0.014% | 0.91 |
| Ions born [1/s] | 2.598e20 | 2.598e20 | 2.598e20 | 1.000 |
| Electron heating [MW] | 1.216 | 1.120 | 1.227 | 0.91 |
| Ion heating [MW] | 1.282 | 1.245 | 1.232 | 1.01 |
| Thermalised [kW] | | 133 | 118 | 1.13 |
| Fast ions | 1.29e19 | 1.08e19 | 1.09e19 | 0.99 |
| Fast-ion energy [kJ] | 61.0 | 52.9 | 54.9 | 0.96 |
| Fast-ion current [kA] | 156 | 69.5 | 58.8 | 1.18 |
| Driven current, with shielding [kA] | 109 | 38.9 | 37.3 | 1.04 |

Against the ASCOT5 run without orbits, with freeofcharge's current of the local pitch:

| Quantity | freeofcharge before | freeofcharge, local pitch | ASCOT5 without orbits | ratio |
|---|---|---|---|---|
| Electron heating [MW] | 1.216 | 1.120 | 1.171 | 0.96 |
| Ion heating [MW] | 1.282 | 1.245 | 1.304 | 0.95 |
| Thermalised [kW] | | 133 | 128 | 1.04 |
| Fast ions | 1.29e19 | 1.08e19 | 1.13e19 | 0.96 |
| Fast-ion energy [kJ] | 61.0 | 52.9 | 57.2 | 0.92 |
| Fast-ion current [kA] | 156 | 133 | 136 | 0.98 |
| Driven current, with shielding [kA] | 109 | 80.6 | 92.7 | 0.87 |

The errors are the binomial standard errors of BBNBI5's 50000 markers. Shell by shell, the
deposited power of the two codes differs by at most 4% of its peak (BBNBI5's noise); the slowing-down
profiles of ASCOT5, from 1000 markers in 20 shells, carry a noise of 10% to 20% per shell, which
the heating and the density of a shell share (it is which births the 1000 markers sample).

### What agrees

- **Deposition and shine-through.** Both codes use Suzuki's beam-stopping fit (freeofcharge's
  `SuzukiStopping` equals ASCOT5's `suzuki_sigmav` to 1e-10) along the same lines, so they agree
  within BBNBI5's Monte Carlo error at all three densities, and the birth profiles agree shell by
  shell.
- **Heating and fast-ion content, without orbits.** ASCOT5's heating moments and thermalised
  energy add up to 104% of the birth power (2.603 MW of 2.501 MW) without orbits and 103% with
  them: the moments of the binned distribution overshoot by that much. Scaled to the birth power,
  ASCOT5 without orbits gives 1.125 MW to the electrons, 1.253 MW to the ions and 1.09e19 fast
  ions, against freeofcharge's 1.120 MW, 1.245 MW and 1.08e19: agreement to 1%. The fast-ion
  energy stays 4% below after the scaling; Stix's drags assume `v_Ti << v << v_Te`, which ASCOT5's
  full Rosenbluth coefficients do not (at the edge, at 125 eV, a 60 keV deuteron has a third of
  the electron thermal speed).
- **The current of the local pitch** agrees with the run without orbits to 2%: the pitch-angle
  scattering of the ions, with their own Coulomb logarithm, decays Gaffey's first Legendre term as
  in ASCOT5.

### What changed, and why

- **A Coulomb logarithm per species.** The first benchmark used the electrons' logarithm (NRL, 16.0
  in the centre) for all drags. freeofcharge now evaluates it per species and speed as ASCOT5 does,
  `ln(lambda_D / max(b_classical, b_quantum))` (`coulomb_logarithms`): 16.7 for the electrons
  and 21.1 for the deuterons at 60 keV in the centre. The ion drag and the pitch-angle scattering
  grow by about a third, the critical energy by 16%, so the ions take more of the heating and
  thermalise sooner: the fast-ion content falls from 14% above ASCOT5 (without orbits) to 4%
  below. The first benchmark's 4% agreement of the heating was partly a cancellation: the
  too-small ion drag had less heating go to the ions while the slowing down to rest gave them the
  last 5% of the beam energy.
- **The same end.** The fast ions now stop being fast at ASCOT5's end condition, `max(2 T_i, 1
  keV)`, and the rest of their energy is reported as `thermalised_power` (133 kW against ASCOT5's
  118 kW with orbits and 128 kW without).
- **The current of the orbits.** The beam is born with a mean pitch of 0.55; 44% of the ions are
  born on trapped orbits (zero orbit width, on the shell's surface), which carry no current over a
  bounce. The passing ones carry their transit-averaged pitch, and the pitch-angle scattering
  averaged over the orbit decays the current faster than in a uniform field: its first eigenvalue
  is 1.4 to 2.5 times Gaffey's on these surfaces (`freeofcharge.trapping`, after Cordey, Nucl.
  Fusion 16, 499, 1976). The fast-ion current goes from 2.66 to 1.18 times ASCOT5's.

### What does not agree, and why

- **The current with orbits (18% above ASCOT5).** freeofcharge keeps every ion on its birth
  surface. A 60 keV deuteron in 2.5 T has a Larmor radius of 2 cm and banana widths of several
  cm, not small next to the central shells: ASCOT5's current density is flat inside rho = 0.2,
  where freeofcharge's is 20% to 50% above it shell by shell, and near the axis the orbits are
  potatoes, not bananas. That the orbit width is the cause is likely but not shown. The test is
  a field scan in ASCOT5: in this Solov'ev case the field scales the orbit widths as 1/B and
  changes nothing else, so freeofcharge's answer stays the same. A first try (300 markers each, at
  2.5 T and 10 T) gave 44 and 52 kA, but about half of the markers reached the CPU-time limit
  before thermalising (20 of 1000 in the stored run), so those currents are cut short and do not
  settle the question. The ASCOT5 current with orbits is also noisy: 44% of the markers carry
  almost none, and the profile varies by tens of percent per shell.
- **The electron heating with orbits (9% below ASCOT5).** Following the orbits moves 5% of the
  heating from the ions to the electrons in ASCOT5 (1.227 against 1.171 MW to the electrons); a
  local model has no such shift.
- **The driven current** compares two shielding models, not the slowing down. freeofcharge's
  trapped-electron correction `G` is Sauter's `L31` with the electron collisionality (the
  default `sauter_trapped_electron_correction`), from the trapped fraction of each surface. a5py's
  `currentdrive` moment uses the same `L31` but evaluates the collisionality with
  `(R / r)^(3/4)` where Sauter's formula has `(r / R)^(-3/2)`, which makes it nearly
  collisionless here; it also leaves out the `Z` of Sauter's collisionality. With a5py's
  collisionality in freeofcharge's `G` (`nu_e*` times `eps^2.25 / Z_eff`) the driven current of
  the local pitch is 0.98 of ASCOT5's without orbits instead of 0.87, and the one of the orbits
  1.19 instead of 1.04, the ratio of the fast-ion currents: the 4% agreement with orbits was the
  18% of the orbit width against the 11% of the shielding. Compare the fast-ion currents.

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

The field scan for the orbit width: `--toroidal-field` scales the Solov'ev flux with B (q stays
the same), so freeofcharge's results do not change (a test checks this) while the orbit widths
go as 1/B. One run per field, each with its own work folder and results file, for example

```bash
for b in 2.5 5 10; do
  OMP_NUM_THREADS=8 python benchmarks/ascot5/run_ascot5.py --ascot5 /path/to/ascot5/build \
      --work /tmp/ascot5-field-$b --out /tmp/ascot5-field-$b.json --toroidal-field $b \
      --slowing-down-markers 1000 --cpu-time 600
done
```

If the orbit width causes the 18%, ASCOT5's fast-ion current with orbits rises towards
freeofcharge's 69.5 kA as B grows. Give the markers enough CPU time (`--cpu-time`): at 10 T,
120 s stopped about half of them before they thermalised.

## References

- E. Hirvijoki et al., ASCOT: Solving the kinetic equation of minority particle species in tokamak
  plasmas, Comput. Phys. Commun. 185, 1310 (2014).
- J. Varje et al., High-performance orbit-following code ASCOT5 for Monte Carlo simulations in
  fusion plasmas, arXiv:1908.02482 (2019).
- S. Suzuki et al., Attenuation of high-energy neutral hydrogen beams in high-density plasmas,
  Plasma Phys. Control. Fusion 40, 2097 (1998).
- J. G. Cordey, Effects of particle trapping on the slowing-down of fast ions in a toroidal plasma,
  Nucl. Fusion 16, 499 (1976).
- O. Sauter, C. Angioni and Y. R. Lin-Liu, Neoclassical conductivity and bootstrap current
  formulas for general axisymmetric equilibria and arbitrary collisionality regime, Phys. Plasmas
  6, 2834 (1999).
- Y. R. Lin-Liu and F. L. Hinton, Trapped electron correction to beam driven current in general
  tokamak equilibria, Phys. Plasmas 4, 4179 (1997).
- M. Honda, M. Kikuchi and M. Azumi, Collisionality dependence of the trapped-electron correction
  to the neutral-beam driven current, Nucl. Fusion 52, 023021 (2012).
