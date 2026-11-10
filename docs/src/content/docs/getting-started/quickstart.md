---
title: Quickstart
description: Deposit a neutral beam in a tokamak plasma, slow down its fast ions and read off the heating and current.
---

First, ensure that `freeofcharge` is [installed](/freeofcharge/getting-started/installation/).

A 16.5 MW beam of 1 MeV deuterium atoms, tangent to $R = 5.3$ m, in an ITER-sized
equilibrium with a deuterium plasma of $10^{20}\,\mathrm{m^{-3}}$ and 20 keV in the centre and 2 % helium:

```python
import astropy.units as u
import freeofcharge as foc

# or foc.Equilibrium.from_geqdsk(path)
equilibrium = foc.Equilibrium.solovev(6.2 * u.m, 1.8 * u.m, 1.7, 5.3 * u.T)
plasma = foc.Plasma.parabolic(
    1e20 * u.m**-3, 20 * u.keV, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)}
)
origin, direction = foc.tangential_injection(5.3 * u.m, 12 * u.m)
beam = foc.NeutralBeam.gaussian(
    origin,
    direction,
    1 * u.MeV,
    16.5 * u.MW,
    width=30 * u.cm,
    divergence=5 * u.mrad,
    markers=1000,
    seed=1,
)

deposition = foc.deposit(
    beam, equilibrium, plasma, step=2 * u.cm, births_per_marker=2, seed=2
)
print(f"shine-through {100 * deposition.shine_through_fraction:.2f} %")

surfaces = equilibrium.flux_surfaces(deposition.rho_edges)
steady = foc.slow_down(deposition, plasma, surfaces)
electrons, ions = steady.electron_heating.sum(), steady.ion_heating.sum()
print(f"electrons {electrons / 1e6:.1f} MW, ions {ions / 1e6:.1f} MW")
print(f"driven current {steady.driven_current / 1e6:.2f} MA")
```

```text
shine-through 0.08 %
electrons 10.2 MW, ions 6.3 MW
driven current 0.59 MA
```

It runs in a few seconds on a laptop.

The physical settings take astropy Quantities (`pip install astropy`), strings such as
`"20 keV"` (as in a configuration file) or plain numbers in the documented units: SI, eV
for energies and temperatures (which may also be given in K) and rad for angles, so
`Plasma.parabolic(1e20, 2e4, ...)` is the same plasma. A value of the wrong dimension
raises an error.

1. `Equilibrium.solovev(R0, a, kappa, B0)` is an analytic equilibrium; `Plasma.parabolic`
   gives parabolic density and temperature profiles in $\rho = \sqrt{\psi_n}$, with each ion's
   share of the electron density.
2. `NeutralBeam.gaussian` draws the markers, streams of atoms with a weight in atoms per
   second, about the axis from `tangential_injection`.
3. `deposit` traces each marker as a straight line attenuated with Suzuki's stopping cross
   sections: the birth profile on flux shells (`deposition.power`, `birth_rate`), sampled
   fast-ion births with their pitch, and the shine-through, with where it hits the wall.
4. `slow_down` slows the fast ions down on their flux surfaces (Stix): the heating of the
   electrons and ions, the fast-ion density and pressure, and the beam-driven current of
   their orbits (17% of these ions are born trapped and carry none).

`deposition.to_dataset(surfaces)` and `steady.to_dataset()` return the profiles as xarray
datasets for plotting, for example with plasma-plots. More markers and a shorter `step`
reduce the sampling noise of the profiles.

The [examples](/freeofcharge/examples/) go through each step and what it checks; the
[API reference](/freeofcharge/api/freeofcharge/) lists every function.
