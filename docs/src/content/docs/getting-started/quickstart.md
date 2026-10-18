---
title: Quickstart
description: Deposit a neutral beam in a tokamak plasma and slow down its fast ions.
---

First, ensure that `freeofcharge` is [installed](/freeofcharge/getting-started/installation/).

## A beam in an equilibrium

```python
import freeofcharge as foc

# or foc.Equilibrium.from_geqdsk(path)
equilibrium = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3)
plasma = foc.Plasma.parabolic(1e20, 2e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)})
origin, direction = foc.tangential_injection(5.3, 12.0)
beam = foc.NeutralBeam.gaussian(
    origin, direction, 1e6, 16.5e6, divergence=0.005, markers=2000
)

deposition = foc.deposit(beam, equilibrium, plasma)
print(f"shine-through {100 * deposition.shine_through_fraction:.2f} %")
```

`deposit` traces every marker as a straight line, attenuated with Suzuki's stopping cross
sections, and returns the birth profile on flux shells, sampled fast-ion births and the
shine-through with the points where it meets the wall.

## Heating and current

```python
surfaces = equilibrium.flux_surfaces(deposition.rho_edges)
steady = foc.slow_down(deposition, plasma, surfaces)
print(steady.electron_heating.sum(), steady.ion_heating.sum(), steady.driven_current)
steady.to_dataset()  # profiles as an xarray dataset, for plasma-plots
```

See the [API reference](/freeofcharge/api/freeofcharge/) for every module and function.
