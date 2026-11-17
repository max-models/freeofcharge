"""The live demo of the docs: a neutral beam in an ITER-sized tokamak, deposited and slowed down.

It runs in the browser (Pyodide); `run` takes the values of the sliders and returns the
figures. The equilibrium and its flux surfaces are computed once, when the page loads.
"""

import numpy as np

import freeofcharge as foc
from freeofcharge.slowing_down import critical_velocity

_ELEMENTARY_CHARGE = 1.602176634e-19

# 25 shells in rho: coarse enough for a few hundred births per run, fine enough for a profile
_RHO_EDGES = np.linspace(0.0, 1.0, 26)
_RHO = 0.5 * (_RHO_EDGES[1:] + _RHO_EDGES[:-1])
_EQUILIBRIUM = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, points=65)
_SURFACES = _EQUILIBRIUM.flux_surfaces(_RHO_EDGES, resolution=200, field_points=128)
_IONS = {"D": (2, 1, 0.96), "He": (4, 2, 0.02)}
_POWER = 16.5e6


def _layout(y_title):
    return {
        "xaxis": {"title": {"text": "$\\rho$"}, "range": [0, 1]},
        "yaxis": {"title": {"text": y_title}, "rangemode": "tozero"},
        "legend": {"orientation": "h", "y": 1.12},
        "margin": {"t": 40},
    }


def run(energy=1000.0, density=1.0e20, temperature=20.0, tangency_radius=5.3):
    """Deposit a 16.5 MW deuterium beam and slow its fast ions down.

    Args:
        energy: Energy of the beam atoms [keV].
        density: Central electron density [m^-3].
        temperature: Central electron and ion temperature [keV].
        tangency_radius: Major radius the beam axis is tangent to [m].

    Returns:
        The heating and current-density profiles (``figure``), the shine-through,
        the heating of electrons and ions, the critical energy on the axis and the
        driven current (``numbers``).
    """
    plasma = foc.Plasma.parabolic(density, 1e3 * temperature, ions=_IONS)
    origin, direction = foc.tangential_injection(tangency_radius, 12.0)
    beam = foc.NeutralBeam.gaussian(
        origin,
        direction,
        1e3 * energy,
        _POWER,
        width=0.3,
        divergence=5e-3,
        markers=200,
        seed=1,
    )
    deposition = foc.deposit(
        beam,
        _EQUILIBRIUM,
        plasma,
        step=0.05,
        rho_edges=_RHO_EDGES,
        births_per_marker=4,
        seed=2,
    )
    steady = foc.slow_down(deposition, plasma, _SURFACES)

    # the energy below which the ions take more of the beam's energy than the electrons
    v_c = critical_velocity(
        plasma.electron_density(0.0),
        plasma.electron_temperature(0.0),
        plasma.ion_densities(0.0),
        plasma.ion_masses,
        plasma.ion_charges,
    )
    critical_energy = 0.5 * beam.mass * float(v_c) ** 2 / _ELEMENTARY_CHARGE

    volume = _SURFACES.volume
    heating = {
        "data": [
            {
                "x": _RHO,
                "y": deposition.power / volume / 1e6,
                "mode": "lines",
                "name": "deposited",
                "line": {"dash": "dot"},
            },
            {
                "x": _RHO,
                "y": steady.electron_heating / volume / 1e6,
                "mode": "lines",
                "name": "electrons",
            },
            {
                "x": _RHO,
                "y": steady.ion_heating / volume / 1e6,
                "mode": "lines",
                "name": "ions",
            },
        ],
        "layout": _layout("power density [MW/m³]"),
    }
    current = {
        "data": [
            {
                "x": _RHO,
                "y": steady.current_density / 1e6,
                "mode": "lines",
                "name": "driven current",
            }
        ],
        "layout": _layout("current density [MA/m²]"),
    }
    electrons = float(steady.electron_heating.sum())
    ions = float(steady.ion_heating.sum())
    return {
        "figure": [heating, current],
        "numbers": {
            "Shine-through": f"{100 * deposition.shine_through_fraction:.2g} %",
            "Electron heating": f"{electrons / 1e6:.1f} MW",
            "Ion heating": f"{ions / 1e6:.1f} MW",
            "Critical energy (axis)": f"{critical_energy / 1e3:.0f} keV",
            "Driven current": f"{steady.driven_current / 1e6:.2f} MA",
        },
    }
