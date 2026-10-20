r"""Classical slowing down of fast ions (Stix), local to their flux surface.

A fast ion of speed ``v`` with ``v_Ti << v << v_Te`` loses energy to the
electrons and to the ions at the rate

.. math::

    \frac{dv}{dt} = -\frac{v^3 + v_c^3}{\tau_s v^2},

with the Spitzer slowing-down time on the electrons

.. math::

    \tau_s = \frac{3 (2\pi)^{3/2} \varepsilon_0^2 m_b T_e^{3/2}}
                  {n_e Z_b^2 e^4 m_e^{1/2} \ln\Lambda}

and the critical velocity, where the electron and ion drags are equal,

.. math::

    v_c^3 = \frac{3\sqrt{\pi}}{4} m_e v_{Te}^3
            \sum_j \frac{n_j Z_j^2}{n_e m_j}, \qquad v_{Te} = \sqrt{2 T_e / m_e}

(Stix, Plasma Phys. 14, 367, 1972). Its pitch ``xi = v_par / v`` relaxes by
pitch-angle scattering on the ions, ``d<xi>/dt = -nu_D <xi>``, with the
deflection rate of a fast particle on cold targets ``nu_D = sum_j Gamma_j / v^3``
(Helander and Sigmar, *Collisional Transport in Magnetized Plasmas*, 2002),
while the ions' energy drag is ``sum_j (m_b / m_j) Gamma_j / v^2``. Their ratio,

.. math::

    \beta = \frac{\sum_j n_j Z_j^2}{\sum_j n_j Z_j^2 m_b / m_j}
          = \frac{Z_\mathrm{eff}}{[Z]},

gives ``nu_D = beta v_c^3 / (tau_s v^3)`` and, dividing by ``dv/dt``,

.. math::

    \langle\xi\rangle(v) = \xi_0 \left[\frac{v^3 (v_0^3 + v_c^3)}
                                         {v_0^3 (v^3 + v_c^3)}\right]^{\beta/3},

the form of Gaffey (J. Plasma Phys. 16, 149, 1976). The exponent ``beta / 3``
follows from the derivation above; it has not been compared with Gaffey's.

Everything is local: an ion slows down on the flux surface where it was born
(no orbit width, no losses, no charge exchange), and its energy at
thermalisation is neglected. The beam-driven current is the fast-ion current
shielded by the electrons, ``j = j_f (1 - Z_b / Z_eff (1 - G))``, with a
trapped-electron correction ``G``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import xarray as xr
from maybempi import MPI
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy import constants

from freeofcharge.deposition import allreduce

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces
    from freeofcharge.profiles import Plasma

__all__ = [
    "SlowingDown",
    "coulomb_logarithm",
    "critical_velocity",
    "ion_heating_fraction",
    "pitch_angle_exponent",
    "pitch_at_speed",
    "slowing_down_integrals",
    "slowing_down_time",
    "slow_down",
    "trapped_electron_correction",
]

_E = ELEMENTARY_CHARGE
_ME = ELECTRON_MASS
_EPS0 = constants.epsilon_0

#: Gauss-Legendre nodes and weights on [0, 1], for the speed integrals.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)
_NODES = 0.5 * (_NODES + 1.0)
_WEIGHTS = 0.5 * _WEIGHTS


def coulomb_logarithm(ne, te):
    """Return the electron Coulomb logarithm (NRL formulary, ``T_e > 10 eV``).

    Args:
        ne: Electron density [m^-3].
        te: Electron temperature [eV].
    """
    ne_cm3 = xp.asarray(ne, dtype=float) * 1.0e-6
    return 24.0 - xp.log(xp.sqrt(ne_cm3) / xp.asarray(te, dtype=float))


def slowing_down_time(mass, charge_number, ne, te, ln_lambda=None):
    """Return the Spitzer slowing-down time on the electrons [s].

    Args:
        mass: Fast-ion mass [kg].
        charge_number: Fast-ion charge number.
        ne: Electron density [m^-3].
        te: Electron temperature [eV].
        ln_lambda: Coulomb logarithm; from :func:`coulomb_logarithm` if None.
    """
    if ln_lambda is None:
        ln_lambda = coulomb_logarithm(ne, te)
    te_joule = xp.asarray(te, dtype=float) * _E
    return (
        3.0
        * (2.0 * xp.pi) ** 1.5
        * _EPS0**2
        * mass
        * te_joule**1.5
        / (xp.asarray(ne) * charge_number**2 * _E**4 * xp.sqrt(_ME) * ln_lambda)
    )


def _z_over_mass(ne, ion_densities, ion_masses, ion_charges):
    """Return ``sum_j n_j Z_j^2 / (n_e m_j)`` [1/kg]."""
    return sum(
        xp.asarray(n) * z**2 / m
        for n, m, z in zip(ion_densities, ion_masses, ion_charges, strict=True)
    ) / xp.asarray(ne)


def critical_velocity(ne, te, ion_densities, ion_masses, ion_charges):
    """Return the critical velocity ``v_c`` [m/s], independent of the fast ion.

    Args:
        ne: Electron density [m^-3].
        te: Electron temperature [eV].
        ion_densities: Density of each thermal ion species [m^-3].
        ion_masses: Their masses [kg].
        ion_charges: Their charge numbers.
    """
    v_te = xp.sqrt(2.0 * xp.asarray(te, dtype=float) * _E / _ME)
    z_over_m = _z_over_mass(ne, ion_densities, ion_masses, ion_charges)
    return (0.75 * xp.sqrt(xp.pi) * _ME * v_te**3 * z_over_m) ** (1.0 / 3.0)


def pitch_angle_exponent(mass, ne, ion_densities, ion_masses, ion_charges):
    """Return ``beta = Z_eff / [Z]``, the pitch-angle scattering of the ions.

    ``[Z] = sum_j n_j Z_j^2 m_b / (n_e m_j)``, so ``beta = 1`` for a beam in a
    plasma of its own isotope.
    """
    zeff_ne = sum(
        xp.asarray(n) * z**2 for n, z in zip(ion_densities, ion_charges, strict=True)
    )
    z_bracket = mass * _z_over_mass(ne, ion_densities, ion_masses, ion_charges)
    return zeff_ne / xp.asarray(ne) / z_bracket


def _log_integral(w):
    """Return ``int_0^w 2u / (1 + u^3) du``."""
    w = xp.asarray(w, dtype=float)
    sqrt3 = xp.sqrt(3.0)
    antiderivative = (
        -2.0 / 3.0 * xp.log1p(w)
        + xp.log(w * w - w + 1.0) / 3.0
        + 2.0 / sqrt3 * xp.arctan((2.0 * w - 1.0) / sqrt3)
    )
    return antiderivative + 2.0 / sqrt3 * xp.pi / 6.0


def ion_heating_fraction(energy_ratio):
    """Return the share of a fast ion's energy that goes to the ions.

    ``(1/x) int_0^x dy / (1 + y^(3/2))`` with ``x = E_0 / E_c``.
    """
    x = xp.asarray(energy_ratio, dtype=float)
    return _log_integral(xp.sqrt(x)) / x


def pitch_at_speed(speed_ratio, birth_speed_ratio, birth_pitch, beta):
    """Return the mean pitch at ``v / v_c`` of ions born at ``v_0 / v_c``."""
    u3 = xp.asarray(speed_ratio, dtype=float) ** 3
    u03 = xp.asarray(birth_speed_ratio, dtype=float) ** 3
    return birth_pitch * (u3 * (u03 + 1.0) / (u03 * (u3 + 1.0))) ** (beta / 3.0)


def slowing_down_integrals(birth_speed, birth_pitch, v_c, tau_s, beta):
    """Return the time integrals of one fast ion from birth to thermalisation.

    Over the slowing-down trajectory ``dt = tau_s v^2 dv / (v^3 + v_c^3)``.

    Returns:
        ``(time, kinetic, parallel)``: the time to slow down to rest [s], the
        time integral of ``v^2`` [m^2/s] (``m/2`` times it is the energy
        content per unit birth rate) and of the parallel velocity
        ``v <xi>(v)`` [m].
    """
    v0 = xp.asarray(birth_speed, dtype=float)
    vc = xp.asarray(v_c, dtype=float)
    x = v0 / vc
    time = tau_s / 3.0 * xp.log1p(x**3)
    # int_0^x u^4 / (1 + u^3) du = x^2 / 2 - int_0^x u / (1 + u^3) du
    kinetic = tau_s * vc**2 * (0.5 * x**2 - 0.5 * _log_integral(x))
    u = x[..., None] * xp.asarray(_NODES)
    pitch = pitch_at_speed(
        u, x[..., None], xp.asarray(birth_pitch)[..., None], xp.asarray(beta)[..., None]
    )
    parallel = (
        tau_s
        * vc
        * x
        * xp.sum(xp.asarray(_WEIGHTS) * u**3 * pitch / (u**3 + 1.0), axis=-1)
    )
    return time, kinetic, parallel


def trapped_electron_correction(zeff, epsilon):
    """Return the trapped-electron correction ``G`` of the beam-driven current.

    The large-aspect-ratio fit ``G = (1.55 + 0.85/Z) sqrt(eps) - (0.20 +
    1.55/Z) eps``, as quoted for neutral-beam current drive (Start and Cordey;
    Wesson, *Tokamaks*). The coefficients have not been checked against the
    original papers; Lin-Liu and Hinton (Phys. Plasmas 4, 4179, 1997) give the
    general-geometry form.
    """
    z = xp.asarray(zeff, dtype=float)
    eps = xp.asarray(epsilon, dtype=float)
    return (1.55 + 0.85 / z) * xp.sqrt(eps) - (0.20 + 1.55 / z) * eps


#: The signature of a trapped-electron correction ``G(Z_eff, epsilon)``.
TrappedElectronCorrection = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SlowingDown:
    """Steady-state fast ions per shell, summed over all ranks."""

    rho_edges: np.ndarray
    electron_heating: np.ndarray
    """Power to the electrons in each shell [W]."""
    ion_heating: np.ndarray
    """Power to the ions in each shell [W]."""
    fast_ion_density: np.ndarray
    """[m^-3]."""
    fast_ion_pressure: np.ndarray
    """Isotropic estimate, two thirds of the energy density [Pa]."""
    fast_ion_current_density: np.ndarray
    """Parallel current density of the fast ions [A/m^2] (positive along B)."""
    current_density: np.ndarray
    """Net beam-driven parallel current density, with the electron shielding
    [A/m^2]."""
    driven_current: float
    """Sum of the current density times the shell areas [A]."""
    stored_energy: float
    """Fast-ion energy content [J]."""

    def to_dataset(self) -> xr.Dataset:
        """Return the profiles on the shell centres as an xarray dataset."""
        rho = 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])
        heating = np.stack([self.electron_heating, self.ion_heating])
        return xr.Dataset(
            {
                "heating": (
                    ("species", "rho"),
                    heating,
                    {"long_name": "heating", "units": "W"},
                ),
                "fast_ion_density": (
                    "rho",
                    self.fast_ion_density,
                    {"long_name": "fast-ion density", "units": "1/m^3"},
                ),
                "fast_ion_pressure": (
                    "rho",
                    self.fast_ion_pressure,
                    {"long_name": "fast-ion pressure", "units": "Pa"},
                ),
                "fast_ion_current_density": (
                    "rho",
                    self.fast_ion_current_density,
                    {"long_name": "fast-ion current density", "units": "A/m^2"},
                ),
                "current_density": (
                    "rho",
                    self.current_density,
                    {"long_name": "driven current density", "units": "A/m^2"},
                ),
            },
            coords={
                "rho": ("rho", rho, {"long_name": "rho_pol"}),
                "species": ["electrons", "ions"],
            },
            attrs={
                "driven_current": self.driven_current,
                "stored_energy": self.stored_energy,
            },
        )


def slow_down(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    *,
    ln_lambda=None,
    trapped_electron_correction: TrappedElectronCorrection = trapped_electron_correction,
    comm: MPI.Comm | None = None,
) -> SlowingDown:
    """Return the steady state of the births of ``deposition``, shell by shell.

    Every birth slows down on its own shell with the local ``tau_s``, ``v_c``
    and ``beta``; ``surfaces`` must have the shells of the deposition.

    Args:
        deposition: Births (this rank's) and their shells.
        plasma: Profiles.
        surfaces: Volumes, areas and aspect ratios of the shells.
        ln_lambda: Coulomb logarithm; from :func:`coulomb_logarithm` if None.
        trapped_electron_correction: ``G(Z_eff, epsilon)``.
        comm: Communicator over which the births are split; serial if None.
    """
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    n_shells = len(surfaces.rho_edges) - 1
    mass = deposition.mass_number * ATOMIC_MASS
    births = deposition.births
    # births within half a step outside the last closed surface see its values
    rho = xp.minimum(deposition.birth_rho, 1.0)
    ne = plasma.electron_density(rho)
    te = plasma.electron_temperature(rho)
    ions = plasma.ion_densities(rho)
    speed = xp.sqrt(xp.sum(births.velocities**2, axis=1))
    energy = 0.5 * mass * speed**2  # J
    tau_s = slowing_down_time(mass, 1, ne, te, ln_lambda)
    v_c = critical_velocity(ne, te, ions, plasma.ion_masses, plasma.ion_charges)
    beta = pitch_angle_exponent(mass, ne, ions, plasma.ion_masses, plasma.ion_charges)
    to_ions = ion_heating_fraction(speed**2 / v_c**2)
    time, kinetic, parallel = slowing_down_integrals(
        speed, deposition.birth_pitch, v_c, tau_s, beta
    )
    w = births.weights
    shell = xp.clip(
        surfaces.shell_index(xp.minimum(rho, 0.999999 * surfaces.rho_edges[-1])),
        0,
        n_shells - 1,
    )
    sums = xp.algorithms.segment_sum(
        xp.stack(
            [
                w * energy * (1.0 - to_ions),
                w * energy * to_ions,
                w * time,
                w * 0.5 * mass * kinetic,
                w * parallel,
            ],
            axis=-1,
        ),
        shell,
        n_shells,
    )
    sums = xp.to_numpy(allreduce(sums, comm))
    volume = surfaces.volume
    zeff = xp.to_numpy(plasma.zeff(surfaces.rho))
    g = xp.to_numpy(trapped_electron_correction(zeff, surfaces.inverse_aspect_ratio))
    j_fast = _E * sums[:, 4] / volume
    j_net = j_fast * (1.0 - (1.0 - g) / zeff)
    return SlowingDown(
        rho_edges=surfaces.rho_edges,
        electron_heating=sums[:, 0],
        ion_heating=sums[:, 1],
        fast_ion_density=sums[:, 2] / volume,
        fast_ion_pressure=2.0 / 3.0 * sums[:, 3] / volume,
        fast_ion_current_density=j_fast,
        current_density=j_net,
        driven_current=float(np.sum(j_net * surfaces.area)),
        stored_energy=float(np.sum(sums[:, 3])),
    )
