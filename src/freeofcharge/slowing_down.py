r"""Classical slowing down of fast ions (Stix), local to their flux surface.

A fast ion of speed ``v`` with ``v_Ti << v << v_Te`` loses energy to the
electrons and to the ions at the rate

.. math::

    \frac{dv}{dt} = -\frac{v^3 + v_c^3}{\tau_s v^2},

with the Spitzer slowing-down time on the electrons

.. math::

    \tau_s = \frac{3 (2\pi)^{3/2} \varepsilon_0^2 m_b T_e^{3/2}}
                  {n_e Z_b^2 e^4 m_e^{1/2} \ln\Lambda_e}

and the critical velocity, where the electron and ion drags are equal,

.. math::

    v_c^3 = \frac{3\sqrt{\pi}}{4} m_e v_{Te}^3
            \sum_j \frac{n_j Z_j^2}{n_e m_j}
            \frac{\ln\Lambda_j}{\ln\Lambda_e}, \qquad v_{Te} = \sqrt{2 T_e / m_e}

(Stix, Plasma Phys. 14, 367, 1972). Its pitch ``xi = v_par / v`` relaxes by
pitch-angle scattering on the ions, ``d<xi>/dt = -nu_D <xi>``, with the
deflection rate of a fast particle on cold targets ``nu_D = sum_j Gamma_j / v^3``,
``Gamma_j = n_j Z_j^2 Z_b^2 e^4 ln(Lambda_j) / (4 pi eps_0^2 m_b^2)`` (Helander
and Sigmar, *Collisional Transport in Magnetized Plasmas*, 2002), while the
ions' energy drag is ``sum_j (m_b / m_j) Gamma_j / v^2``. Their ratio,

.. math::

    \beta = \frac{\sum_j n_j Z_j^2 \ln\Lambda_j}
                 {\sum_j n_j Z_j^2 \ln\Lambda_j\, m_b / m_j},

equal to ``Z_eff / [Z]`` for equal logarithms, gives ``nu_D = beta v_c^3 /
(tau_s v^3)`` and, dividing by ``dv/dt``,

.. math::

    \langle\xi\rangle(v) = \xi_0 \left[\frac{v^3 (v_0^3 + v_c^3)}
                                         {v_0^3 (v^3 + v_c^3)}\right]^{\beta/3},

the first Legendre term of the form of Gaffey (J. Plasma Phys. 16, 149, 1976),
the ``l``-th term decaying with the exponent ``l (l + 1) beta / 6`` from the
eigenvalues ``l (l + 1) / 2`` of the Lorentz operator. The exponent is derived
here and checked against the equations of motion and a Monte Carlo of the
scattering (tests), not symbol by symbol against Gaffey's paper. These closed
forms hold for constant Coulomb logarithms.

Each species has its own Coulomb logarithm (:func:`coulomb_logarithms`, as
ASCOT5 evaluates it); the ion ones grow with the speed of the fast ion, so
:func:`slow_down` integrates the slowing down numerically
(:func:`slowing_down_moments`), with the drags and the deflection rate at
each speed.

The ions slow down on the flux surface where they are born (no orbit width,
no losses, no charge exchange) until their energy falls to
``max(thermal_factor T_i, minimum_energy)``, where they join the thermal ions
with what is left (zero by default: to rest). Their current is that of their
orbits: trapped ions carry none, and pitch-angle scattering is averaged over
the orbit (:mod:`freeofcharge.trapping`). The beam-driven current is the
fast-ion current shielded by the electrons, ``j = j_f (1 - Z_b / Z_eff (1 -
G))``, with the trapped-electron correction ``G``.
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
from freeofcharge.trapping import pitch_modes

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces
    from freeofcharge.profiles import Plasma

__all__ = [
    "SlowingDown",
    "collision_coefficients",
    "coulomb_logarithm",
    "coulomb_logarithms",
    "critical_velocity",
    "electron_collisionality",
    "ion_heating_fraction",
    "l31_coefficient",
    "pitch_angle_exponent",
    "pitch_at_speed",
    "sauter_trapped_electron_correction",
    "slowing_down_integrals",
    "slowing_down_moments",
    "slowing_down_time",
    "slow_down",
    "start_cordey_fit",
    "start_cordey_trapped_electron_correction",
]

_E = ELEMENTARY_CHARGE
_ME = ELECTRON_MASS
_EPS0 = constants.epsilon_0
_HBAR = constants.hbar

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


def coulomb_logarithms(
    speed, mass, charge_number, ne, te, ion_densities, ion_masses, ion_charges, ti
):
    r"""Return the Coulomb logarithms of a fast ion on the electrons and each ion.

    .. math::

        \ln\Lambda_{b} = \ln\frac{\lambda_D}{\max(b_\mathrm{cl}, b_\mathrm{qm})},
        \quad b_\mathrm{cl} = \frac{|Z_b Z| e^2}{4\pi\varepsilon_0 m_r \bar u^2},
        \quad b_\mathrm{qm} = \frac{\hbar}{2 m_r \bar u},
        \quad \bar u^2 = v^2 + \frac{2T}{m},

    with the Debye length of electrons and ions together and the reduced mass
    ``m_r`` (Hirvijoki et al., Comput. Phys. Commun. 185, 1310, 2014; ASCOT5's
    ``mccc_coefs_clog``). For the electrons ``u`` is their thermal speed and
    the logarithm is about the NRL one; for the ions it grows with ``v``.

    Args:
        speed: Speed of the fast ion [m/s].
        mass: Its mass [kg].
        charge_number: Its charge number.
        ne: Electron density [m^-3].
        te: Electron temperature [eV].
        ion_densities: Density of each ion species [m^-3].
        ion_masses: Their masses [kg].
        ion_charges: Their charge numbers.
        ti: Ion temperature [eV].

    Returns:
        ``(ln_e, [ln_j, ...])`` broadcast over the arguments.
    """
    te_j = xp.asarray(te, dtype=float) * _E
    ti_j = xp.asarray(ti, dtype=float) * _E
    ne = xp.asarray(ne, dtype=float)
    v2 = xp.asarray(speed, dtype=float) ** 2
    inverse_square = ne / te_j + sum(
        xp.asarray(n, dtype=float) * z**2 / ti_j
        for n, z in zip(ion_densities, ion_charges, strict=True)
    )
    debye = xp.sqrt(_EPS0 / (_E**2 * inverse_square))

    def log(target_mass, target_charge, temperature):
        u2 = v2 + 2.0 * temperature / target_mass
        reduced = mass * target_mass / (mass + target_mass)
        classical = (
            abs(charge_number * target_charge)
            * _E**2
            / (4.0 * np.pi * _EPS0 * reduced * u2)
        )
        quantum = _HBAR / (2.0 * reduced * xp.sqrt(u2))
        return xp.log(debye / xp.maximum(classical, quantum))

    electrons = log(_ME, 1, te_j)
    ions = [log(m, z, ti_j) for m, z in zip(ion_masses, ion_charges, strict=True)]
    return electrons, ions


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
    """Return the critical velocity ``v_c`` [m/s] for equal Coulomb logarithms.

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
    plasma of its own isotope (equal Coulomb logarithms).
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
    """Return the time integrals of one fast ion from birth to rest, closed forms.

    Over the slowing-down trajectory ``dt = tau_s v^2 dv / (v^3 + v_c^3)``,
    with constant ``v_c``, ``tau_s`` and ``beta``.

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


def collision_coefficients(
    speed,
    mass,
    charge_number,
    ne,
    te,
    ion_densities,
    ion_masses,
    ion_charges,
    ti,
    ln_lambda=None,
):
    """Return the drags and the deflection of a fast ion at ``speed``.

    ``dv/dt = -electron_drag v - ion_drag / v^2`` and ``nu_D = deflection /
    v^3``: ``electron_drag = 1 / tau_s`` [1/s], ``ion_drag = sum_j (m_b / m_j)
    Gamma_j`` and ``deflection = sum_j Gamma_j`` [m^3/s^3], each with its own
    Coulomb logarithm (:func:`coulomb_logarithms`), or all with ``ln_lambda``
    if given. With equal logarithms ``ion_drag tau_s = v_c^3``.
    """
    if ln_lambda is None:
        ln_e, ln_ions = coulomb_logarithms(
            speed,
            mass,
            charge_number,
            ne,
            te,
            ion_densities,
            ion_masses,
            ion_charges,
            ti,
        )
    else:
        ln_e = xp.asarray(ln_lambda, dtype=float)
        ln_ions = [ln_e] * len(ion_masses)
    electron_drag = 1.0 / slowing_down_time(mass, charge_number, ne, te, ln_e)
    gamma = [
        xp.asarray(n)
        * (z * charge_number) ** 2
        * _E**4
        * ln_j
        / (4.0 * np.pi * _EPS0**2 * mass**2)
        for n, z, ln_j in zip(ion_densities, ion_charges, ln_ions, strict=True)
    ]
    ion_drag = sum(mass / m * g for m, g in zip(ion_masses, gamma, strict=True))
    deflection = sum(gamma)
    return electron_drag, ion_drag, deflection


def slowing_down_moments(speed, electron_drag, ion_drag, deflection, rates):
    r"""Return the time integrals over slowing-down trajectories sampled in speed.

    Each row of ``speed`` runs from the birth speed down to the end speed,
    equally spaced in ``ln v`` (an odd number of points); ``electron_drag``,
    ``ion_drag`` and ``deflection`` (see :func:`collision_coefficients`) are
    given on it. Simpson's rule in ``ln v`` integrates ``dt = dv / (a_e v +
    a_i / v^2)``, the trapezoidal rule the cumulative ``int nu_D dt``.

    Args:
        speed: Speeds ``(N, G)``, decreasing along each row [m/s].
        electron_drag: ``a_e`` [1/s], broadcast to ``speed``.
        ion_drag: ``a_i`` [m^3/s^3], broadcast to ``speed``.
        deflection: ``nu_D v^3`` [m^3/s^3], broadcast to ``speed``.
        rates: Decay rates ``kappa`` of the pitch modes in units of ``nu_D``,
            ``(N, M)``.

    Returns:
        ``(time, kinetic, ion_loss, parallel)``: the time to slow down [s], the
        time integral of ``v^2`` [m^2/s], the energy lost to the ions per unit
        mass [m^2/s^2], and ``int v exp(-kappa_k int nu_D dt') dt`` [m] for
        each rate, ``(N, M)``.
    """
    v = xp.asarray(speed, dtype=float)
    points = v.shape[1]
    if points % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of speeds")
    log_step = xp.log(v[:, 0] / v[:, -1]) / (points - 1)
    simpson = np.full(points, 2.0)
    simpson[1::2] = 4.0
    simpson[0] = simpson[-1] = 1.0
    weights = log_step[:, None] * xp.asarray(simpson / 3.0)
    # dt / d(ln v)
    dt = v / (electron_drag * v + ion_drag / v**2)
    time = xp.sum(weights * dt, axis=-1)
    kinetic = xp.sum(weights * v**2 * dt, axis=-1)
    ion_loss = xp.sum(weights * ion_drag / v * dt, axis=-1)
    # int nu_D dt from the birth speed down to each speed
    rate = deflection / v**3 * dt
    half = 0.5 * log_step[:, None] * (rate[:, 1:] + rate[:, :-1])
    exponent = xp.concatenate(
        [xp.zeros((v.shape[0], 1)), xp.cumsum(half, axis=-1)], axis=-1
    )
    rates = xp.asarray(rates, dtype=float)
    parallel = xp.stack(
        [
            xp.sum(weights * v * dt * xp.exp(-rates[:, k, None] * exponent), axis=-1)
            for k in range(rates.shape[1])
        ],
        axis=-1,
    )
    return time, kinetic, ion_loss, parallel


# ----------------------------------------------------------------------
# Trapped-electron correction of the shielding
# ----------------------------------------------------------------------


def start_cordey_fit(zeff, epsilon):
    """Return ``G = (1.55 + 0.85/Z) sqrt(eps) - (0.20 + 1.55/Z) eps``.

    Mikkelsen and Singer's fit (Nucl. Technol./Fusion 4, 237, 1983) to the
    collisionless values of Start and Cordey (Phys. Fluids 23, 1477, 1980) for
    circular surfaces at large aspect ratio. Within 6% of
    :func:`l31_coefficient` without collisions and ``f_t = 1.46 sqrt(eps) -
    0.46 eps^(3/2)`` up to ``eps = 0.3`` (Honda et al., Nucl. Fusion 52,
    023021, 2012, compare the models).
    """
    z = xp.asarray(zeff, dtype=float)
    eps = xp.asarray(epsilon, dtype=float)
    return (1.55 + 0.85 / z) * xp.sqrt(eps) - (0.20 + 1.55 / z) * eps


def l31_coefficient(zeff, trapped_fraction, collisionality=None):
    """Return Sauter's bootstrap coefficient ``L31``, equal to ``G``.

    ``L31 = (1 + 1.4/(Z+1)) X - 1.9/(Z+1) X^2 + 0.3/(Z+1) X^3 + 0.2/(Z+1) X^4``
    with ``X = f_t / (1 + (1 - 0.1 f_t) sqrt(nu_e*) + 0.5 (1 - f_t) nu_e* /
    Z)`` (Sauter, Angioni and Lin-Liu, Phys. Plasmas 6, 2834, 1999, eqs. 14).
    The trapped-electron correction of the beam-driven current is ``G = L31``
    (Lin-Liu and Hinton, Phys. Plasmas 4, 4179, 1997; Honda et al., Nucl.
    Fusion 52, 023021, 2012); collisions reduce it. No collisions if
    ``collisionality`` is None.
    """
    z = xp.asarray(zeff, dtype=float)
    ft = xp.asarray(trapped_fraction, dtype=float)
    nu = xp.asarray(0.0 if collisionality is None else collisionality, dtype=float)
    x = ft / (1.0 + (1.0 - 0.1 * ft) * xp.sqrt(nu) + 0.5 * (1.0 - ft) * nu / z)
    z1 = z + 1.0
    return (1.0 + 1.4 / z1) * x - 1.9 / z1 * x**2 + 0.3 / z1 * x**3 + 0.2 / z1 * x**4


def electron_collisionality(ne, te, zeff, safety_factor, major_radius, epsilon):
    """Return Sauter's ``nu_e* = 6.921e-18 q R n_e Z ln(Lambda_e) / (T_e^2 eps^1.5)``.

    With ``ln(Lambda_e) = 31.3 - ln(sqrt(n_e) / T_e)`` (Sauter et al. 1999,
    eqs. 18); SI units and ``T_e`` in eV.
    """
    ne = xp.asarray(ne, dtype=float)
    te = xp.asarray(te, dtype=float)
    ln_e = 31.3 - xp.log(xp.sqrt(ne) / te)
    return (
        6.921e-18
        * xp.asarray(safety_factor)
        * xp.asarray(major_radius)
        * ne
        * xp.asarray(zeff)
        * ln_e
        / (te**2 * xp.asarray(epsilon) ** 1.5)
    )


def sauter_trapped_electron_correction(surfaces: FluxSurfaces, plasma: Plasma):
    """Return ``G`` of each shell: :func:`l31_coefficient` with collisions.

    The trapped fraction, safety factor and major radius of the surfaces, the
    plasma at the shell centres.
    """
    rho = surfaces.rho
    zeff = xp.to_numpy(plasma.zeff(rho))
    ne = xp.to_numpy(plasma.electron_density(rho))
    te = xp.to_numpy(plasma.electron_temperature(rho))
    nu = electron_collisionality(
        ne,
        te,
        zeff,
        surfaces.safety_factor,
        surfaces.major_radius,
        surfaces.inverse_aspect_ratio,
    )
    return np.asarray(l31_coefficient(zeff, surfaces.trapped_fraction, nu))


def start_cordey_trapped_electron_correction(surfaces: FluxSurfaces, plasma: Plasma):
    """Return ``G`` of each shell from :func:`start_cordey_fit` (no collisions)."""
    zeff = xp.to_numpy(plasma.zeff(surfaces.rho))
    return np.asarray(start_cordey_fit(zeff, surfaces.inverse_aspect_ratio))


#: The signature of a trapped-electron correction: ``G`` per shell.
TrappedElectronCorrection = Callable[["FluxSurfaces", "Plasma"], np.ndarray]


@dataclass(frozen=True)
class SlowingDown:
    """Steady-state fast ions per shell, summed over all ranks."""

    rho_edges: np.ndarray
    electron_heating: np.ndarray
    ion_heating: np.ndarray
    fast_ion_density: np.ndarray
    fast_ion_pressure: np.ndarray
    fast_ion_current_density: np.ndarray
    current_density: np.ndarray
    driven_current: float
    stored_energy: float
    thermalised_power: float = 0.0
    trapped_births: float = 0.0
    """Power to the electrons in each shell [W]."""
    """Power to the ions in each shell, while the ions are fast [W]."""
    """[m^-3]."""
    """Isotropic estimate, two thirds of the energy density [Pa]."""
    """Parallel current density of the fast ions [A/m^2] (positive along B)."""
    """Net beam-driven parallel current density, with the electron shielding
    [A/m^2]."""
    """Sum of the current density times the shell areas [A]."""
    """Fast-ion energy content [J]."""
    """Energy rate the ions still have when they join the thermal ions [W]."""
    """Share of the births on trapped orbits (zero orbit width)."""

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
                "thermalised_power": self.thermalised_power,
                "trapped_births": self.trapped_births,
            },
        )


def slow_down(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    *,
    ln_lambda=None,
    orbit_average: bool = True,
    thermal_factor: float = 0.0,
    minimum_energy: float = 0.0,
    trapped_electron_correction: TrappedElectronCorrection = (
        sauter_trapped_electron_correction
    ),
    modes: int = 8,
    speed_points: int = 201,
    comm: MPI.Comm | None = None,
) -> SlowingDown:
    """Return the steady state of the births of ``deposition``, shell by shell.

    Every birth slows down on its own shell with the local electron and ion
    drags and pitch-angle scattering; ``surfaces`` must have the shells of the
    deposition.

    Args:
        deposition: Births (this rank's) and their shells.
        plasma: Profiles.
        surfaces: Volumes, areas and fields of the shells.
        ln_lambda: One Coulomb logarithm for all species (a number or one
            per birth); each its own (:func:`coulomb_logarithms`) if None.
        orbit_average: Carry the current of the orbits, trapped ions none and
            scattering averaged over the orbit (:mod:`freeofcharge.trapping`);
            if False, the local pitch with Gaffey's decay, as if the field were
            uniform.
        thermal_factor: The ions are thermal below ``thermal_factor T_i`` ...
        minimum_energy: ... or below this energy [eV], whichever is higher.
        trapped_electron_correction: ``G`` per shell.
        modes: Pitch modes of the orbit average.
        speed_points: Points of the speed integrals (equal steps in ``ln v``),
            odd.
        comm: Communicator over which the births are split; serial if None.
    """
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    n_shells = len(surfaces.rho_edges) - 1
    mass = deposition.mass_number * ATOMIC_MASS
    births = deposition.births
    # births within half a step outside the last closed surface see its values
    rho = xp.minimum(xp.asarray(deposition.birth_rho), 1.0)
    ne = plasma.electron_density(rho)
    te = plasma.electron_temperature(rho)
    ti = plasma.ion_temperature(rho)
    ions = plasma.ion_densities(rho)
    speed = xp.sqrt(xp.sum(births.velocities**2, axis=1))
    energy = 0.5 * mass * speed**2  # J
    end_energy = xp.minimum(
        xp.maximum(thermal_factor * ti, minimum_energy) * _E, energy
    )
    # to rest: the time below 1e-4 v_0 and the current carried there vanish
    ratio = xp.maximum(xp.sqrt(end_energy / energy), 1.0e-4)
    grid = xp.linspace(0.0, 1.0, speed_points)
    speeds = speed[:, None] * ratio[:, None] ** grid

    def column(values):
        return xp.asarray(values)[:, None]

    electron_drag, ion_drag, deflection = collision_coefficients(
        speeds,
        mass,
        1,
        column(ne),
        column(te),
        [column(n) for n in ions],
        plasma.ion_masses,
        plasma.ion_charges,
        column(ti),
        None if ln_lambda is None else xp.reshape(xp.asarray(ln_lambda), (-1, 1)),
    )
    shell = xp.clip(
        surfaces.shell_index(xp.minimum(rho, 0.999999 * surfaces.rho_edges[-1])),
        0,
        n_shells - 1,
    )
    if orbit_average:
        pitch = pitch_modes(surfaces, modes=modes)
        shell_np = xp.to_numpy(shell)
        mode_amplitudes = pitch.amplitudes(
            shell_np,
            xp.to_numpy(deposition.birth_pitch),
            xp.to_numpy(deposition.birth_field),
        )
        amplitudes = xp.asarray(mode_amplitudes)
        rates = xp.asarray(pitch.rates[shell_np])
        trapped = xp.asarray(np.all(mode_amplitudes == 0.0, axis=-1))
    else:
        amplitudes = xp.asarray(deposition.birth_pitch)[:, None]
        rates = xp.ones((speed.shape[0], 1))
        trapped = xp.zeros(speed.shape[0], dtype=bool)
    time, kinetic, ion_loss, parallel_modes = slowing_down_moments(
        speeds, electron_drag, ion_drag, deflection, rates
    )
    parallel = xp.sum(amplitudes * parallel_modes, axis=-1)
    to_ions = mass * ion_loss
    to_electrons = energy - end_energy - to_ions
    w = births.weights
    sums = xp.algorithms.segment_sum(
        xp.stack(
            [
                w * to_electrons,
                w * to_ions,
                w * time,
                w * 0.5 * mass * kinetic,
                w * parallel,
                w * end_energy,
                w * trapped,
                w,
            ],
            axis=-1,
        ),
        shell,
        n_shells,
    )
    sums = xp.to_numpy(allreduce(sums, comm))
    volume = surfaces.volume
    zeff = xp.to_numpy(plasma.zeff(surfaces.rho))
    g = np.asarray(trapped_electron_correction(surfaces, plasma))
    j_fast = _E * sums[:, 4] / volume
    j_net = j_fast * (1.0 - (1.0 - g) / zeff)
    total_births = float(np.sum(sums[:, 7]))
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
        thermalised_power=float(np.sum(sums[:, 5])),
        trapped_births=float(np.sum(sums[:, 6])) / total_births
        if total_births > 0.0
        else 0.0,
    )
