r"""Time-dependent slowing down: the response to a beam power that varies in time.

`freeofcharge.slow_down` gives the steady state of a constant source. A source
whose power changes in time, $P(t) = s(t)\,P_\mathrm{ref}$ with fixed birth
profiles (switch-on, switch-off, short blips, modulation), is followed with the
slowing-down Green's function: every fast ion slows down deterministically from
its birth, $dv/dt = -a_e v - a_i / v^2$ (classical drag, Stix, Plasma Phys. 14,
367, 1972; with one Coulomb logarithm $dv/dt = -(v^3 + v_c^3) / (\tau_s v^2)$
and the age at speed $v$, $\tau(v) = \frac{\tau_s}{3}\ln\frac{v_0^3 + v_c^3}{v^3 + v_c^3}$).
A moment $Q$ of the fast ions at time $t$ (their number, energy, heating of
the electrons and ions, parallel current, the power they bring to the thermal
ions) is then the convolution of the source with the response $K_Q(\tau)$ of
the births of a unit source at age $\tau$,

$$
Q(t) = \int_0^\infty s(t - \tau)\, K_Q(\tau)\, d\tau,
$$

and the steady state is $\int_0^\infty K_Q\,d\tau$ (Duhamel's principle: the
fast ions do not act on each other or on the plasma, so the response is
linear and time invariant; e.g. Heidbrink and Sadler, Nucl. Fusion 34, 535,
1994, for the use of the classical slowing down after beam blips).

`slowing_down_response` follows the births of a deposition along the same
trajectories as `slow_down` (the same drags with each species' Coulomb
logarithm, the same pitch modes for the current, the same end energy) and
keeps, per shell, the cumulative kernels $C_Q(\tau) = \int_0^\tau K_Q\,d\tau'$
on a grid of ages. `SlowingDownResponse.evolve` takes the source piecewise
constant on the steps of a uniform time grid, where the convolution is exact,
$Q(t_n) = \sum_{k \ge 1} s_{n-k}\,[C_Q(k\Delta t) - C_Q((k - 1)\Delta t)]$.
The electron heating of a birth is the rest of its energy loss after the ions'
share, so the energy balance $dW/dt = P_\mathrm{birth} - P_e - P_i - P_\mathrm{th}$
holds at every time. The same assumptions as in `slow_down` apply: zero orbit
width, no losses, fixed plasma profiles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from maybempi import MPI
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import to_unit
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
from scipy.signal import fftconvolve

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.deposition import allreduce
from freeofcharge.equilibrium import rho_coordinate
from freeofcharge.slowing_down import (
    TrappedElectronCorrection,
    collision_coefficients,
    sauter_trapped_electron_correction,
)
from freeofcharge.trapping import pitch_modes

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces
    from freeofcharge.profiles import Plasma

__all__ = [
    "SlowingDownResponse",
    "slowing_down_response",
    "time_dependent_slow_down",
]

_E = ELEMENTARY_CHARGE

#: The moments of the fast ions, in the order of `SlowingDownResponse.cumulative`.
QUANTITIES = (
    "fast_ions",
    "fast_ion_energy",
    "electron_heating",
    "ion_heating",
    "parallel",
    "thermalised_power",
)


def _interpolate_rows(x, y, ages, chunk_points: int = 2_000_000):
    r"""Return ``y`` of each row at ``ages``, constant beyond the row's last ``x``.

    ``x`` ``(N, G)`` increases along each row, ``y`` is ``(Q, N, G)``; the
    result is ``(Q, N, K)``. All rows go through one `numpy.interp` call, the
    rows shifted apart along the axis.
    """
    n, _ = x.shape
    out = np.empty((y.shape[0], n, ages.size))
    span = float(np.max(x[:, -1], initial=0.0)) + float(ages[-1]) + 1.0
    rows = max(1, chunk_points // max(ages.size, 1))
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        shift = span * np.arange(stop - start)[:, None]
        nodes = (x[start:stop] + shift).ravel()
        queries = (np.minimum(ages[None, :], x[start:stop, -1:]) + shift).ravel()
        for q in range(y.shape[0]):
            out[q, start:stop] = np.interp(
                queries, nodes, y[q, start:stop].ravel()
            ).reshape(stop - start, ages.size)
    return out


@dataclass(frozen=True)
class SlowingDownResponse:
    r"""The cumulative response of each shell to a unit source switched on at age 0.

    ``cumulative[q, k, s]`` is $C_q(\tau_k) = \int_0^{\tau_k} K_q\,d\tau$ for the
    births of shell ``s`` at their reference rates (the deposition's weights),
    for the moments `QUANTITIES`: the number of fast ions, their energy [J], the
    heating of the electrons and of the ions [W], $\sum w \int v_\parallel\,dt$
    [m/s] and the power that joins the thermal ions [W]. Its last age is the
    steady state, equal to `freeofcharge.slow_down`'s.
    """

    rho_edges: np.ndarray
    age: np.ndarray
    cumulative: np.ndarray
    volume: np.ndarray
    area: np.ndarray
    shielding: np.ndarray
    birth_power: float
    slowing_down_time: np.ndarray
    thermalisation_time: np.ndarray
    radial_coordinate: str = "poloidal"
    """Edges of the shells."""
    r"""Ages $\tau_k$ [s], from 0 to the longest slowing down, equally spaced."""
    """Cumulative kernels ``(quantity, age, shell)``."""
    r"""Shell volumes [$\mathrm{m^3}$]."""
    r"""Shell cross-section areas [$\mathrm{m^2}$]."""
    r"""$1 - Z_b (1 - G) / Z_\mathrm{eff}$ per shell: the driven current over the fast ions'."""
    """Energy rate of the births at the reference rates [W], all ranks."""
    r"""Birth-weighted mean of the Spitzer time $\tau_s = 1/a_e$ at birth, per shell [s]."""
    """Birth-weighted mean time from birth to thermalisation, per shell [s]."""
    r"""The flux $\rho$ is the root of (`freeofcharge.rho_coordinate`)."""

    @property
    def steady_state(self) -> np.ndarray:
        """Return the steady state of each moment per shell, ``(quantity, shell)``."""
        return self.cumulative[:, -1]

    def _at_ages(self, ages):
        """Return the cumulative kernels at ``ages``, ``(quantity, len(ages), shell)``."""
        ages = np.asarray(ages, dtype=float)
        index = np.clip(np.searchsorted(self.age, ages, side="right") - 1, 0, None)
        index = np.minimum(index, self.age.size - 2)
        step = self.age[index + 1] - self.age[index]
        weight = np.clip((ages - self.age[index]) / step, 0.0, 1.0)[None, :, None]
        return (1.0 - weight) * self.cumulative[:, index] + weight * self.cumulative[
            :, index + 1
        ]

    @method(
        equations={
            "convolution": r"Q(t_n) = \sum_{k=1}^{n} s_{n-k} \left[C_Q(k\Delta t) - C_Q((k-1)\Delta t)\right] + s_\mathrm{init} \left[C_Q(\infty) - C_Q(n\Delta t)\right]",
            "source": r"s(t) = s_m \text{ on } [t_m, t_{m+1}), \qquad s_m = P(t_m + \Delta t/2) / P_\mathrm{ref}",
            "moments": r"n_f = \frac{Q_n}{V}, \quad p_f = \frac{2}{3}\frac{Q_W}{V}, \quad j_f = \frac{e Q_\parallel}{V}, \quad j = j_f \left(1 - \frac{Z_b}{Z_\mathrm{eff}}(1 - G)\right)",
        },
        assumptions=(
            r"the source piecewise constant on the time steps (exact there); before $t_0$ it is $s_\mathrm{init}$",
            r"$C_Q$ linear between the ages of the response",
        ),
        references=(
            "[Stix, Plasma Phys. 14, 367 (1972)](https://doi.org/10.1088/0032-1028/14/4/002)",
            "[Heidbrink and Sadler, Nucl. Fusion 34, 535 (1994)](https://doi.org/10.1088/0029-5515/34/4/I07)",
        ),
    )
    @sp.profile("freeofcharge.evolve_slowing_down", tags=PROFILE_TAGS)
    def evolve(
        self,
        time,
        power: Callable | np.ndarray | float,
        *,
        initial_power: float = 0.0,
        reference_power: float | None = None,
    ) -> xr.Dataset:
        r"""Return the fast ions per shell and time under the source power ``power``.

        Args:
            time: Equally spaced times $t_n$ [s].
            power: The source power [W]: a callable of time, evaluated at the
                midpoints of the steps, or one value per time, the power on
                $[t_n, t_{n+1})$ (the last value is not used), or a number.
            initial_power: The constant power before $t_0$ [W]: 0 starts
                without fast ions, ``reference_power`` in steady state.
            reference_power: The power [W] at which the deposition's births
                have their weights; its birth power (the deposited power,
                without shine-through) if None. Pass the injected power to give
                ``power`` as an injected power.

        Returns:
            Dataset over ``(time, rho)`` with the heating of the ``species``
            (electrons, ions) [W] and its density, the fast-ion density,
            energy, pressure, current density and driven current density, the
            power that joins the thermal ions, and the whole-plasma
            ``stored_energy``, ``driven_current``, ``total_heating`` and the
            ``birth_power`` over ``time``.
        """
        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must be one-dimensional with at least two points")
        step = np.diff(time)
        dt = float(time[-1] - time[0]) / (time.size - 1)
        if dt <= 0.0 or not np.allclose(step, dt, rtol=1e-6, atol=0.0):
            raise ValueError("time must be increasing and equally spaced")
        if callable(power):
            source = np.asarray(power(time + 0.5 * dt), dtype=float)
            source = np.broadcast_to(source, time.shape).astype(float)
        else:
            source = np.broadcast_to(np.asarray(power, dtype=float), time.shape)
            source = source.astype(float)
        reference = self.birth_power if reference_power is None else reference_power
        scale = source / reference
        cumulative = self._at_ages(dt * np.arange(time.size))
        increments = np.diff(cumulative, axis=1, prepend=cumulative[:, :1])
        moments = fftconvolve(scale[None, :, None], increments, axes=1)[:, : time.size]
        if initial_power:
            moments += (initial_power / reference) * (
                self.steady_state[:, None, :] - cumulative
            )
        count, energy, to_electrons, to_ions, parallel, thermalised = moments
        heating = np.stack([to_electrons, to_ions])
        j_fast = _E * parallel / self.volume
        j_net = j_fast * self.shielding
        rho = 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])
        dims = ("time", "rho")
        return xr.Dataset(
            {
                "birth_power": (
                    "time",
                    source,
                    {"long_name": "birth power", "units": "W"},
                ),
                "heating": (
                    ("species", *dims),
                    heating,
                    {"long_name": "heating", "units": "W"},
                ),
                "heating_density": (
                    ("species", *dims),
                    heating / self.volume,
                    {"long_name": "heating", "units": "W/m^3"},
                ),
                "fast_ion_density": (
                    dims,
                    count / self.volume,
                    {"long_name": "fast-ion density", "units": "1/m^3"},
                ),
                "fast_ion_energy": (
                    dims,
                    energy,
                    {"long_name": "fast-ion energy", "units": "J"},
                ),
                "fast_ion_pressure": (
                    dims,
                    2.0 / 3.0 * energy / self.volume,
                    {"long_name": "fast-ion pressure", "units": "Pa"},
                ),
                "fast_ion_current_density": (
                    dims,
                    j_fast,
                    {"long_name": "fast-ion current density", "units": "A/m^2"},
                ),
                "current_density": (
                    dims,
                    j_net,
                    {"long_name": "driven current density", "units": "A/m^2"},
                ),
                "thermalised_power": (
                    dims,
                    thermalised,
                    {"long_name": "thermalised power", "units": "W"},
                ),
                "stored_energy": (
                    "time",
                    energy.sum(axis=-1),
                    {"long_name": "fast-ion energy", "units": "J"},
                ),
                "driven_current": (
                    "time",
                    np.nansum(j_net * self.area, axis=-1)
                    if np.any(np.isfinite(self.shielding))
                    else np.full(time.size, np.nan),
                    {"long_name": "driven current", "units": "A"},
                ),
                "total_heating": (
                    ("species", "time"),
                    heating.sum(axis=-1),
                    {"long_name": "heating", "units": "W"},
                ),
                "slowing_down_time": (
                    "rho",
                    self.slowing_down_time,
                    {"long_name": r"$\tau_s$", "units": "s"},
                ),
                "thermalisation_time": (
                    "rho",
                    self.thermalisation_time,
                    {"long_name": "thermalisation time", "units": "s"},
                ),
            },
            coords={
                "time": ("time", time, {"long_name": "time", "units": "s"}),
                "rho": ("rho", rho, rho_coordinate(self.radial_coordinate)),
                "species": ["electrons", "ions"],
            },
            attrs={
                "radial_coordinate": self.radial_coordinate,
                "reference_power": reference,
                "initial_power": initial_power,
            },
        )

    @method(
        equations={
            "gain": r"H_Q(f) = \frac{\int_0^\infty K_Q(\tau)\, e^{-2\pi i f \tau}\, d\tau}{\int_0^\infty K_Q\, d\tau} \approx \frac{\sum_k \Delta C_{Q,k}\, e^{-2\pi i f \bar\tau_k}}{C_Q(\infty)}",
        },
        assumptions=(r"the kernel at the midpoint $\bar\tau_k$ of each age step",),
    )
    def transfer_function(self, frequency) -> xr.Dataset:
        r"""Return the response to a sinusoidal source relative to the steady one.

        A source $s = 1 + \epsilon\, e^{2\pi i f t}$ makes each moment
        $Q = Q_\infty (1 + \epsilon\, H_Q(f)\, e^{2\pi i f t})$: $|H_Q|$ is the
        modulation amplitude relative to the source's and $-\arg H_Q$ the phase
        delay.

        Args:
            frequency: Frequencies [Hz].

        Returns:
            Complex gains over ``(frequency, rho)`` (``heating`` also over
            ``species``), and of the whole-plasma ``stored_energy``,
            ``driven_current`` and ``total_heating`` over ``frequency``; NaN
            where the steady value is zero.
        """
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
        increments = np.diff(self.cumulative, axis=1)
        middle = 0.5 * (self.age[1:] + self.age[:-1])
        phase = np.exp(-2j * np.pi * frequency[:, None] * middle[None, :])
        transform = np.einsum("fk,qks->qfs", phase, increments)
        steady = self.steady_state[:, None, :]

        def ratio(numerator, denominator):
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(denominator != 0.0, numerator / denominator, np.nan)

        gain = ratio(transform, steady)
        current = transform[4] * self.shielding * self.area / self.volume
        steady_current = steady[4] * self.shielding * self.area / self.volume
        rho = 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])
        dims = ("frequency", "rho")
        attrs = {"long_name": "gain", "units": ""}
        return xr.Dataset(
            {
                "fast_ion_density": (dims, gain[0], attrs),
                "fast_ion_energy": (dims, gain[1], attrs),
                "heating": (("species", *dims), gain[2:4], attrs),
                "fast_ion_current_density": (dims, gain[4], attrs),
                "thermalised_power": (dims, gain[5], attrs),
                "stored_energy": (
                    "frequency",
                    ratio(transform[1].sum(-1), steady[1].sum(-1)),
                    attrs,
                ),
                "driven_current": (
                    "frequency",
                    ratio(np.nansum(current, -1), np.nansum(steady_current, -1)),
                    attrs,
                ),
                "total_heating": (
                    ("species", "frequency"),
                    ratio(transform[2:4].sum(-1), steady[2:4].sum(-1)),
                    attrs,
                ),
            },
            coords={
                "frequency": (
                    "frequency",
                    frequency,
                    {"long_name": "frequency", "units": "Hz"},
                ),
                "rho": ("rho", rho, rho_coordinate(self.radial_coordinate)),
                "species": ["electrons", "ions"],
            },
        )


@method(
    equations={
        "drag": r"\frac{dv}{dt} = -a_e v - \frac{a_i}{v^2}, \qquad \frac{d\tau}{d\ln v} = -\frac{v}{a_e v + a_i / v^2}",
        "kernels": r"C_Q(\tau) = \sum_b w_b\, c_{Q,b}(\min(\tau, \tau_b)), \qquad c_n = \tau, \quad c_W = \int_0^\tau E\,d\tau', \quad c_i = m_b \int_0^\tau \frac{a_i}{v}\,d\tau', \quad c_e = E_0 - E(\tau) - c_i",
        "thermalised": r"C_\mathrm{th}(\tau) = \sum_b w_b\, E_{\mathrm{end},b}\, H(\tau - \tau_b)",
        "current": r"C_\parallel(\tau) = \sum_b w_b \int_0^{\tau} v \sum_k A_{b,k}\, e^{-\kappa_k \int_0^{\tau'} \nu_D\, d\tau''}\, d\tau'",
        "spitzer_limit": r"\tau(v) = \frac{\tau_s}{3}\ln\frac{v_0^3 + v_c^3}{v^3 + v_c^3} \quad \text{(one Coulomb logarithm)}",
    },
    assumptions=(
        r"the births of the deposition at fixed relative rates; zero orbit width, fixed plasma",
        r"the trajectories, drags, pitch modes and end energy as in `freeofcharge.slow_down`; cumulative Simpson in $\ln v$",
        r"the heating rates are the kernels of the energy given to each species, so their cumulative is the energy given up to that age",
    ),
    references=(
        "[Stix, Plasma Phys. 14, 367 (1972)](https://doi.org/10.1088/0032-1028/14/4/002)",
        "[Gaffey, J. Plasma Phys. 16, 149 (1976)](https://doi.org/10.1017/S0022377800020134)",
        "[Heidbrink and Sadler, Nucl. Fusion 34, 535 (1994)](https://doi.org/10.1088/0029-5515/34/4/I07)",
    ),
    components=(collision_coefficients, pitch_modes),
)
@sp.profile("freeofcharge.slowing_down_response", tags=PROFILE_TAGS)
def slowing_down_response(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    *,
    ln_lambda=None,
    orbit_average: bool = True,
    thermal_factor: float = 0.0,
    minimum_energy: float = 0.0,
    trapped_electron_correction: TrappedElectronCorrection | None = None,
    modes: int = 8,
    speed_points: int = 201,
    age_points: int = 2001,
    comm: MPI.Comm | None = None,
) -> SlowingDownResponse:
    r"""Return the cumulative response kernels of the births of ``deposition`` per shell.

    The arguments are those of `freeofcharge.slow_down`; ``age_points`` ages,
    equally spaced from 0 to the longest slowing down, carry the kernels.

    Each birth's trajectory is sampled at ``speed_points`` speeds from its
    birth speed to its end speed, equally spaced in $\ln v$; its age, energy,
    the energy it has given to the ions and electrons and its
    $\int v_\parallel\,dt$ are integrated along it (cumulative Simpson), then
    interpolated linearly to the ages and summed over the births of each
    shell, each with its weight.
    """
    minimum_energy = to_unit(minimum_energy, "eV", "minimum_energy")
    if orbit_average and not surfaces.axisymmetric:
        raise ValueError(
            "the orbit average uses the bounce averages of axisymmetric surfaces; "
            "on 3D surfaces pass orbit_average=False for the local pitch"
        )
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    if speed_points % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of speeds")
    plasma.check_radial_coordinate(surfaces.radial_coordinate)
    n_shells = len(surfaces.rho_edges) - 1
    mass = deposition.mass_number * ATOMIC_MASS
    births = deposition.births
    rho = xp.minimum(xp.asarray(deposition.birth_rho), 1.0)
    ne = plasma.electron_density(rho)
    te = plasma.electron_temperature(rho)
    ti = plasma.ion_temperature(rho)
    ions = plasma.ion_densities(rho)
    speed = xp.sqrt(xp.sum(births.velocities**2, axis=1))
    energy = 0.5 * mass * speed**2
    end_energy = xp.minimum(
        xp.maximum(thermal_factor * ti, minimum_energy) * _E, energy
    )
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
    shell = xp.to_numpy(
        xp.clip(
            surfaces.shell_index(xp.minimum(rho, 0.999999 * surfaces.rho_edges[-1])),
            0,
            n_shells - 1,
        )
    )
    if orbit_average:
        pitch = pitch_modes(surfaces, modes=modes)
        amplitudes = pitch.amplitudes(
            shell,
            xp.to_numpy(deposition.birth_pitch),
            xp.to_numpy(deposition.birth_field),
        )
        rates = pitch.rates[shell]
    else:
        amplitudes = xp.to_numpy(deposition.birth_pitch)[:, None]
        rates = np.ones((amplitudes.shape[0], 1))

    v = xp.to_numpy(speeds)
    shape = v.shape
    a_e = np.broadcast_to(xp.to_numpy(electron_drag), shape)
    a_i = np.broadcast_to(xp.to_numpy(ion_drag), shape)
    nu = np.broadcast_to(xp.to_numpy(deflection), shape)
    log_step = (np.log(v[:, 0] / v[:, -1]) / (speed_points - 1))[:, None]

    def accumulate(integrand):
        return log_step * cumulative_simpson(integrand, dx=1.0, axis=-1, initial=0.0)

    dt = v / (a_e * v + a_i / v**2)  # -d(age)/d(ln v)
    age = accumulate(dt)
    kinetic = accumulate(v**2 * dt)
    to_ions = mass * accumulate(a_i / v * dt)
    scattering = log_step * cumulative_trapezoid(
        nu / v**3 * dt, dx=1.0, axis=-1, initial=0.0
    )
    parallel = np.zeros(shape)
    for k in range(rates.shape[1]):
        parallel += amplitudes[:, k, None] * accumulate(
            v * dt * np.exp(-rates[:, k, None] * scattering)
        )
    energy_np = xp.to_numpy(energy)
    to_electrons = energy_np[:, None] - 0.5 * mass * v**2 - to_ions
    end = xp.to_numpy(end_energy)
    weights = xp.to_numpy(births.weights)

    longest = float(np.max(age[:, -1], initial=0.0))
    if comm is not None and comm.Get_size() > 1:
        longest = comm.allreduce(longest, op=MPI.MAX)
    ages = np.linspace(0.0, longest if longest > 0.0 else 1.0, age_points)

    # per birth the cumulative kernels: the age while fast, the time integral of
    # the energy, the energy given to the electrons and to the ions so far (the
    # integrals of the heating rates) and the time integral of v_parallel
    stored = 0.5 * mass * kinetic
    cumulative_nodes = np.stack([age, stored, to_electrons, to_ions, parallel])
    values = _interpolate_rows(age, cumulative_nodes, ages)
    thermalised = end[:, None] * (ages[None, :] >= age[:, -1:])
    per_birth = np.concatenate([values, thermalised[None]], axis=0)
    one_hot = np.zeros((n_shells, weights.size))
    one_hot[shell, np.arange(weights.size)] = weights
    cumulative = np.einsum("sb,qbk->qks", one_hot, per_birth)
    totals = one_hot @ np.stack(
        [
            np.ones_like(weights),
            1.0 / a_e[:, 0],
            age[:, -1],
            energy_np,
        ],
        axis=-1,
    )
    cumulative = xp.to_numpy(allreduce(xp.asarray(cumulative), comm))
    totals = xp.to_numpy(allreduce(xp.asarray(totals), comm))
    rate = totals[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        tau_s = np.where(rate > 0.0, totals[:, 1] / rate, np.nan)
        lifetime = np.where(rate > 0.0, totals[:, 2] / rate, np.nan)

    zeff = xp.to_numpy(plasma.zeff(surfaces.rho))
    if trapped_electron_correction is None and not surfaces.axisymmetric:
        g = np.full(n_shells, np.nan)
    else:
        correction = (
            sauter_trapped_electron_correction
            if trapped_electron_correction is None
            else trapped_electron_correction
        )
        g = np.asarray(correction(surfaces, plasma))
    return SlowingDownResponse(
        rho_edges=surfaces.rho_edges,
        age=ages,
        cumulative=cumulative,
        volume=np.asarray(surfaces.volume),
        area=np.asarray(surfaces.area),
        shielding=1.0 - (1.0 - g) / zeff,
        birth_power=float(np.sum(totals[:, 3])),
        slowing_down_time=tau_s,
        thermalisation_time=lifetime,
        radial_coordinate=surfaces.radial_coordinate,
    )


def time_dependent_slow_down(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    time,
    power: Callable | np.ndarray | float,
    *,
    initial_power: float = 0.0,
    reference_power: float | None = None,
    **options,
) -> xr.Dataset:
    """Return the fast ions per shell and time under the source power ``power``.

    `slowing_down_response` with ``options`` (the arguments of
    `freeofcharge.slow_down` and ``age_points``), then
    `SlowingDownResponse.evolve` with ``time``, ``power``, ``initial_power``
    and ``reference_power``.
    """
    response = slowing_down_response(deposition, plasma, surfaces, **options)
    return response.evolve(
        time, power, initial_power=initial_power, reference_power=reference_power
    )
