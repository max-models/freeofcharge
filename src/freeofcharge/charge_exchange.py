r"""Charge-exchange losses of the fast ions on the background neutrals.

A fast ion that meets a thermal hydrogen atom of the background (recycled at the
wall, its density $n_0(\rho)$ peaked at the edge, typically
$10^{15}$-$10^{17}\ \mathrm{m^{-3}}$) can capture its electron and leave the
plasma as a fast neutral (Heidbrink and Sadler, Nucl. Fusion 34, 535, 1994,
sec. 4): a loss channel that matters at low density and the signal of neutral
particle analysers. Along the slowing down of `freeofcharge.slow_down` the
ion is lost at the rate

$$
\nu_\mathrm{cx}(v) = n_0\, \sigma_\mathrm{cx}(E / A_b)\, v,
$$

with the H$^+$ + H(1s) cross section of Janev and Smith (Atomic and
Plasma-Material Interaction Data for Fusion 4, IAEA, 1993;
`plasmacoll.charge_exchange`) at the energy per nucleon $E / A_b$, the neutrals
at rest. It survives to speed $v$ with the probability

$$
S(v) = \exp\left(-\int_{v}^{v_0} \frac{\nu_\mathrm{cx}}{a_e v' + a_i / v'^2}\,dv'\right),
$$

on the trajectory $dv/dt = -a_e v - a_i / v^2$ of `collision_coefficients`
(Stix, Plasma Phys. 14, 367, 1972). Every moment of `slow_down` is weighted by
$S$: the fast-ion density, pressure and current, the heating of electrons and
ions, the thermalised power $S(v_\mathrm{end}) E_\mathrm{end}$. The power carried
away by the fast neutrals is

$$
P_\mathrm{cx} = \int S\,\nu_\mathrm{cx}\, \tfrac12 m_b v^2\,dt,
$$

so that $P_e + P_i + P_\mathrm{th} + P_\mathrm{cx} = P_\mathrm{birth}$; the power to the
electrons is the rest of this balance, as in `slow_down`, and its direct
integral checks it. The emission $S \nu_\mathrm{cx}\,dt$ binned in the
energy at capture is the spectrum of the fast neutrals.

Assumptions: zero orbit width (the ion stays on its birth surface and sees the
neutrals there); the neutrals cold (relative speed $v$; their thermal speed
matters only near the end of the slowing down); every fast neutral leaves the
plasma (no re-ionisation, an upper bound of the loss); the cold ion the neutral
leaves behind is thermal and not counted; ground-state H(1s) targets and the
hydrogen data at the same speed for D and T.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from maybempi import MPI
from methodcard import method
from plasmacoll.charge_exchange import proton_hydrogen_charge_exchange
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import to_array, to_unit

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.deposition import allreduce
from freeofcharge.slowing_down import (
    SlowingDown,
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
    "ChargeExchangeSlowingDown",
    "NeutralDensity",
    "charge_exchange_moments",
    "charge_exchange_rate",
    "exponential_neutral_density",
    "slow_down_with_charge_exchange",
]

_E = ELEMENTARY_CHARGE

#: A neutral density: a number (uniform), one value per shell, or a function of $\rho$ [$\mathrm{m^{-3}}$].
NeutralDensity = float | np.ndarray | Callable[[np.ndarray], np.ndarray]


@method(
    equations={
        "profile": r"n_0(\rho) = n_{0,\mathrm{edge}}\, \exp\!\left(-\frac{1 - \rho}{\lambda}\right)",
    },
    assumptions=(
        r"$\lambda$ in units of $\rho$; a decay into the plasma from the last closed surface, as a recycling source gives",
    ),
)
def exponential_neutral_density(
    edge_density: float, decay_length: float
) -> Callable[[np.ndarray], np.ndarray]:
    r"""Return $n_0(\rho) = n_{0,\mathrm{edge}} \exp(-(1 - \rho) / \lambda)$ as a function of $\rho$.

    Args:
        edge_density: The neutral density at $\rho = 1$ [$\mathrm{m^{-3}}$].
        decay_length: $\lambda$, the e-folding length in $\rho$ (positive).
    """
    edge = to_unit(edge_density, "m^-3", "edge_density")
    if decay_length <= 0.0:
        raise ValueError("decay_length must be > 0")

    def density(rho):
        rho = xp.asarray(rho, dtype=float)
        return edge * xp.exp(-(1.0 - xp.minimum(rho, 1.0)) / decay_length)

    return density


@method(
    equations={
        "rate": r"\nu_\mathrm{cx} = n_0\, \sigma_\mathrm{cx}\!\left(\frac{m_b v^2}{2 A_b e}\right) v",
    },
    assumptions=(
        r"neutrals at rest; $\sigma_\mathrm{cx}$ against the energy per nucleon in eV/u",
    ),
    references=(
        "R. K. Janev and J. J. Smith, Atomic and Plasma-Material Interaction Data for Fusion 4 (IAEA, Vienna, 1993)",
    ),
    components=("plasmacoll.charge_exchange.proton_hydrogen_charge_exchange",),
)
def charge_exchange_rate(speed, neutral_density, cross_section=None):
    r"""Return the charge-exchange rate $\nu_\mathrm{cx} = n_0 \sigma_\mathrm{cx} v$ [1/s].

    Args:
        speed: Speed of the fast ion [m/s].
        neutral_density: $n_0$ [$\mathrm{m^{-3}}$], broadcast to ``speed``.
        cross_section: $\sigma_\mathrm{cx}$ [$\mathrm{m^2}$] as a function of
            the energy per nucleon [eV/u]; plasmacoll's H$^+$ + H(1s) fit if None.
    """
    sigma = proton_hydrogen_charge_exchange if cross_section is None else cross_section
    v = xp.asarray(speed, dtype=float)
    energy_per_nucleon = 0.5 * ATOMIC_MASS * v**2 / _E
    return (
        xp.asarray(neutral_density, dtype=float)
        * xp.asarray(sigma(energy_per_nucleon))
        * v
    )


def _cumulative_integral(values, step):
    r"""Return $\int_{x_0}^{x_k} f\,dx$ at every point, fourth order on equal steps.

    Each interval by the cubic through its four nearest points:
    $h/24\,(-f_{k-1} + 13 f_k + 13 f_{k+1} - f_{k+2})$ inside, the one-sided
    $h/24\,(9 f_0 + 19 f_1 - 5 f_2 + f_3)$ at the ends.
    """
    f = values
    inner = -f[:, :-3] + 13.0 * f[:, 1:-2] + 13.0 * f[:, 2:-1] - f[:, 3:]
    first = 9.0 * f[:, :1] + 19.0 * f[:, 1:2] - 5.0 * f[:, 2:3] + f[:, 3:4]
    last = f[:, -4:-3] - 5.0 * f[:, -3:-2] + 19.0 * f[:, -2:-1] + 9.0 * f[:, -1:]
    pieces = step[:, None] / 24.0 * xp.concatenate([first, inner, last], axis=-1)
    return xp.concatenate(
        [xp.zeros((f.shape[0], 1)), xp.cumsum(pieces, axis=-1)], axis=-1
    )


@method(
    equations={
        "survival": r"S_k = \exp\left(-\int_{\ln v_k}^{\ln v_0} \nu_\mathrm{cx} \frac{dt}{d\ln v}\,d\ln v\right), \qquad \frac{dt}{d\ln v} = \frac{v}{a_e v + a_i / v^2}",
        "moments": r"\int S\,dt, \quad \int S v^2\,dt, \quad \int S \frac{a_i}{v}\,dt, \quad \int S a_e v^2\,dt, \quad \int S \nu_\mathrm{cx} \frac{v^2}{2}\,dt, \quad \int S v\, e^{-\kappa_k \int \nu_D\,dt'}\,dt",
        "closed_form": r"\nu_\mathrm{cx}, a_e \text{ constant}, a_i = 0: \quad S = \left(\frac{v}{v_0}\right)^{\nu_\mathrm{cx}/a_e}, \qquad \frac{P_\mathrm{cx}}{E_0} = \frac{\nu_\mathrm{cx}}{\nu_\mathrm{cx} + 2 a_e}\left[1 - \left(\frac{v}{v_0}\right)^{2 + \nu_\mathrm{cx}/a_e}\right]",
    },
    order=4,
    assumptions=(
        r"speeds equally spaced in $\ln v$, an odd number of them (at least 5)",
        r"the moments by Simpson's rule, the cumulative $\int \nu_\mathrm{cx}\,dt$ by local cubics (fourth order), the cumulative scattering by the trapezoidal rule as in `slowing_down_moments`",
    ),
    references=(
        "[Stix, Plasma Phys. 14, 367 (1972)](https://doi.org/10.1088/0032-1028/14/4/002)",
        "[Heidbrink and Sadler, Nucl. Fusion 34, 535 (1994)](https://doi.org/10.1088/0029-5515/34/4/I07)",
    ),
    components=("freeofcharge.slowing_down.slowing_down_moments",),
)
def charge_exchange_moments(speed, electron_drag, ion_drag, deflection, rates, cx_rate):
    r"""Return the survival-weighted time integrals over slowing-down trajectories.

    As `freeofcharge.slowing_down.slowing_down_moments` (same arguments, same
    speed grid), with every integrand weighted by the survival probability
    $S$ against charge exchange at the rate ``cx_rate`` [1/s] (broadcast to
    ``speed``). With ``cx_rate`` zero the first four results are those of
    `slowing_down_moments`.

    Returns:
        ``(time, kinetic, ion_loss, parallel, electron_loss, cx_loss, survival)``:
        the first four as in `slowing_down_moments`; the energy lost to the
        electrons and to charge exchange per unit mass [$\mathrm{m^2/s^2}$];
        and $S$ at every speed of the grid, ``(N, G)``.
    """
    v = xp.asarray(speed, dtype=float)
    points = v.shape[1]
    if points % 2 == 0 or points < 5:
        raise ValueError("need an odd number of speeds, at least 5")
    log_step = xp.log(v[:, 0] / v[:, -1]) / (points - 1)
    simpson = np.full(points, 2.0)
    simpson[1::2] = 4.0
    simpson[0] = simpson[-1] = 1.0
    weights = log_step[:, None] * xp.asarray(simpson / 3.0)
    dt = v / (electron_drag * v + ion_drag / v**2)
    loss = xp.asarray(cx_rate, dtype=float) * dt
    loss = loss * xp.ones_like(v)
    survival = xp.exp(-_cumulative_integral(loss, log_step))
    kept = weights * survival * dt
    time = xp.sum(kept, axis=-1)
    kinetic = xp.sum(kept * v**2, axis=-1)
    ion_loss = xp.sum(kept * ion_drag / v, axis=-1)
    electron_loss = xp.sum(kept * electron_drag * v**2, axis=-1)
    cx_loss = xp.sum(weights * survival * loss * 0.5 * v**2, axis=-1)
    rate = deflection / v**3 * dt
    half = 0.5 * log_step[:, None] * (rate[:, 1:] + rate[:, :-1])
    exponent = xp.concatenate(
        [xp.zeros((v.shape[0], 1)), xp.cumsum(half, axis=-1)], axis=-1
    )
    rates = xp.asarray(rates, dtype=float)
    parallel = xp.stack(
        [
            xp.sum(kept * v * xp.exp(-rates[:, k, None] * exponent), axis=-1)
            for k in range(rates.shape[1])
        ],
        axis=-1,
    )
    return (
        time,
        kinetic,
        ion_loss,
        parallel,
        electron_loss,
        cx_loss,
        survival,
    )


@dataclass(frozen=True)
class ChargeExchangeSlowingDown:
    """The steady state of `slow_down` with charge-exchange losses, summed over all ranks."""

    slowing_down: SlowingDown
    cx_loss: np.ndarray
    birth_power: np.ndarray
    birth_rate: np.ndarray
    cx_rate: np.ndarray
    neutral_density: np.ndarray
    spectrum_edges: np.ndarray
    fast_neutral_spectrum: np.ndarray
    """The survival-weighted moments, in the form of `slow_down`'s result
    (its ``power_balance`` leaves out the charge-exchange loss)."""
    """Power carried away by the fast neutrals, by birth shell [W]."""
    """Energy rate of the births in each shell [W]."""
    """Births per second in each shell [1/s]."""
    """Fast neutrals per second, by birth shell [1/s]."""
    r"""$n_0$ at the shell centres [$\mathrm{m^{-3}}$]."""
    """Energy bin edges of the spectrum [eV]."""
    """Fast neutrals per second and eV, by birth shell and energy at capture,
    ``(shells, bins)`` [1/(s eV)]."""

    @property
    def cx_power(self) -> float:
        """Return the total power lost to charge exchange [W]."""
        return float(np.sum(self.cx_loss))

    @property
    def lost_fraction(self) -> float:
        """Return the share of the birth power lost to charge exchange."""
        return self.cx_power / self.slowing_down.birth_power

    @property
    def cx_probability(self) -> np.ndarray:
        r"""Return $P_\mathrm{cx}$ per shell: the share of its births that become fast neutrals."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.birth_rate > 0.0, self.cx_rate / self.birth_rate, 0.0)

    @property
    def power_balance(self) -> float:
        r"""Return $(P_e + P_i + P_\mathrm{th} + P_\mathrm{cx}) / P_\mathrm{birth} - 1$."""
        s = self.slowing_down
        heating = float(np.sum(s.electron_heating) + np.sum(s.ion_heating))
        return (heating + s.thermalised_power + self.cx_power) / s.birth_power - 1.0

    def to_dataset(self, surfaces: FluxSurfaces | None = None) -> xr.Dataset:
        """Return `SlowingDown.to_dataset` with the losses and the fast-neutral spectrum."""
        data = self.slowing_down.to_dataset(surfaces)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(
                self.birth_power > 0.0, self.cx_loss / self.birth_power, 0.0
            )
        data["cx_loss"] = (
            "rho",
            self.cx_loss,
            {"long_name": "charge-exchange loss", "units": "W"},
        )
        if surfaces is not None:
            data["cx_loss_density"] = (
                "rho",
                self.cx_loss / surfaces.volume,
                {"long_name": "charge-exchange loss", "units": "W/m^3"},
            )
        data["cx_power_fraction"] = (
            "rho",
            share,
            {"long_name": "lost share of the birth power", "units": ""},
        )
        data["cx_probability"] = (
            "rho",
            self.cx_probability,
            {"long_name": "charge-exchange probability", "units": ""},
        )
        data["neutral_density"] = (
            "rho",
            self.neutral_density,
            {"long_name": "neutral density", "units": "1/m^3"},
        )
        centres = 0.5 * (self.spectrum_edges[1:] + self.spectrum_edges[:-1])
        data = data.assign_coords(
            energy=("energy", centres, {"long_name": "energy", "units": "eV"})
        )
        data["fast_neutral_spectrum"] = (
            "energy",
            self.fast_neutral_spectrum.sum(axis=0),
            {"long_name": "fast-neutral flux", "units": "1/(s eV)"},
        )
        data["fast_neutral_spectrum_by_shell"] = (
            ("rho", "energy"),
            self.fast_neutral_spectrum,
            {"long_name": "fast-neutral flux", "units": "1/(s eV)"},
        )
        data.attrs.update(cx_power=self.cx_power, cx_lost_fraction=self.lost_fraction)
        return data


def _neutrals_per_birth(neutral_density, rho, shell, n_shells):
    """Return $n_0$ at each birth: a number, one value per shell or a function of rho."""
    if callable(neutral_density):
        return xp.asarray(neutral_density(rho), dtype=float) * xp.ones_like(rho)
    values = np.asarray(neutral_density, dtype=float)
    if values.ndim == 0:
        return xp.full(rho.shape, float(values))
    if values.shape != (n_shells,):
        raise ValueError(
            f"neutral_density needs one value per shell ({n_shells}), not {values.shape}"
        )
    return xp.asarray(values)[shell]


def _neutrals_on_shells(neutral_density, rho_centres):
    """Return $n_0$ at the shell centres, for the dataset."""
    if callable(neutral_density):
        return np.asarray(
            xp.to_numpy(xp.asarray(neutral_density(rho_centres))), dtype=float
        )
    return np.broadcast_to(
        np.asarray(neutral_density, dtype=float), rho_centres.shape
    ).copy()


#: Optical depth $\int \nu_\mathrm{cx}\,dt$ beyond which a trajectory is cut ($S < e^{-40}$).
_DEPTH_CUT = 40.0
#: The largest growth of the optical depth over one step of the speed grid.
_DEPTH_STEP = 0.2
#: The most points of the speed grid.
_MAX_POINTS = 4001


def _column(values):
    """``values`` as a column, to broadcast over the speed grid."""
    return xp.asarray(values)[:, None]


def _worst_step(depth) -> float:
    """The largest growth of the optical depth over a step, while $S$ matters."""
    matters = depth[:, :-1] < _DEPTH_CUT
    return float(xp.max(xp.where(matters, xp.diff(depth, axis=-1), 0.0)))


def _speed_grid(speed, log_end, points, coefficients, cx_rate):
    r"""Return the speed grid that resolves the survival, its coefficients and rates.

    Starts from ``points`` speeds from ``speed`` down to ``speed`` $e^{\text{log\_end}}$,
    equally spaced in $\ln v$, as `slow_down` takes them. Where the optical
    depth $\int \nu_\mathrm{cx}\,dt$ passes `_DEPTH_CUT`, the trajectory is cut
    there; where it grows by more than `_DEPTH_STEP` over a step (while $S$
    matters), the grid is refined. Without charge exchange the grid is
    unchanged.

    Returns:
        ``(speeds, (electron_drag, ion_drag, deflection), cx_rate, truncated)``.
    """
    truncated = xp.zeros(speed.shape, dtype=bool)

    def evaluate(points, log_end):
        speeds = speed[:, None] * xp.exp(
            log_end[:, None] * xp.linspace(0.0, 1.0, points)
        )
        drags = coefficients(speeds)
        rate = cx_rate(speeds) * xp.ones_like(speeds)
        dt = speeds / (drags[0] * speeds + drags[1] / speeds**2)
        depth = _cumulative_integral(rate * dt, -log_end / (points - 1))
        return speeds, drags, rate, depth

    speeds, drags, rate, depth = evaluate(points, log_end)
    for _ in range(12):
        if speed.shape[0] == 0:
            break
        beyond = depth[:, -1] > _DEPTH_CUT + 1.0
        worst = _worst_step(depth)
        cut = bool(xp.any(beyond))
        refine = worst > _DEPTH_STEP and points < _MAX_POINTS
        if not (cut or refine):
            break
        if cut:
            k = xp.maximum(xp.argmax(depth > _DEPTH_CUT, axis=-1), 1)
            low = xp.take_along_axis(depth, (k - 1)[:, None], axis=-1)[:, 0]
            high = xp.take_along_axis(depth, k[:, None], axis=-1)[:, 0]
            fraction = xp.clip((_DEPTH_CUT - low) / (high - low), 0.0, 1.0)
            position = (k - 1 + fraction) / (points - 1)
            log_end = xp.where(beyond, log_end * position, log_end)
            truncated = truncated | beyond
        if refine:
            wanted = int(np.ceil((points - 1) * worst / _DEPTH_STEP))
            points = min(_MAX_POINTS, wanted + 1 + wanted % 2)
        speeds, drags, rate, depth = evaluate(points, log_end)
    if speed.shape[0] and _worst_step(depth) > _DEPTH_STEP:
        worst = _worst_step(depth)
        warnings.warn(
            f"the speed grid ({points} points) does not resolve the charge-exchange "
            f"survival: the optical depth grows by {worst:.2g} over a step",
            stacklevel=3,
        )
    return speeds, drags, rate, truncated


def _fast_neutral_spectrum(
    energy, end_ratio, survival, weights, shell, n_shells, edges
):
    r"""Return the fast neutrals per second in each (shell, energy bin) [1/s].

    The captured share above the energy $E$, $1 - S(E)$, is interpolated in
    $\ln E$ between the grid points of each trajectory ($E_k = E_0 (v_k/v_0)^2$)
    and differenced at the bin edges, so the bins add up to $1 - S$ of the
    energies they cover.
    """
    n, points = survival.shape
    spectrum = np.zeros((n_shells, len(edges) - 1))
    if n == 0:
        return spectrum
    captured = 1.0 - survival
    span = 2.0 * np.log(end_ratio)  # ln(E_end / E_0), negative
    with np.errstate(divide="ignore", invalid="ignore"):
        position = np.log(np.maximum(edges, 1.0e-300)[None, :] / energy[:, None])
        position = np.where(span[:, None] < 0.0, position / span[:, None], 0.0)
    position = np.clip(position * (points - 1), 0.0, points - 1.0)
    k = np.minimum(np.floor(position).astype(int), points - 2)
    fraction = position - k
    at_edges = (1.0 - fraction) * np.take_along_axis(captured, k, axis=-1) + (
        fraction * np.take_along_axis(captured, k + 1, axis=-1)
    )
    np.add.at(spectrum, shell, weights[:, None] * (at_edges[:, :-1] - at_edges[:, 1:]))
    return spectrum


@method(
    equations={
        "rate": r"\nu_\mathrm{cx} = n_0(\rho_b)\, \sigma_\mathrm{cx}(E / A_b)\, v",
        "balance": r"\sum_k \left(P_{e,k} + P_{i,k} + P_{\mathrm{cx},k}\right) + P_\mathrm{th} = \sum_b w_b E_b",
        "spectrum": r"\frac{dN}{dE\,dt} = \frac{1}{\Delta E_j} \sum_b w_b \sum_{k:\,E_k \in \Delta E_j} q_k S_k \nu_{\mathrm{cx},k} \frac{dt}{d\ln v}\Big|_k",
    },
    assumptions=(
        r"as `slow_down`: zero orbit width, the plasma and the neutrals of the birth surface",
        r"cold neutrals; every fast neutral leaves the plasma (no re-ionisation); the cold ion left behind is not counted",
    ),
    references=(
        "[Stix, Plasma Phys. 14, 367 (1972)](https://doi.org/10.1088/0032-1028/14/4/002)",
        "[Heidbrink and Sadler, Nucl. Fusion 34, 535 (1994)](https://doi.org/10.1088/0029-5515/34/4/I07)",
        "R. K. Janev and J. J. Smith, Atomic and Plasma-Material Interaction Data for Fusion 4 (IAEA, Vienna, 1993)",
    ),
    components=(
        charge_exchange_rate,
        charge_exchange_moments,
        collision_coefficients,
        pitch_modes,
    ),
)
@sp.profile("freeofcharge.slow_down_with_charge_exchange", tags=PROFILE_TAGS)
def slow_down_with_charge_exchange(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    neutral_density: NeutralDensity,
    *,
    cross_section: Callable | None = None,
    spectrum_edges=None,
    ln_lambda=None,
    orbit_average: bool = True,
    thermal_factor: float = 0.0,
    minimum_energy: float = 0.0,
    trapped_electron_correction: TrappedElectronCorrection | None = None,
    modes: int = 8,
    speed_points: int = 201,
    comm: MPI.Comm | None = None,
) -> ChargeExchangeSlowingDown:
    r"""Return the steady state of the births of ``deposition`` with charge-exchange losses.

    `freeofcharge.slow_down` with every fast ion surviving charge exchange on
    the neutrals of its birth surface with the probability
    $S = \exp(-\int \nu_\mathrm{cx}\,dt)$; with ``neutral_density`` zero the
    results are those of `slow_down`. The keyword arguments shared with
    `slow_down` mean the same.

    Args:
        deposition: Births (this rank's) and their shells.
        plasma: Profiles.
        surfaces: Volumes, areas and fields of the shells.
        neutral_density: $n_0$ [$\mathrm{m^{-3}}$]: a number, one value per
            shell, or a function of $\rho$ (e.g. `exponential_neutral_density`).
        cross_section: $\sigma_\mathrm{cx}$ [$\mathrm{m^2}$] against the energy
            per nucleon [eV/u]; plasmacoll's H$^+$ + H(1s) fit if None.
        spectrum_edges: Energy bins of the fast-neutral spectrum [eV]; 100
            equal bins up to the largest birth energy if None.
        ln_lambda: As `slow_down`.
        orbit_average: As `slow_down`.
        thermal_factor: As `slow_down`.
        minimum_energy: As `slow_down` [eV].
        trapped_electron_correction: As `slow_down`.
        modes: As `slow_down`.
        speed_points: As `slow_down` (odd, at least 5).
        comm: As `slow_down`.
    """
    minimum_energy = to_unit(minimum_energy, "eV", "minimum_energy")
    if orbit_average and not surfaces.axisymmetric:
        raise ValueError(
            "the orbit average uses the bounce averages of axisymmetric surfaces; "
            "on 3D surfaces pass orbit_average=False for the local pitch"
        )
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    if "unspecified" not in (
        surfaces.radial_coordinate,
        deposition.radial_coordinate,
    ) and (surfaces.radial_coordinate != deposition.radial_coordinate):
        raise ValueError(
            f"the surfaces' rho is {surfaces.radial_coordinate}, "
            f"the deposition's {deposition.radial_coordinate}"
        )
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
    energy = 0.5 * mass * speed**2  # J
    end_energy = xp.minimum(
        xp.maximum(thermal_factor * ti, minimum_energy) * _E, energy
    )
    shell = xp.clip(
        surfaces.shell_index(xp.minimum(rho, 0.999999 * surfaces.rho_edges[-1])),
        0,
        n_shells - 1,
    )
    neutrals = _neutrals_per_birth(neutral_density, rho, shell, n_shells)
    ratio = xp.maximum(xp.sqrt(end_energy / energy), 1.0e-4)
    speeds, (electron_drag, ion_drag, deflection), cx_rate, truncated = _speed_grid(
        speed,
        xp.log(ratio),
        speed_points,
        lambda v: collision_coefficients(
            v,
            mass,
            1,
            _column(ne),
            _column(te),
            [_column(n) for n in ions],
            plasma.ion_masses,
            plasma.ion_charges,
            _column(ti),
            None if ln_lambda is None else xp.reshape(xp.asarray(ln_lambda), (-1, 1)),
        ),
        # the cross section is per nucleon: the speed sets it, whatever the isotope
        lambda v: charge_exchange_rate(v, _column(neutrals), cross_section),
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
    time, kinetic, ion_loss, parallel_modes, _, cx_loss, survival = (
        charge_exchange_moments(
            speeds, electron_drag, ion_drag, deflection, rates, cx_rate
        )
    )
    parallel = xp.sum(amplitudes * parallel_modes, axis=-1)
    to_ions = mass * ion_loss
    to_cx = mass * cx_loss
    # a truncated trajectory ends where S < exp(-40): what is left there is negligible
    final_energy = xp.where(truncated, 0.5 * mass * speeds[:, -1] ** 2, end_energy)
    thermalised = survival[:, -1] * final_energy
    to_electrons = energy - thermalised - to_ions - to_cx
    w = births.weights
    sums = xp.algorithms.segment_sum(
        xp.stack(
            [
                w * to_electrons,
                w * to_ions,
                w * time,
                w * 0.5 * mass * kinetic,
                w * parallel,
                w * thermalised,
                w * trapped,
                w,
                w * energy,
                w * to_cx,
                w * (1.0 - survival[:, -1]),
            ],
            axis=-1,
        ),
        shell,
        n_shells,
    )
    sums = xp.to_numpy(allreduce(sums, comm))
    if spectrum_edges is None:
        local_max = float(xp.max(energy)) / _E if speed.shape[0] else 0.0
        top = comm.allreduce(local_max, op=MPI.MAX) if comm is not None else local_max
        spectrum_edges = np.linspace(0.0, top if top > 0.0 else 1.0, 101)
    edges = np.asarray(to_array(spectrum_edges, "eV", "spectrum_edges"), dtype=float)
    spectrum = _fast_neutral_spectrum(
        xp.to_numpy(energy) / _E,
        xp.to_numpy(speeds[:, -1] / speeds[:, 0]),
        xp.to_numpy(survival),
        xp.to_numpy(w),
        xp.to_numpy(shell),
        n_shells,
        edges,
    )
    spectrum = allreduce(spectrum, comm) / np.diff(edges)
    volume = surfaces.volume
    zeff = xp.to_numpy(plasma.zeff(surfaces.rho))
    j_fast = _E * sums[:, 4] / volume
    if trapped_electron_correction is None and not surfaces.axisymmetric:
        warnings.warn(
            "3D surfaces: the trapped-electron correction of the driven current "
            "(Sauter, Start-Cordey) assumes axisymmetry, so current_density and "
            "driven_current are NaN; pass trapped_electron_correction to use one",
            stacklevel=2,
        )
        g = np.full(n_shells, np.nan)
    else:
        correction = (
            sauter_trapped_electron_correction
            if trapped_electron_correction is None
            else trapped_electron_correction
        )
        g = np.asarray(correction(surfaces, plasma))
    j_net = j_fast * (1.0 - (1.0 - g) / zeff)
    total_births = float(np.sum(sums[:, 7]))
    steady = SlowingDown(
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
        radial_coordinate=surfaces.radial_coordinate,
        birth_power=float(np.sum(sums[:, 8])),
    )
    rho_centres = 0.5 * (surfaces.rho_edges[1:] + surfaces.rho_edges[:-1])
    return ChargeExchangeSlowingDown(
        slowing_down=steady,
        cx_loss=sums[:, 9],
        birth_power=sums[:, 8],
        birth_rate=sums[:, 7],
        cx_rate=sums[:, 10],
        neutral_density=_neutrals_on_shells(neutral_density, rho_centres),
        spectrum_edges=edges,
        fast_neutral_spectrum=spectrum,
    )
