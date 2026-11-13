r"""Beam-target and thermonuclear fusion neutron rates, shell by shell.

A fast ion of the beam slowing down in a plasma with $n_t$ target ions per
$\mathrm{m^3}$ fuses at the rate $n_t \langle\sigma v_\mathrm{rel}\rangle_b(E)$,
so over its slowing down it makes

$$
N = n_t \int_{t_0}^{t_\mathrm{end}} \langle\sigma v_\mathrm{rel}\rangle_b\,dt
  = n_t \int_{v_\mathrm{end}}^{v_0} \frac{\langle\sigma v_\mathrm{rel}\rangle_b(v)}
                                            {a_e v + a_i / v^2}\,dv
$$

reactions, with the Stix drag $dv/dt = -(a_e v + a_i / v^2)$ of
`freeofcharge.slowing_down.collision_coefficients` (each species with its own
Coulomb logarithm, as `freeofcharge.slow_down` does). A steady birth rate
$w$ [1/s] gives $w N$ reactions per second. The reactivity
$\langle\sigma v_\mathrm{rel}\rangle_b$ of one fast ion in the target is
`plasmacoll.fusion.FusionReaction.beam_reactivity`: the cross sections of
Bosch and Hale (Nucl. Fusion 32, 611, 1992) averaged over an isotropic
Maxwellian target at $T_i$, or for a target at rest ($\sigma v_b$, the usual
approximation for $v_b \gg v_{Ti}$). The thermonuclear rate of the same
reaction is $n_1 n_2 \langle\sigma v\rangle(T_i) / (1 + \delta_{12})$, with the
reactivity of Bosch and Hale's Table VII.

Assumptions: the fast ions slow down on their birth shell (zero orbit width,
no losses, no charge exchange), as in `freeofcharge.slow_down`, and the
reactions are counted on that shell; the slowing down is that of the mean
speed (no energy diffusion); a fast ion counts until its energy falls to the
larger of ``thermal_factor`` $\times\,T_i$ (the mean thermal energy
$\tfrac32 T_i$ by default) and ``minimum_energy``, after which it is part of
the thermal ions, so neither rate counts it twice; the fast ions do not fuse
with each other (beam-beam), and the fusion products are not followed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from maybempi import MPI
from methodcard import ConvergenceProblem, method
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.fusion import FUSION_REACTIONS, FusionReaction
from plasmacoll.units import to_unit
from scipy.interpolate import RegularGridInterpolator

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.deposition import allreduce
from freeofcharge.equilibrium import rho_coordinate
from freeofcharge.slowing_down import collision_coefficients

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces
    from freeofcharge.profiles import Plasma

__all__ = [
    "NeutronRates",
    "neutron_rates",
    "slowing_down_yield",
    "thermonuclear_rate",
]

#: The nuclei of the reactions: (mass number, charge number).
_NUCLEI = {"D": (2, 1), "T": (3, 1), "3He": (3, 2)}


def _closed_form_yield(v0, v1, a_e, a_i, sigma):
    r"""$\sigma \int v\,dt$ from $v_0$ to $v_1$ with constant drags, closed form."""
    c = (a_i / a_e) ** (1.0 / 3.0)

    def f(x):
        # int_0^x du / (1 + u^3)
        return np.log((x + 1.0) ** 2 / (x**2 - x + 1.0)) / 6.0 + (
            np.arctan((2.0 * x - 1.0) / np.sqrt(3.0)) + np.pi / 6.0
        ) / np.sqrt(3.0)

    return sigma / a_e * ((v0 - v1) - c * (f(v0 / c) - f(v1 / c)))


def _yield_problem() -> ConvergenceProblem:
    """A constant cross section with constant drags, against the closed form."""
    a_e, a_i, v0, v1, sigma = 1.0, 2.0, 3.0, 0.5, 0.7
    exact = _closed_form_yield(v0, v1, a_e, a_i, sigma)
    span = np.log(v0 / v1)

    def error(log_step: float) -> float:
        points = round(span / log_step) + 1
        speed = np.exp(np.linspace(np.log(v0), np.log(v1), points))[None, :]
        value = slowing_down_yield(speed, a_e, a_i, sigma * speed)
        return float(abs(value[0] - exact))

    return ConvergenceProblem(error, steps=[span / 16, span / 32, span / 64])


@method(
    equations={
        "yield": r"Y = \int \langle\sigma v_\mathrm{rel}\rangle_b\,dt = \int_{\ln v_\mathrm{end}}^{\ln v_0} \frac{\langle\sigma v_\mathrm{rel}\rangle_b(v)\, v}{a_e v + a_i/v^2}\,d\ln v",
        "closed_form": r"Y = \frac{\sigma}{a_e}\Big[v_0 - v_1 - v_c\big(F(v_0/v_c) - F(v_1/v_c)\big)\Big], \quad F(x) = \int_0^x \frac{du}{1 + u^3}, \quad v_c^3 = \frac{a_i}{a_e} \quad (\sigma \text{ constant, cold target})",
    },
    order=4,
    assumptions=(
        r"speeds equally spaced in $\ln v$, an odd number of them; Simpson's rule, as `slowing_down_moments`",
        r"the order is checked with a constant cross section and constant drags",
    ),
    problem=_yield_problem,
)
def slowing_down_yield(speed, electron_drag, ion_drag, reactivity):
    r"""Return $\int \langle\sigma v_\mathrm{rel}\rangle_b\,dt$ [$\mathrm{m^3}$] along each slowing down.

    Args:
        speed: Speeds ``(N, G)``, decreasing along each row and equally spaced
            in $\ln v$, an odd number of them [m/s].
        electron_drag: $a_e$ [1/s], broadcast to ``speed``.
        ion_drag: $a_i$ [$\mathrm{m^3/s^3}$], broadcast to ``speed``.
        reactivity: $\langle\sigma v_\mathrm{rel}\rangle_b$ at ``speed``
            [$\mathrm{m^3/s}$].

    Returns:
        The integral for each row, ``(N,)``; times the target density, the
        reactions per fast ion.
    """
    v = np.asarray(speed, dtype=float)
    points = v.shape[1]
    if points % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of speeds")
    log_step = np.log(v[:, 0] / v[:, -1]) / (points - 1)
    simpson = np.full(points, 2.0)
    simpson[1::2] = 4.0
    simpson[0] = simpson[-1] = 1.0
    dt = v / (electron_drag * v + ion_drag / v**2)
    return log_step * np.sum(simpson / 3.0 * reactivity * dt, axis=-1)


@dataclass
class NeutronRates:
    """Beam-target and thermonuclear reaction rates per shell, summed over all ranks."""

    rho_edges: np.ndarray
    beam_target: np.ndarray
    thermonuclear: np.ndarray
    volume: np.ndarray
    reaction: str
    neutron: bool = True
    birth_rate: float = 0.0
    radial_coordinate: str = "poloidal"
    """Reactions of the fast ions born in each shell [1/s]."""
    """Reactions of the thermal ions in each shell [1/s]."""
    r"""Shell volumes [$\mathrm{m^3}$]."""
    """The reaction, a key of `plasmacoll.fusion.FUSION_REACTIONS`."""
    """Whether each reaction makes a neutron."""
    """Fast ions born per second, all ranks."""
    r"""The flux $\rho$ is the root of (`freeofcharge.rho_coordinate`)."""

    @property
    def total_beam_target(self) -> float:
        """Return the beam-target reactions per second of the whole plasma."""
        return float(np.sum(self.beam_target))

    @property
    def total_thermonuclear(self) -> float:
        """Return the thermonuclear reactions per second of the whole plasma."""
        return float(np.sum(self.thermonuclear))

    @property
    def reactions_per_ion(self) -> float:
        """Return the beam-target reactions per fast ion born."""
        return self.total_beam_target / self.birth_rate if self.birth_rate else 0.0

    def to_dataset(self) -> xr.Dataset:
        """Return the rates and their densities on the shell centres."""
        rho = 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])
        what = "neutron" if self.neutron else "reaction"

        def variable(values, name, units):
            return ("rho", values, {"long_name": name, "units": units})

        return xr.Dataset(
            {
                "beam_target_rate": variable(
                    self.beam_target, f"beam-target {what} rate", "1/s"
                ),
                "thermonuclear_rate": variable(
                    self.thermonuclear, f"thermonuclear {what} rate", "1/s"
                ),
                "beam_target_emissivity": variable(
                    self.beam_target / self.volume,
                    f"beam-target {what} emissivity",
                    "1/(m^3 s)",
                ),
                "thermonuclear_emissivity": variable(
                    self.thermonuclear / self.volume,
                    f"thermonuclear {what} emissivity",
                    "1/(m^3 s)",
                ),
            },
            coords={"rho": ("rho", rho, rho_coordinate(self.radial_coordinate))},
            attrs={
                "reaction": self.reaction,
                "radial_coordinate": self.radial_coordinate,
                "total_beam_target": self.total_beam_target,
                "total_thermonuclear": self.total_thermonuclear,
                "birth_rate": self.birth_rate,
            },
        )


def _reaction(reaction: str | FusionReaction) -> FusionReaction:
    """The reaction object of a key of `FUSION_REACTIONS`."""
    if isinstance(reaction, FusionReaction):
        return reaction
    if reaction not in FUSION_REACTIONS:
        raise ValueError(
            f"unknown reaction {reaction!r}; one of {sorted(FUSION_REACTIONS)}"
        )
    return FUSION_REACTIONS[reaction]


def _species(plasma: Plasma, nucleus: str) -> str | None:
    """The name of the ion species of ``plasma`` that is ``nucleus``, if any."""
    mass_number, charge = _NUCLEI[nucleus]
    for name, ion in plasma.ions.items():
        if ion.mass_number == mass_number and ion.charge == charge:
            return name
    return None


@method(
    equations={
        "rate": r"R_k = \frac{n_1 n_2}{1 + \delta_{12}}\,\langle\sigma v\rangle(T_i)\,V_k \quad \text{at the shell centre}",
    },
    assumptions=(
        r"both reactants Maxwellian at $T_i$; Bosch and Hale's reactivity (Table VII)",
        r"the midpoint rule over each shell",
    ),
    references=(
        "[Bosch and Hale, Nucl. Fusion 32, 611 (1992)](https://doi.org/10.1088/0029-5515/32/4/I07)",
    ),
)
def thermonuclear_rate(
    plasma: Plasma, surfaces: FluxSurfaces, reaction: str | FusionReaction = "D(d,n)3He"
) -> np.ndarray:
    """Return the thermonuclear reactions per second of each shell [1/s].

    The reactants are the ion species of ``plasma`` with their mass and charge
    numbers (D: 2, 1; T: 3, 1; 3He: 3, 2); zero if one is missing.
    """
    fusion = _reaction(reaction)
    first = _species(plasma, fusion.target)
    second = _species(plasma, fusion.projectile)
    if first is None or second is None:
        return np.zeros(len(surfaces.rho_edges) - 1)
    rho = surfaces.rho
    ions = dict(zip(plasma.ions, plasma.ion_densities(rho), strict=True))
    n1, n2 = (
        np.asarray(xp.to_numpy(ions[name]), dtype=float) for name in (first, second)
    )
    identical = 2.0 if fusion.target == fusion.projectile else 1.0
    ti = np.asarray(xp.to_numpy(plasma.ion_temperature(rho)), dtype=float)
    return n1 * n2 / identical * fusion.reactivity(ti) * surfaces.volume


def _reactivity_table(
    fusion: FusionReaction,
    beam: str,
    energies: np.ndarray,
    temperatures: np.ndarray | None,
    points: int,
):
    r"""$\langle\sigma v_\mathrm{rel}\rangle_b(E, T)$ tabulated in $\ln E$ and $T$, as a function."""
    low, high = 0.999 * energies.min(), 1.001 * energies.max()
    log_e = np.linspace(np.log(low), np.log(high), points)
    if temperatures is None:
        log_k = np.log(fusion.beam_reactivity(np.exp(log_e), beam=beam))

        def cold(energy, temperature):
            return np.exp(np.interp(np.log(energy), log_e, log_k))

        return cold
    t_low, t_high = temperatures.min(), temperatures.max()
    grid_t = np.linspace(t_low, max(t_high, t_low + 1.0), 33)
    log_k = np.log(
        fusion.beam_reactivity(np.exp(log_e)[:, None], grid_t[None, :], beam=beam)
    )
    table = RegularGridInterpolator((log_e, grid_t), log_k)

    def warm(energy, temperature):
        energy, temperature = np.broadcast_arrays(energy, temperature)
        query = np.stack([np.log(energy), temperature], axis=-1)
        return np.exp(table(query))

    return warm


@method(
    equations={
        "births": r"R_k^\mathrm{bt} = \sum_{b \in k} w_b\, n_t(\rho_b)\, Y_b, \qquad Y_b = \int_{v_\mathrm{end}}^{v_0} \frac{\langle\sigma v_\mathrm{rel}\rangle_b(v, T_i)}{a_e v + a_i/v^2}\,dv",
        "end_energy": r"E_\mathrm{end} = \min\!\left(\max(f_\mathrm{th}\, T_i,\ E_\mathrm{min}),\ E_0\right), \qquad f_\mathrm{th} = \texttt{thermal\_factor} = \tfrac32",
        "table": r"\ln\langle\sigma v_\mathrm{rel}\rangle_b \text{ tabulated on 257 points in } \ln E \text{ and 33 in } T_i, \text{ linear in between}",
    },
    assumptions=(
        r"zero orbit width: the fast ions slow down and react on their birth shell; no losses, no charge exchange, no energy diffusion",
        r"an isotropic Maxwellian target at $T_i$ (`maxwellian_target`), or at rest",
        r"no beam-beam reactions; fast ions below $E_\mathrm{end}$ count as thermal",
    ),
    references=(
        "[Bosch and Hale, Nucl. Fusion 32, 611 (1992)](https://doi.org/10.1088/0029-5515/32/4/I07)",
        "[Stix, Plasma Phys. 14, 367 (1972)](https://doi.org/10.1088/0032-1028/14/4/002)",
    ),
    components=(collision_coefficients, slowing_down_yield, thermonuclear_rate),
)
@sp.profile("freeofcharge.neutron_rates", tags=PROFILE_TAGS)
def neutron_rates(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    *,
    reaction: str | FusionReaction = "D(d,n)3He",
    target: str | None = None,
    maxwellian_target: bool = True,
    ln_lambda=None,
    thermal_factor: float = 1.5,
    minimum_energy: float = 0.0,
    speed_points: int = 101,
    table_points: int = 257,
    comm: MPI.Comm | None = None,
) -> NeutronRates:
    r"""Return the beam-target and thermonuclear rates of ``reaction``, shell by shell.

    Every birth of ``deposition`` slows down on its shell with the drags of
    `freeofcharge.slow_down` and reacts with the target ions there.

    Args:
        deposition: Births (this rank's) and their shells; the beam ions must
            be one reactant (mass number 2: D, 3: T).
        plasma: Profiles.
        surfaces: Volumes of the shells of the deposition.
        reaction: A key of `plasmacoll.fusion.FUSION_REACTIONS`, e.g.
            ``"D(d,n)3He"`` or ``"T(d,n)4He"``, or a reaction.
        target: The ion species of ``plasma`` the beam reacts with; the one
            that is the other reactant if None.
        maxwellian_target: Average over a Maxwellian target at $T_i$; if False
            the target is at rest.
        ln_lambda: One Coulomb logarithm for all species; each its own if None.
        thermal_factor: The fast ions join the thermal ions below
            ``thermal_factor`` $\times\,T_i$ ...
        minimum_energy: ... or below this energy [eV], whichever is higher.
        speed_points: Points of the slowing-down integral (equal steps in
            $\ln v$), odd.
        table_points: Energies of the reactivity table.
        comm: Communicator over which the births are split; serial if None.
    """
    minimum_energy = to_unit(minimum_energy, "eV", "minimum_energy")
    fusion = _reaction(reaction)
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    plasma.check_radial_coordinate(surfaces.radial_coordinate)
    beam = {2: "D", 3: "T"}.get(deposition.mass_number, "")
    if beam not in (fusion.target, fusion.projectile):
        raise ValueError(
            f"{fusion.key}: a beam of mass number {deposition.mass_number} is not a reactant"
        )
    other = fusion.projectile if beam == fusion.target else fusion.target
    if target is None:
        target = _species(plasma, other)
        if target is None:
            raise ValueError(f"the plasma has no {other} ions for {fusion.key}")
    elif target not in plasma.ions:
        raise ValueError(f"the plasma has no ion species {target!r}")
    elif (plasma.ions[target].mass_number, plasma.ions[target].charge) != _NUCLEI[
        other
    ]:
        raise ValueError(f"{fusion.key}: the target must be {other}, not {target!r}")
    n_shells = len(surfaces.rho_edges) - 1
    mass = deposition.mass_number * ATOMIC_MASS
    births = deposition.births
    rho = xp.minimum(xp.asarray(deposition.birth_rho), 1.0)
    ne, te, ti = (
        plasma.electron_density(rho),
        plasma.electron_temperature(rho),
        plasma.ion_temperature(rho),
    )
    ions = plasma.ion_densities(rho)
    n_target = ions[list(plasma.ions).index(target)]
    speed = xp.sqrt(xp.sum(births.velocities**2, axis=1))
    energy = 0.5 * mass * speed**2 / ELEMENTARY_CHARGE
    end_energy = xp.minimum(xp.maximum(thermal_factor * ti, minimum_energy), energy)
    ratio = xp.maximum(xp.sqrt(end_energy / energy), 1.0e-4)
    speeds = speed[:, None] * ratio[:, None] ** xp.linspace(0.0, 1.0, speed_points)

    def column(values):
        return xp.asarray(values)[:, None]

    electron_drag, ion_drag, _ = collision_coefficients(
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
    speeds_np = np.asarray(xp.to_numpy(speeds), dtype=float)
    ti_np = np.asarray(xp.to_numpy(ti), dtype=float)
    yields = np.zeros(speeds_np.shape[0])
    if speeds_np.shape[0] > 0:
        # the energy a fast ion has in the target's frame, with the beam's mass
        energies = 0.5 * mass * speeds_np**2 / ELEMENTARY_CHARGE
        energies *= fusion.masses[0 if beam == fusion.target else 1] / mass
        table = _reactivity_table(
            fusion,
            beam,
            energies,
            ti_np if maxwellian_target else None,
            table_points,
        )
        reactivity = table(energies, ti_np[:, None])
        yields = slowing_down_yield(
            speeds_np,
            np.asarray(xp.to_numpy(electron_drag)),
            np.asarray(xp.to_numpy(ion_drag)),
            reactivity,
        )
    shell = xp.clip(
        surfaces.shell_index(xp.minimum(rho, 0.999999 * surfaces.rho_edges[-1])),
        0,
        n_shells - 1,
    )
    w = births.weights
    sums = xp.algorithms.segment_sum(
        xp.stack([w * n_target * xp.asarray(yields), w], axis=-1), shell, n_shells
    )
    sums = xp.to_numpy(allreduce(sums, comm))
    return NeutronRates(
        rho_edges=surfaces.rho_edges,
        beam_target=sums[:, 0],
        thermonuclear=thermonuclear_rate(plasma, surfaces, fusion),
        volume=np.asarray(surfaces.volume),
        reaction=fusion.key,
        neutron=fusion.neutron,
        birth_rate=float(np.sum(sums[:, 1])),
        radial_coordinate=surfaces.radial_coordinate,
    )
