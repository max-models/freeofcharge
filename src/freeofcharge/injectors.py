r"""Neutral-beam injectors: boxes of positive-ion sources and their energy components.

A positive-ion source extracts atomic and molecular ions (for hydrogen
$\mathrm{H^+}$, $\mathrm{H_2^+}$, $\mathrm{H_3^+}$); after the neutraliser the
molecules have broken up into atoms at a half and a third of the full energy
$E$. `BeamSource` describes one source: where it sits and aims, $E$, its neutral
power and the shares $f_k$ of that power in the full, half and third energy
components, its width and divergence and the isotope. Component $k$ carries the
power $f_k P$ in atoms of energy $E / k$, so

$$
\dot N_k = \frac{f_k P}{e E / k}
$$

atoms per second. `InjectorBox` groups sources that converge on a crossover
point (the port), and `deposit_sources` deposits every component of every
source on one equilibrium and tabulates shine-through, wall hits and birth
profiles per source and component.

`w7x_injectors` builds W7-X-like injectors from `W7XInjectors`, one parameter
set. Only some of its numbers are published (hydrogen at 55 keV, about 1.8 MW
neutral power per source, two boxes NI20 and NI21 of four sources each, two
sources of NI21 in the campaign OP1.2b); the geometry, the power fractions, the
width and the divergence are **stand-ins**, not W7-X data: those are IPP's.
Replace them with `dataclasses.replace` when the real values are at hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import cunumpy as xp
import numpy as np
import xarray as xr
from maybempi import MPI
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS
from plasmacoll.units import to_array, to_unit

from freeofcharge.beam import NeutralBeam
from freeofcharge.deposition import Deposition, deposit
from freeofcharge.equilibrium import (
    FluxSurfaces,
    MagneticEquilibrium,
    rho_coordinate,
)
from freeofcharge.profiles import Plasma

__all__ = [
    "COMPONENTS",
    "BeamSource",
    "InjectorBox",
    "SourceDepositions",
    "W7XInjectors",
    "aim_direction",
    "deposit_sources",
    "w7x_injectors",
]

COMPONENTS = ("full", "half", "third")
"""The energy components of a positive-ion source, at $E$, $E/2$ and $E/3$."""


@dataclass(frozen=True)
class BeamSource:
    r"""One positive-ion source: position, aim, energy components and optics.

    The lengths, angles, energy and power take quantities (``"55 keV"``,
    ``"1.8 MW"``, ``"1 deg"``) as well as numbers in m, rad, eV and W.
    """

    name: str
    position: np.ndarray
    direction: np.ndarray
    energy: float
    power: float
    power_fractions: tuple[float, float, float] = (1.0, 0.0, 0.0)
    mass_number: int = 1
    width: float | tuple[float, float] = 0.0
    divergence: float | tuple[float, float] = 0.0
    halo_fraction: float = 0.0
    halo_divergence: float | tuple[float, float] = 0.0
    """Name, unique among the sources deposited together."""
    """Centre of the source grid [m]."""
    """Axis of the source (any length)."""
    """Full energy $E$ of the atoms [eV]."""
    """Neutral power of all components [W]."""
    """Shares of ``power`` in the full, half and third energy components."""
    """Isotope (1 H, 2 D)."""
    """Source half-width (standard deviation) [m], as `NeutralBeam.gaussian`."""
    """Core divergence (standard deviation) [rad]."""
    """Share of the markers in the halo."""
    """Halo divergence [rad]."""

    def __post_init__(self) -> None:
        """Convert the quantities and check the power fractions."""
        direction = np.asarray(self.direction, dtype=float)
        for name, value in (
            ("position", to_array(self.position, "m", "position")),
            ("direction", direction / np.linalg.norm(direction)),
            ("energy", to_unit(self.energy, "eV", "energy")),
            ("power", to_unit(self.power, "W", "power")),
            ("width", to_unit(self.width, "m", "width")),
            ("divergence", to_unit(self.divergence, "rad", "divergence")),
            (
                "halo_divergence",
                to_unit(self.halo_divergence, "rad", "halo_divergence"),
            ),
            ("power_fractions", tuple(float(f) for f in self.power_fractions)),
        ):
            object.__setattr__(self, name, value)
        fractions = np.asarray(self.power_fractions)
        if fractions.shape != (3,) or np.any(fractions < 0.0):
            raise ValueError("power_fractions are three shares, none negative")
        if not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"power_fractions sum to {fractions.sum()}, not 1")

    @method(
        equations={
            "components": r"E_k = \frac{E}{k}, \qquad P_k = f_k P, \qquad \dot N_k = \frac{f_k P}{e E_k}, \qquad k = 1, 2, 3",
        },
        assumptions=(
            r"each component a `NeutralBeam.gaussian` of the source's width and divergence, the same for all three",
            r"components with $f_k = 0$ are left out",
        ),
    )
    def beams(
        self,
        markers: int = 1000,
        *,
        seed: int | None = None,
        comm: MPI.Comm | None = None,
    ) -> dict[str, NeutralBeam]:
        r"""Return a beam per energy component (``"full"``, ``"half"``, ``"third"``).

        Each has ``markers`` markers over all ranks (the same for each, so
        each profile has the same statistics) and the weights of its power.
        """
        result = {}
        for k, (component, fraction) in enumerate(
            zip(COMPONENTS, self.power_fractions, strict=True)
        ):
            if fraction == 0.0:
                continue
            result[component] = NeutralBeam.gaussian(
                self.position,
                self.direction,
                self.energy / (k + 1),
                fraction * self.power,
                mass_number=self.mass_number,
                width=self.width,
                divergence=self.divergence,
                halo_fraction=self.halo_fraction,
                halo_divergence=self.halo_divergence,
                markers=markers,
                seed=None if seed is None else seed + k,
                comm=comm,
            )
        return result

    def beam(
        self,
        markers: int = 1000,
        *,
        seed: int | None = None,
        comm: MPI.Comm | None = None,
    ) -> NeutralBeam:
        """Return all energy components as one beam (``markers`` per component)."""
        return NeutralBeam.concatenate(
            list(self.beams(markers, seed=seed, comm=comm).values())
        )


@dataclass(frozen=True)
class InjectorBox:
    """A beam box: sources whose axes cross at one point, the port."""

    name: str
    sources: tuple[BeamSource, ...]
    """Name of the box."""
    """Its sources."""

    @property
    def power(self) -> float:
        """Return the neutral power of all sources [W]."""
        return float(sum(source.power for source in self.sources))

    @classmethod
    @method(
        equations={
            "axis": r"\hat d_s \propto \hat a + \tan\alpha_s\, \hat h + \tan\beta_s\, \hat v, \qquad \hat h = \frac{\hat a \times \hat z}{|\hat a \times \hat z|}, \quad \hat v = \hat h \times \hat a",
            "position": r"\mathbf{x}_s = \mathbf{x}_\mathrm{c} - L\, \hat d_s",
        },
        assumptions=(
            r"the source axes cross at $\mathbf{x}_\mathrm{c}$; source $s$ is turned by $\alpha_s$ horizontally and $\beta_s$ vertically from the box axis $\hat a$",
        ),
    )
    def converging(
        cls,
        name: str,
        crossover,
        axis,
        angles: Sequence[tuple[float, float]],
        distance: float,
        **source,
    ) -> InjectorBox:
        r"""Return a box whose sources aim through the ``crossover`` point.

        Args:
            name: Box name; the sources are ``"<name>/S<i>"``, from 1.
            crossover: Where the source axes cross [m].
            axis: Box axis (any length), into the plasma.
            angles: Per source $(\alpha, \beta)$, its horizontal and vertical
                angle from the box axis [rad].
            distance: From each source to the crossover [m].
            **source: The other `BeamSource` fields, the same for all.
        """
        crossover = to_array(crossover, "m", "crossover")
        distance = to_unit(distance, "m", "distance")
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        horizontal = np.cross(axis, [0.0, 0.0, 1.0])
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(horizontal, axis)
        sources = []
        for i, (alpha, beta) in enumerate(angles):
            alpha = to_unit(alpha, "rad", "angle")
            beta = to_unit(beta, "rad", "angle")
            direction = axis + np.tan(alpha) * horizontal + np.tan(beta) * vertical
            direction /= np.linalg.norm(direction)
            sources.append(
                BeamSource(
                    name=f"{name}/S{i + 1}",
                    position=crossover - distance * direction,
                    direction=direction,
                    **source,
                )
            )
        return cls(name, tuple(sources))


def aim_direction(
    target, tangency_radius: float, *, sense: int = 1, tilt: float = 0.0
) -> np.ndarray:
    r"""Return the direction of a line through ``target`` with tangency radius $R_t$.

    In the horizontal plane the line passes ``target`` heading inwards, with
    $\cos\gamma = R_t / R_\mathrm{target}$ the angle $\gamma$ to the toroidal
    direction (``sense`` $+1$ along $+\varphi$), then is tilted by ``tilt``
    [rad] upwards.
    """
    target = to_array(target, "m", "target")
    tangency_radius = to_unit(tangency_radius, "m", "tangency_radius")
    tilt = to_unit(tilt, "rad", "tilt")
    radius = float(np.hypot(target[0], target[1]))
    if not 0.0 <= tangency_radius <= radius:
        raise ValueError("the tangency radius must lie between 0 and the target's R")
    phi = np.arctan2(target[1], target[0])
    e_r = np.array([np.cos(phi), np.sin(phi), 0.0])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
    cos_g = tangency_radius / radius
    horizontal = sense * cos_g * e_phi - np.sqrt(1.0 - cos_g**2) * e_r
    return np.cos(tilt) * horizontal + np.array([0.0, 0.0, np.sin(tilt)])


@dataclass(frozen=True)
class W7XInjectors:
    r"""The parameters of `w7x_injectors`: published numbers and **stand-ins**.

    Published (Lazerson et al., EPS 2019 and IAEA FEC 2020, abstracts):
    ``energy``, ``mass_number``, ``source_power`` and the layout of two boxes
    with four sources each, two of NI21 in OP1.2b. Everything else is a
    defensible stand-in, **not W7-X data**: two boxes injecting in opposite
    toroidal directions, each aiming its box axis through a point near the
    magnetic axis of DESC's W7-X example in the triangular plane
    ($\varphi = 36^\circ$ and $108^\circ$), tangentially-ish (tangency radius
    ``tangency_radius``), with the sources converging on a crossover point
    ``crossover_distance`` before that point. Replace them with IPP's geometry
    (`dataclasses.replace`, or `InjectorBox` and `BeamSource` directly).
    """

    energy: float = 55e3
    mass_number: int = 1
    source_power: float = 1.8e6
    power_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2)
    width: float = 0.1
    divergence: float = 0.012
    boxes: tuple[str, ...] = ("NI20", "NI21")
    targets: tuple[tuple[float, float, float], ...] = (
        (5.26, np.radians(108.0), 0.0),
        (5.26, np.radians(36.0), 0.0),
    )
    senses: tuple[int, ...] = (-1, 1)
    tangency_radius: float = 4.3
    crossover_distance: float = 1.5
    source_distance: float = 6.0
    source_angles: tuple[tuple[float, float], ...] = (
        (-0.015, 0.02),
        (0.015, 0.02),
        (-0.015, -0.02),
        (0.015, -0.02),
    )
    """Full energy [eV] (published: 55 keV in OP1.2b)."""
    """Hydrogen (published for OP1.2b; deuterium planned)."""
    """Neutral power per source [W] (published: about 1.8 MW in hydrogen)."""
    """Power shares of the full, half and third energy (stand-in)."""
    """Source half-width [m] (stand-in)."""
    """Core divergence [rad] (stand-in, about 0.7 degrees)."""
    """Box names (published)."""
    r"""Aim point of each box axis, $(R, \varphi, Z)$ [m, rad, m] (stand-in)."""
    r"""Toroidal sense of each box, $+1$ along $+\varphi$ (stand-in)."""
    """Tangency radius of the box axes [m] (stand-in)."""
    """From the crossover point to the aim point [m] (stand-in)."""
    """From each source to the crossover point [m] (stand-in)."""
    r"""$(\alpha, \beta)$ of the four sources from the box axis [rad] (stand-in)."""


def w7x_injectors(
    parameters: W7XInjectors | None = None,
    *,
    sources: dict[str, Sequence[int]] | None = None,
) -> tuple[InjectorBox, ...]:
    """Return W7-X-like injector boxes; see `W7XInjectors` for what is a stand-in.

    Args:
        parameters: The parameter set; `W7XInjectors()` if None.
        sources: Which sources (from 1) of which box, e.g. ``{"NI21": [1, 2]}``
            for the two of OP1.2b; all of all boxes if None.
    """
    p = W7XInjectors() if parameters is None else parameters
    boxes = []
    for name, (r, phi, z), sense in zip(p.boxes, p.targets, p.senses, strict=True):
        if sources is not None and name not in sources:
            continue
        target = np.array([r * np.cos(phi), r * np.sin(phi), z])
        axis = aim_direction(target, p.tangency_radius, sense=sense)
        box = InjectorBox.converging(
            name,
            target - p.crossover_distance * axis,
            axis,
            p.source_angles,
            p.source_distance,
            energy=p.energy,
            power=p.source_power,
            power_fractions=p.power_fractions,
            mass_number=p.mass_number,
            width=p.width,
            divergence=p.divergence,
        )
        if sources is not None:
            chosen = tuple(box.sources[i - 1] for i in sources[name])
            box = InjectorBox(name, chosen)
        boxes.append(box)
    return tuple(boxes)


@dataclass
class SourceDepositions:
    """The deposition of each energy component of each source."""

    depositions: dict[tuple[str, str], Deposition] = field(default_factory=dict)
    """By ``(source name, component)``."""

    @property
    def sources(self) -> list[str]:
        """Return the source names, in order."""
        return list(dict.fromkeys(source for source, _ in self.depositions))

    def total(self) -> Deposition:
        """Return all of them as one `Deposition`, e.g. for `freeofcharge.slow_down`."""
        return Deposition.combine(list(self.depositions.values()))

    def to_dataset(self, surfaces: FluxSurfaces | None = None) -> xr.Dataset:
        """Return profiles and powers over ``(source, component[, rho])``.

        Missing components (a zero power share) are NaN. The wall-hit power is
        NaN without a wall.
        """
        sources = self.sources
        first = next(iter(self.depositions.values()))
        n_shells = len(first.rho_edges) - 1
        shape = (len(sources), len(COMPONENTS))
        profiles = {
            name: np.full((*shape, n_shells), np.nan)
            for name in ("birth_rate", "power")
        }
        scalars = {
            name: np.full(shape, np.nan)
            for name in (
                "injected_power",
                "absorbed_power",
                "shine_through_power",
                "shine_through_fraction",
                "wall_hit_power",
            )
        }
        for (source, component), d in self.depositions.items():
            index = (sources.index(source), COMPONENTS.index(component))
            profiles["birth_rate"][index] = d.birth_rate
            profiles["power"][index] = d.power
            scalars["injected_power"][index] = d.injected_power
            scalars["absorbed_power"][index] = d.absorbed_power
            scalars["shine_through_power"][index] = d.shine_through_power
            scalars["shine_through_fraction"][index] = d.shine_through_fraction
            if d.wall_hits is not None:
                hits = d.wall_hits
                speed2 = xp.sum(hits.velocities**2, axis=1)
                energy = 0.5 * d.mass_number * ATOMIC_MASS * speed2
                scalars["wall_hit_power"][index] = float(xp.sum(hits.weights * energy))
        dims = ("source", "component")
        units = {
            "injected_power": ("injected power", "W"),
            "absorbed_power": ("absorbed power", "W"),
            "shine_through_power": ("shine-through power", "W"),
            "shine_through_fraction": ("shine-through fraction", ""),
            "wall_hit_power": ("power onto the wall (this rank's hits)", "W"),
        }
        variables: dict[str, tuple] = {
            "birth_rate": (
                (*dims, "rho"),
                profiles["birth_rate"],
                {"long_name": "birth rate", "units": "1/s"},
            ),
            "power": (
                (*dims, "rho"),
                profiles["power"],
                {"long_name": "deposited power", "units": "W"},
            ),
        }
        for name, values in scalars.items():
            long_name, unit = units[name]
            variables[name] = (dims, values, {"long_name": long_name, "units": unit})
        if surfaces is not None:
            variables["power_density"] = (
                (*dims, "rho"),
                profiles["power"] / surfaces.volume,
                {"long_name": "deposited power density", "units": "W/m^3"},
            )
        rho = 0.5 * (first.rho_edges[1:] + first.rho_edges[:-1])
        return xr.Dataset(
            variables,
            coords={
                "source": sources,
                "component": list(COMPONENTS),
                "rho": ("rho", rho, rho_coordinate(first.radial_coordinate)),
            },
            attrs={"radial_coordinate": first.radial_coordinate},
        )


def _flatten(sources: Iterable[BeamSource | InjectorBox]) -> list[BeamSource]:
    """Return the sources of boxes and single sources, in order."""
    flat: list[BeamSource] = []
    for item in sources:
        flat.extend(item.sources if isinstance(item, InjectorBox) else (item,))
    names = [source.name for source in flat]
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique: {names}")
    return flat


def deposit_sources(
    sources: Iterable[BeamSource | InjectorBox],
    equilibrium: MagneticEquilibrium,
    plasma: Plasma,
    *,
    markers: int = 1000,
    seed: int | None = None,
    comm: MPI.Comm | None = None,
    **kwargs,
) -> SourceDepositions:
    """Deposit every energy component of every source.

    Args:
        sources: Sources and boxes of sources.
        equilibrium: Equilibrium.
        plasma: Profiles.
        markers: Markers per component and source.
        seed: Seed; each source and component gets its own stream from it.
        comm: Communicator over which the markers are split.
        **kwargs: Passed to `freeofcharge.deposit` (``step``, ``rho_edges``,
            ``births_per_marker``, ``stopping``, ...).
    """
    result = SourceDepositions()
    for i, source in enumerate(_flatten(sources)):
        beams = source.beams(
            markers, seed=None if seed is None else seed + 10 * i, comm=comm
        )
        for j, (component, beam) in enumerate(beams.items()):
            result.depositions[(source.name, component)] = deposit(
                beam,
                equilibrium,
                plasma,
                seed=None if seed is None else seed + 10 * i + 5 + j,
                comm=comm,
                **kwargs,
            )
    return result
