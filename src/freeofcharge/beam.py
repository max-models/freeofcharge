"""Neutral beam markers: fast hydrogen atoms on straight lines.

A marker is a stream of atoms: a position and a velocity (Cartesian, ``z``
vertical) and a weight in atoms per second, held in a
`plasmacoll.ParticleArrays`. With MPI every rank holds its own share of
the markers. `NeutralBeam.from_dataset` takes the markers a beamline
model hands on, e.g. beamtracer's at the port.

The physical settings of `tangential_injection` and `NeutralBeam.gaussian`
take astropy Quantities (``60 * u.keV``), strings such as ``"60 keV"`` or
plain numbers in SI units, eV for the energy and rad for the angles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cunumpy as xp
import numpy as np
from maybempi import MPI
from methodcard import method
from plasmacoll import ParticleArrays, make_rng
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import to_array, to_unit

__all__ = ["NeutralBeam", "tangential_injection"]


@method(
    equations={
        "start": r"\mathbf{x}_\mathrm{start} = \mathbf{x}_\mathrm{tan} - \sqrt{R_\mathrm{start}^2 - R_\mathrm{tan}^2}\;\hat d",
    },
    assumptions=(
        r"a horizontal axis tangent to $R = R_\mathrm{tan}$, starting at $R = R_\mathrm{start} > R_\mathrm{tan}$",
    ),
)
def tangential_injection(
    tangency_radius: float,
    start_radius: float,
    *,
    height: float = 0.0,
    toroidal_angle: float = 0.0,
    sense: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the start point [m] and direction of a horizontal beam axis.

    The axis is tangent to the circle ``R = tangency_radius`` [m] at the
    toroidal angle ``toroidal_angle`` [rad] and starts at ``R = start_radius``
    [m], at ``Z = height`` [m]; ``sense = +1`` points along ``+phi`` at the
    tangency point. The lengths and the angle take quantities (``"5.3 m"``,
    ``30 * u.deg``).
    """
    tangency_radius = to_unit(tangency_radius, "m", "tangency_radius")
    start_radius = to_unit(start_radius, "m", "start_radius")
    height = to_unit(height, "m", "height")
    toroidal_angle = to_unit(toroidal_angle, "rad", "toroidal_angle")
    if start_radius <= tangency_radius:
        raise ValueError("the start radius must exceed the tangency radius")
    if sense not in (1, -1):
        raise ValueError("sense must be +1 or -1")
    cos_p, sin_p = np.cos(toroidal_angle), np.sin(toroidal_angle)
    tangent_point = np.array([tangency_radius * cos_p, tangency_radius * sin_p, height])
    direction = sense * np.array([-sin_p, cos_p, 0.0])
    length = np.sqrt(start_radius**2 - tangency_radius**2)
    return tangent_point - length * direction, direction


@dataclass
class NeutralBeam:
    """Neutral markers of one isotope (positions ``(N, 3)``, weights in atoms/s)."""

    markers: ParticleArrays
    mass_number: int
    """Atomic mass number of the beam atoms (1 H, 2 D, 3 T)."""

    @property
    def mass(self) -> float:
        """Return the atom mass [kg]."""
        return self.mass_number * ATOMIC_MASS

    @property
    def size(self) -> int:
        """Return the number of local markers."""
        return int(self.markers.weights.shape[0])

    @property
    def speeds(self):
        """Return the marker speeds [m/s]."""
        return xp.sqrt(xp.sum(self.markers.velocities**2, axis=1))

    @property
    def energies(self):
        """Return the marker energies [eV]."""
        return 0.5 * self.mass * self.speeds**2 / ELEMENTARY_CHARGE

    @property
    def directions(self):
        """Return the unit directions ``(N, 3)``."""
        return self.markers.velocities / self.speeds[:, None]

    def power(self, comm: MPI.Comm | None = None) -> float:
        """Return the beam power over all ranks [W]."""
        local = float(xp.sum(self.markers.weights * self.energies)) * ELEMENTARY_CHARGE
        return local if comm is None else comm.allreduce(local)

    @classmethod
    def concatenate(cls, beams: Sequence[NeutralBeam]) -> NeutralBeam:
        """Join beams of one isotope (e.g. energy components)."""
        mass_numbers = {beam.mass_number for beam in beams}
        if len(mass_numbers) != 1:
            raise ValueError(f"beams of different isotopes: {sorted(mass_numbers)}")
        return cls(
            ParticleArrays(
                xp.concatenate([b.markers.positions for b in beams]),
                xp.concatenate([b.markers.velocities for b in beams]),
                xp.concatenate([b.markers.weights for b in beams]),
            ),
            mass_numbers.pop(),
        )

    @classmethod
    @method(
        equations={
            "positions": r"\mathbf{x} = \mathbf{x}_0 + w_h N_1 \hat h + w_v N_2 \hat v",
            "directions": r"\hat d \propto \hat a + \tan\theta_h \hat h + \tan\theta_v \hat v, \qquad \hat h = \hat a\times\hat z, \quad \hat v = \hat h\times\hat a",
            "weight": r"w = \frac{P}{E e N}",
        },
        assumptions=(
            r"mono-energetic; $N_{1,2}$ standard normal, $\theta_{h,v}$ Gaussian with the core or (for `halo_fraction`) the halo divergence",
            r"the weights sum to the power $P$ over all ranks",
        ),
    )
    def gaussian(
        cls,
        origin,
        direction,
        energy: float,
        power: float,
        *,
        mass_number: int = 2,
        width: float | tuple[float, float] = 0.0,
        divergence: float | tuple[float, float] = 0.0,
        halo_fraction: float = 0.0,
        halo_divergence: float | tuple[float, float] = 0.0,
        markers: int = 10000,
        seed: int | None = None,
        comm: MPI.Comm | None = None,
    ) -> NeutralBeam:
        """Return a mono-energetic beam with Gaussian width and divergence.

        The marker positions spread about ``origin`` with standard deviation
        ``width`` [m] across the axis, and their directions with standard
        deviations ``divergence`` [rad] in the two planes containing the axis:
        each a scalar or ``(horizontal, vertical)``, horizontal being the
        plane of the axis and ``z``-normal. A share ``halo_fraction`` of the
        markers has the ``halo_divergence`` instead (core and halo, as BBNBI
        describes a beamlet). The markers are split between the ranks of
        ``comm``; the weights give ``power`` [W] over all of them.

        ``origin``, ``energy``, ``power``, ``width`` and the divergences take
        astropy Quantities or strings (``"1 MeV"``, ``"16.5 MW"``, ``"5 mrad"``)
        as well as numbers in the units below.

        Args:
            origin: Centre of the source plane [m].
            direction: Beam axis (any length).
            energy: Atom energy [eV].
            power: Beam power [W].
            mass_number: Isotope.
            width: Source half-width (standard deviation) [m].
            divergence: Core divergence (standard deviation) [rad].
            halo_fraction: Share of markers in the halo.
            halo_divergence: Halo divergence [rad].
            markers: Number of markers over all ranks.
            seed: Seed of the random streams (per rank).
            comm: Communicator; serial if None.
        """
        origin = to_array(origin, "m", "origin")
        energy = to_unit(energy, "eV", "energy")
        power = to_unit(power, "W", "power")
        width = to_unit(width, "m", "width")
        divergence = to_unit(divergence, "rad", "divergence")
        halo_divergence = to_unit(halo_divergence, "rad", "halo_divergence")
        rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
        local = markers // size + (1 if rank < markers % size else 0)
        rng = make_rng(seed, rank)

        axis = np.asarray(direction, dtype=float)
        axis = axis / np.linalg.norm(axis)
        horizontal = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(horizontal) < 1.0e-12:
            horizontal = np.array([1.0, 0.0, 0.0])
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(horizontal, axis)

        def pair(value: float | tuple[float, float]) -> tuple[float, float]:
            values = np.broadcast_to(np.asarray(value, dtype=float), (2,))
            return float(values[0]), float(values[1])

        width_h, width_v = pair(width)
        core_h, core_v = pair(divergence)
        halo_h, halo_v = pair(halo_divergence)
        normal = xp.asarray(rng.standard_normal((local, 4)))
        halo = xp.asarray(rng.random(local)) < halo_fraction
        angle_h = normal[:, 0] * xp.where(halo, halo_h, core_h)
        angle_v = normal[:, 1] * xp.where(halo, halo_v, core_v)

        axis_x = xp.asarray(axis)
        horizontal_x = xp.asarray(horizontal)
        vertical_x = xp.asarray(vertical)
        directions = (
            axis_x[None, :]
            + xp.tan(angle_h)[:, None] * horizontal_x[None, :]
            + xp.tan(angle_v)[:, None] * vertical_x[None, :]
        )
        directions /= xp.sqrt(xp.sum(directions**2, axis=1))[:, None]
        positions = (
            xp.asarray(np.asarray(origin, dtype=float))[None, :]
            + (width_h * normal[:, 2])[:, None] * horizontal_x[None, :]
            + (width_v * normal[:, 3])[:, None] * vertical_x[None, :]
        )
        speed = np.sqrt(2.0 * energy * ELEMENTARY_CHARGE / (mass_number * ATOMIC_MASS))
        rate = power / (energy * ELEMENTARY_CHARGE) / markers
        return cls(
            ParticleArrays(positions, speed * directions, xp.full(local, rate)),
            mass_number,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset,
        *,
        mass_number: int | None = None,
        comm: MPI.Comm | None = None,
    ) -> NeutralBeam:
        """Return the neutral markers of a dataset, e.g. a beamline's at its port.

        The dataset holds ``position`` and ``velocity`` over ``(component, marker)``
        and ``weight`` [atoms/s] over ``marker``, as beamtracer writes them
        (``beamtracer.beamline.Markers.to_dataset``); with a ``state``, only the
        neutrals (``"0"``) are taken. ``mass_number`` defaults to the ``mass``
        attribute in atomic masses. The markers are split between the ranks of
        ``comm``.
        """
        keep = np.ones(dataset.sizes["marker"], dtype=bool)
        if "state" in dataset:
            keep &= np.asarray(dataset["state"].values).astype(str) == "0"
        if mass_number is None:
            if "mass" not in dataset.attrs:
                raise ValueError("the dataset has no mass; give the mass_number")
            mass_number = round(float(dataset.attrs["mass"]) / ATOMIC_MASS)
        index = np.flatnonzero(keep)
        if comm is not None:
            index = index[comm.Get_rank() :: comm.Get_size()]
        positions = dataset["position"].transpose("marker", "component").values[index]
        velocities = dataset["velocity"].transpose("marker", "component").values[index]
        weights = dataset["weight"].values[index]
        return cls(
            ParticleArrays(
                xp.asarray(positions, dtype=float),
                xp.asarray(velocities, dtype=float),
                xp.asarray(weights, dtype=float),
            ),
            int(mass_number),
        )
