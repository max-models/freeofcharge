r"""Deposition of a neutral beam in the plasma: births and shine-through.

Every marker flies on a straight line from its position. Along it the line is
cut into steps of length ``ds``; at each step's midpoint inside the last closed
flux surface the optical depth grows by ``n_e sigma_s ds``, so the stream left
after step ``k`` is ``w exp(-tau_k)`` and step ``k`` ionises
``w (exp(-tau_{k-1}) - exp(-tau_k))`` atoms per second. Those are tallied on
shells of ``rho`` (the birth profile, without sampling noise). Births are then
sampled: for each of ``births_per_marker`` draws, the depth ``tau*`` where the
cumulative ionised share reaches a uniform random fraction, the birth point by
linear interpolation of ``tau`` within its step, and the weight ``w (1 -
exp(-tau_total)) / births_per_marker``. A fast ion starts with the atom's
velocity.

With a first wall, a line enters the vessel at the first step inside the wall
and ends at the first step after that outside it: the shine-through hits the
wall there (to within ``ds``). Without a wall the line runs to its end.

Charge exchange is part of ``sigma_s``: an ion that becomes a fast neutral
again counts as stopped, not followed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cunumpy as xp
import numpy as np
import xarray as xr
from maybempi import MPI
from plasmacoll import ParticleArrays, make_rng
from plasmacoll.constants import ELEMENTARY_CHARGE

from freeofcharge.beam import NeutralBeam
from freeofcharge.equilibrium import Equilibrium, FluxSurfaces
from freeofcharge.profiles import Plasma
from freeofcharge.stopping import StoppingModel, SuzukiStopping

__all__ = ["Deposition", "deposit"]


@dataclass
class Deposition:
    """The fast ions born in the plasma and the beam that shines through.

    The profiles and powers are sums over all ranks; the births and wall hits
    are this rank's.
    """

    births: ParticleArrays
    birth_rho: object
    birth_pitch: object
    birth_field: object
    mass_number: int
    rho_edges: np.ndarray
    birth_rate: np.ndarray
    power: np.ndarray
    injected_power: float
    absorbed_power: float
    shine_through_power: float
    wall_hits: ParticleArrays | None
    """Birth positions ``(N, 3)``, velocities and weights [ions/s]."""
    """``rho`` of each birth."""
    """``v_par / v`` of each birth (positive along B)."""
    """``|B|`` at each birth [T]."""
    """Ions born per second in each shell."""
    """Beam power deposited in each shell [W]."""
    """Where the shine-through meets the wall: positions, velocities, weights
    [atoms/s]; None without a wall."""

    @property
    def shine_through_fraction(self) -> float:
        """Return the share of the injected power that is not absorbed."""
        return self.shine_through_power / self.injected_power

    def power_density(self, surfaces: FluxSurfaces) -> np.ndarray:
        """Return the deposited power per volume of each shell [W/m^3]."""
        return self.power / surfaces.volume

    def to_dataset(self, surfaces: FluxSurfaces | None = None) -> xr.Dataset:
        """Return the profiles on the shell centres, with densities if ``surfaces``."""
        rho = 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])
        variables = {
            "birth_rate": (
                "rho",
                self.birth_rate,
                {"long_name": "birth rate", "units": "1/s"},
            ),
            "power": (
                "rho",
                self.power,
                {"long_name": "deposited power", "units": "W"},
            ),
        }
        if surfaces is not None:
            variables["power_density"] = (
                "rho",
                self.power_density(surfaces),
                {"long_name": "deposited power density", "units": "W/m^3"},
            )
            variables["birth_density"] = (
                "rho",
                self.birth_rate / surfaces.volume,
                {"long_name": "birth rate density", "units": "1/(m^3 s)"},
            )
        return xr.Dataset(
            variables,
            coords={"rho": ("rho", rho, {"long_name": "rho_pol"})},
            attrs={
                "injected_power": self.injected_power,
                "absorbed_power": self.absorbed_power,
                "shine_through_power": self.shine_through_power,
                "shine_through_fraction": self.shine_through_fraction,
            },
        )


def allreduce(values, comm: MPI.Comm | None):
    """Sum an array over the ranks of ``comm`` in place and return it."""
    if comm is not None and comm.Get_size() > 1:
        with xp.mpi.mpi_buffer(values, send=True, recv=True) as buffer:
            comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
    return values


def _path_length(beam: NeutralBeam, equilibrium: Equilibrium) -> float:
    """Return a length every line needs to cross the whole equilibrium grid."""
    positions = xp.to_numpy(beam.markers.positions)
    r_start = float(np.hypot(positions[:, 0], positions[:, 1]).max(initial=0.0))
    z_start = float(np.abs(positions[:, 2]).max(initial=0.0))
    lower, upper = equilibrium.grid.lower_bound, equilibrium.grid.upper_bound
    radius = max(r_start, upper[0])
    return 2.0 * radius + (upper[1] - lower[1]) + z_start


def deposit(
    beam: NeutralBeam,
    equilibrium: Equilibrium,
    plasma: Plasma,
    stopping: StoppingModel | None = None,
    *,
    step: float = 0.01,
    rho_edges=None,
    births_per_marker: int = 1,
    max_length: float | None = None,
    seed: int | None = None,
    comm: MPI.Comm | None = None,
    chunk_points: int = 2_000_000,
) -> Deposition:
    """Return the deposition of ``beam`` in the plasma.

    Args:
        beam: This rank's markers.
        equilibrium: Flux, boundary and (optional) wall.
        plasma: Profiles.
        stopping: Stopping model; :class:`SuzukiStopping` if None.
        step: Step ``ds`` along the lines [m].
        rho_edges: Shells of the profiles; 50 equal ones if None.
        births_per_marker: Births sampled per marker (0 for profiles only).
        max_length: Length of the lines [m]; long enough to cross the grid if None.
        seed: Seed of the birth sampling (per rank).
        comm: Communicator over which the markers are split; serial if None.
        chunk_points: Line points held at once (memory).
    """
    stopping = SuzukiStopping() if stopping is None else stopping
    rho_edges = (
        np.linspace(0.0, 1.0, 51) if rho_edges is None else np.asarray(rho_edges, float)
    )
    n_shells = len(rho_edges) - 1
    shells = FluxSurfaces(rho_edges, np.zeros(0), np.zeros(0), np.zeros(0))
    length = _path_length(beam, equilibrium) if max_length is None else max_length
    n_steps = int(np.ceil(length / step))
    chunk = max(1, chunk_points // n_steps)
    rank = 0 if comm is None else comm.Get_rank()
    rng = make_rng(seed, rank)

    energies = beam.energies
    energy_per_amu = energies / beam.mass_number
    directions = beam.directions
    s_mid = (xp.arange(n_steps) + 0.5) * step

    tallies = xp.zeros((n_shells, 2))
    totals = xp.zeros(2)  # absorbed and shine-through power [W]
    birth_parts: list[tuple] = []
    hit_parts: list[tuple] = []

    for start in range(0, beam.size, chunk):
        sl = slice(start, min(start + chunk, beam.size))
        origin = beam.markers.positions[sl]
        direction = directions[sl]
        weight = beam.markers.weights[sl]
        points = origin[:, None, :] + s_mid[None, :, None] * direction[:, None, :]
        r = xp.hypot(points[..., 0], points[..., 1])
        z = points[..., 2]

        if equilibrium.wall is not None:
            in_wall = equilibrium.inside_wall(r, z)
            entered = xp.cumsum(in_wall, axis=1) > 0
            leaving = entered & ~in_wall
            alive = xp.cumsum(leaving, axis=1) == 0
            has_hit = xp.any(leaving, axis=1)
            hit_step = xp.argmax(leaving, axis=1)
        else:
            alive = xp.ones(r.shape, dtype=bool)
            has_hit = xp.zeros(r.shape[0], dtype=bool)
            hit_step = xp.zeros(r.shape[0], dtype=int)

        in_plasma = equilibrium.inside(r, z) & alive
        row, col = xp.nonzero(in_plasma)
        rho = equilibrium.rho(r[row, col], z[row, col])
        sigma = stopping(
            energy_per_amu[sl][row],
            plasma.electron_density(rho),
            plasma.electron_temperature(rho),
            plasma.ion_densities(rho),
            plasma.ion_mass_numbers,
            plasma.ion_charges,
        )
        dtau = xp.zeros(r.shape)
        dtau[row, col] = plasma.electron_density(rho) * sigma * step
        tau = xp.cumsum(dtau, axis=1)
        tau_before = tau - dtau
        absorbed = weight[:, None] * (xp.exp(-tau_before) - xp.exp(-tau))
        shell = xp.clip(
            shells.shell_index(xp.minimum(rho, 0.999999 * rho_edges[-1])),
            0,
            n_shells - 1,
        )
        rate = absorbed[row, col]
        tallies += xp.algorithms.segment_sum(
            xp.stack([rate, rate * energies[sl][row] * ELEMENTARY_CHARGE], axis=-1),
            shell,
            n_shells,
        )
        tau_total = tau[:, -1]
        stopped = 1.0 - xp.exp(-tau_total)
        totals += (
            xp.stack(
                [
                    xp.sum(weight * stopped * energies[sl]),
                    xp.sum(weight * (1.0 - stopped) * energies[sl]),
                ]
            )
            * ELEMENTARY_CHARGE
        )

        if equilibrium.wall is not None:
            hits = xp.nonzero(has_hit)[0]
            hit_parts.append(
                (
                    points[hits, hit_step[hits]],
                    beam.markers.velocities[sl][hits],
                    (weight * (1.0 - stopped))[hits],
                )
            )

        absorbing = xp.nonzero(tau_total > 0.0)[0]
        if births_per_marker > 0 and int(absorbing.shape[0]) > 0:
            tau_a, dtau_a, before_a = (
                tau[absorbing],
                dtau[absorbing],
                tau_before[absorbing],
            )
            for _ in range(births_per_marker):
                u = xp.asarray(rng.random(int(absorbing.shape[0])))
                target = -xp.log1p(-u * stopped[absorbing])
                k = xp.argmax(tau_a >= target[:, None], axis=1)
                rows = xp.arange(int(absorbing.shape[0]))
                fraction = (target - before_a[rows, k]) / dtau_a[rows, k]
                distance = (k + fraction) * step
                birth_parts.append(
                    (
                        origin[absorbing] + distance[:, None] * direction[absorbing],
                        beam.markers.velocities[sl][absorbing],
                        weight[absorbing] * stopped[absorbing] / births_per_marker,
                    )
                )

    allreduce(tallies, comm)
    allreduce(totals, comm)
    tallies, totals = xp.to_numpy(tallies), xp.to_numpy(totals)

    def join(parts: list[tuple]) -> ParticleArrays:
        if not parts:
            return ParticleArrays(xp.zeros((0, 3)), xp.zeros((0, 3)), xp.zeros(0))
        return ParticleArrays(
            *(xp.concatenate([p[i] for p in parts]) for i in range(3))
        )

    births = join(birth_parts)
    positions = births.positions
    birth_r = xp.hypot(positions[:, 0], positions[:, 1])
    birth_rho = equilibrium.rho(birth_r, positions[:, 2])
    if births.weights.shape[0] > 0:
        field = equilibrium.magnetic_field_cartesian(positions)
        speed = xp.sqrt(xp.sum(births.velocities**2, axis=1))
        b_norm = xp.sqrt(xp.sum(field**2, axis=1))
        birth_pitch = xp.sum(births.velocities * field, axis=1) / (speed * b_norm)
    else:
        birth_pitch = xp.zeros(0)
        b_norm = xp.zeros(0)

    return Deposition(
        births=births,
        birth_rho=birth_rho,
        birth_pitch=birth_pitch,
        birth_field=b_norm,
        mass_number=beam.mass_number,
        rho_edges=rho_edges,
        birth_rate=tallies[:, 0],
        power=tallies[:, 1],
        injected_power=beam.power(comm),
        absorbed_power=float(totals[0]),
        shine_through_power=float(totals[1]),
        wall_hits=None if equilibrium.wall is None else join(hit_parts),
    )
