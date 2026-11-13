r"""Deposition of a neutral beam in the plasma: births and shine-through.

Every marker flies on a straight line from its position. Along it the line is
cut into steps of length $\Delta s$; at each step's midpoint inside the last closed
flux surface the optical depth grows by $n_e \sigma_s \Delta s$, so the stream left
after step $k$ is $w\,e^{-\tau_k}$ and step $k$ ionises
$w\,(e^{-\tau_{k-1}} - e^{-\tau_k})$ atoms per second. Those are tallied on
shells of $\rho$ (the birth profile, without sampling noise). Births are then
sampled: for each of ``births_per_marker`` draws, the depth $\tau^*$ where the
cumulative ionised share reaches a uniform random fraction, the birth point by
linear interpolation of $\tau$ within its step, and the weight
$w\,(1 - e^{-\tau_\mathrm{total}})$ / ``births_per_marker``. A fast ion starts with the atom's
velocity.

With a first wall, a line enters the vessel at the first step inside the wall
and ends at the first step after that outside it: the shine-through hits the
wall there (to within $\Delta s$). Without a wall the line runs to its end.

Charge exchange is part of $\sigma_s$: an ion that becomes a fast neutral
again counts as stopped, not followed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cunumpy as xp
import numpy as np
import xarray as xr
from maybempi import MPI
from methodcard import method
from plasmacoll import ParticleArrays, make_rng
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.units import to_unit

from freeofcharge.beam import NeutralBeam
from freeofcharge.equilibrium import (
    FluxSurfaces,
    MagneticEquilibrium,
    radial_coordinate_of,
    rho_coordinate,
)
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
    radial_coordinate: str = "poloidal"
    transmission: object = None
    """Birth positions ``(N, 3)``, velocities and weights [ions/s]."""
    r"""$\rho$ of each birth."""
    r"""$v_\parallel / v$ of each birth (positive along $\mathbf{B}$)."""
    """$|B|$ at each birth [T]."""
    """Ions born per second in each shell."""
    """Beam power deposited in each shell [W]."""
    """Where the shine-through meets the wall: positions, velocities, weights
    [atoms/s]; None without a wall."""
    r"""The flux $\rho$ is the root of, the equilibrium's (`freeofcharge.rho_coordinate`)."""
    r"""$e^{-\tau_\mathrm{total}}$ of each of this rank's markers, in the beam's order:
    the share of its stream that shines through (`freeofcharge.shine_through`)."""

    @classmethod
    def combine(cls, depositions: Sequence[Deposition]) -> Deposition:
        """Return the sum of depositions on the same shells, of one isotope.

        For example the energy components and sources of an injector, to be
        slowed down together: profiles and powers add up, births and wall hits
        are joined.
        """
        if not depositions:
            raise ValueError("nothing to combine")
        first = depositions[0]
        for other in depositions[1:]:
            if not np.array_equal(other.rho_edges, first.rho_edges):
                raise ValueError("the depositions have different shells")
            if other.mass_number != first.mass_number:
                raise ValueError("the depositions are of different isotopes")
            if other.radial_coordinate != first.radial_coordinate:
                raise ValueError("the depositions have different radial coordinates")

        def join(parts: list[ParticleArrays]) -> ParticleArrays:
            return ParticleArrays(
                xp.concatenate([p.positions for p in parts]),
                xp.concatenate([p.velocities for p in parts]),
                xp.concatenate([p.weights for p in parts]),
            )

        hits = [d.wall_hits for d in depositions if d.wall_hits is not None]
        transmissions = [d.transmission for d in depositions]
        return cls(
            births=join([d.births for d in depositions]),
            birth_rho=xp.concatenate([xp.asarray(d.birth_rho) for d in depositions]),
            birth_pitch=xp.concatenate(
                [xp.asarray(d.birth_pitch) for d in depositions]
            ),
            birth_field=xp.concatenate(
                [xp.asarray(d.birth_field) for d in depositions]
            ),
            mass_number=first.mass_number,
            rho_edges=first.rho_edges,
            birth_rate=np.sum([d.birth_rate for d in depositions], axis=0),
            power=np.sum([d.power for d in depositions], axis=0),
            injected_power=sum(d.injected_power for d in depositions),
            absorbed_power=sum(d.absorbed_power for d in depositions),
            shine_through_power=sum(d.shine_through_power for d in depositions),
            wall_hits=join(hits) if hits else None,
            radial_coordinate=first.radial_coordinate,
            transmission=None
            if any(t is None for t in transmissions)
            else xp.concatenate([xp.asarray(t) for t in transmissions]),
        )

    @property
    def shine_through_fraction(self) -> float:
        """Return the share of the injected power that is not absorbed."""
        return self.shine_through_power / self.injected_power

    def power_density(self, surfaces: FluxSurfaces) -> np.ndarray:
        r"""Return the deposited power per volume of each shell [$\mathrm{W/m^3}$]."""
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
            coords={"rho": ("rho", rho, rho_coordinate(self.radial_coordinate))},
            attrs={
                "radial_coordinate": self.radial_coordinate,
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


def _path_length(beam: NeutralBeam, equilibrium: MagneticEquilibrium) -> float:
    """Return a length every line needs to cross the whole equilibrium box."""
    positions = xp.to_numpy(beam.markers.positions)
    lower, upper = equilibrium.bounds
    if positions.shape[0] > 0:
        lower = np.minimum(lower, positions.min(axis=0))
        upper = np.maximum(upper, positions.max(axis=0))
    return float(np.linalg.norm(upper - lower))


def _step_window(positions, directions, equilibrium, step: float, n_steps: int):
    r"""Return per line the first and the end step that can be in the plasma.

    The steps whose midpoints lie in the cylinder $R \le R_\max$,
    $Z_\min \le Z \le Z_\max$ around the equilibrium's box, padded by 2 % of its
    size and one step; a line that misses it gets an empty window.
    """
    lower, upper = (np.asarray(b, dtype=float) for b in equilibrium.bounds)
    pad = 0.02 * float(np.max(upper - lower)) + step
    r_max = float(max(np.abs(lower[:2]).max(), np.abs(upper[:2]).max())) + pad
    z_lo, z_hi = lower[2] - pad, upper[2] + pad
    p, d = xp.asarray(positions), xp.asarray(directions)
    # R(s)^2 <= r_max^2: a s^2 + b s + c <= 0
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1])
    c = p[:, 0] ** 2 + p[:, 1] ** 2 - r_max**2
    root = xp.sqrt(xp.maximum(b**2 - 4.0 * a * c, 0.0))
    vertical = a < 1.0e-12
    safe_a = xp.where(vertical, 1.0, a)
    s_in = xp.where(
        vertical, xp.where(c <= 0.0, -xp.inf, xp.inf), (-b - root) / (2.0 * safe_a)
    )
    s_out = xp.where(
        vertical, xp.where(c <= 0.0, xp.inf, -xp.inf), (-b + root) / (2.0 * safe_a)
    )
    s_out = xp.where(b**2 - 4.0 * a * c < 0.0, -xp.inf, s_out)
    # Z_min <= Z(s) <= Z_max
    flat = xp.abs(d[:, 2]) < 1.0e-12
    safe_dz = xp.where(flat, 1.0, d[:, 2])
    t1, t2 = (z_lo - p[:, 2]) / safe_dz, (z_hi - p[:, 2]) / safe_dz
    within = (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi)
    z_in = xp.where(flat, xp.where(within, -xp.inf, xp.inf), xp.minimum(t1, t2))
    z_out = xp.where(flat, xp.where(within, xp.inf, -xp.inf), xp.maximum(t1, t2))
    s_in, s_out = xp.maximum(s_in, z_in), xp.minimum(s_out, z_out)
    first = xp.clip(
        xp.floor(xp.where(xp.isfinite(s_in), s_in, -1.0) / step), 0, n_steps
    )
    last = xp.clip(
        xp.ceil(xp.where(xp.isfinite(s_out), s_out, n_steps * step) / step), 0, n_steps
    )
    empty = s_out <= s_in
    first = xp.where(empty, n_steps, first).astype(int)
    last = xp.where(empty, 0, last).astype(int)
    return first, last


@method(
    equations={
        "attenuation": r"\frac{dI}{ds} = -n_e\,\sigma_s\, I \quad\Rightarrow\quad I = w\, e^{-\tau}, \qquad \tau(s) = \int_0^s n_e \sigma_s\, ds'",
        "optical_depth": r"\tau_k = \sum_{i \le k} n_e(s_i) \sigma_s(s_i)\, \Delta s, \qquad s_i = \left(i + \tfrac{1}{2}\right) \Delta s",
        "births": r"\dot N_k = w \left( e^{-\tau_{k-1}} - e^{-\tau_k} \right)",
        "powers": r"P_\mathrm{abs} = w \left(1 - e^{-\tau_\mathrm{total}}\right) E, \qquad P_\mathrm{shine} = w\, e^{-\tau_\mathrm{total}} E",
        "sampling": r"\tau^* = -\ln\!\left(1 - u\,(1 - e^{-\tau_\mathrm{total}})\right), \qquad w_\mathrm{birth} = \frac{w (1 - e^{-\tau_\mathrm{total}})}{n_b}",
        "pitch": r"\xi = \frac{\mathbf{v}\cdot\mathbf{B}}{v B}",
    },
    assumptions=(
        r"straight lines from the markers; midpoints outside the last closed surface do not attenuate",
        r"charge exchange counts as stopped (part of $\sigma_s$)",
        r"the wall crossing and the boundary are resolved to one step; a birth may lie half a step outside",
        r"$u$ uniform; the birth point by linear interpolation of $\tau$ within its step",
    ),
    components=(SuzukiStopping,),
)
def deposit(
    beam: NeutralBeam,
    equilibrium: MagneticEquilibrium,
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
    r"""Return the deposition of ``beam`` in the plasma.

    Args:
        beam: This rank's markers.
        equilibrium: Flux, boundary and (optional) wall.
        plasma: Profiles.
        stopping: Stopping model; `SuzukiStopping` if None.
        step: Step $\Delta s$ along the lines [m]; a quantity such as ``"2 cm"``.
        rho_edges: Shells of the profiles; 50 equal ones if None.
        births_per_marker: Births sampled per marker (0 for profiles only).
        max_length: Length of the lines [m] (or a quantity); long enough to cross
            the grid if None.
        seed: Seed of the birth sampling (per rank).
        comm: Communicator over which the markers are split; serial if None.
        chunk_points: Line points held at once (memory).
    """
    radial_coordinate = radial_coordinate_of(equilibrium)
    plasma.check_radial_coordinate(radial_coordinate)
    step = to_unit(step, "m", "step")
    max_length = to_unit(max_length, "m", "max_length")
    stopping = SuzukiStopping() if stopping is None else stopping
    rho_edges = (
        np.linspace(0.0, 1.0, 51) if rho_edges is None else np.asarray(rho_edges, float)
    )
    n_shells = len(rho_edges) - 1
    shells = FluxSurfaces(rho_edges, np.zeros(0), np.zeros(0), np.zeros(0))
    length = _path_length(beam, equilibrium) if max_length is None else max_length
    n_steps = int(np.ceil(length / step))
    directions = beam.directions
    # without a wall only the steps near the plasma matter: a window per line
    window = (
        None
        if equilibrium.has_wall
        else _step_window(
            beam.markers.positions, directions, equilibrium, step, n_steps
        )
    )
    longest = n_steps if window is None else max(1, int(xp.max(window[1] - window[0])))
    chunk = max(1, chunk_points // longest)
    rank = 0 if comm is None else comm.Get_rank()
    rng = make_rng(seed, rank)

    energies = beam.energies
    energy_per_amu = energies / beam.mass_number

    tallies = xp.zeros((n_shells, 2))
    totals = xp.zeros(2)  # absorbed and shine-through power [W]
    birth_parts: list[tuple] = []
    hit_parts: list[tuple] = []
    transmission_parts: list = []

    for start in range(0, beam.size, chunk):
        sl = slice(start, min(start + chunk, beam.size))
        origin = beam.markers.positions[sl]
        direction = directions[sl]
        weight = beam.markers.weights[sl]
        if window is None:
            first, last = 0, n_steps
        else:
            first = int(xp.min(window[0][sl]))
            last = max(int(xp.max(window[1][sl])), first + 1)
        s_mid = (xp.arange(first, last) + 0.5) * step
        points = origin[:, None, :] + s_mid[None, :, None] * direction[:, None, :]
        shape = points.shape[:2]

        if equilibrium.has_wall:
            in_wall = equilibrium.inside_wall_at(points)
            entered = xp.cumsum(in_wall, axis=1) > 0
            leaving = entered & ~in_wall
            alive = xp.cumsum(leaving, axis=1) == 0
            has_hit = xp.any(leaving, axis=1)
            hit_step = xp.argmax(leaving, axis=1)
        else:
            alive = xp.ones(shape, dtype=bool)
            has_hit = xp.zeros(shape[0], dtype=bool)
            hit_step = xp.zeros(shape[0], dtype=int)

        both = getattr(equilibrium, "rho_and_inside_at", None)
        if both is None:
            row, col = xp.nonzero(equilibrium.inside_at(points) & alive)
            rho = equilibrium.rho_at(points[row, col])
        else:
            # one evaluation for both where the equilibrium offers it
            rho_all, inside = both(points)
            row, col = xp.nonzero(inside & alive)
            rho = rho_all[row, col]
        ne = plasma.electron_density(rho)
        sigma = stopping(
            energy_per_amu[sl][row],
            ne,
            plasma.electron_temperature(rho),
            plasma.ion_densities(rho),
            plasma.ion_mass_numbers,
            plasma.ion_charges,
        )
        dtau = xp.zeros(shape)
        dtau[row, col] = ne * sigma * step
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
        transmission_parts.append(1.0 - stopped)
        totals += (
            xp.stack(
                [
                    xp.sum(weight * stopped * energies[sl]),
                    xp.sum(weight * (1.0 - stopped) * energies[sl]),
                ]
            )
            * ELEMENTARY_CHARGE
        )

        if equilibrium.has_wall:
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
                distance = (first + k + fraction) * step
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
    birth_rho = equilibrium.rho_at(positions)
    if births.weights.shape[0] > 0:
        field = equilibrium.field_at(positions)
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
        wall_hits=join(hit_parts) if equilibrium.has_wall else None,
        radial_coordinate=radial_coordinate,
        transmission=xp.concatenate(transmission_parts)
        if transmission_parts
        else xp.zeros(0),
    )
