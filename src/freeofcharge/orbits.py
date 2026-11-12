r"""Guiding-centre orbits of the fast ions with Coulomb collisions: confinement, losses, wall loads.

Each fast ion is followed as a guiding centre $\mathbf{X}$ with parallel velocity
$v_\parallel$ and magnetic moment per unit mass $\mu = v_\perp^2 / (2B)$ in physical
(Cartesian) space,

$$
\dot{\mathbf{X}} = \frac{v_\parallel \mathbf{B}^* + \frac{\mu}{q/m}\,\hat b \times \nabla B}{B^*_\parallel},
\qquad
\dot v_\parallel = -\mu\,\frac{\mathbf{B}^* \cdot \nabla B}{B^*_\parallel},
\qquad
\mathbf{B}^* = \mathbf{B} + \frac{v_\parallel}{q/m}\,\nabla\times\hat b,
\quad B^*_\parallel = \hat b \cdot \mathbf{B}^*,
$$

(Littlejohn, J. Plasma Phys. 29, 111, 1983; Cary and Brizard, Rev. Mod. Phys. 81, 693,
2009): the parallel motion with the mirror force, the $\nabla B$ and curvature drifts
(in $\nabla\times\hat b$), $\mu$ an invariant and the energy
$\tfrac12 v_\parallel^2 + \mu B$ conserved by the flow. No electric field. The field
is the cubic spline interpolant of a `freeofcharge.GriddedEquilibrium`,
differentiated exactly, so $\nabla B$ and $\nabla\times\hat b$ are those of the
interpolated field; any other equilibrium, `freeofcharge.CoilField` (the coils' field
outside $\rho = 1$) included, is sampled on such a grid first (`orbit_field`). The
orbits are integrated by RK4 in equal steps of at most ``step`` metres along the
orbit, in a compiled kernel (``freeofcharge.kernels.guiding_centre_push``).

Physical coordinates are the simple first choice: no inverse map, no special
treatment of the magnetic axis. Most guiding-centre codes push in flux coordinates
instead (exact field derivatives from the map, no localisation); there the axis
needs a polar chart, which matters for central deposition. In physical space the
cost is the interpolation, and near the axis nothing changes.

Between pushes, Coulomb collisions with the thermal plasma act by Monte Carlo
(`collide`), with the drags and the deflection of `freeofcharge.slowing_down`
(`collision_coefficients`: the same Coulomb logarithms, $\tau_s$, critical speed):
the deterministic slowing down $dv/dt = -a_e v - a_i / v^2$ exactly over the step,
the energy diffusion that keeps a Maxwellian at $T_e$ and $T_i$ in equilibrium, and
Lorentz pitch-angle scattering (Boozer and Kuo-Petravic, Phys. Fluids 24, 851, 1981).
A marker is thermalised (removed, its energy to the thermal ions) below the end
energy of `freeofcharge.slow_down`, and lost when its guiding centre (optionally its
Larmor circle) crosses the wall, or leaves $\rho$ = ``rho_limit`` without a wall.

The moments are time integrals along the orbits, as in `slow_down`: markers are
streams [ions/s], so the time each one spends in a shell, times its weight, is the
steady-state number of fast ions there.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import to_unit

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.equilibrium import rho_coordinate
from freeofcharge.gridded_equilibrium import GriddedEquilibrium
from freeofcharge.kernels.guiding_centre_push import (
    guiding_centre_push_numpy as _numpy_push,
)
from freeofcharge.kernels.guiding_centre_push import kernel as _push
from freeofcharge.slowing_down import collision_coefficients

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces, MagneticEquilibrium
    from freeofcharge.profiles import Plasma

__all__ = [
    "FATES",
    "GuidingCentres",
    "OrbitFollowing",
    "collide",
    "follow_orbits",
    "guiding_centres",
    "orbit_field",
    "revolved_wall",
    "trace_orbits",
    "wall_clearance",
    "wall_distance",
]

_E = ELEMENTARY_CHARGE

_RECORD = 64
"""Steps a marker takes near the wall per call of the kernel, all tested."""
_CROSSED = 0.005
"""Interpolated distance beyond the wall [m] at which the kernel stops near it."""

#: What became of a marker: its ``fate`` code is the index.
FATES = ("alive", "thermalised", "wall", "rho_limit", "grid", "max_time")
_ALIVE, _THERMALISED, _WALL, _RHO_LIMIT, _GRID, _MAX_TIME = range(6)


def revolved_wall(outline, n_phi: int = 128):
    """Return a closed `simgeometry` triangle mesh: the polygon ``outline`` revolved about $z$.

    Args:
        outline: ``(n, 2)`` vertices $(R, Z)$ of a closed polygon [m], e.g. the
            wall of an `freeofcharge.Equilibrium` (``equilibrium.wall.vertices``).
        n_phi: Toroidal segments.
    """
    from simgeometry.objects.triangle_mesh import TriangleMesh

    outline = np.asarray(outline, dtype=float)
    if np.allclose(outline[0], outline[-1]):
        outline = outline[:-1]
    n = outline.shape[0]
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    r, z = outline[:, 0], outline[:, 1]
    vertices = np.stack(
        [
            (r[None, :] * np.cos(phi)[:, None]).ravel(),
            (r[None, :] * np.sin(phi)[:, None]).ravel(),
            np.broadcast_to(z, (n_phi, n)).ravel(),
        ],
        axis=-1,
    )
    j, i = np.meshgrid(np.arange(n_phi), np.arange(n), indexing="ij")
    j, i = j.ravel(), i.ravel()
    a = j * n + i
    b = j * n + (i + 1) % n
    c = ((j + 1) % n_phi) * n + (i + 1) % n
    d = ((j + 1) % n_phi) * n + i
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return TriangleMesh(vertices, faces, closed=True)


def orbit_field(
    equilibrium: MagneticEquilibrium,
    resolution: int | tuple[int, int, int] = (96, 16, 96),
    *,
    box=None,
    margin: float = 0.05,
    **kwargs,
) -> GriddedEquilibrium:
    r"""Return ``equilibrium`` sampled on a cubic-spline grid for the pusher.

    A `GriddedEquilibrium` with ``order`` 3 is returned as it is. Others are sampled
    over $(R, Z)$ = ``box`` (``((R_min, R_max), (Z_min, Z_max))`` [m], e.g. covering
    the wall), or the plasma's box when None; ``kwargs`` go to `GriddedEquilibrium`.
    An axisymmetric equilibrium needs few toroidal nodes (the cylindrical components
    do not depend on $\varphi$). A `freeofcharge.CoilField` gives the coils' field
    outside $\rho = 1$ (sampled with the compiled Biot–Savart kernel).
    """
    if isinstance(equilibrium, GriddedEquilibrium) and equilibrium.order == 3:
        return equilibrium
    if box is not None:
        (r_lo, r_hi), (z_lo, z_hi) = box
        kwargs.setdefault("r_range", (float(r_lo), float(r_hi)))
        kwargs.setdefault("z_range", (float(z_lo), float(z_hi)))
    return GriddedEquilibrium(equilibrium, resolution, margin=margin, order=3, **kwargs)


@dataclass
class GuidingCentres:
    r"""Guiding-centre markers: positions, parallel velocities, magnetic moments, weights.

    ``mu`` is per unit mass, $v_\perp^2 / (2B)$ [$\mathrm{m^2\,s^{-2}\,T^{-1}}$];
    ``weights`` [ions/s] (streams, as the births of `freeofcharge.deposit`).
    """

    positions: np.ndarray
    v_parallel: np.ndarray
    mu: np.ndarray
    weights: np.ndarray
    mass: float
    charge_number: int = 1
    """``(n, 3)`` [m]."""
    """[m/s], positive along $\\mathbf{B}$."""
    """[$\\mathrm{m^2\\,s^{-2}\\,T^{-1}}$]."""
    """[1/s]."""
    """[kg]."""

    @property
    def charge_over_mass(self) -> float:
        """Return $q/m$ [C/kg]."""
        return self.charge_number * _E / self.mass

    def energy(self, field: GriddedEquilibrium) -> np.ndarray:
        r"""Return $m(\tfrac12 v_\parallel^2 + \mu B)$ [J] at the guiding centres."""
        b = np.linalg.norm(xp.to_numpy(field.field_at(self.positions)), axis=-1)
        return self.mass * (0.5 * self.v_parallel**2 + self.mu * b)


@method(
    equations={
        "centre": r"\mathbf{X} = \mathbf{x} - \frac{\hat b \times \mathbf{v}}{\Omega}, \qquad \Omega = \frac{qB}{m}",
        "invariants": r"v_\parallel = \mathbf{v} \cdot \hat b, \qquad \mu = \frac{v^2 - v_\parallel^2}{2 B(\mathbf{X})}",
    },
    assumptions=(
        r"first order in the Larmor radius: $\hat b$ and $B$ at the particle, $\mu$ with $B$ at the guiding centre (the energy is kept exactly)",
    ),
)
def guiding_centres(
    positions,
    velocities,
    weights,
    field: GriddedEquilibrium,
    mass: float,
    charge_number: int = 1,
) -> GuidingCentres:
    r"""Return the guiding centres of particles at ``positions`` with ``velocities``.

    $\mathbf{X} = \mathbf{x} - \hat b \times \mathbf{v} / \Omega$, $v_\parallel$ the
    velocity along $\hat b$ at the particle, and $\mu$ such that the kinetic energy
    $\tfrac12 v_\parallel^2 + \mu B(\mathbf{X})$ is the particle's.

    Args:
        positions: ``(n, 3)`` [m].
        velocities: ``(n, 3)`` [m/s].
        weights: ``(n,)`` [1/s].
        field: The gridded field.
        mass: Ion mass [kg].
        charge_number: Ion charge number.
    """
    x = np.asarray(xp.to_numpy(xp.asarray(positions, dtype=float)), dtype=float)
    v = np.asarray(xp.to_numpy(xp.asarray(velocities, dtype=float)), dtype=float)
    b = np.asarray(xp.to_numpy(field.field_at(x)), dtype=float)
    strength = np.linalg.norm(b, axis=-1)
    unit = b / strength[:, None]
    omega = charge_number * _E * strength / mass
    centres = x - np.cross(unit, v) / omega[:, None]
    v_par = np.sum(v * unit, axis=-1)
    b_centre = np.linalg.norm(xp.to_numpy(field.field_at(centres)), axis=-1)
    # off the grid (NaN) the particle's field
    b_centre = np.where(np.isfinite(b_centre), b_centre, strength)
    mu = np.maximum(np.sum(v * v, axis=-1) - v_par**2, 0.0) / (2.0 * b_centre)
    return GuidingCentres(
        positions=centres,
        v_parallel=v_par,
        mu=mu,
        weights=np.asarray(xp.to_numpy(xp.asarray(weights, dtype=float)), dtype=float),
        mass=float(mass),
        charge_number=int(charge_number),
    )


def _c_order(array) -> np.ndarray:
    """A C-ordered float copy (Pyccel checks the strides, also of length-1 axes)."""
    return np.array(array, dtype=float, order="C", copy=True)


def _strength(field: GriddedEquilibrium, points) -> np.ndarray:
    return np.linalg.norm(np.asarray(xp.to_numpy(field.field_at(points))), axis=-1)


class _Pusher:
    """The kernel's arrays of one gridded field, and pushes of marker subsets."""

    def __init__(
        self, field: GriddedEquilibrium, rho_edges, qm: float, wall=None
    ) -> None:
        self.field = field
        coefficients, self.grid, self.shape = field.spline_arrays
        self.coefficients = coefficients.reshape(-1)
        self.rho_edges = np.ascontiguousarray(rho_edges, dtype=float)
        self.qm = float(qm)
        self.wall = np.zeros(1) if wall is None else _c_order(wall.reshape(-1))

    def clearance_at(self, points) -> np.ndarray:
        """Return the interpolated wall clearance at ``points`` ``(m, 3)`` (inf without one)."""
        points = np.asarray(points, dtype=float)
        if self.wall.shape[0] == 1:
            return np.full(points.shape[0], np.inf)
        value, ok = _numpy_push.scalar_at(
            self.wall, 1, 0, self.grid, self.shape, *points.T
        )
        return np.where(ok, value, -np.inf)

    @sp.profile("freeofcharge.orbit_push", tags=PROFILE_TAGS)
    def push(
        self,
        state,
        mu,
        duration,
        max_step,
        rho2_stop,
        clearance_stop=-np.inf,
        clearance_high=np.inf,
        record: int = 1,
    ):
        """Push the columns of ``state`` ``(4, m)``; return the kernel's outputs.

        ``(state, previous, elapsed, status, shell_time, shell_vpar, path, taken)``;
        ``path`` holds the start of the first ``record`` steps if ``record`` > 1.
        """
        m = state.shape[1]
        state = _c_order(state)
        previous = np.zeros((3, m))
        elapsed = np.zeros(m)
        status = np.zeros(m, dtype=np.int64)
        n_shells = len(self.rho_edges) - 1
        shell_time = np.zeros((n_shells, m))
        shell_vpar = np.zeros((n_shells, m))
        path = np.zeros((max(int(record), 1), 3, m))
        taken = np.zeros(m, dtype=np.int64)
        if m:
            _push(
                state,
                _c_order(mu),
                _c_order(duration),
                _c_order(max_step),
                self.coefficients,
                self.grid,
                self.shape,
                self.qm,
                float(rho2_stop),
                self.wall,
                float(clearance_stop),
                float(clearance_high),
                self.rho_edges,
                previous,
                elapsed,
                status,
                shell_time,
                shell_vpar,
                path,
                taken,
            )
        return state, previous, elapsed, status, shell_time, shell_vpar, path, taken


def trace_orbits(
    markers: GuidingCentres,
    field: GriddedEquilibrium,
    duration: float,
    *,
    samples: int = 200,
    step: float = 0.05,
) -> xr.Dataset:
    r"""Return collisionless orbits sampled at ``samples`` times, in plasma-plots' layout.

    A Dataset of ``(t, marker)`` variables ``x``, ``y``, ``z`` [m], ``R``, ``rho``,
    ``v_parallel`` [m/s], ``energy`` [eV] and ``B`` [T] (as plasma-plots' orbit
    products), NaN after a marker leaves the grid. No wall: for the conservation
    checks and pictures of single orbits.

    Args:
        markers: The guiding centres.
        field: The gridded field.
        duration: Time to follow [s].
        samples: Output times (equally spaced, the first at 0).
        step: Largest step along the orbit [m].
    """
    pusher = _Pusher(field, np.array([0.0, 1.0]), markers.charge_over_mass)
    n = len(markers.weights)
    state = np.stack([*markers.positions.T, markers.v_parallel]).astype(float)
    speed = np.sqrt(
        markers.v_parallel**2 + 2.0 * markers.mu * _strength(field, markers.positions)
    )
    max_step = step / np.maximum(speed, 1e-30)
    interval = duration / max(samples - 1, 1)
    out = np.full((samples, 4, n), np.nan)
    out[0] = state
    alive = np.ones(n, dtype=bool)
    for k in range(1, samples):
        state, _, _, status, *_ = pusher.push(
            state, markers.mu, np.where(alive, interval, 0.0), max_step, np.inf
        )
        alive &= status == 0
        out[k] = np.where(alive, state, np.nan)
    positions = np.moveaxis(out[:, :3], 1, -1)
    b = _strength(field, positions)
    energy = markers.mass * (0.5 * out[:, 3] ** 2 + markers.mu * b) / _E
    dims = ("t", "marker")
    return xr.Dataset(
        {
            "x": (dims, out[:, 0], {"units": "m"}),
            "y": (dims, out[:, 1], {"units": "m"}),
            "z": (dims, out[:, 2], {"units": "m"}),
            "R": (dims, np.hypot(out[:, 0], out[:, 1]), {"units": "m"}),
            "rho": (
                dims,
                np.asarray(xp.to_numpy(field.rho_at(positions))),
                rho_coordinate(field.radial_coordinate),
            ),
            "v_parallel": (
                dims,
                out[:, 3],
                {"long_name": r"$v_\parallel$", "units": "m/s"},
            ),
            "energy": (dims, energy, {"long_name": "energy", "units": "eV"}),
            "B": (dims, b, {"long_name": r"$|B|$", "units": "T"}),
        },
        coords={
            "t": ("t", interval * np.arange(samples), {"units": "s"}),
            "marker": np.arange(n),
        },
    )


@method(
    equations={
        "drag": r"\frac{dv}{dt} = -a_e v - \frac{a_i}{v^2} \;\Rightarrow\; v^3(t + \Delta t) = \left(v^3 + \frac{a_i}{a_e}\right) e^{-3 a_e \Delta t} - \frac{a_i}{a_e}",
        "electrons": r"\mathbf{v}' = \mathbf{v}_\mathrm{drag} + \sqrt{\frac{T_e}{m}\left(1 - e^{-2 a_e \Delta t}\right)}\,\mathbf{N}_3",
        "ions": r"v' = v + \left(-\frac{D_i}{v}\,\Delta t + \sqrt{2 D_i \Delta t}\,N\right)\hat{\mathbf{v}}, \qquad D_i = \frac{a_i T_i}{m v^3}",
        "pitch": r"\xi' = \xi (1 - \nu_D \Delta t) \pm \sqrt{(1 - \xi^2)\,\nu_D \Delta t}, \qquad \nu_D = \frac{\sum_j \Gamma_j}{v^3}",
        "split": r"\Delta E_i = \Delta E_\mathrm{drag} \frac{a_i / \bar v}{a_e \bar v^2 + a_i / \bar v} + \Delta E_{i,\mathrm{diff}}, \qquad \bar v = \tfrac12 (v + v_\mathrm{drag})",
    },
    assumptions=(
        r"$a_e$, $a_i$, $\nu_D$ of `collision_coefficients` (the Coulomb logarithms and drags of `slow_down`) at the speed before the step, constant over it",
        r"the diffusion from the fluctuation-dissipation relation of the high-speed drags: the electrons' as an isotropic Ornstein-Uhlenbeck kick of the velocity (exact for constant $a_e$, a Maxwellian at $T_e$ stationary; it scatters the pitch a little too), the ions' along the velocity (a Maxwellian at $T_i$ stationary for $v \gg v_{Ti}$, so markers are removed near the thermal energy)",
        r"Euler–Maruyama for the ions' diffusion and the pitch ($\xi$ clipped to $[-1, 1]$); a random sign for the pitch kick; the operators applied in turn: drag, electrons, ions, pitch",
    ),
    references=(
        "[Boozer and Kuo-Petravic, Phys. Fluids 24, 851 (1981)](https://doi.org/10.1063/1.863445)",
        "[Hirvijoki et al., Comput. Phys. Commun. 185, 1310 (2014)](https://doi.org/10.1016/j.cpc.2014.01.014)",
        "Helander and Sigmar, Collisional Transport in Magnetized Plasmas (2002), ch. 3",
    ),
    components=(collision_coefficients,),
)
def collide(
    speed,
    pitch,
    dt,
    electron_drag,
    ion_drag,
    deflection,
    te,
    ti,
    mass: float,
    rng: np.random.Generator,
    *,
    energy_diffusion: bool = True,
):
    r"""Return the speed, pitch and energy to the electrons and ions after one collision step.

    Args:
        speed: Speeds $v$ [m/s].
        pitch: $\xi = v_\parallel / v$.
        dt: Time steps [s].
        electron_drag: $a_e$ [1/s] (`collision_coefficients`).
        ion_drag: $a_i$ [$\mathrm{m^3/s^3}$].
        deflection: $\sum_j \Gamma_j$ [$\mathrm{m^3/s^3}$], so $\nu_D = $ ``deflection`` $/ v^3$.
        te: Electron temperature [eV].
        ti: Ion temperature [eV].
        mass: Ion mass [kg].
        rng: Random generator.
        energy_diffusion: Include the energy diffusion; without it the speed
            follows the drag of `slow_down` exactly.

    Returns:
        ``(speed, pitch, to_electrons, to_ions)``, the energies [J] lost by the ion.
    """
    v = np.asarray(speed, dtype=float)
    a_e, a_i = np.asarray(electron_drag, float), np.asarray(ion_drag, float)
    xi = np.clip(np.asarray(pitch, dtype=float), -1.0, 1.0)
    ratio = a_i / a_e
    v_cubed = (v**3 + ratio) * np.exp(-3.0 * a_e * dt) - ratio
    v_drag = np.cbrt(np.maximum(v_cubed, 0.0))
    v_mid = 0.5 * (v + v_drag)
    with np.errstate(invalid="ignore", divide="ignore"):
        ion_share = (a_i / v_mid) / (a_e * v_mid**2 + a_i / v_mid)
    ion_share = np.where(np.isfinite(ion_share), ion_share, 0.0)
    drag_loss = 0.5 * mass * (v**2 - v_drag**2)
    to_ions = ion_share * drag_loss
    to_electrons = drag_loss - to_ions
    v_new = v_drag
    if energy_diffusion:
        # electrons: an isotropic kick of the velocity vector, the variance of the
        # Ornstein-Uhlenbeck process with the drag a_e and the temperature T_e
        thermal = np.asarray(te) * _E / mass * -np.expm1(-2.0 * a_e * dt)
        kick = np.sqrt(thermal)[..., None] * rng.standard_normal((*v.shape, 3))
        parallel = xi * v_drag + kick[..., 0]
        perpendicular = np.sqrt(np.maximum(1.0 - xi**2, 0.0)) * v_drag + kick[..., 1]
        v_e = np.sqrt(parallel**2 + perpendicular**2 + kick[..., 2] ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            xi = np.where(v_e > 0.0, parallel / v_e, xi)
        to_electrons = to_electrons + 0.5 * mass * (v_drag**2 - v_e**2)
        # ions: along the velocity, with the drift that keeps a Maxwellian at T_i
        with np.errstate(invalid="ignore", divide="ignore"):
            d_i = np.where(v > 0.0, a_i * np.asarray(ti) * _E / (mass * v**3), 0.0)
            drift = np.where(v > 0.0, -d_i / v, 0.0)
        v_new = np.abs(
            v_e + drift * dt + np.sqrt(2.0 * d_i * dt) * rng.standard_normal(v.shape)
        )
        to_ions = to_ions + 0.5 * mass * (v_e**2 - v_new**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = np.where(v > 0.0, np.asarray(deflection) / v**3 * dt, 0.0)
    sign = np.where(rng.random(v.shape) < 0.5, -1.0, 1.0)
    xi_new = xi * (1.0 - nu) + sign * np.sqrt(np.maximum(1.0 - xi**2, 0.0) * nu)
    return v_new, np.clip(xi_new, -1.0, 1.0), to_electrons, to_ions


@dataclass
class OrbitFollowing:
    """The result of `follow_orbits`: the markers' fates and the steady-state moments.

    The moments are per shell of ``rho_edges`` [W, 1/m^3, ...] with the volumes of
    ``surfaces`` (NaN densities without them); the powers are totals [W].
    """

    markers: xr.Dataset
    rho_edges: np.ndarray
    electron_heating: np.ndarray
    ion_heating: np.ndarray
    particles: np.ndarray
    energy_content: np.ndarray
    parallel_flow: np.ndarray
    volume: np.ndarray
    birth_power: float
    thermalised_power: float
    lost_power: float
    numerical_power: float
    alive_power: float
    radial_coordinate: str = "unspecified"
    traces: xr.Dataset | None = None
    wall: Any = field(default=None, repr=False)
    """Per marker (dimension ``marker``): birth and end positions, energies,
    pitches, times, the ``fate`` (index into `FATES`) and the wall face hit."""
    """Shell edges."""
    """Power to the electrons per shell [W]."""
    """Power to the ions per shell [W]."""
    """Steady-state number of fast ions per shell."""
    """Steady-state fast-ion energy per shell [J]."""
    r"""$\sum w \int v_\parallel\,dt$ per shell [m]."""
    """Shell volumes [m^3] (NaN without ``surfaces``)."""
    """Energy rate of the births [W]."""
    """Energy rate of the markers when thermalised [W]."""
    """Energy rate of the markers when lost (wall, rho limit, grid) [W]."""
    """Energy change of the markers in the orbit integration (an error) [W]."""
    """Energy rate still carried by markers stopped at ``max_time`` [W]."""

    @property
    def power_balance(self) -> float:
        r"""Return $(P_e + P_i + P_\mathrm{th} + P_\mathrm{lost} + P_\mathrm{end} - P_\mathrm{num}) / P_\mathrm{birth} - 1$, zero to roundoff."""
        total = (
            np.sum(self.electron_heating)
            + np.sum(self.ion_heating)
            + self.thermalised_power
            + self.lost_power
            + self.alive_power
            - self.numerical_power
        )
        return float(total / self.birth_power - 1.0)

    @property
    def lost_fraction(self) -> float:
        """Return the share of the birth power that is lost."""
        return self.lost_power / self.birth_power

    def fate_fractions(self) -> dict[str, float]:
        """Return the share of the births (by weight) of each fate."""
        w = self.markers.weight.values
        fate = self.markers.fate.values
        return {
            name: float(np.sum(w[fate == k]) / np.sum(w))
            for k, name in enumerate(FATES)
        }

    def power_fractions(self) -> dict[str, float]:
        """Return the birth power's shares: heating, thermalised, lost, still alive, numerical."""
        p = self.birth_power
        return {
            "electrons": float(np.sum(self.electron_heating) / p),
            "ions": float(np.sum(self.ion_heating) / p),
            "thermalised": self.thermalised_power / p,
            "lost": self.lost_power / p,
            "alive": self.alive_power / p,
            "numerical": self.numerical_power / p,
        }

    @property
    def stored_energy(self) -> float:
        """Return the fast-ion energy content [J]."""
        return float(np.sum(self.energy_content))

    def to_dataset(self) -> xr.Dataset:
        """Return the profiles on the shell centres, as `freeofcharge.SlowingDown.to_dataset`."""
        rho = 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])
        heating = np.stack([self.electron_heating, self.ion_heating])
        return xr.Dataset(
            {
                "heating": (
                    ("species", "rho"),
                    heating,
                    {"long_name": "heating", "units": "W"},
                ),
                "heating_density": (
                    ("species", "rho"),
                    heating / self.volume,
                    {"long_name": "heating", "units": "W/m^3"},
                ),
                "fast_ion_density": (
                    "rho",
                    self.particles / self.volume,
                    {"long_name": "fast-ion density", "units": "1/m^3"},
                ),
                "fast_ion_pressure": (
                    "rho",
                    2.0 / 3.0 * self.energy_content / self.volume,
                    {"long_name": "fast-ion pressure", "units": "Pa"},
                ),
                "fast_ion_current_density": (
                    "rho",
                    _E * self.parallel_flow / self.volume,
                    {"long_name": "fast-ion current density", "units": "A/m^2"},
                ),
            },
            coords={
                "rho": ("rho", rho, rho_coordinate(self.radial_coordinate)),
                "species": ["electrons", "ions"],
            },
            attrs={
                "radial_coordinate": self.radial_coordinate,
                "stored_energy": self.stored_energy,
                "birth_power": self.birth_power,
                "thermalised_power": self.thermalised_power,
                "lost_power": self.lost_power,
                "numerical_power": self.numerical_power,
                "alive_power": self.alive_power,
            },
        )

    def losses(self) -> xr.Dataset:
        """Return the lost markers (all loss fates): where, when, energy, pitch, power."""
        fate = self.markers.fate.values
        lost = np.isin(fate, (_WALL, _RHO_LIMIT, _GRID))
        return self.markers.isel(marker=np.nonzero(lost)[0])

    @method(
        equations={
            "load": r"q_f = \frac{1}{A_f} \sum_{b \to f} w_b E_b",
        },
        assumptions=(
            r"the guiding centre's (or Larmor circle's) first crossing of the wall's triangulation; energy at the crossing",
            r"a steady state: weights in ions/s give powers in W",
        ),
    )
    def wall_loads(self, *, theta_bins: int = 64, phi_bins: int = 180) -> xr.Dataset:
        r"""Return the power on each face of the wall and, on a Fourier wall, a map.

        Dataset over ``face``: ``power`` [W], ``area`` [m^2], ``wall_load``
        $q_f = P_f / A_f$ [W/m^2], ``markers`` (hits) and the face centres ``X``,
        ``Y``, ``Z``; ``vertices`` and ``faces`` of the mesh (for a 3D surface
        plot). On a wall with angles (simgeometry's `FourierSurface`), also
        ``wall_load_map`` over ``(theta, phi)`` bins: the power in each bin over its
        area.
        """
        if self.wall is None:
            raise ValueError("no wall: the losses are at rho_limit (see losses())")
        mesh = getattr(self.wall, "mesh", self.wall)
        areas = np.asarray(mesh.face_areas, dtype=float)
        faces = np.asarray(mesh.faces)
        vertices = np.asarray(mesh.vertices)
        hit = self.markers.fate.values == _WALL
        face = self.markers.face.values[hit].astype(np.int64)
        power = (
            self.markers.weight.values[hit] * self.markers.end_energy.values[hit] * _E
        )
        per_face = np.bincount(face, weights=power, minlength=len(areas))
        counts = np.bincount(face, minlength=len(areas))
        centres = vertices[faces].mean(axis=1)
        data = {
            "power": ("face", per_face, {"long_name": "power", "units": "W"}),
            "area": ("face", areas, {"long_name": "area", "units": "m^2"}),
            "wall_load": (
                "face",
                per_face / areas,
                {"long_name": "wall load", "units": "W/m^2"},
            ),
            "markers": ("face", counts, {"long_name": "markers"}),
            "X": ("face", centres[:, 0], {"units": "m"}),
            "Y": ("face", centres[:, 1], {"units": "m"}),
            "Z": ("face", centres[:, 2], {"units": "m"}),
            "vertices": (("vertex", "xyz"), vertices, {"units": "m"}),
            "faces": (("face", "corner"), faces),
        }
        coords: dict[str, Any] = {"xyz": ["x", "y", "z"]}
        triangulate = getattr(self.wall, "triangulate", None)
        if triangulate is not None:
            _, angles = triangulate(*self.wall.resolution)
            theta_f = np.mod(np.asarray(angles)[:, :, 0].mean(axis=1), 2 * np.pi)
            phi_f = np.mod(np.asarray(angles)[:, :, 1].mean(axis=1), 2 * np.pi)
            t_edges = np.linspace(0.0, 2 * np.pi, theta_bins + 1)
            p_edges = np.linspace(0.0, 2 * np.pi, phi_bins + 1)
            area_map, _, _ = np.histogram2d(
                theta_f, phi_f, [t_edges, p_edges], weights=areas
            )
            power_map, _, _ = np.histogram2d(
                theta_f, phi_f, [t_edges, p_edges], weights=per_face
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                load_map = np.where(area_map > 0, power_map / area_map, np.nan)
            data["wall_load_map"] = (
                ("theta", "phi"),
                load_map,
                {"long_name": "wall load", "units": "W/m^2"},
            )
            coords["theta"] = (
                "theta",
                0.5 * (t_edges[1:] + t_edges[:-1]),
                {"long_name": r"$\theta$", "units": "rad"},
            )
            coords["phi"] = (
                "phi",
                0.5 * (p_edges[1:] + p_edges[:-1]),
                {"long_name": r"$\varphi$", "units": "rad"},
            )
        return xr.Dataset(
            data,
            coords=coords,
            attrs={
                "lost_power": float(np.sum(power)),
                "peak_wall_load": float(np.max(per_face / areas))
                if len(areas)
                else 0.0,
            },
        )


def _closest_on_triangles(p, a, b, c) -> np.ndarray:
    """The closest points of triangles ``(a, b, c)`` to points ``p`` (Ericson's regions), all ``(..., 3)``."""

    def dot(u, v):
        return np.sum(u * v, axis=-1)

    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(invalid="ignore", divide="ignore"):
        on_ab = a + (d1 / (d1 - d3))[..., None] * ab
        on_ac = a + (d2 / (d2 - d6))[..., None] * ac
        on_bc = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None] * (c - b)
        total = va + vb + vc
        inner = a + (vb / total)[..., None] * ab + (vc / total)[..., None] * ac
    conditions = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
    ]
    choices = [a, b, on_ab, c, on_ac, on_bc]
    result = inner
    for condition, choice in zip(conditions[::-1], choices[::-1], strict=True):
        result = np.where(condition[..., None], choice, result)
    return result


def wall_distance(
    wall, points, *, neighbours: int = 12, near: float = 0.3
) -> np.ndarray:
    """Return the distance of ``points`` ``(n, 3)`` to a triangulated wall, positive inside.

    Exact for the triangles among the ``neighbours`` with the nearest centres (all
    points closer than ``near`` [m]); farther away the distance to the nearest
    face centre. The sign from ``wall.inside_at`` (a closed wall).
    """
    from scipy.spatial import KDTree

    mesh = getattr(wall, "mesh", wall)
    vertices = np.asarray(mesh.vertices, dtype=float)
    corners = vertices[np.asarray(mesh.faces)]
    tree = KDTree(corners.mean(axis=1))
    points = np.asarray(points, dtype=float)
    distance, nearest = tree.query(points, k=neighbours)
    distance = np.asarray(distance, dtype=float).reshape(len(points), -1)
    nearest = np.asarray(nearest, dtype=np.int64).reshape(len(points), -1)
    result = distance[:, 0].copy()
    close = np.nonzero(distance[:, 0] < near)[0]
    for lo in range(0, len(close), 65536):
        part = close[lo : lo + 65536]
        tri = corners[nearest[part]]  # (m, k, 3 corners, 3)
        closest = _closest_on_triangles(
            points[part, None, :], tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        )
        result[part] = np.min(
            np.linalg.norm(closest - points[part, None, :], axis=-1), axis=1
        )
    inside = np.asarray(xp.to_numpy(wall.inside_at(points)), dtype=bool)
    return np.where(inside, result, -result)


def wall_clearance(
    wall, field: GriddedEquilibrium, *, check_points: int = 4000, seed: int = 0
) -> np.ndarray:
    r"""Return the wall's signed distance on the field's grid, as spline coefficients.

    The distance (`wall_distance`) at the grid's nodes over one field period,
    interpolated like the field: the kernel stops a marker near the wall from it.
    The wall must repeat with the field's period (W7-X's stand-in, an axisymmetric
    wall); random points over the whole torus check that the interpolated sign
    agrees with the wall's ``inside_at`` (a ValueError otherwise).
    """
    r = np.asarray(field.samples["R"])
    z = np.asarray(field.samples["Z"])
    phi = np.arange(field._shape[1]) * field.period / field._shape[1]
    rr, pp, zz = np.meshgrid(r, phi, z, indexing="ij")
    nodes = np.stack([rr * np.cos(pp), rr * np.sin(pp), zz], axis=-1).reshape(-1, 3)
    values = wall_distance(wall, nodes).reshape(rr.shape)
    coefficients = field.spline_coefficients(values)
    rng = np.random.default_rng(seed)
    radius = rng.uniform(r[0], r[-1], check_points)
    angle = rng.uniform(0.0, 2.0 * np.pi, check_points)
    points = np.stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            rng.uniform(z[0], z[-1], check_points),
        ],
        -1,
    )
    pusher = _Pusher(field, np.array([0.0, 1.0]), 1.0, coefficients)
    clearance = pusher.clearance_at(points)
    inside = np.asarray(xp.to_numpy(wall.inside_at(points)), dtype=bool)
    clear = np.abs(clearance) > 2.0 * max(r[1] - r[0], z[1] - z[0])
    wrong = clear & (inside != (clearance > 0))
    if np.any(wrong):
        raise ValueError(
            f"the wall does not repeat with the field's period ({wrong.sum()} of "
            f"{clear.sum()} test points on the wrong side); pass clearance=None"
        )
    return coefficients


def _mesh_hits(mesh, starts, ends):
    """First hits of segments with a triangle mesh: ``(hit, point, face)``.

    A segment that ends outside a closed mesh without a hit (it passed exactly
    through an edge or a vertex, where the triangle tests round off) is a hit at
    its end, on the face with the nearest centre.
    """
    hits = mesh.first_hit(starts, ends)
    hit = np.asarray(hits.hit, dtype=bool).copy()
    point = np.asarray(hits.point, dtype=float).copy()
    face = np.asarray(hits.face, dtype=np.int64).copy()
    if getattr(mesh, "closed", False) and np.any(~hit):
        rest = np.nonzero(~hit)[0]
        outside = ~np.asarray(mesh.inside_at(ends[rest]), dtype=bool)
        missed = rest[outside]
        if len(missed):
            centres = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].mean(axis=1)
            gap = ends[missed, None, :] - centres[None, :, :]
            hit[missed] = True
            point[missed] = ends[missed]
            face[missed] = np.argmin(np.sum(gap * gap, axis=-1), axis=1)
    return hit, point, face


def _local_plasma(plasma: Plasma, rho):
    rho = np.minimum(rho, 1.0)
    return (
        np.asarray(xp.to_numpy(plasma.electron_density(rho))),
        np.asarray(xp.to_numpy(plasma.electron_temperature(rho))),
        np.asarray(xp.to_numpy(plasma.ion_temperature(rho))),
        [np.asarray(xp.to_numpy(n)) for n in plasma.ion_densities(rho)],
    )


def _perpendicular_basis(b_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each row of ``b_unit`` and to each other."""
    helper = np.where(
        (np.abs(b_unit[:, 2]) < 0.9)[:, None],
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    e1 = np.cross(b_unit, helper)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    return e1, np.cross(b_unit, e1)


def _finish_at_limit(finish, index, starts, ends, field, rho_limit) -> None:
    """Record the markers ``index`` lost where the step ``starts -> ends`` reaches ``rho_limit``."""
    if len(index) == 0:
        return
    rho_a = np.asarray(xp.to_numpy(field.rho_at(starts)))
    rho_b = np.asarray(xp.to_numpy(field.rho_at(ends)))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.clip((rho_limit - rho_a) / (rho_b - rho_a), 0.0, 1.0)
    fraction = np.where(np.isfinite(fraction), fraction, 1.0)
    points = starts + fraction[:, None] * (ends - starts)
    off = ~np.isfinite(rho_b) & ~(rho_a >= rho_limit)
    finish(index[~off], _RHO_LIMIT, points[~off])
    finish(index[off], _GRID, ends[off])


def _check_wall(wall, field: GriddedEquilibrium, safe_rho: float) -> None:
    """Raise if the wall reaches inside ``safe_rho`` or leaves the grid."""
    mesh = getattr(wall, "mesh", wall)
    vertices = np.asarray(mesh.vertices, dtype=float)
    rho = np.asarray(xp.to_numpy(field.rho_at(vertices)))
    if np.isfinite(safe_rho) and np.any(np.isfinite(rho) & (rho <= safe_rho)):
        raise ValueError(
            f"the wall reaches inside rho = {safe_rho} (safe_rho); lower safe_rho"
        )
    if not np.all(np.isfinite(rho)):
        warnings.warn(
            "parts of the wall are off the field's grid: markers reaching the grid's "
            "edge first are lost there (fate 'grid'); sample the field over the wall",
            stacklevel=3,
        )


@method(
    equations={
        "orbit": r"\dot{\mathbf{X}} = \frac{v_\parallel \mathbf{B}^* + \frac{\mu}{q/m}\,\hat b \times \nabla B}{B^*_\parallel}, \quad \dot v_\parallel = -\mu\,\frac{\mathbf{B}^* \cdot \nabla B}{B^*_\parallel}, \quad \mathbf{B}^* = \mathbf{B} + \frac{v_\parallel}{q/m}\nabla\times\hat b",
        "curl": r"\nabla\times\hat b = \frac{\nabla\times\mathbf{B} + \hat b \times \nabla B}{B}",
        "rk4": r"\Delta t = \frac{\Delta t_c}{\lceil \Delta t_c\, v / \Delta s \rceil}, \qquad \text{RK4 in } (\mathbf{X}, v_\parallel)",
        "collision_step": r"\Delta t_c = c \min\left(\frac{1}{a_e + a_i / v^3},\ \frac{1}{\nu_D}\right)",
        "moments": r"N_k = \sum_b w_b \int_{\rho \in k} dt, \qquad W_k = \sum_b w_b \int_{\rho \in k} E\,dt, \qquad P_{e,k} = \sum_b w_b\, \Delta E_{e,b}",
        "balance": r"\sum_k (P_{e,k} + P_{i,k}) + P_\mathrm{th} + P_\mathrm{lost} + P_\mathrm{end} = P_\mathrm{birth} + P_\mathrm{num}",
    },
    assumptions=(
        r"guiding centres in physical space on the cubic-spline interpolant of the field (exact derivatives of it); no electric field",
        r"$\mu$ an invariant between collisions; energy conserved by the flow, to the RK4 error ($P_\mathrm{num}$)",
        r"collisions only inside $\rho = 1$, with the plasma at the guiding centre; operator split from the orbit, once per $\Delta t_c$",
        r"lost when the segment of a step crosses the wall (`first_hit`, exact for the triangulation), by the guiding centre or, with `larmor_points`, a spoke to the Larmor circle at its end; without a wall at $\rho$ = `rho_limit` (linear in $\rho$ along the step)",
        r"the compiled kernel steps without wall tests while the interpolated distance to the wall (`wall_clearance`) exceeds `clearance` (or inside $\rho$ = `safe_rho`); closer, every step is tested in NumPy. `clearance` must exceed the error of the interpolated distance, the sagitta of a step and, with `larmor_points`, a Larmor radius",
        r"time in a shell scored at the start of each step; the energy over a push interval the mean of its values before and after the collision; the last step of a thermalised marker cut where its energy reaches the end energy (linear in time)",
    ),
    references=(
        "[Littlejohn, J. Plasma Phys. 29, 111 (1983)](https://doi.org/10.1017/S002237780000060X)",
        "[Cary and Brizard, Rev. Mod. Phys. 81, 693 (2009)](https://doi.org/10.1103/RevModPhys.81.693)",
        "[Hirvijoki et al., Comput. Phys. Commun. 185, 1310 (2014)](https://doi.org/10.1016/j.cpc.2014.01.014) (ASCOT5's guiding-centre and collision model)",
    ),
    components=(guiding_centres, collide),
)
def follow_orbits(
    deposition: Deposition,
    field: GriddedEquilibrium,
    plasma: Plasma | None = None,
    *,
    surfaces: FluxSurfaces | None = None,
    wall=None,
    rho_limit: float = 1.0,
    safe_rho: float | None = None,
    clearance: float | None = 0.03,
    step: float = 0.05,
    max_time: float = 1.0,
    collision_step: float = 0.05,
    interval: float = 1.0e-4,
    thermal_factor: float = 2.0,
    minimum_energy: float = 0.0,
    energy_diffusion: bool = True,
    ln_lambda=None,
    larmor_points: int = 0,
    seed: int | None = None,
    markers: GuidingCentres | None = None,
) -> OrbitFollowing:
    r"""Follow the births of ``deposition`` as guiding centres until thermalised or lost.

    Args:
        deposition: The births (positions, velocities, weights [ions/s]), their
            isotope and the shells (``rho_edges``) of the moments.
        field: The field, a cubic `GriddedEquilibrium` (see `orbit_field`) covering
            the wall.
        plasma: Profiles for the collisions; None follows the orbits without
            collisions (prompt losses) up to ``max_time``.
        surfaces: The shells' volumes for the densities (`FluxSurfaces` of
            ``deposition.rho_edges``); NaN densities without them.
        wall: A simgeometry surface with ``first_hit(starts, ends)``
            (`TriangleMesh`, `FourierSurface`; `revolved_wall` for a tokamak); None
            ends the orbits at $\rho$ = ``rho_limit``.
        rho_limit: Loss radius without a wall.
        safe_rho: With a wall, the kernel pushes without wall tests only inside
            this $\rho$ (also where the wall is closer than ``clearance``); outside
            it every step is tested (in NumPy). None: no limit in $\rho$, or 1 if
            ``clearance`` is None.
        clearance: With a wall, every step closer to the wall than this [m] (the
            interpolated distance of `wall_clearance` on the field's grid; the wall
            must repeat with the field's period) is tested; None: no distance
            field, ``safe_rho`` only.
        step: Largest step along the orbit [m] (a length, so the time step grows as
            the ions slow down).
        max_time: Longest time a marker is followed [s].
        collision_step: $c$, the collision step as a fraction of the shortest
            collision time.
        interval: Push interval without collisions (outside $\rho = 1$ or without
            ``plasma``) [s].
        thermal_factor: Thermalised below ``thermal_factor`` $\times\, T_i$ ...
        minimum_energy: ... or below this energy [eV], whichever is higher
            (`slow_down`'s end energy).
        energy_diffusion: Include the energy diffusion of `collide`.
        ln_lambda: One Coulomb logarithm for all species; each its own if None.
        larmor_points: Points on the Larmor circle tested against the wall at the
            end of each step near it (0: the guiding centre only).
        seed: Seed of the collisions' random numbers.
        markers: Guiding centres to follow instead of those of the births (same
            order and weights).
    """
    minimum_energy = to_unit(minimum_energy, "eV", "minimum_energy")
    field = orbit_field(field)
    mass = deposition.mass_number * ATOMIC_MASS
    if markers is None:
        births = deposition.births
        markers = guiding_centres(
            births.positions, births.velocities, births.weights, field, mass
        )
    rng = np.random.default_rng(seed)
    rho_edges = np.asarray(deposition.rho_edges, dtype=float)
    n_shells = len(rho_edges) - 1
    qm = markers.charge_over_mass
    wall_coefficients = None
    clearance_stop = -np.inf
    if wall is None:
        rho2_stop = rho_limit**2
    else:
        if safe_rho is None:
            safe_rho = 1.0 if clearance is None else np.inf
        rho2_stop = safe_rho**2
        _check_wall(wall, field, safe_rho)
        if clearance is not None:
            wall_coefficients = wall_clearance(wall, field)
            clearance_stop = float(clearance)
    pusher = _Pusher(field, rho_edges, qm, wall_coefficients)
    n = len(markers.weights)
    w = markers.weights
    state = np.stack([*markers.positions.T, markers.v_parallel]).astype(float)
    mu = markers.mu.astype(float).copy()
    energy = markers.mass * (
        0.5 * state[3] ** 2 + mu * _strength(field, markers.positions)
    )
    birth_energy = energy.copy()
    birth_pitch = state[3] / np.sqrt(2.0 * energy / mass)
    birth_rho = np.asarray(xp.to_numpy(field.rho_at(markers.positions)))
    time = np.zeros(n)
    fate = np.zeros(n, dtype=np.int64)
    end = {
        "position": np.full((n, 3), np.nan),
        "energy": np.full(n, np.nan),
        "pitch": np.full(n, np.nan),
        "face": np.full(n, -1, dtype=np.int64),
    }
    tallies = {
        name: np.zeros(n_shells)
        for name in ("electrons", "ions", "particles", "energy", "flow")
    }
    powers = {"thermalised": 0.0, "lost": 0.0, "numerical": 0.0, "alive": 0.0}
    if not (np.all(np.isfinite(birth_rho)) and np.all(np.isfinite(energy))):
        raise ValueError("births off the field's grid: sample the field over them")

    def finish(index, code, positions, face=None):
        """Record the end of the markers ``index`` with fate ``code``."""
        if len(index) == 0:
            return
        fate[index] = code
        end["position"][index] = positions
        b = _strength(field, positions)
        b = np.where(np.isfinite(b), b, _strength(field, state[:3, index].T))
        e = mass * (0.5 * state[3, index] ** 2 + mu[index] * b)
        e = np.where(np.isfinite(e), e, energy[index])
        powers["numerical"] += float(np.sum(w[index] * (e - energy[index])))
        energy[index] = e
        end["energy"][index] = e / _E
        end["pitch"][index] = state[3, index] / np.sqrt(2.0 * e / mass)
        if code in (_WALL, _RHO_LIMIT, _GRID):
            powers["lost"] += float(np.sum(w[index] * e))
        elif code == _THERMALISED:
            powers["thermalised"] += float(np.sum(w[index] * e))
        else:
            powers["alive"] += float(np.sum(w[index] * e))
        if face is not None:
            end["face"][index] = face

    def score(index, round_time, round_vpar, factor, mean_energy):
        """Add one round's time integrals of the markers ``index``."""
        weight = w[index] * factor
        tallies["particles"] += round_time @ weight
        tallies["energy"] += round_time @ (weight * mean_energy)
        tallies["flow"] += round_vpar @ weight

    mesh = None if wall is None else getattr(wall, "mesh", wall)

    @sp.profile("freeofcharge.orbit_wall_test", tags=PROFILE_TAGS)
    def wall_test(index, starts, ends):
        """Return the markers of ``index`` that hit the wall on ``starts -> ends``."""
        hit, point, face = _mesh_hits(mesh, starts, ends)
        if larmor_points and np.any(~hit):
            rest = np.nonzero(~hit)[0]
            centre = ends[rest]
            b = np.asarray(xp.to_numpy(field.field_at(centre)))
            strength = np.linalg.norm(b, axis=1)
            ok = np.isfinite(strength)
            strength = np.where(ok, strength, 1.0)
            radius = np.sqrt(2.0 * mu[index[rest]] * strength) / (qm * strength)
            unit = np.where(ok[:, None], b / strength[:, None], [0.0, 0.0, 1.0])
            e1, e2 = _perpendicular_basis(unit)
            angles = 2.0 * np.pi * np.arange(larmor_points) / larmor_points
            ring = centre[:, None, :] + radius[:, None, None] * (
                np.cos(angles)[None, :, None] * e1[:, None, :]
                + np.sin(angles)[None, :, None] * e2[:, None, :]
            )
            spoke_hit, spoke_point, spoke_face = _mesh_hits(
                mesh,
                np.broadcast_to(centre[:, None, :], ring.shape).reshape(-1, 3),
                ring.reshape(-1, 3),
            )
            spoke_hit = spoke_hit.reshape(len(rest), -1) & ok[:, None]
            distance = np.linalg.norm(
                spoke_point.reshape(len(rest), -1, 3) - centre[:, None, :], axis=-1
            )
            distance = np.where(spoke_hit, distance, np.inf)
            nearest = np.argmin(distance, axis=1)
            any_hit = spoke_hit.any(axis=1)
            sel = rest[any_hit]
            pick = np.nonzero(any_hit)[0] * larmor_points + nearest[any_hit]
            hit[sel] = True
            point[sel] = spoke_point[pick]
            face[sel] = spoke_face[pick]
        return hit, point, face

    def path_hits(index, path, taken, ends):
        """First wall hits on the recorded steps: ``(hit, point, face, step)``.

        ``path`` ``(m, K, 3)`` holds the start of each of the ``taken`` steps of the
        markers ``index``; the last step ends at ``ends``.
        """
        m = len(index)
        hit = np.zeros(m, dtype=bool)
        point = np.full((m, 3), np.nan)
        face = np.full(m, -1, dtype=np.int64)
        segment = np.zeros(m, dtype=np.int64)
        count = np.minimum(taken, path.shape[1])
        if not np.any(count):
            return hit, point, face, segment
        owner = np.repeat(np.arange(m), count)
        step = np.concatenate([np.arange(c) for c in count])
        starts = path[owner, step]
        following = np.minimum(step + 1, path.shape[1] - 1)
        last = step + 1 >= count[owner]
        stops = np.where(last[:, None], ends[owner], path[owner, following])
        any_hit, any_point, any_face = wall_test(index[owner], starts, stops)
        # the earliest step of each marker that hits
        order = np.lexsort((step, ~any_hit, owner))
        first = order[np.r_[True, owner[order][1:] != owner[order][:-1]]]
        first = first[any_hit[first]]
        hit[owner[first]] = True
        point[owner[first]] = any_point[first]
        face[owner[first]] = any_face[first]
        segment[owner[first]] = step[first]
        return hit, point, face, segment

    with sp.region("freeofcharge.follow_orbits", tags=PROFILE_TAGS):
        while True:
            alive = np.nonzero(fate == _ALIVE)[0]
            if len(alive) == 0:
                break
            positions = state[:3, alive].T
            rho = np.asarray(xp.to_numpy(field.rho_at(positions)))
            speed = np.sqrt(2.0 * energy[alive] / mass)
            duration = np.full(len(alive), float(interval))
            collisional = np.zeros(len(alive), dtype=bool)
            local = None
            if plasma is not None:
                collisional = rho < 1.0
                ne, te, ti, ions = _local_plasma(plasma, rho)
                coefficients = collision_coefficients(
                    speed,
                    mass,
                    1,
                    ne,
                    te,
                    ions,
                    plasma.ion_masses,
                    plasma.ion_charges,
                    ti,
                    ln_lambda,
                )
                a_e, a_i, deflection = (
                    np.asarray(xp.to_numpy(c)) for c in coefficients
                )
                rate = np.maximum(a_e + a_i / speed**3, deflection / speed**3)
                duration = np.where(collisional, collision_step / rate, duration)
                local = (a_e, a_i, deflection, te, ti)
            duration = np.minimum(duration, max_time - time[alive])
            max_step = step / speed
            start_energy = energy[alive].copy()
            round_time = np.zeros((n_shells, len(alive)))
            round_vpar = np.zeros((n_shells, len(alive)))
            # push: the kernel away from the wall, NumPy step by step near it
            remaining = duration.copy()
            shrink = np.ones(len(alive))
            todo = np.arange(len(alive))
            while len(todo):
                index = alive[todo]
                pushed = pusher.push(
                    state[:, index],
                    mu[index],
                    remaining[todo],
                    max_step[todo],
                    rho2_stop,
                    clearance_stop,
                )
                new, previous, elapsed, status, shell_time, shell_vpar, *_ = pushed
                state[:, index] = new
                round_time[:, todo] += shell_time
                round_vpar[:, todo] += shell_vpar
                remaining[todo] -= elapsed
                time[index] += elapsed
                stopped = status != 0
                if wall is None:
                    crossed = todo[status == 1]
                    _finish_at_limit(
                        finish,
                        alive[crossed],
                        previous[:, status == 1].T,
                        state[:3, alive[crossed]].T,
                        field,
                        rho_limit,
                    )
                    finish(
                        alive[todo[status == 2]],
                        _GRID,
                        state[:3, alive[todo[status == 2]]].T,
                    )
                    todo = np.array([], dtype=np.int64)
                    continue
                # with a wall: near it the kernel records every step of a few
                # dozen, and each of them is tested against the wall
                outside = todo[stopped]
                path = previous[:, stopped].T[:, None, :]
                taken = (status[stopped] == 1).astype(np.int64)
                step_dt = np.zeros(len(outside))
                back = []
                while len(outside):
                    index = alive[outside]
                    ends = state[:3, index].T
                    hit, point, face, segment = path_hits(index, path, taken, ends)
                    # the steps after the one that hit are not taken
                    time[index[hit]] -= (taken[hit] - segment[hit] - 1) * step_dt[hit]
                    finish(index[hit], _WALL, point[hit], face[hit])
                    keep = ~hit
                    outside = outside[keep]
                    index = alive[outside]
                    # off the grid without a hit: lost at the grid's edge
                    on_grid = np.isfinite(
                        np.asarray(xp.to_numpy(field.rho_at(state[:3, index].T)))
                    )
                    finish(index[~on_grid], _GRID, state[:3, index[~on_grid]].T)
                    outside = outside[on_grid]
                    index = alive[outside]
                    done = remaining[outside] <= 1e-12 * np.maximum(
                        duration[outside], 1e-300
                    )
                    rho2 = (
                        np.asarray(xp.to_numpy(field.rho_at(state[:3, index].T))) ** 2
                    )
                    inside = (rho2 < rho2_stop) & ~done
                    inside &= pusher.clearance_at(state[:3, index].T) >= clearance_stop
                    back.append(outside[inside])
                    outside = outside[~inside & ~done]
                    if not len(outside):
                        break
                    index = alive[outside]
                    near_step = max_step[outside] * shrink[outside]
                    near_time = np.minimum(remaining[outside], _RECORD * near_step)
                    step_dt = near_time / np.ceil(near_time / (near_step * (1 + 1e-9)))
                    pushed = pusher.push(
                        state[:, index],
                        mu[index],
                        near_time,
                        near_step * (1.0 + 1e-9),
                        np.inf,
                        -_CROSSED,
                        clearance_stop,
                        record=_RECORD,
                    )
                    new, _, elapsed, status, shell_time, shell_vpar, path, taken = (
                        pushed
                    )
                    path = np.moveaxis(path, 2, 0)  # (marker, step, xyz)
                    state[:, index] = new
                    round_time[:, outside] += shell_time
                    round_vpar[:, outside] += shell_vpar
                    remaining[outside] -= elapsed
                    time[index] += elapsed
                    # a stage off the grid: shorter steps towards the wall, and
                    # lost at the grid's edge only when they are tiny
                    off_stage = status == 2
                    shrink[outside[off_stage]] *= 0.25
                    grid_loss = off_stage & (shrink[outside] < 1e-3)
                    finish(index[grid_loss], _GRID, state[:3, index[grid_loss]].T)
                    outside = outside[~grid_loss]
                    path, taken = path[~grid_loss], taken[~grid_loss]
                    step_dt = step_dt[~grid_loss]
                todo = np.concatenate(back) if back else np.array([], dtype=np.int64)
            # energy after the push: the change is the integration error
            still = fate[alive] == _ALIVE
            idx = alive[still]
            b_new = _strength(field, state[:3, idx].T)
            e_new = mass * (0.5 * state[3, idx] ** 2 + mu[idx] * b_new)
            powers["numerical"] += float(np.sum(w[idx] * (e_new - energy[idx])))
            energy[idx] = e_new
            # the time integrals: the mean energy over the round, and only the time
            # down to the end energy for the thermalised
            factor = np.ones(len(alive))
            end_of_round = start_energy.copy()
            if local is not None:
                a_e, a_i, deflection, te, ti = local
                hit_coll = still & collisional
                j = alive[hit_coll]
                speed_now = np.sqrt(2.0 * energy[j] / mass)
                pitch = np.clip(state[3, j] / speed_now, -1.0, 1.0)
                sel = np.nonzero(hit_coll)[0]
                v_new, xi_new, to_e, to_i = collide(
                    speed_now,
                    pitch,
                    duration[sel],
                    a_e[sel],
                    a_i[sel],
                    deflection[sel],
                    te[sel],
                    ti[sel],
                    mass,
                    rng,
                    energy_diffusion=energy_diffusion,
                )
                shell = np.searchsorted(rho_edges, rho[sel], side="right") - 1
                shell = np.clip(shell, 0, n_shells - 1)
                b_here = b_new[hit_coll[still]]
                before = energy[j]
                after = 0.5 * mass * v_new**2
                end_energy = np.maximum(thermal_factor * ti[sel], minimum_energy) * _E
                cold = after < end_energy
                # thermalised: the step only down to the end energy (linear in time)
                with np.errstate(invalid="ignore", divide="ignore"):
                    part = np.clip((before - end_energy) / (before - after), 0.0, 1.0)
                part = np.where(cold & np.isfinite(part), part, 1.0)
                after = before - part * (before - after)
                v_new = np.sqrt(2.0 * after / mass)
                tallies["electrons"] += np.bincount(
                    shell, w[j] * part * to_e, minlength=n_shells
                )
                tallies["ions"] += np.bincount(
                    shell, w[j] * part * to_i, minlength=n_shells
                )
                state[3, j] = xi_new * v_new
                mu[j] = v_new**2 * (1.0 - xi_new**2) / (2.0 * b_here)
                energy[j] = after
                factor[sel] = part
                end_of_round[sel] = after
                finish(j[cold], _THERMALISED, state[:3, j[cold]].T)
            score(
                alive,
                round_time,
                round_vpar,
                factor,
                0.5 * (start_energy + end_of_round),
            )
            late = np.nonzero((fate == _ALIVE) & (time >= max_time * (1.0 - 1e-12)))[0]
            finish(late, _MAX_TIME, state[:3, late].T)

    volume = (
        np.full(n_shells, np.nan)
        if surfaces is None
        else np.asarray(surfaces.volume, float)
    )
    end_rho = np.asarray(xp.to_numpy(field.rho_at(end["position"])))
    dims = "marker"
    dataset = xr.Dataset(
        {
            "weight": (dims, w, {"long_name": "weight", "units": "1/s"}),
            "birth_x": (dims, markers.positions[:, 0], {"units": "m"}),
            "birth_y": (dims, markers.positions[:, 1], {"units": "m"}),
            "birth_z": (dims, markers.positions[:, 2], {"units": "m"}),
            "birth_rho": (dims, birth_rho, rho_coordinate(field.radial_coordinate)),
            "birth_energy": (
                dims,
                birth_energy / _E,
                {"long_name": "birth energy", "units": "eV"},
            ),
            "birth_pitch": (
                dims,
                birth_pitch,
                {"long_name": r"$v_\parallel / v$ at birth"},
            ),
            "fate": (
                dims,
                fate,
                {"long_name": "fate", "flag_meanings": " ".join(FATES)},
            ),
            "time": (dims, time, {"long_name": "time", "units": "s"}),
            "end_x": (dims, end["position"][:, 0], {"units": "m"}),
            "end_y": (dims, end["position"][:, 1], {"units": "m"}),
            "end_z": (dims, end["position"][:, 2], {"units": "m"}),
            "end_rho": (dims, end_rho, rho_coordinate(field.radial_coordinate)),
            "end_energy": (
                dims,
                end["energy"],
                {"long_name": "end energy", "units": "eV"},
            ),
            "end_pitch": (
                dims,
                end["pitch"],
                {"long_name": r"$v_\parallel / v$ at the end"},
            ),
            "face": (dims, end["face"], {"long_name": "wall face"}),
        },
        coords={"marker": np.arange(n)},
    )
    return OrbitFollowing(
        markers=dataset,
        rho_edges=rho_edges,
        electron_heating=tallies["electrons"],
        ion_heating=tallies["ions"],
        particles=tallies["particles"],
        energy_content=tallies["energy"],
        parallel_flow=tallies["flow"],
        volume=volume,
        birth_power=float(np.sum(w * birth_energy)),
        thermalised_power=powers["thermalised"],
        lost_power=powers["lost"],
        numerical_power=powers["numerical"],
        alive_power=powers["alive"],
        radial_coordinate=field.radial_coordinate,
        wall=wall,
    )
