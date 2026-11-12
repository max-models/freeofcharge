r"""Stellarator coils: their vacuum field, and the coils as obstacles for the beam.

A `CoilSet` holds closed filaments, each a polyline of points with one current. It
comes from a CoilStellaration coilset through plasma-plots
(``plasma_plots.from_coilset``, the optional extra ``coils``), from its Dataset, or
from arrays. It gives

* the Biot–Savart field of the filaments anywhere (`CoilSet.field_at`), exact for
  straight segments, so the error is that of the polyline: second order in the
  spacing of the points;
* `CoilField`: an equilibrium inside its last closed surface and the coils' vacuum
  field outside, as one `freeofcharge.MagneticEquilibrium`, for orbits and wall
  loads between the plasma and the wall;
* `normal_field_error`: $|\mathbf{B}\cdot\hat n| / |\mathbf{B}|$ of the coils on
  a surface, CoilStellaration's field-error metric;
* `PlasmaBoundary`: a ConStellaration boundary (Fourier series of $R$ and $Z$) with
  its inside, for the beam access when there is no equilibrium;
* the coils as obstacles: each coil a tube of radius ``radius`` around its
  filament (the winding pack), `CoilSet.first_hit` gives the distance along lines
  to the first tube they enter and `CoilSet.clearance` the shortest distance
  between line segments and the filaments; `blocked_fraction` and
  `injection_access` use them for a beam and for a scan of injection lines.

The filaments are in metres, the currents in amperes; the points are Cartesian with
$z$ vertical.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from methodcard import method
from plasmacoll.units import to_unit

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.equilibrium import FluxSurfaces, MagneticEquilibrium
from freeofcharge.kernels.biot_savart_polylines import kernel as biot_savart_polylines
from freeofcharge.kernels.segment_distance import kernel as segment_distance

__all__ = [
    "CoilField",
    "CoilSet",
    "PlasmaBoundary",
    "blocked_fraction",
    "injection_access",
    "normal_field_error",
]

MU_0 = 4.0e-7 * np.pi * 1.00000000055
"""Vacuum permeability [H/m] (CODATA 2018)."""

_MAX_PAIRS = 2_000_000
"""Most point–segment pairs held at once."""


def _dot(a, b):
    return xp.sum(a * b, axis=-1)


def _chunks(n: int, m: int):
    """Slices of ``n`` items such that each slice times ``m`` stays below the limit."""
    size = max(1, _MAX_PAIRS // max(1, m))
    return (slice(lo, min(lo + size, n)) for lo in range(0, n, size))


def _sphere_entry(w, d, r2):
    """Distance along unit ``d`` from ``w`` (relative to the centre) into a ball."""
    b = _dot(w, d)
    c = _dot(w, w) - r2
    disc = b * b - c
    root = xp.sqrt(xp.maximum(disc, 0.0))
    t_in, t_out = -b - root, -b + root
    t = xp.where(t_in >= 0.0, t_in, xp.where(t_out >= 0.0, 0.0, xp.inf))
    return xp.where(disc >= 0.0, t, xp.inf)


def _cylinder_entry(w, d, axis, length, r2):
    """Distance along unit ``d`` into the finite cylinder around ``axis`` (unit) ``[0, length]``."""
    wa, da = _dot(w, axis), _dot(d, axis)
    w_perp = w - wa[..., None] * axis
    d_perp = d - da[..., None] * axis
    a = _dot(d_perp, d_perp)
    b = _dot(w_perp, d_perp)
    c = _dot(w_perp, w_perp) - r2
    safe = xp.where(a > 0.0, a, 1.0)
    disc = b * b - a * c
    root = xp.sqrt(xp.maximum(disc, 0.0))
    t_in, t_out = (-b - root) / safe, (-b + root) / safe
    parallel = a <= 0.0
    # parallel to the axis: inside the infinite cylinder for all t or never
    t_in = xp.where(parallel, xp.where(c <= 0.0, -xp.inf, xp.inf), t_in)
    t_out = xp.where(parallel, xp.where(c <= 0.0, xp.inf, -xp.inf), t_out)
    t = xp.where(t_in >= 0.0, t_in, xp.where(t_out >= 0.0, 0.0, xp.inf))
    t = xp.where(parallel | (disc >= 0.0), t, xp.inf)
    finite_t = xp.where(xp.isfinite(t), t, 0.0)
    u = wa + finite_t * da
    return xp.where((u >= 0.0) & (u <= length), t, xp.inf)


def _segment_distance(p0, u, q0, v):
    """Shortest distance between the segments ``p0 + s u`` and ``q0 + t v``, ``s, t`` in [0, 1].

    The closest points as in Ericson, Real-Time Collision Detection (2005), 5.1.9.
    """
    r = p0 - q0
    a, e = _dot(u, u), _dot(v, v)
    f, c, b = _dot(v, r), _dot(u, r), _dot(u, v)
    denom = a * e - b * b
    safe_a = xp.where(a > 0.0, a, 1.0)
    safe_e = xp.where(e > 0.0, e, 1.0)
    s = xp.where(
        denom > 1e-14 * a * e, (b * f - c * e) / xp.where(denom > 0, denom, 1.0), 0.0
    )
    s = xp.clip(s, 0.0, 1.0)
    t = (b * s + f) / safe_e
    s = xp.where(t < 0.0, xp.clip(-c / safe_a, 0.0, 1.0), s)
    s = xp.where(t > 1.0, xp.clip((b - c) / safe_a, 0.0, 1.0), s)
    t = xp.clip(t, 0.0, 1.0)
    gap = p0 + s[..., None] * u - q0 - t[..., None] * v
    return xp.sqrt(_dot(gap, gap))


class CoilSet:
    """Closed coil filaments with their currents.

    Args:
        filaments: Points along each coil, ``(n_coils, n_points, 3)`` [m]; each
            polyline closes from its last point back to the first.
        currents: One current per coil ``(n_coils,)`` [A].
        nfp: Number of field periods (only stored).
    """

    def __init__(self, filaments, currents, *, nfp: int = 1) -> None:
        """Check the shapes and store the segments."""
        filaments = xp.asarray(filaments, dtype=float)
        currents = xp.atleast_1d(xp.asarray(currents, dtype=float))
        if filaments.ndim != 3 or filaments.shape[-1] != 3 or filaments.shape[1] < 3:
            raise ValueError(
                "filaments must be (n_coils, n_points >= 3, 3), "
                f"got {tuple(filaments.shape)}"
            )
        if currents.shape != (filaments.shape[0],):
            raise ValueError(
                f"need one current per coil ({filaments.shape[0]}), got {tuple(currents.shape)}"
            )
        self.filaments = filaments
        self.currents = currents
        self.nfp = int(nfp)
        ends = xp.roll(filaments, -1, axis=1)
        n_points = filaments.shape[1]
        self.starts = filaments.reshape(-1, 3)
        self.ends = ends.reshape(-1, 3)
        self.segment_currents = xp.repeat(currents, n_points)
        self.segment_coils = xp.repeat(xp.arange(filaments.shape[0]), n_points)
        # for the kernels: nodes and edges component-major (3, m), C-ordered
        edges = self.ends - self.starts
        self._nodes = xp.array(self.starts.T, order="C")
        self._edges = xp.array(edges.T, order="C")
        self._inverse_length2 = 1.0 / xp.maximum(_dot(edges, edges), 1e-300)

    @property
    def n_coils(self) -> int:
        """Return the number of coils."""
        return int(self.filaments.shape[0])

    @property
    def lengths(self):
        """Return the length of each coil's polyline [m]."""
        steps = xp.roll(self.filaments, -1, axis=1) - self.filaments
        return xp.sum(xp.sqrt(_dot(steps, steps)), axis=1)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> CoilSet:
        """Return the coils of a Dataset from ``plasma_plots.from_coilset``.

        It has ``X``, ``Y``, ``Z`` over ``(coil, s)`` and ``current`` over ``coil``.
        """
        points = np.stack(
            [dataset[axis].transpose("coil", "s").values for axis in "XYZ"], axis=-1
        )
        return cls(
            points,
            dataset["current"].values,
            nfp=int(dataset.attrs.get("nfp", 1)),
        )

    @classmethod
    def from_coilset(cls, coilset, *, n_points: int = 256) -> CoilSet:
        """Return every coil of a coilset in CoilStellaration's layout.

        Expanded over the field periods and the stellarator symmetry by
        ``plasma_plots.from_coilset`` (the optional extra ``coils``), which takes a
        dict, its JSON string or a DESC ``CoilSet``.

        Args:
            coilset: The coilset.
            n_points: Points along each coil.
        """
        try:
            import plasma_plots
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise ImportError(
                "reading coilsets needs plasma-plots: pip install freeofcharge[coils]"
            ) from error
        return cls.from_dataset(plasma_plots.from_coilset(coilset, n_points=n_points))

    @classmethod
    def planar_circular(
        cls,
        n_coils: int,
        major_radius: float,
        coil_radius: float,
        current: float,
        *,
        n_points: int = 128,
        offset: float = 0.0,
    ) -> CoilSet:
        r"""Return ``n_coils`` planar circular coils around a circular torus.

        Coil $k$ lies in the plane $\varphi_k = $ ``offset`` $+ 2\pi k / N$, centred on
        $R$ = ``major_radius``, with radius ``coil_radius``; a positive current gives
        a field along $+\varphi$ inside. The lengths take quantities.
        """
        major_radius = to_unit(major_radius, "m", "major_radius")
        coil_radius = to_unit(coil_radius, "m", "coil_radius")
        offset = to_unit(offset, "rad", "offset")
        phi = offset + 2.0 * np.pi * np.arange(n_coils) / n_coils
        s = 2.0 * np.pi * np.arange(n_points) / n_points
        r = major_radius + coil_radius * np.cos(s)
        z = coil_radius * np.sin(s)
        points = np.stack(
            [
                r[None, :] * np.cos(phi)[:, None],
                r[None, :] * np.sin(phi)[:, None],
                np.broadcast_to(z, (n_coils, n_points)),
            ],
            axis=-1,
        )
        # (R, Z) circulating counter-clockwise seen along +phi drives B along +phi
        return cls(points[:, ::-1], np.full(n_coils, float(current)), nfp=n_coils)

    @staticmethod
    def _kernel_points(points):
        """Return ``points`` ``(..., 3)`` as a new C-ordered ``(3, n)`` array, and their shape.

        A copy, not ``ascontiguousarray``: the transpose of ``(1, 3)`` is flagged
        contiguous but keeps strides that Pyccel's order check rejects.
        """
        points = xp.asarray(points, dtype=float)
        return xp.array(points.reshape(-1, 3).T, order="C"), points.shape[:-1]

    @method(
        equations={
            "biot_savart": r"\mathbf{B}(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_k I_k \oint \frac{d\mathbf{l}_k \times (\mathbf{x} - \mathbf{l}_k)}{|\mathbf{x} - \mathbf{l}_k|^3}",
            "segment": r"\mathbf{B}_\mathrm{seg} = \frac{\mu_0 I}{4\pi}\, \frac{r_a + r_b}{r_a r_b\,(r_a r_b + \mathbf{r}_a \cdot \mathbf{r}_b)}\; \mathbf{r}_a \times \mathbf{r}_b, \qquad \mathbf{r}_{a,b} = \mathbf{x} - \mathbf{x}_{a,b}",
        },
        assumptions=(
            r"filaments without cross-section: straight segments between the stored points, exact per segment (Hanson and Hirshman, Phys. Plasmas 9, 4410 (2002))",
            r"a segment adds nothing at its own end points (the field diverges on a filament)",
        ),
    )
    @sp.profile("freeofcharge.coil_field", tags=PROFILE_TAGS)
    def field_at(self, points):
        r"""Return the coils' vacuum field $\mathbf{B}$ [T] at Cartesian ``points`` ``(..., 3)``."""
        points, shape = self._kernel_points(points)
        field = xp.zeros_like(points)
        if points.shape[1]:
            biot_savart_polylines(
                points, self._nodes, self.filaments.shape[1], self.currents, field
            )
        return (MU_0 / (4.0 * np.pi) * field.T).reshape(*shape, 3)

    @sp.profile("freeofcharge.coil_distance", tags=PROFILE_TAGS)
    def distance_at(self, points):
        """Return the distance of Cartesian ``points`` ``(..., 3)`` to the nearest filament [m]."""
        points, shape = self._kernel_points(points)
        result = xp.zeros(points.shape[1])
        if points.shape[1]:
            segment_distance(
                points, self._nodes, self._edges, self._inverse_length2, result
            )
        return result.reshape(shape)

    @method(
        equations={
            "capsule": r"C_j = \{\mathbf{x} : \min_{0 \le u \le 1} |\mathbf{x} - \mathbf{a}_j - u(\mathbf{b}_j - \mathbf{a}_j)| \le r\}",
            "hit": r"t^* = \min_j \min\{t \ge 0 : \mathbf{x}_0 + t\hat d \in C_j\}",
        },
        assumptions=(
            r"each coil a tube of circular cross-section $r$ around its polyline: the union of the capsules of its segments",
            r"straight lines; a start inside a tube hits at $t = 0$",
        ),
    )
    def first_hit(self, starts, directions, radius):
        """Return where lines first enter a coil tube.

        Args:
            starts: Start points ``(N, 3)`` [m].
            directions: Directions ``(N, 3)``, normalized here.
            radius: Tube radius around the filaments [m]; takes quantities.

        Returns:
            The distance along each line to the first tube, ``inf`` for none, and
            the index of that coil, -1 for none.
        """
        radius = to_unit(radius, "m", "radius")
        starts = xp.asarray(starts, dtype=float).reshape(-1, 3)
        directions = xp.asarray(directions, dtype=float).reshape(-1, 3)
        directions = directions / xp.sqrt(_dot(directions, directions))[:, None]
        edge = self.ends - self.starts
        length = xp.sqrt(_dot(edge, edge))
        axis = edge / xp.maximum(length, 1e-300)[:, None]
        r2 = radius * radius
        distance = xp.full(starts.shape[0], xp.inf)
        coil = xp.full(starts.shape[0], -1, dtype=int)
        for part in _chunks(starts.shape[0], edge.shape[0]):
            w = starts[part, None, :] - self.starts[None]
            d = xp.broadcast_to(directions[part, None, :], w.shape)
            t = xp.minimum(
                _sphere_entry(w, d, r2),
                _cylinder_entry(w, d, axis[None], length[None], r2),
            )
            nearest = xp.argmin(t, axis=1)
            best = xp.take_along_axis(t, nearest[:, None], axis=1)[:, 0]
            distance[part] = best
            coil[part] = xp.where(xp.isfinite(best), self.segment_coils[nearest], -1)
        return distance, coil

    @method(
        equations={
            "clearance": r"c = \min_j \min_{0 \le s \le L,\; 0 \le u \le 1} |\mathbf{x}_0 + s\hat d - \mathbf{a}_j - u(\mathbf{b}_j - \mathbf{a}_j)|",
        },
        assumptions=(
            r"closest points of two segments as in Ericson, Real-Time Collision Detection (2005), 5.1.9",
        ),
    )
    def clearance(self, starts, directions, lengths):
        """Return the shortest distance between line segments and the filaments [m].

        Args:
            starts: Start points ``(N, 3)`` [m].
            directions: Directions ``(N, 3)``, normalized here.
            lengths: Length of each segment ``(N,)`` or one for all [m].
        """
        starts = xp.asarray(starts, dtype=float).reshape(-1, 3)
        directions = xp.asarray(directions, dtype=float).reshape(-1, 3)
        directions = directions / xp.sqrt(_dot(directions, directions))[:, None]
        lengths = xp.broadcast_to(xp.asarray(lengths, dtype=float), starts.shape[:1])
        steps = directions * lengths[:, None]
        edge = self.ends - self.starts
        result = xp.empty(starts.shape[0])
        for part in _chunks(starts.shape[0], edge.shape[0]):
            gaps = _segment_distance(
                starts[part, None, :],
                steps[part, None, :],
                self.starts[None],
                edge[None],
            )
            result[part] = xp.min(gaps, axis=1)
        return result


@method(
    equations={
        "error": r"\epsilon = \frac{|\mathbf{B}_\mathrm{coils} \cdot \hat n|}{|\mathbf{B}_\mathrm{coils}|}, \qquad \hat n \propto \partial_\theta \mathbf{x} \times \partial_\zeta \mathbf{x}",
    },
    assumptions=(
        r"the surface on a uniform, periodic $(\theta, \zeta)$ grid; tangents from central differences",
    ),
    components=(CoilSet.field_at,),
)
def normal_field_error(coils: CoilSet, surface):
    r"""Return $|\mathbf{B}\cdot\hat n| / |\mathbf{B}|$ of the coils on a closed surface.

    The mean over a uniform grid is CoilStellaration's
    ``normalized_field_error/mean``.

    Args:
        coils: The coils.
        surface: Points ``(n_theta, n_zeta, 3)`` [m] periodic in both directions
            over the whole torus, or a Dataset from ``plasma_plots.from_boundary``
            with ``zeta`` over the whole torus.

    Returns:
        The error on the grid ``(n_theta, n_zeta)``; a DataArray for a Dataset.
    """
    dataset = surface if isinstance(surface, xr.Dataset) else None
    if dataset is not None:
        points = np.stack(
            [
                dataset[axis].isel(rho=0).transpose("theta", "zeta").values
                for axis in "XYZ"
            ],
            axis=-1,
        )
    else:
        points = xp.to_numpy(xp.asarray(surface, dtype=float))
    tangents = [np.roll(points, -1, axis) - np.roll(points, 1, axis) for axis in (0, 1)]
    normal = np.cross(tangents[0], tangents[1])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    field = xp.to_numpy(coils.field_at(points))
    error = np.abs(np.sum(field * normal, axis=-1)) / np.linalg.norm(field, axis=-1)
    if dataset is None:
        return xp.asarray(error)
    return xr.DataArray(
        error,
        dims=("theta", "zeta"),
        coords={"theta": dataset.theta, "zeta": dataset.zeta},
        attrs={"label": r"$|B_n| / |B|$", "long_name": "normal field error"},
    )


class CoilField:
    r"""An equilibrium inside its last closed surface and the coils' field outside.

    A `freeofcharge.MagneticEquilibrium`: $\rho$, the boundary, the wall and the
    flux surfaces are the equilibrium's; $\mathbf{B}$ is the equilibrium's inside
    and the coils' vacuum field outside (or everywhere, ``inside="coils"``). At
    $\rho = 1$ the two differ by the coils' field error and the plasma's own
    currents; `jump` measures it.

    Args:
        equilibrium: The plasma equilibrium.
        coils: The coils.
        inside: ``"equilibrium"`` or ``"coils"``: the field inside.
        current_scale: Factor on the coil currents, e.g. to match the
            equilibrium's field strength.
    """

    def __init__(
        self,
        equilibrium: MagneticEquilibrium,
        coils: CoilSet,
        *,
        inside: str = "equilibrium",
        current_scale: float = 1.0,
    ) -> None:
        """Store the parts."""
        if inside not in ("equilibrium", "coils"):
            raise ValueError(f"inside must be 'equilibrium' or 'coils', got {inside!r}")
        self.equilibrium = equilibrium
        self.coils = coils
        self.inside = inside
        self.current_scale = float(current_scale)

    @property
    def has_wall(self) -> bool:
        """Return whether the equilibrium has a first wall."""
        return self.equilibrium.has_wall

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the equilibrium's box [m]."""
        return self.equilibrium.bounds

    def rho_at(self, points):
        r"""Return the equilibrium's $\rho$."""
        return self.equilibrium.rho_at(points)

    def inside_at(self, points):
        """Return whether the points are inside the last closed surface."""
        return self.equilibrium.inside_at(points)

    def inside_wall_at(self, points):
        """Return whether the points are inside the first wall."""
        return self.equilibrium.inside_wall_at(points)

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the equilibrium's shells."""
        return self.equilibrium.flux_surfaces(rho_edges, **kwargs)

    def coil_field_at(self, points):
        r"""Return the coils' $\mathbf{B}$ [T], scaled by ``current_scale``."""
        return self.current_scale * self.coils.field_at(points)

    def field_at(self, points):
        r"""Return $\mathbf{B}$ [T]: the equilibrium's inside, the coils' outside."""
        points = xp.asarray(points, dtype=float)
        if self.inside == "coils":
            return self.coil_field_at(points)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        inside = xp.asarray(self.equilibrium.inside_at(flat)).reshape(-1)
        field = xp.zeros_like(flat)
        if bool(xp.any(inside)):
            field[inside] = xp.asarray(self.equilibrium.field_at(flat[inside])).reshape(
                -1, 3
            )
        if not bool(xp.all(inside)):
            field[~inside] = self.coil_field_at(flat[~inside])
        return field.reshape(*shape, 3)

    def jump(self, points):
        r"""Return $|\mathbf{B}_\mathrm{coils} - \mathbf{B}_\mathrm{eq}| / |\mathbf{B}_\mathrm{eq}|$ at ``points``.

        On points of the last closed surface this is the jump crossing it.
        """
        inner = xp.asarray(self.equilibrium.field_at(points))
        difference = self.coil_field_at(points) - inner
        return xp.sqrt(_dot(difference, difference) / _dot(inner, inner))


class PlasmaBoundary:
    r"""A plasma boundary as a Fourier surface, ConStellaration's ``SurfaceRZFourier``.

    $R = \sum_{m,n} r^c_{mn} \cos(m\theta - N_\mathrm{fp} n\varphi)$ (plus $r^s_{mn}$ sines)
    and $Z = \sum_{m,n} z^s_{mn} \sin(m\theta - N_\mathrm{fp} n\varphi)$ (plus $z^c_{mn}$
    cosines), $m = 0 \dots M$ along the first axis and $n = -N \dots N$ along the second,
    $\varphi$ the cylindrical angle. It tells whether points are inside, which is all
    `blocked_fraction` and `injection_access` need when there is no equilibrium.

    Args:
        r_cos: ``(M + 1, 2N + 1)`` [m].
        z_sin: ``(M + 1, 2N + 1)`` [m].
        n_field_periods: Number of field periods.
        r_sin: Optional, as ``r_cos``.
        z_cos: Optional, as ``r_cos``.
        n_theta: Points of the cross-section polygon.
    """

    def __init__(
        self,
        r_cos,
        z_sin,
        n_field_periods: int,
        *,
        r_sin=None,
        z_cos=None,
        n_theta: int = 256,
    ) -> None:
        """Check the shapes and cache a box around the surface."""
        self.r_cos = np.asarray(r_cos, dtype=float)
        self.z_sin = np.asarray(z_sin, dtype=float)
        if (
            self.r_cos.ndim != 2
            or self.r_cos.shape[1] % 2 != 1
            or self.z_sin.shape != self.r_cos.shape
        ):
            raise ValueError(
                "r_cos and z_sin must be (M + 1, 2N + 1) arrays of one shape, "
                f"got {self.r_cos.shape} and {self.z_sin.shape}"
            )
        self.r_sin = None if r_sin is None else np.asarray(r_sin, dtype=float)
        self.z_cos = None if z_cos is None else np.asarray(z_cos, dtype=float)
        self.nfp = int(n_field_periods)
        self.n_theta = int(n_theta)
        theta = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        phi = np.linspace(0.0, 2 * np.pi / self.nfp, 32, endpoint=False)
        r, z = self.rz(theta[:, None], phi[None, :])
        r_max = float(r.max())
        self._bounds = (
            np.array([-r_max, -r_max, float(z.min())]),
            np.array([r_max, r_max, float(z.max())]),
        )

    @classmethod
    def from_boundary(cls, boundary, **kwargs) -> PlasmaBoundary:
        """Return the surface of ConStellaration's ``boundary.json`` (a dict or its JSON string)."""
        if isinstance(boundary, (str, bytes)):
            boundary = json.loads(boundary)
        return cls(
            boundary["r_cos"],
            boundary["z_sin"],
            boundary["n_field_periods"],
            r_sin=boundary.get("r_sin"),
            z_cos=boundary.get("z_cos"),
            **kwargs,
        )

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box holding the surface [m]."""
        return self._bounds

    def rz(self, theta, phi):
        """Return $R$ and $Z$ [m] at broadcast angles ``theta`` and ``phi``."""
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
        n_tor = (self.r_cos.shape[1] - 1) // 2
        m = np.arange(self.r_cos.shape[0])[:, None]
        n = np.arange(-n_tor, n_tor + 1)[None, :]
        angle = m * theta[..., None, None] - self.nfp * n * phi[..., None, None]
        cos, sin = np.cos(angle), np.sin(angle)
        r = np.sum(self.r_cos * cos, axis=(-2, -1))
        z = np.sum(self.z_sin * sin, axis=(-2, -1))
        if self.r_sin is not None:
            r = r + np.sum(self.r_sin * sin, axis=(-2, -1))
        if self.z_cos is not None:
            z = z + np.sum(self.z_cos * cos, axis=(-2, -1))
        return r, z

    @method(
        equations={
            "inside": r"(R, Z) \text{ inside the polygon } \{(R(\theta_j, \varphi), Z(\theta_j, \varphi))\}_{j}",
        },
        assumptions=(
            r"each cross-section at constant $\varphi$ a simple closed curve, resolved by `n_theta` points; even–odd rule",
        ),
    )
    def inside_at(self, points):
        """Return whether Cartesian ``points`` ``(..., 3)`` are inside the surface."""
        points = xp.to_numpy(xp.asarray(points, dtype=float))
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        r = np.hypot(flat[:, 0], flat[:, 1])
        z = flat[:, 2]
        phi = np.arctan2(flat[:, 1], flat[:, 0])
        lower, upper = self._bounds
        candidates = np.nonzero((r <= upper[0]) & (z >= lower[2]) & (z <= upper[2]))[0]
        result = np.zeros(flat.shape[0], dtype=bool)
        theta = np.linspace(0.0, 2 * np.pi, self.n_theta, endpoint=False)
        size = max(1, _MAX_PAIRS // (self.n_theta * self.r_cos.size))
        for lo in range(0, candidates.size, size):
            rows = candidates[lo : lo + size]
            r0, z0 = self.rz(theta[None, :], phi[rows, None])
            r1, z1 = np.roll(r0, -1, axis=1), np.roll(z0, -1, axis=1)
            zp, rp = z[rows, None], r[rows, None]
            crosses = (z0 > zp) != (z1 > zp)
            safe = np.where(z1 != z0, z1 - z0, 1.0)
            r_cut = r0 + (zp - z0) * (r1 - r0) / safe
            result[rows] = np.sum(crosses & (rp < r_cut), axis=1) % 2 == 1
        return xp.asarray(result.reshape(shape))


def _entry(
    inside: Callable, starts, directions, max_length: float, step: float
) -> np.ndarray:
    """Distance along each line to the first point inside, ``inf`` for none.

    Sampled every ``step`` and bisected to a thousandth of it.
    """
    s = xp.arange(0.0, max_length + step, step)
    points = starts[:, None, :] + s[None, :, None] * directions[:, None, :]
    flags = xp.asarray(inside(points)).reshape(points.shape[:2])
    has = xp.any(flags, axis=1)
    first = xp.argmax(flags, axis=1)
    hi = s[first]
    lo = xp.maximum(hi - step, 0.0)
    rows = xp.nonzero(has & (first > 0))[0]
    for _ in range(10):
        mid = 0.5 * (lo[rows] + hi[rows])
        flag = xp.asarray(
            inside(starts[rows] + mid[:, None] * directions[rows])
        ).reshape(-1)
        hi[rows] = xp.where(flag, mid, hi[rows])
        lo[rows] = xp.where(flag, lo[rows], mid)
    return xp.where(has, hi, xp.inf)


@method(
    equations={
        "blocked": r"f = \frac{\sum_i w_i\, [t_{\mathrm{coil},i} < t_{\mathrm{plasma},i}]}{\sum_i w_i}",
    },
    assumptions=(
        r"straight marker lines from their start positions; the plasma entry sampled every `step` and bisected",
        r"markers that miss the plasma count as blocked when they hit a coil anywhere",
    ),
    components=(CoilSet.first_hit,),
)
def blocked_fraction(
    beam,
    coils: CoilSet,
    equilibrium: MagneticEquilibrium | PlasmaBoundary,
    radius: float,
    *,
    max_length: float = 20.0,
    step: float = 0.02,
) -> dict:
    """Return the share of a beam that hits a coil tube before the plasma.

    Args:
        beam: A `freeofcharge.NeutralBeam`.
        coils: The coils.
        equilibrium: The plasma's last closed surface: an equilibrium or a
            `PlasmaBoundary`.
        radius: Coil tube radius [m]; takes quantities.
        max_length: Longest line followed [m].
        step: Sampling step for the plasma entry [m].

    Returns:
        ``fraction`` (of the weight, on this rank), and per marker ``coil_distance``,
        ``plasma_distance`` (``inf`` for none), ``coil`` (-1 for none) and ``blocked``.
    """
    starts = xp.asarray(beam.markers.positions, dtype=float)
    directions = xp.asarray(beam.directions, dtype=float)
    coil_distance, coil = coils.first_hit(starts, directions, radius)
    plasma_distance = _entry(
        equilibrium.inside_at, starts, directions, max_length, step
    )
    blocked = coil_distance < xp.minimum(plasma_distance, max_length)
    weights = xp.asarray(beam.markers.weights, dtype=float)
    total = float(xp.sum(weights))
    fraction = float(xp.sum(xp.where(blocked, weights, 0.0))) / total if total else 0.0
    return {
        "fraction": fraction,
        "coil_distance": coil_distance,
        "plasma_distance": plasma_distance,
        "coil": coil,
        "blocked": blocked,
    }


@method(
    equations={
        "axis": r"\mathbf{x}(t) = \mathbf{x}_\mathrm{tan} + (t - L)\,\hat d, \qquad \hat d = \cos\alpha\, \hat\varphi + \sin\alpha\, \hat z",
        "clearance": r"c = \min_{0 \le t \le t_\mathrm{plasma}} d(\mathbf{x}(t), \text{filaments}) - r_\mathrm{coil}",
        "admissible": r"c \ge r_\mathrm{beam}",
    },
    assumptions=(
        r"the beam a cylinder of radius $r_\mathrm{beam}$ around its axis up to the plasma entry, no divergence",
        r"the axis passes through the point at radius $R_\mathrm{tan}$ in the plane $Z$ = `height`, there tangent to the circle $R = R_\mathrm{tan}$, tilted by $\alpha$ out of the horizontal plane",
    ),
    components=(CoilSet.clearance,),
)
def injection_access(
    coils: CoilSet,
    equilibrium: MagneticEquilibrium | PlasmaBoundary,
    tangency_radii,
    toroidal_angles,
    vertical_angles=(0.0,),
    *,
    start_distance: float,
    coil_radius: float,
    beam_radius: float = 0.0,
    height: float = 0.0,
    sense: int = 1,
    step: float = 0.02,
) -> xr.Dataset:
    r"""Return which injection lines reach the plasma between the coils.

    Each axis is tangent to the circle $R$ = tangency radius at the toroidal angle
    and the height ``height``, tilted by the vertical angle, and starts
    ``start_distance`` before the tangency point. Its clearance is the distance
    from the axis, up to where it enters the plasma, to the nearest filament minus
    ``coil_radius``; the line is admissible if that is at least ``beam_radius``
    and it reaches the plasma.

    Args:
        coils: The coils.
        equilibrium: The plasma's last closed surface: an equilibrium or a
            `PlasmaBoundary`.
        tangency_radii: Tangency radii [m].
        toroidal_angles: Toroidal angles of the tangency points [rad].
        vertical_angles: Tilts out of the horizontal plane [rad].
        start_distance: Distance from the start to the tangency point [m].
        coil_radius: Coil tube radius [m].
        beam_radius: Beam radius [m].
        height: Height of the tangency points [m].
        sense: ``+1`` towards $+\varphi$ at the tangency point, ``-1`` against.
        step: Sampling step for the plasma entry [m].

    Returns:
        A Dataset over ``(tangency_radius, toroidal_angle, vertical_angle)``:
        ``clearance`` [m], ``plasma_distance`` (from the start, ``inf`` for none)
        [m] and ``admissible``.
    """
    start_distance = to_unit(start_distance, "m", "start_distance")
    coil_radius = to_unit(coil_radius, "m", "coil_radius")
    beam_radius = to_unit(beam_radius, "m", "beam_radius")
    height = to_unit(height, "m", "height")
    if sense not in (1, -1):
        raise ValueError("sense must be +1 or -1")
    radii = np.atleast_1d(np.asarray(tangency_radii, dtype=float))
    phis = np.atleast_1d(np.asarray(toroidal_angles, dtype=float))
    alphas = np.atleast_1d(np.asarray(vertical_angles, dtype=float))
    r, phi, alpha = np.meshgrid(radii, phis, alphas, indexing="ij")
    tangent = np.stack([r * np.cos(phi), r * np.sin(phi), np.full_like(r, height)], -1)
    direction = np.stack(
        [
            -sense * np.cos(alpha) * np.sin(phi),
            sense * np.cos(alpha) * np.cos(phi),
            np.sin(alpha),
        ],
        -1,
    )
    starts = xp.asarray((tangent - start_distance * direction).reshape(-1, 3))
    directions = xp.asarray(direction.reshape(-1, 3))
    max_length = 2.0 * start_distance
    plasma = _entry(equilibrium.inside_at, starts, directions, max_length, step)
    reach = xp.where(xp.isfinite(plasma), plasma, max_length)
    clearance = coils.clearance(starts, directions, reach) - coil_radius
    admissible = xp.isfinite(plasma) & (clearance >= beam_radius)
    dims = ("tangency_radius", "toroidal_angle", "vertical_angle")

    def grid(values):
        return xp.to_numpy(values).reshape(r.shape)

    return xr.Dataset(
        {
            "clearance": (
                dims,
                grid(clearance),
                {"units": "m", "long_name": "clearance to the coils"},
            ),
            "plasma_distance": (
                dims,
                grid(plasma),
                {"units": "m", "long_name": "distance to the plasma"},
            ),
            "admissible": (
                dims,
                grid(admissible),
                {"long_name": "reaches the plasma between the coils"},
            ),
        },
        coords={
            "tangency_radius": ("tangency_radius", radii, {"units": "m"}),
            "toroidal_angle": ("toroidal_angle", phis, {"units": "rad"}),
            "vertical_angle": ("vertical_angle", alphas, {"units": "rad"}),
        },
        attrs={
            "coil_radius": coil_radius,
            "beam_radius": beam_radius,
            "start_distance": start_distance,
            "height": height,
            "sense": sense,
        },
    )
