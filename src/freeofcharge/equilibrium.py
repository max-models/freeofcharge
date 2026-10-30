r"""Equilibria: the protocol the beam codes use, and the axisymmetric one.

:class:`MagneticEquilibrium` is what deposition and slowing down need from an
equilibrium, at Cartesian points ``(..., 3)``: the radius ``rho``, the field,
inside the plasma and the wall, a bounding box, and the shells between radii.
:class:`Equilibrium` is the axisymmetric implementation; 3D ones (GVEC, DESC)
are adapters in their own modules.

The axisymmetric field is

.. math::

    B_R = -\frac{1}{R}\frac{\partial\psi}{\partial Z}, \qquad
    B_Z = \frac{1}{R}\frac{\partial\psi}{\partial R}, \qquad
    B_\phi = \frac{F(\psi)}{R},

with ``psi`` the poloidal flux per radian [Wb/rad] and ``(R, phi, Z)``
right-handed. G-EQDSK files from different codes differ in these signs (the
COCOS convention); :func:`Equilibrium.from_geqdsk` takes the file as it is.
Radii are ``rho = sqrt(psi_n)``, ``psi_n = (psi - psi_axis) / (psi_boundary -
psi_axis)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cunumpy as xp
import numpy as np
from scipy.interpolate import RectBivariateSpline
from simgeometry.grid import Grid
from simgeometry.objects.polygon import Polygon

from freeofcharge import geqdsk

__all__ = ["Equilibrium", "FluxSurfaces", "MagneticEquilibrium"]


class _StarShapedPolygon:
    """Point-in-polygon for a polygon that is star-shaped about ``centre``.

    Every ray from the centre crosses the boundary once, so a point is inside
    exactly when it is on the centre's side of the one edge its angle falls on:
    a binary search over the vertex angles instead of a ray-casting loop over
    all edges, with the same answer away from the boundary itself.
    """

    def __init__(self, vertices: np.ndarray, centre: tuple[float, float]) -> None:
        cx, cy = centre
        x, y = vertices[:, 0], vertices[:, 1]
        if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0.0:
            vertices = vertices[::-1]  # counter-clockwise
        theta = np.mod(np.arctan2(vertices[:, 1] - cy, vertices[:, 0] - cx), 2 * np.pi)
        start = int(np.argmin(theta))
        vertices = np.roll(vertices, -start, axis=0)
        theta = np.roll(theta, -start)
        if not np.all(np.diff(theta) > 0.0):
            raise ValueError("the polygon is not star-shaped about the centre")
        self.centre = (float(cx), float(cy))
        self._theta = xp.asarray(theta)
        edge = np.roll(vertices, -1, axis=0) - vertices
        self._ax, self._ay = xp.asarray(vertices[:, 0]), xp.asarray(vertices[:, 1])
        self._ex, self._ey = xp.asarray(edge[:, 0]), xp.asarray(edge[:, 1])

    @classmethod
    def try_build(cls, vertices, centre) -> _StarShapedPolygon | None:
        """Return the fast test, or None when the polygon is not star-shaped."""
        try:
            return cls(np.asarray(vertices, dtype=float), centre)
        except ValueError:
            return None

    def contains(self, r, z):
        """Return whether the points are strictly inside (same shape as ``r``)."""
        cx, cy = self.centre
        theta = xp.mod(xp.arctan2(z - cy, r - cx), 2.0 * np.pi)
        # edge k runs from vertex k to k + 1; angles below vertex 0 belong to the
        # closing edge from the last vertex
        k = xp.searchsorted(self._theta, theta, side="right") - 1
        k[k < 0] = self._theta.shape[0] - 1
        return self._ex[k] * (z - self._ay[k]) - self._ey[k] * (r - self._ax[k]) > 0.0


@dataclass(frozen=True)
class FluxSurfaces:
    """Volumes and shapes of the shells between the radii ``rho_edges``."""

    rho_edges: np.ndarray
    volume: np.ndarray
    area: np.ndarray
    inverse_aspect_ratio: np.ndarray
    field_strength: np.ndarray | None = field(default=None, repr=False)
    surface_weight: np.ndarray | None = field(default=None, repr=False)
    safety_factor: np.ndarray | None = None
    major_radius: np.ndarray | None = None
    """Shell edges in ``rho``, increasing from 0 to 1."""
    """Volume of each shell [m^3]."""
    """Poloidal cross-section area of each shell [m^2]."""
    """``(R_max - R_min) / (R_max + R_min)`` of the surface at each shell centre
    (averaged over the toroidal angle in 3D)."""
    """``|B|`` [T] at points on the surface at each shell centre, shape
    ``(shells, points)``: around it in 2D, on a ``(theta, zeta)`` grid in 3D."""
    """Weights of those points in the flux-surface average, normalised to a sum
    of one: ``dl_p / B_p`` in 2D (``dl_p`` the poloidal arc length), the
    Jacobian in 3D."""
    """``q = 1 / iota`` at each shell centre; in 2D ``F / (2 pi) * oint dl_p /
    (R^2 B_p)``."""
    """``(R_max + R_min) / 2`` of the surface at each shell centre [m]."""

    @property
    def rho(self) -> np.ndarray:
        """Return the shell centres."""
        return 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])

    @classmethod
    def from_flux_coordinates(
        cls,
        shape,
        nfp: int,
        rho_edges,
        field_points: tuple[int, int] = (32, 32),
        radial_points: int = 8,
    ) -> FluxSurfaces:
        """Return the shells of a 3D equilibrium in flux coordinates.

        ``shape(rho, theta, zeta)`` returns, at flat arrays of flux coordinates,
        ``R``, ``dR/drho``, ``dR/dtheta``, ``dZ/drho``, ``dZ/dtheta``, ``|B|``
        and ``iota`` (``zeta`` the toroidal angle up to sign). Volumes and the
        toroidally averaged cross-section areas come from Gauss-Legendre
        quadrature in ``rho`` within each shell and the trapezoidal rule on a
        ``field_points`` ``(theta, zeta)`` grid over one of the ``nfp`` field
        periods; the field strength and the averaging weights (the Jacobian
        ``R dA``) from that grid on the surface through each shell centre.
        """
        rho_edges = np.asarray(rho_edges, dtype=float)
        n_theta, n_zeta = field_points
        period = 2.0 * np.pi / nfp
        theta, zeta = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False),
            np.linspace(0.0, period, n_zeta, endpoint=False),
            indexing="ij",
        )
        theta, zeta = theta.ravel(), zeta.ravel()
        cell = (2.0 * np.pi / n_theta) * (period / n_zeta)
        nodes, weights = np.polynomial.legendre.leggauss(radial_points)
        n_shells = len(rho_edges) - 1
        volume, area = np.zeros(n_shells), np.zeros(n_shells)
        for k in range(n_shells):
            lo, hi = rho_edges[k], rho_edges[k + 1]
            for node, weight in zip(nodes, weights, strict=True):
                rho = np.full(theta.size, 0.5 * (lo + hi) + 0.5 * (hi - lo) * node)
                r, r_r, r_t, z_r, z_t, *_ = shape(rho, theta, zeta)
                jac_rz = np.abs(r_r * z_t - r_t * z_r)
                factor = 0.5 * (hi - lo) * weight * cell
                volume[k] += factor * np.sum(r * jac_rz) * nfp
                area[k] += factor * np.sum(jac_rz) / period

        centres = 0.5 * (rho_edges[1:] + rho_edges[:-1])
        strength = np.empty((n_shells, theta.size))
        surface_weight = np.empty_like(strength)
        epsilon, major, iota = (np.empty(n_shells) for _ in range(3))
        for k, centre in enumerate(centres):
            rho = np.full(theta.size, centre)
            r, r_r, r_t, z_r, z_t, strength[k], iotas = shape(rho, theta, zeta)
            jacobian = np.abs(r * (r_r * z_t - r_t * z_r))
            surface_weight[k] = jacobian / jacobian.sum()
            slices = r.reshape(n_theta, n_zeta)
            r_min, r_max = slices.min(axis=0), slices.max(axis=0)
            epsilon[k] = np.mean((r_max - r_min) / (r_max + r_min))
            major[k] = np.mean(0.5 * (r_max + r_min))
            iota[k] = np.mean(iotas)
        return cls(
            rho_edges,
            volume,
            area,
            epsilon,
            field_strength=strength,
            surface_weight=surface_weight,
            safety_factor=1.0 / iota,
            major_radius=major,
        )

    @property
    def rotational_transform(self) -> np.ndarray | None:
        """Return ``iota = 1 / q`` at each shell centre."""
        return None if self.safety_factor is None else 1.0 / self.safety_factor

    def surface_average(self, values) -> np.ndarray:
        """Return the flux-surface average of ``values`` given at the field points."""
        if self.surface_weight is None:
            raise ValueError("these surfaces have no field points")
        return np.sum(self.surface_weight * values, axis=-1)

    @property
    def trapped_fraction(self) -> np.ndarray:
        r"""Return the trapped-particle fraction of each surface centre.

        .. math::

            f_t = 1 - \frac{3}{4} \langle B^2 \rangle \int_0^{1/B_\max}
                  \frac{\lambda\, d\lambda}{\langle\sqrt{1 - \lambda B}\rangle}

        (Lin-Liu and Miller, Phys. Plasmas 2, 1666, 1995), by Gauss-Legendre
        quadrature in ``lambda``.
        """
        if self.field_strength is None or self.surface_weight is None:
            raise ValueError("these surfaces have no field points")
        b = self.field_strength / self.field_strength.max(axis=1, keepdims=True)
        nodes, weights = np.polynomial.legendre.leggauss(64)
        # lambda B_max = 1 - s^2 resolves the square-root end point
        s = 0.5 * (nodes + 1.0)
        lam = 1.0 - s**2
        root = np.sqrt(np.maximum(1.0 - lam[None, :, None] * b[:, None, :], 0.0))
        average = np.sum(self.surface_weight[:, None, :] * root, axis=-1)
        integral = np.sum(0.5 * weights * lam * 2.0 * s / average, axis=-1)
        return 1.0 - 0.75 * self.surface_average(b**2) * integral

    def shell_index(self, rho):
        """Return the shell of each ``rho``, -1 outside ``[0, 1)``."""
        edges = xp.asarray(self.rho_edges)
        index = xp.searchsorted(edges, xp.asarray(rho), side="right") - 1
        return xp.where((index >= 0) & (index < len(self.rho_edges) - 1), index, -1)


@runtime_checkable
class MagneticEquilibrium(Protocol):
    """What the beam codes need from an equilibrium, at Cartesian ``points``.

    ``points`` has shape ``(..., 3)`` [m]; the results have its leading shape
    (``(..., 3)`` for the field).
    """

    @property
    def has_wall(self) -> bool:
        """Return whether there is a first wall (otherwise all points are inside)."""
        ...

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper corners of a box holding the plasma [m]."""
        ...

    def rho_at(self, points) -> Any:
        """Return the radius ``rho``, 0 on the axis and 1 on the boundary."""
        ...

    def field_at(self, points) -> Any:
        """Return ``B`` [T]."""
        ...

    def inside_at(self, points) -> Any:
        """Return whether the points are inside the last closed flux surface."""
        ...

    def inside_wall_at(self, points) -> Any:
        """Return whether the points are inside the first wall."""
        ...

    def flux_surfaces(self, rho_edges) -> FluxSurfaces:
        """Return the shells between the radii ``rho_edges``."""
        ...


def _cylindrical(points):
    """Return ``(R, Z)`` of Cartesian ``points`` ``(..., 3)``."""
    points = xp.asarray(points, dtype=float)
    return xp.hypot(points[..., 0], points[..., 1]), points[..., 2]


class Equilibrium:
    """The flux ``psi(R, Z)`` on a rectilinear grid, ``F(psi_n)``, boundary and wall.

    Args:
        r: Increasing major radii of the grid [m].
        z: Increasing heights of the grid [m].
        psi: Flux on the grid, shape ``(len(r), len(z))`` [Wb/rad].
        psi_axis: Flux on the magnetic axis.
        psi_boundary: Flux on the last closed flux surface.
        f: ``F = R B_phi`` on a uniform grid of ``psi_n`` from 0 to 1 [T m];
            the last value holds outside.
        axis: ``(R, Z)`` of the magnetic axis [m].
        boundary: Last closed flux surface, vertices ``(n, 2)`` in ``(R, Z)``.
        wall: First wall, vertices ``(n, 2)``, or None.
        method: Interpolation of ``psi`` and its derivatives
            (``RegularGridInterpolator`` methods: "linear", "cubic", ...).
    """

    def __init__(
        self,
        r,
        z,
        psi,
        psi_axis: float,
        psi_boundary: float,
        f,
        *,
        axis: tuple[float, float],
        boundary,
        wall=None,
        method: str = "cubic",
    ) -> None:
        """Set up the interpolation of ``psi`` and its gradient."""
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)
        psi = np.asarray(psi, dtype=float)
        if psi.shape != (len(r), len(z)):
            raise ValueError(f"psi has shape {psi.shape}, expected {(len(r), len(z))}")
        self.grid = Grid(2, (r[0], z[0]), (r[-1], z[-1]), (len(r), len(z)))
        self.r = r
        self.z = z
        self.psi_grid = psi
        self.psi_axis = float(psi_axis)
        self.psi_boundary = float(psi_boundary)
        self.f = np.asarray(f, dtype=float)
        self.axis = (float(axis[0]), float(axis[1]))
        self.boundary = Polygon(vertices=np.asarray(boundary, dtype=float))
        self.wall = (
            None if wall is None else Polygon(vertices=np.asarray(wall, dtype=float))
        )
        # The boundary (and usually the wall) is star-shaped about the magnetic
        # axis: test points against the one edge in their direction.
        self._fast_boundary = _StarShapedPolygon.try_build(boundary, self.axis)
        self._fast_wall = (
            None if wall is None else _StarShapedPolygon.try_build(wall, self.axis)
        )
        # The gradient at the nodes from a cubic spline (fourth-order accurate),
        # then interpolated like psi.
        spline = RectBivariateSpline(r, z, psi, kx=3, ky=3)
        dpsi_dr = spline(r, z, dx=1)
        dpsi_dz = spline(r, z, dy=1)
        self._interpolator = xp.scipy.interpolate.RegularGridInterpolator(
            (xp.asarray(r), xp.asarray(z)),
            xp.asarray(np.stack([psi, dpsi_dr, dpsi_dz], axis=-1)),
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._f_psi_n = xp.linspace(0.0, 1.0, len(self.f))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def solovev(
        cls,
        major_radius: float,
        minor_radius: float,
        elongation: float = 1.0,
        toroidal_field: float = 1.0,
        q_axis: float = 1.0,
        *,
        wall_psi_n: float | None = 1.2,
        points: int = 129,
        surface_points: int = 256,
    ) -> Equilibrium:
        r"""Return the Solov'ev equilibrium ``psi = c (R^2 Z^2 + kappa^2/4 (R^2 - R_0^2)^2)``.

        It solves the Grad-Shafranov equation with ``F = B_0 R_0`` constant and
        ``mu_0 p' = -2 c (1 + kappa^2)``; ``c = B_0 / (2 R_0^2 kappa q_0)`` sets
        the safety factor on the axis. The last closed surface passes through
        ``R_0 + a`` on the outboard midplane and through ``sqrt(2 R_0^2 - (R_0
        + a)^2)`` on the inboard one, so the shape (triangular, wider inboard
        than ``R_0 - a``) is fixed by ``a / R_0``; the wall is the surface
        ``psi_n = wall_psi_n`` (none if None).
        """
        r0, a, kappa = major_radius, minor_radius, elongation
        r_inner_sq = 2.0 * r0**2 - (r0 + a) ** 2
        if r_inner_sq <= 0.0:
            raise ValueError("minor radius too large: the boundary would reach R = 0")
        c = toroidal_field / (2.0 * r0**2 * kappa * q_axis)

        def flux(rr, zz):
            return c * (rr**2 * zz**2 + 0.25 * kappa**2 * (rr**2 - r0**2) ** 2)

        psi_boundary = flux(r0 + a, 0.0)
        # The outermost surface reaches R^2 = R_0^2 -+ sqrt(4 psi / (c kappa^2))
        # on the midplane.
        psi_max = psi_boundary * max(1.0, wall_psi_n or 1.0)
        half_width = np.sqrt(4.0 * psi_max / (c * kappa**2))
        if half_width >= r0**2:
            raise ValueError("the wall surface would reach R = 0")
        r_in, r_out = np.sqrt(r0**2 - half_width), np.sqrt(r0**2 + half_width)
        # On the surface Z^2 = (psi / c - kappa^2 / 4 (R^2 - R_0^2)^2) / R^2.
        r_fine = np.linspace(r_in, r_out, 2001)
        z_sq = (psi_max / c - 0.25 * kappa**2 * (r_fine**2 - r0**2) ** 2) / r_fine**2
        z_max = np.sqrt(np.maximum(z_sq, 0.0).max())
        margin = 0.05 * (r_out - r_in)
        r = np.linspace(max(r_in - margin, 0.5 * r_in), r_out + margin, points)
        z = np.linspace(-z_max - margin, z_max + margin, points)
        rr, zz = np.meshgrid(r, z, indexing="ij")
        circle = np.stack([np.full(4, r0), np.zeros(4)], axis=1) + np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        ) * np.array([0.5 * a, 0.5 * a * kappa])
        provisional = cls(
            r,
            z,
            flux(rr, zz),
            0.0,
            psi_boundary,
            np.full(2, toroidal_field * r0),
            axis=(r0, 0.0),
            boundary=circle,
        )
        boundary = provisional.flux_surface(1.0, surface_points)
        wall = (
            None
            if wall_psi_n is None
            else provisional.flux_surface(wall_psi_n, surface_points)
        )
        return cls(
            r,
            z,
            flux(rr, zz),
            0.0,
            psi_boundary,
            np.full(2, toroidal_field * r0),
            axis=(r0, 0.0),
            boundary=boundary,
            wall=wall,
        )

    @classmethod
    def from_geqdsk(cls, path: str | Path, **kwargs) -> Equilibrium:
        """Read a G-EQDSK file; ``kwargs`` go to the constructor."""
        data = geqdsk.read(path)
        wall = data.limiter if len(data.limiter) > 2 else None
        return cls(
            data.r,
            data.z,
            data.psi,
            data.psi_axis,
            data.psi_boundary,
            data.fpol,
            axis=data.axis,
            boundary=data.boundary,
            wall=wall,
            **kwargs,
        )

    def to_geqdsk(self, path: str | Path, *, description: str = "freeofcharge") -> None:
        """Write the equilibrium as a G-EQDSK file (pressure and q left zero)."""
        n_r = len(self.r)
        psi_n = np.linspace(0.0, 1.0, n_r)
        geqdsk.write(
            path,
            geqdsk.GEqdsk(
                r=self.r,
                z=self.z,
                psi=self.psi_grid,
                psi_axis=self.psi_axis,
                psi_boundary=self.psi_boundary,
                fpol=np.interp(psi_n, np.linspace(0.0, 1.0, len(self.f)), self.f),
                axis=self.axis,
                boundary=self.boundary.vertices,
                limiter=np.zeros((0, 2)) if self.wall is None else self.wall.vertices,
                description=description,
            ),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, r, z):
        r = xp.asarray(r, dtype=float)
        z = xp.asarray(z, dtype=float)
        values = self._interpolator(xp.stack([r.ravel(), z.ravel()], axis=-1))
        return [values[:, k].reshape(r.shape) for k in range(3)]

    def psi(self, r, z):
        """Return ``psi`` at ``(r, z)`` (NaN outside the grid)."""
        return self._evaluate(r, z)[0]

    def psi_normalized(self, r, z):
        """Return ``psi_n`` at ``(r, z)``."""
        return (self.psi(r, z) - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    def rho(self, r, z):
        """Return ``rho = sqrt(psi_n)`` at ``(r, z)``."""
        return xp.sqrt(xp.maximum(self.psi_normalized(r, z), 0.0))

    def inside(self, r, z):
        """Return whether the points are inside the last closed flux surface."""
        r = xp.asarray(r, dtype=float)
        z = xp.asarray(z, dtype=float)
        if self._fast_boundary is not None:
            return self._fast_boundary.contains(r, z)
        return self.boundary.contains_points(xp.stack([r.ravel(), z.ravel()])).reshape(
            r.shape
        )

    def inside_wall(self, r, z):
        """Return whether the points are inside the first wall (all if none)."""
        r = xp.asarray(r, dtype=float)
        z = xp.asarray(z, dtype=float)
        if self.wall is None:
            return xp.ones(r.shape, dtype=bool)
        if self._fast_wall is not None:
            return self._fast_wall.contains(r, z)
        return self.wall.contains_points(xp.stack([r.ravel(), z.ravel()])).reshape(
            r.shape
        )

    def magnetic_field(self, r, z):
        """Return ``(B_R, B_Z, B_phi)`` at ``(r, z)`` [T]."""
        psi, dpsi_dr, dpsi_dz = self._evaluate(r, z)
        r = xp.asarray(r, dtype=float)
        psi_n = (psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)
        f = xp.interp(xp.clip(psi_n, 0.0, 1.0), self._f_psi_n, xp.asarray(self.f))
        return -dpsi_dz / r, dpsi_dr / r, f / r

    # ------------------------------------------------------------------
    # MagneticEquilibrium, at Cartesian points
    # ------------------------------------------------------------------

    @property
    def has_wall(self) -> bool:
        """Return whether there is a first wall."""
        return self.wall is not None

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the box swept by the ``(R, Z)`` grid about the axis [m]."""
        lower, upper = self.grid.lower_bound, self.grid.upper_bound
        r_max = float(upper[0])
        return (
            np.array([-r_max, -r_max, float(lower[1])]),
            np.array([r_max, r_max, float(upper[1])]),
        )

    def rho_at(self, points):
        """Return ``rho`` at Cartesian ``points`` ``(..., 3)``."""
        return self.rho(*_cylindrical(points))

    def inside_at(self, points):
        """Return whether Cartesian ``points`` are inside the last closed surface."""
        return self.inside(*_cylindrical(points))

    def inside_wall_at(self, points):
        """Return whether Cartesian ``points`` are inside the first wall."""
        return self.inside_wall(*_cylindrical(points))

    def field_at(self, points):
        """Return ``B`` [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``."""
        points = xp.asarray(points, dtype=float)
        x, y = points[..., 0], points[..., 1]
        r, zc = _cylindrical(points)
        b_r, b_z, b_phi = self.magnetic_field(r, zc)
        cos_phi, sin_phi = x / r, y / r
        return xp.stack(
            [b_r * cos_phi - b_phi * sin_phi, b_r * sin_phi + b_phi * cos_phi, b_z],
            axis=-1,
        )

    # ------------------------------------------------------------------
    # Flux surfaces
    # ------------------------------------------------------------------

    def flux_surface(
        self, psi_n: float, points: int = 256, iterations: int = 60
    ) -> np.ndarray:
        """Return the surface ``psi_n`` as vertices ``(points, 2)`` in ``(R, Z)``.

        Bisection along rays from the axis to the edge of the grid, so the
        surface must be star-shaped about the axis and ``psi_n`` must increase
        along every ray up to it.
        """
        theta = xp.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
        cos_t, sin_t = xp.cos(theta), xp.sin(theta)
        r0, z0 = self.axis
        lower, upper = self.grid.lower_bound, self.grid.upper_bound
        big = 1.0e30
        reach_r = xp.where(
            cos_t > 0, (upper[0] - r0) / xp.where(cos_t > 0, cos_t, 1), big
        )
        reach_r = xp.minimum(
            reach_r,
            xp.where(cos_t < 0, (lower[0] - r0) / xp.where(cos_t < 0, cos_t, 1), big),
        )
        reach_z = xp.where(
            sin_t > 0, (upper[1] - z0) / xp.where(sin_t > 0, sin_t, 1), big
        )
        reach_z = xp.minimum(
            reach_z,
            xp.where(sin_t < 0, (lower[1] - z0) / xp.where(sin_t < 0, sin_t, 1), big),
        )
        high = 0.999999 * xp.minimum(reach_r, reach_z)
        if bool(
            xp.any(self.psi_normalized(r0 + high * cos_t, z0 + high * sin_t) < psi_n)
        ):
            raise ValueError(f"the surface psi_n = {psi_n} leaves the grid")
        low = xp.zeros_like(high)
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            above = self.psi_normalized(r0 + mid * cos_t, z0 + mid * sin_t) >= psi_n
            high = xp.where(above, mid, high)
            low = xp.where(above, low, mid)
        distance = 0.5 * (low + high)
        return xp.to_numpy(
            xp.stack([r0 + distance * cos_t, z0 + distance * sin_t], axis=1)
        )

    def flux_surfaces(
        self, rho_edges, resolution: int = 400, field_points: int = 256
    ) -> FluxSurfaces:
        """Return the volumes, areas, shapes and fields of the shells.

        The volumes and areas are sums over a ``resolution``-squared grid of
        cells covering the last closed surface; the field strength, the
        flux-surface average weights and the safety factor come from
        ``field_points`` points on the surface through each shell centre.
        """
        rho_edges = np.asarray(rho_edges, dtype=float)
        vertices = self.boundary.vertices
        r_lo, z_lo = vertices.min(axis=0)
        r_hi, z_hi = vertices.max(axis=0)
        dr = (r_hi - r_lo) / resolution
        dz = (z_hi - z_lo) / resolution
        rc = xp.asarray(r_lo + dr * (np.arange(resolution) + 0.5))
        zc = xp.asarray(z_lo + dz * (np.arange(resolution) + 0.5))
        rr, zz = xp.meshgrid(rc, zc, indexing="ij")
        rr, zz = rr.ravel(), zz.ravel()
        shells = FluxSurfaces(rho_edges, np.zeros(0), np.zeros(0), np.zeros(0))
        index = xp.where(self.inside(rr, zz), shells.shell_index(self.rho(rr, zz)), -1)
        n_shells = len(rho_edges) - 1
        sums = xp.algorithms.segment_sum(
            xp.stack([2.0 * np.pi * rr * dr * dz, xp.full(rr.shape, dr * dz)], axis=-1),
            index,
            n_shells,
        )
        centres = 0.5 * (rho_edges[1:] + rho_edges[:-1])
        epsilon = np.empty(n_shells)
        major = np.empty(n_shells)
        q = np.empty(n_shells)
        strength = np.empty((n_shells, field_points))
        weight = np.empty((n_shells, field_points))
        for k, rho in enumerate(centres):
            surface = self.flux_surface(rho**2, points=field_points)
            r_min, r_max = surface[:, 0].min(), surface[:, 0].max()
            epsilon[k] = (r_max - r_min) / (r_max + r_min)
            major[k] = 0.5 * (r_max + r_min)
            b_r, b_z, b_phi = (
                xp.to_numpy(b)
                for b in self.magnetic_field(surface[:, 0], surface[:, 1])
            )
            b_pol = np.hypot(b_r, b_z)
            # arc length of each point: half the distance between its neighbours
            arc = 0.5 * np.hypot(
                *(np.roll(surface, -1, axis=0) - np.roll(surface, 1, axis=0)).T
            )
            strength[k] = np.sqrt(b_pol**2 + b_phi**2)
            weight[k] = arc / b_pol / np.sum(arc / b_pol)
            q[k] = np.sum(arc * np.abs(b_phi) / (surface[:, 0] * b_pol)) / (2.0 * np.pi)
        sums = xp.to_numpy(sums)
        return FluxSurfaces(
            rho_edges,
            sums[:, 0],
            sums[:, 1],
            epsilon,
            field_strength=strength,
            surface_weight=weight,
            safety_factor=q,
            major_radius=major,
        )
