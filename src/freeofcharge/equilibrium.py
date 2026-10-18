r"""Axisymmetric equilibria: the poloidal flux on an (R, Z) grid.

The field is

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

from dataclasses import dataclass
from pathlib import Path

import cunumpy as xp
import numpy as np
from scipy.interpolate import RectBivariateSpline
from simgeometry.grid import Grid
from simgeometry.objects.polygon import Polygon

from freeofcharge import geqdsk

__all__ = ["Equilibrium", "FluxSurfaces"]


@dataclass(frozen=True)
class FluxSurfaces:
    """Volumes and shapes of the shells between the radii ``rho_edges``."""

    rho_edges: np.ndarray
    """Shell edges in ``rho``, increasing from 0 to 1."""
    volume: np.ndarray
    """Volume of each shell [m^3]."""
    area: np.ndarray
    """Poloidal cross-section area of each shell [m^2]."""
    inverse_aspect_ratio: np.ndarray
    """``(R_max - R_min) / (R_max + R_min)`` of the surface at each shell centre."""

    @property
    def rho(self) -> np.ndarray:
        """Return the shell centres."""
        return 0.5 * (self.rho_edges[1:] + self.rho_edges[:-1])

    def shell_index(self, rho):
        """Return the shell of each ``rho``, -1 outside ``[0, 1)``."""
        edges = xp.asarray(self.rho_edges)
        index = xp.searchsorted(edges, xp.asarray(rho), side="right") - 1
        return xp.where((index >= 0) & (index < len(self.rho_edges) - 1), index, -1)


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
        return self.boundary.contains_points(xp.stack([r.ravel(), z.ravel()])).reshape(
            r.shape
        )

    def inside_wall(self, r, z):
        """Return whether the points are inside the first wall (all if none)."""
        r = xp.asarray(r, dtype=float)
        z = xp.asarray(z, dtype=float)
        if self.wall is None:
            return xp.ones(r.shape, dtype=bool)
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

    def magnetic_field_cartesian(self, positions):
        """Return ``B`` [T] at Cartesian ``positions`` ``(N, 3)``, shape ``(N, 3)``."""
        positions = xp.asarray(positions, dtype=float)
        x, y, zc = positions[:, 0], positions[:, 1], positions[:, 2]
        r = xp.hypot(x, y)
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

    def flux_surfaces(self, rho_edges, resolution: int = 400) -> FluxSurfaces:
        """Return the volumes, areas and inverse aspect ratios of the shells.

        The volumes and areas are sums over a ``resolution``-squared grid of
        cells covering the last closed surface.
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
        for k, rho in enumerate(centres):
            surface = self.flux_surface(rho**2, points=128)
            r_min, r_max = surface[:, 0].min(), surface[:, 0].max()
            epsilon[k] = (r_max - r_min) / (r_max + r_min)
        sums = xp.to_numpy(sums)
        return FluxSurfaces(rho_edges, sums[:, 0], sums[:, 1], epsilon)
