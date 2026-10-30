r"""3D equilibria from GVEC (github.com/gvec-group/gvec), the optional extra ``gvec``.

GVEC maps flux coordinates ``(rho, theta, zeta)`` to ``(X1, X2)`` with its
``hmap``; only ``which_hmap = 1`` is supported, ``(X1, X2) = (R, Z)`` and

.. math::

    \mathbf{x} = (R \cos\zeta, -R \sin\zeta, Z),

so a Cartesian point fixes ``zeta`` and the inverse map is a 2D Newton
iteration for ``(R, Z)`` in ``(u, v) = (rho cos theta, rho sin theta)``, which
stays regular on the axis. The start is the nearest node of a grid cached per
toroidal slice. The field is

.. math::

    \mathbf{B} = \frac{\chi' - \Phi' \partial_\zeta\lambda}{\sqrt{g}}
                 \mathbf{e}_\theta
               + \frac{\Phi' (1 + \partial_\theta\lambda)}{\sqrt{g}}
                 \mathbf{e}_\zeta .
"""

from __future__ import annotations

from pathlib import Path

import cunumpy as xp
import numpy as np
from scipy.spatial import cKDTree

from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["GVECEquilibrium"]


def _require_gvec():
    try:
        import gvec
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "GVEC equilibria need the optional extra: pip install freeofcharge[gvec]"
        ) from error
    return gvec


class GVECEquilibrium:
    """A GVEC equilibrium as a :class:`~freeofcharge.MagneticEquilibrium`.

    Args:
        state: A ``gvec.State``.
        grid: Nodes ``(rho, theta, zeta)`` of the cached starting grid, ``zeta``
            over one field period.
        tolerance: Newton tolerance on ``(R, Z)`` relative to the minor radius.
        iterations: Most Newton iterations.
    """

    has_wall = False

    def __init__(
        self,
        state,
        *,
        grid: tuple[int, int, int] = (16, 32, 16),
        tolerance: float = 1.0e-10,
        iterations: int = 30,
    ) -> None:
        """Cache the starting grid and the boundary box."""
        self.state = state
        self.nfp = int(state.nfp)
        self.tolerance = tolerance
        self.iterations = iterations
        n_rho, n_theta, n_zeta = grid
        self._rho = np.linspace(0.0, 1.0, n_rho)
        self._theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        self._zeta = np.linspace(0.0, 2.0 * np.pi / self.nfp, n_zeta, endpoint=False)
        rr, tt, zz = np.meshgrid(self._rho, self._theta, self._zeta, indexing="ij")
        rtz = np.stack([rr.ravel(), tt.ravel(), zz.ravel()])
        r_grid = state.evaluate_base_list_rtz_all("X1", rtz)[0].reshape(rr.shape)
        z_grid = state.evaluate_base_list_rtz_all("X2", rtz)[0].reshape(rr.shape)
        self._trees = [
            cKDTree(np.stack([r_grid[..., k].ravel(), z_grid[..., k].ravel()], -1))
            for k in range(n_zeta)
        ]
        # off the axis, where theta is undefined
        starts = np.maximum(rr[..., 0], 0.5 * self._rho[1])
        self._starts = np.stack([starts.ravel(), tt[..., 0].ravel()], axis=-1)
        edge_r, edge_z = r_grid[-1], z_grid[-1]
        self.minor_radius = 0.5 * float(np.mean(edge_r.max(0) - edge_r.min(0)))
        r_max = float(edge_r.max())
        self._bounds = (
            np.array([-r_max, -r_max, float(edge_z.min())]),
            np.array([r_max, r_max, float(edge_z.max())]),
        )

    @classmethod
    def from_files(
        cls, parameterfile: str | Path, statefile: str | Path, **kwargs
    ) -> GVECEquilibrium:
        """Return the equilibrium of a GVEC parameter file and state file."""
        gvec = _require_gvec()
        return cls(gvec.State(str(parameterfile), str(statefile)), **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box holding the last closed surface [m]."""
        return self._bounds

    # ------------------------------------------------------------------
    # The inverse map
    # ------------------------------------------------------------------

    def _evaluate(self, quantity: str, rho, theta, zeta):
        """Return ``Q`` and its ``rho``, ``theta`` and ``zeta`` derivatives."""
        rtz = np.stack([rho, theta, zeta])
        return self.state.evaluate_base_list_rtz_all(quantity, rtz)[:4]

    def flux_coordinates(self, points):
        """Return ``(rho, theta, zeta)`` of Cartesian ``points`` ``(..., 3)``.

        ``rho`` is 1 plus the distance to the last closed surface over the
        minor radius for points outside it.
        """
        points = xp.to_numpy(xp.asarray(points, dtype=float))
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        r_target = np.hypot(flat[:, 0], flat[:, 1])
        z_target = flat[:, 2]
        zeta = np.mod(-np.arctan2(flat[:, 1], flat[:, 0]), 2.0 * np.pi)

        # the start: nearest node in the nearest cached slice
        period = 2.0 * np.pi / self.nfp
        slices = np.rint(np.mod(zeta, period) / period * len(self._zeta)).astype(int)
        slices %= len(self._zeta)
        start = np.empty((flat.shape[0], 2))
        for k, tree in enumerate(self._trees):
            mask = slices == k
            if np.any(mask):
                _, node = tree.query(np.stack([r_target[mask], z_target[mask]], -1))
                start[mask] = self._starts[node]
        u = start[:, 0] * np.cos(start[:, 1])
        v = start[:, 0] * np.sin(start[:, 1])

        scale = self.tolerance * self.minor_radius
        residual = np.full(flat.shape[0], np.inf)
        # points still iterating: not converged, and not stuck on the boundary
        active = np.arange(flat.shape[0])
        for _ in range(self.iterations):
            if active.size == 0:
                break
            ua, va = u[active], v[active]
            # derivatives in (u, v) are regular on the axis: evaluate just off it
            rho = np.clip(np.hypot(ua, va), 1.0e-12, 1.0)
            theta = np.arctan2(va, ua)
            za = zeta[active]
            x1, x1_r, x1_t, _ = self._evaluate("X1", rho, theta, za)
            x2, x2_r, x2_t, _ = self._evaluate("X2", rho, theta, za)
            f1, f2 = x1 - r_target[active], x2 - z_target[active]
            previous = residual[active]
            residual[active] = np.hypot(f1, f2)
            # d/du, d/dv from d/drho, d/dtheta
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            a = cos_t * x1_r - sin_t / rho * x1_t
            b = sin_t * x1_r + cos_t / rho * x1_t
            c = cos_t * x2_r - sin_t / rho * x2_t
            d = sin_t * x2_r + cos_t / rho * x2_t
            det = a * d - b * c
            du = (d * f1 - b * f2) / det
            dv = (a * f2 - c * f1) / det
            # damp steps longer than a quarter of the minor radius in (u, v)
            length = np.hypot(du, dv)
            damping = np.minimum(1.0, 0.25 / np.maximum(length, 1.0e-300))
            ua, va = ua - damping * du, va - damping * dv
            # keep the iterate inside the last closed surface
            norm = np.maximum(np.hypot(ua, va), 1.0)
            u[active], v[active] = ua / norm, va / norm
            on_boundary = rho >= 1.0
            stuck = on_boundary & (residual[active] > 0.9 * previous)
            active = active[(residual[active] >= scale) & ~stuck]
        rho = np.minimum(np.hypot(u, v), 1.0)
        theta = np.mod(np.arctan2(v, u), 2.0 * np.pi)
        rho = np.where(
            residual < max(scale, 1.0e-6 * self.minor_radius),
            rho,
            1.0 + residual / self.minor_radius,
        )
        return rho.reshape(shape), theta.reshape(shape), zeta.reshape(shape)

    # ------------------------------------------------------------------
    # MagneticEquilibrium
    # ------------------------------------------------------------------

    def rho_at(self, points):
        """Return ``rho`` at Cartesian ``points`` ``(..., 3)``."""
        return xp.asarray(self.flux_coordinates(points)[0])

    def inside_at(self, points):
        """Return whether Cartesian ``points`` are inside the last closed surface."""
        return self.rho_at(points) < 1.0

    def inside_wall_at(self, points):
        """Return True everywhere: GVEC equilibria carry no wall."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def _field(self, rho, theta, zeta):
        """Return ``B`` ``(n, 3)`` at flux coordinates."""
        x1, x1_r, x1_t, x1_z = self._evaluate("X1", rho, theta, zeta)
        x2, x2_r, x2_t, x2_z = self._evaluate("X2", rho, theta, zeta)
        _, _, la_t, la_z = self._evaluate("LA", rho, theta, zeta)
        # rho = 0 is outside the profiles' derivative domain in some versions
        rho_p = np.maximum(rho, 1.0e-8)
        phi_p = self.state.evaluate_profile("Phi", rho_p, deriv=1)
        chi_p = self.state.evaluate_profile("chi", rho_p, deriv=1)
        cos_z, sin_z = np.cos(zeta), np.sin(zeta)
        # hmap 1: x = (R cos zeta, -R sin zeta, Z)
        dx_dr = np.stack([cos_z, -sin_z, np.zeros_like(zeta)], axis=-1)
        dx_dz = np.stack([np.zeros_like(zeta)] * 2 + [np.ones_like(zeta)], axis=-1)
        dx_dzeta = np.stack([-x1 * sin_z, -x1 * cos_z, np.zeros_like(zeta)], axis=-1)
        e_rho = x1_r[:, None] * dx_dr + x2_r[:, None] * dx_dz
        e_theta = x1_t[:, None] * dx_dr + x2_t[:, None] * dx_dz
        e_zeta = x1_z[:, None] * dx_dr + x2_z[:, None] * dx_dz + dx_dzeta
        jacobian = np.sum(e_rho * np.cross(e_theta, e_zeta), axis=-1)
        b_theta = (chi_p - phi_p * la_z) / jacobian
        b_zeta = phi_p * (1.0 + la_t) / jacobian
        return b_theta[:, None] * e_theta + b_zeta[:, None] * e_zeta

    def field_at(self, points):
        """Return ``B`` [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``.

        Points outside the last closed surface get the field on it.
        """
        rho, theta, zeta = (c.ravel() for c in self.flux_coordinates(points))
        rho = np.minimum(rho, 1.0)
        field = self._field(rho, theta, zeta)
        return xp.asarray(field.reshape(*xp.asarray(points).shape[:-1], 3))

    def flux_surfaces(
        self,
        rho_edges,
        field_points: tuple[int, int] = (32, 32),
        radial_points: int = 8,
    ) -> FluxSurfaces:
        """Return the volumes, areas, shapes and fields of the shells.

        Volumes and the toroidally averaged cross-section areas by
        Gauss-Legendre quadrature in ``rho`` within each shell and the
        trapezoidal rule on a ``field_points`` ``(theta, zeta)`` grid over one
        field period; the field strength and the averaging weights (the
        Jacobian) on that grid on the surface through each shell centre.
        """
        rho_edges = np.asarray(rho_edges, dtype=float)
        n_theta, n_zeta = field_points
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        zeta = np.linspace(0.0, 2.0 * np.pi / self.nfp, n_zeta, endpoint=False)
        cell = (2.0 * np.pi / n_theta) * (2.0 * np.pi / self.nfp / n_zeta)
        nodes, weights = np.polynomial.legendre.leggauss(radial_points)
        n_shells = len(rho_edges) - 1
        volume, area = np.zeros(n_shells), np.zeros(n_shells)
        for k in range(n_shells):
            lo, hi = rho_edges[k], rho_edges[k + 1]
            for node, weight in zip(nodes, weights, strict=True):
                rho = 0.5 * (lo + hi) + 0.5 * (hi - lo) * node
                tt, zz = np.meshgrid(theta, zeta, indexing="ij")
                rr = np.full(tt.size, rho)
                _, x1_r, x1_t, _ = self._evaluate("X1", rr, tt.ravel(), zz.ravel())
                x1 = self._evaluate("X1", rr, tt.ravel(), zz.ravel())[0]
                _, x2_r, x2_t, _ = self._evaluate("X2", rr, tt.ravel(), zz.ravel())
                jac_rz = np.abs(x1_r * x2_t - x1_t * x2_r)
                factor = 0.5 * (hi - lo) * weight * cell
                volume[k] += factor * np.sum(x1 * jac_rz) * self.nfp
                # the cross section averaged over zeta
                area[k] += factor * np.sum(jac_rz) / (2.0 * np.pi / self.nfp)

        centres = 0.5 * (rho_edges[1:] + rho_edges[:-1])
        strength = np.empty((n_shells, n_theta * n_zeta))
        weight = np.empty_like(strength)
        epsilon, major = np.empty(n_shells), np.empty(n_shells)
        tt, zz = np.meshgrid(theta, zeta, indexing="ij")
        tt, zz = tt.ravel(), zz.ravel()
        for k, rho in enumerate(centres):
            rr = np.full(tt.size, rho)
            field = self._field(rr, tt, zz)
            x1, x1_r, x1_t, _ = self._evaluate("X1", rr, tt, zz)
            _, x2_r, x2_t, _ = self._evaluate("X2", rr, tt, zz)
            jacobian = np.abs(x1 * (x1_r * x2_t - x1_t * x2_r))
            strength[k] = np.linalg.norm(field, axis=-1)
            weight[k] = jacobian / jacobian.sum()
            r_slices = x1.reshape(n_theta, n_zeta)
            r_min, r_max = r_slices.min(axis=0), r_slices.max(axis=0)
            epsilon[k] = float(np.mean((r_max - r_min) / (r_max + r_min)))
            major[k] = float(np.mean(0.5 * (r_max + r_min)))
        iota = self.state.evaluate_profile("iota", np.maximum(centres, 1.0e-8))
        return FluxSurfaces(
            rho_edges,
            volume,
            area,
            epsilon,
            field_strength=strength,
            surface_weight=weight,
            safety_factor=1.0 / np.asarray(iota),
            major_radius=major,
        )
