r"""3D equilibria from DESC (github.com/PlasmaControl/DESC), the optional extra ``desc``.

DESC's ``(R, phi, Z)`` is right-handed and ``zeta = phi``, so a Cartesian point
is ``(R cos phi, R sin phi, Z)``. The inverse map is DESC's own
``map_coordinates`` (Newton iterations in flux coordinates); points it cannot
reach lie outside the last closed surface. The field is DESC's ``B``, in the
``(R, phi, Z)`` basis.
"""

from __future__ import annotations

from pathlib import Path

import cunumpy as xp
import numpy as np

from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["DESCEquilibrium"]


def _require_desc():
    try:
        import desc.examples
        import desc.grid
        import desc.io
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "DESC equilibria need the optional extra: pip install freeofcharge[desc]"
        ) from error
    return desc


class DESCEquilibrium:
    """A DESC equilibrium as a :class:`~freeofcharge.MagneticEquilibrium`.

    Args:
        equilibrium: A ``desc.equilibrium.Equilibrium``.
        tolerance: Newton tolerance of ``map_coordinates``.
        iterations: Most Newton iterations.
    """

    has_wall = False

    def _compute(self, names, rho, theta, zeta) -> dict:
        """Return DESC's quantities ``names`` at flat flux coordinates."""
        grid = _require_desc().grid.Grid(
            np.stack([rho, theta, zeta], axis=-1), sort=False
        )
        data = self.equilibrium.compute(names, grid=grid)
        return {name: np.asarray(data[name]) for name in names}

    def __init__(
        self, equilibrium, *, tolerance: float = 1.0e-10, iterations: int = 40
    ) -> None:
        """Cache the boundary box and the minor radius."""
        self.equilibrium = equilibrium
        self.nfp = int(equilibrium.NFP)
        self.tolerance = tolerance
        self.iterations = iterations
        theta, zeta = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False),
            np.linspace(0.0, 2.0 * np.pi / self.nfp, 16, endpoint=False),
            indexing="ij",
        )
        edge = self._compute(
            ["R", "Z"], np.ones(theta.size), theta.ravel(), zeta.ravel()
        )
        edge_r = edge["R"].reshape(theta.shape)
        self.minor_radius = 0.5 * float(np.mean(edge_r.max(0) - edge_r.min(0)))
        r_max = float(edge_r.max())
        self._bounds = (
            np.array([-r_max, -r_max, float(edge["Z"].min())]),
            np.array([r_max, r_max, float(edge["Z"].max())]),
        )

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> DESCEquilibrium:
        """Return the equilibrium saved by DESC in ``path`` (the last of a family)."""
        desc = _require_desc()
        loaded = desc.io.load(str(path))
        if hasattr(loaded, "__getitem__"):
            loaded = loaded[-1]
        return cls(loaded, **kwargs)

    @classmethod
    def from_example(cls, name: str, **kwargs) -> DESCEquilibrium:
        """Return one of DESC's examples, such as ``"W7-X"`` or ``"SOLOVEV"``."""
        return cls(_require_desc().examples.get(name), **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box holding the last closed surface [m]."""
        return self._bounds

    def flux_coordinates(self, points):
        """Return ``(rho, theta, zeta)`` of Cartesian ``points`` ``(..., 3)``.

        ``rho`` is 1 plus the residual of the inverse map over the minor radius
        for points outside the last closed surface.
        """
        points = xp.to_numpy(xp.asarray(points, dtype=float))
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        cylindrical = np.stack(
            [
                np.hypot(flat[:, 0], flat[:, 1]),
                np.arctan2(flat[:, 1], flat[:, 0]),
                flat[:, 2],
            ],
            axis=-1,
        )
        coordinates, (residual, _) = self.equilibrium.map_coordinates(
            cylindrical,
            ("R", "phi", "Z"),
            period=(np.inf, 2.0 * np.pi, np.inf),
            tol=self.tolerance,
            maxiter=self.iterations,
            full_output=True,
        )
        coordinates = np.asarray(coordinates)
        residual = np.asarray(residual)
        inside = residual < 1.0e-6 * self.minor_radius
        rho = np.where(
            inside,
            np.minimum(coordinates[:, 0], 1.0),
            1.0 + residual / self.minor_radius,
        )
        theta = np.mod(coordinates[:, 1], 2.0 * np.pi)
        zeta = np.mod(cylindrical[:, 1], 2.0 * np.pi)
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
        """Return True everywhere: DESC equilibria carry no wall."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def field_at(self, points):
        """Return ``B`` [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``.

        Points outside the last closed surface get the field on it.
        """
        rho, theta, zeta = (c.ravel() for c in self.flux_coordinates(points))
        b = self._compute(["B"], np.minimum(rho, 1.0), theta, zeta)["B"]
        cos_p, sin_p = np.cos(zeta), np.sin(zeta)
        field = np.stack(
            [
                b[:, 0] * cos_p - b[:, 1] * sin_p,
                b[:, 0] * sin_p + b[:, 1] * cos_p,
                b[:, 2],
            ],
            axis=-1,
        )
        return xp.asarray(field.reshape(*xp.asarray(points).shape[:-1], 3))

    def _shape(self, rho, theta, zeta):
        """Return ``R``, ``dR/drho``, ``dR/dtheta``, ``dZ/drho``, ``dZ/dtheta``, |B|, iota."""
        names = ["R", "R_r", "R_t", "Z_r", "Z_t", "|B|"]
        data = self._compute(names, rho, theta, zeta)
        # iota is a surface quantity: on whole surfaces, then per point
        radii = np.unique(rho)
        surfaces = _require_desc().grid.LinearGrid(
            rho=radii,
            M=self.equilibrium.M_grid,
            N=self.equilibrium.N_grid,
            NFP=self.nfp,
        )
        iota = surfaces.compress(
            np.asarray(self.equilibrium.compute("iota", grid=surfaces)["iota"])
        )
        return (*(data[name] for name in names), np.interp(rho, radii, iota))

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the shells, see :meth:`FluxSurfaces.from_flux_coordinates`."""
        return FluxSurfaces.from_flux_coordinates(
            self._shape, self.nfp, rho_edges, **kwargs
        )
