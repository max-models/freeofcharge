r"""3D equilibria from DESC (github.com/PlasmaControl/DESC), the optional extra ``desc``.

DESC's $(R, \varphi, Z)$ is right-handed and $\zeta = \varphi$, so a Cartesian point
is $(R\cos\varphi, R\sin\varphi, Z)$. The inverse map is DESC's own
``map_coordinates`` (Newton iterations in flux coordinates); points it cannot
reach lie outside the last closed surface. The field is DESC's ``B``, in the
$(R, \varphi, Z)$ basis.
"""

from __future__ import annotations

from pathlib import Path

import cunumpy as xp
import numpy as np
from methodcard import method

from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["DESCEquilibrium"]


def _require_desc():
    try:
        import desc.examples  # pyright: ignore[reportMissingImports]
        import desc.grid  # pyright: ignore[reportMissingImports]
        import desc.io  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "DESC equilibria need the optional extra: pip install freeofcharge[desc]"
        ) from error
    return desc


def _newton_in_plane(spline_r, spline_z, r, z, tree, table, tolerance, iterations):
    r"""Return $\rho$, $\theta$ and the residual [m] of the points ``(r, z)`` of one plane.

    Newton in $(u, v) = \rho(\cos\theta, \sin\theta)$, regular on the axis, on the
    splines of $R$ and $Z$ over $(\rho, \theta)$, from the nearest table node;
    $\rho$ is clipped to 1, so points outside end on the boundary with a
    residual; the residual is returned too.
    """
    _, nearest = tree.query(np.stack([r, z], axis=-1))
    rho, theta = table[0][nearest], table[1][nearest]
    u, v = rho * np.cos(theta), rho * np.sin(theta)
    residual = np.full(r.shape, np.inf)
    for _ in range(iterations):
        rho = np.maximum(np.hypot(u, v), 1.0e-12)
        theta = np.mod(np.arctan2(v, u), 2.0 * np.pi)
        f_r = spline_r.ev(rho, theta) - r
        f_z = spline_z.ev(rho, theta) - z
        residual = np.hypot(f_r, f_z)
        if np.all(residual < tolerance):
            break
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        (r_rho, r_th), (z_rho, z_th) = (
            (s.ev(rho, theta, dx=1), s.ev(rho, theta, dy=1) / rho)
            for s in (spline_r, spline_z)
        )
        a11, a12 = r_rho * cos_t - r_th * sin_t, r_rho * sin_t + r_th * cos_t
        a21, a22 = z_rho * cos_t - z_th * sin_t, z_rho * sin_t + z_th * cos_t
        det = a11 * a22 - a12 * a21
        u = u - (a22 * f_r - a12 * f_z) / det
        v = v - (a11 * f_z - a21 * f_r) / det
        norm = np.maximum(np.hypot(u, v), 1.0)
        u, v = u / norm, v / norm
    return rho, theta, residual


class DESCEquilibrium:
    """A DESC equilibrium as a `freeofcharge.MagneticEquilibrium`.

    Args:
        equilibrium: A ``desc.equilibrium.Equilibrium``.
        tolerance: Newton tolerance of ``map_coordinates``.
        iterations: Most Newton iterations.
    """

    has_wall = False
    radial_coordinate = "toroidal"
    r"""$\rho = \sqrt{\Phi_n}$, the root of the normalised toroidal flux."""

    def _compute(self, names, rho, theta, zeta, *, chunk: int = 512) -> dict:
        """Return DESC's quantities ``names`` at flat flux coordinates.

        DESC's evaluation on scattered nodes holds a basis matrix per node (some
        GB for a few thousand nodes with the field), so tensor-product nodes
        (flux shells) go through one ``LinearGrid`` and the rest in chunks.
        """
        grid_module = _require_desc().grid
        rho, theta, zeta = (
            np.asarray(a, dtype=float).ravel() for a in (rho, theta, zeta)
        )
        radii, i_rho = np.unique(rho, return_inverse=True)
        thetas, i_theta = np.unique(np.mod(theta, 2.0 * np.pi), return_inverse=True)
        zetas, i_zeta = np.unique(
            np.mod(zeta, 2.0 * np.pi / self.nfp), return_inverse=True
        )
        if radii.size * thetas.size * zetas.size <= 4 * rho.size:
            grid = grid_module.LinearGrid(
                rho=radii, theta=thetas, zeta=zetas, NFP=self.nfp
            )
            data = self.equilibrium.compute(names, grid=grid)
            # LinearGrid's order: theta fastest, then rho, then zeta
            index = (i_zeta * radii.size + i_rho) * thetas.size + i_theta
            return {name: np.asarray(data[name])[index] for name in names}
        parts: dict[str, list] = {name: [] for name in names}
        for start in range(0, rho.size, chunk):
            part = slice(start, start + chunk)
            grid = grid_module.Grid(
                np.stack([rho[part], theta[part], zeta[part]], axis=-1), sort=False
            )
            data = self.equilibrium.compute(names, grid=grid)
            for name in names:
                parts[name].append(np.asarray(data[name]))
        return {name: np.concatenate(parts[name]) for name in names}

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

    @property
    def stellarator_symmetric(self) -> bool:
        """Return whether the equilibrium is stellarator symmetric (DESC's ``sym``)."""
        return bool(self.equilibrium.sym)

    def flux_coordinates(self, points):
        r"""Return $(\rho, \theta, \zeta)$ of Cartesian ``points`` ``(..., 3)``.

        $\rho$ is 1 plus the residual of the inverse map over the minor radius
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
        r"""Return $\rho$ at Cartesian ``points`` ``(..., 3)``."""
        return xp.asarray(self.flux_coordinates(points)[0])

    def inside_at(self, points):
        """Return whether Cartesian ``points`` are inside the last closed surface."""
        return self.rho_at(points) < 1.0

    def rho_and_inside_at(self, points):
        r"""Return $\rho$ and `inside_at` from one inversion."""
        rho = self.rho_at(points)
        return rho, rho < 1.0

    def inside_wall_at(self, points):
        """Return True everywhere: DESC equilibria carry no wall."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def field_at(self, points):
        r"""Return $\mathbf{B}$ [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``.

        Points outside the last closed surface get the field on it.
        """
        return self.rho_and_field_at(points)[1]

    def rho_and_field_at(self, points):
        r"""Return $\rho$ and $\mathbf{B}$ [T] at Cartesian ``points``, one inversion for both."""
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
        shape = xp.asarray(points).shape[:-1]
        return xp.asarray(rho.reshape(shape)), xp.asarray(field.reshape(*shape, 3))

    @method(
        equations={
            "table": r"f(\rho, \theta) = \sum_{ij} c_{ij}\, \beta_3(\rho - \rho_i)\, \beta_3(\theta - \theta_j), \quad f \in \{R, Z, B_R, B_\varphi, B_Z\}, \ \zeta = \varphi",
            "newton": r"\begin{pmatrix} \partial_u R & \partial_v R \\ \partial_u Z & \partial_v Z \end{pmatrix} \begin{pmatrix} \Delta u \\ \Delta v \end{pmatrix} = -\begin{pmatrix} R - R_* \\ Z - Z_* \end{pmatrix}, \quad (u, v) = \rho\,(\cos\theta, \sin\theta)",
            "chain": r"\partial_u = \cos\theta\, \partial_\rho - \frac{\sin\theta}{\rho}\, \partial_\theta, \quad \partial_v = \sin\theta\, \partial_\rho + \frac{\cos\theta}{\rho}\, \partial_\theta",
        },
        assumptions=(
            r"one DESC evaluation on a $(\rho, \theta)$ table per plane $\zeta = \varphi$, bicubic splines over it (periodic in $\theta$)",
            r"Newton in $(u, v)$, regular on the axis, from the nearest table node; $\rho$ clipped to 1",
            r"a point is inside if Newton converges to a residual below `tolerance` times the minor radius with $\rho < 1$; otherwise $\rho = 1 + \text{residual}/a$",
        ),
    )
    def rho_and_field_in_planes(
        self,
        phi,
        r,
        z,
        *,
        table: tuple[int, int] = (65, 128),
        iterations: int = 30,
    ):
        r"""Return $\rho$ and $(B_R, B_\varphi, B_Z)$ [T] at points in the planes ``phi``.

        A fast path for many points in a few toroidal planes, such as the
        sampling of `freeofcharge.GriddedEquilibrium`: DESC evaluates $R$, $Z$
        and $\mathbf{B}$ once on a ``table`` of $(\rho, \theta)$ nodes in each
        plane, and each point is inverted by Newton on bicubic splines of that
        table, about a thousand times faster than ``map_coordinates`` (which
        `rho_at` uses). Outside the last closed surface, $\rho$ is 1 plus the
        residual over the minor radius and the field that of the boundary
        point the iteration ends on, as in `flux_coordinates`.

        Args:
            phi: Toroidal angles of the planes [rad], shape ``(P,)``.
            r: Radii of the points in each plane [m], shape ``(P, M)``.
            z: Heights of the points in each plane [m], shape ``(P, M)``.
            table: $(\rho, \theta)$ nodes of the table per plane.
            iterations: Most Newton iterations.

        Returns:
            $\rho$ of shape ``(P, M)`` and the cylindrical field
            $(B_R, B_\varphi, B_Z)$ of shape ``(P, M, 3)``.
        """
        from scipy.interpolate import RectBivariateSpline
        from scipy.spatial import KDTree

        phi = np.atleast_1d(np.asarray(phi, dtype=float))
        r = np.asarray(r, dtype=float).reshape(len(phi), -1)
        z = np.asarray(z, dtype=float).reshape(len(phi), -1)
        n_rho, n_theta = table
        rho_nodes = np.linspace(0.0, 1.0, n_rho)
        theta_nodes = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        # the planes reduced to one field period, each once and sorted, as DESC wants
        planes, plane_of = np.unique(
            np.mod(phi, 2.0 * np.pi / self.nfp), return_inverse=True
        )
        grid = _require_desc().grid.LinearGrid(
            rho=rho_nodes, theta=theta_nodes, zeta=planes, NFP=self.nfp
        )
        data = self.equilibrium.compute(["R", "Z", "B"], grid=grid)
        # DESC's order: theta fastest, then rho, then zeta
        shape = (len(planes), n_rho, n_theta)
        b = np.asarray(data["B"]).reshape(*shape, 3)
        tables = [np.asarray(data[name]).reshape(shape) for name in ("R", "Z")]
        tables += [b[..., k] for k in range(3)]
        pad = 3
        theta_padded = np.concatenate(
            [
                theta_nodes[-pad:] - 2.0 * np.pi,
                theta_nodes,
                theta_nodes[:pad] + 2.0 * np.pi,
            ]
        )
        rho_table, theta_table = (
            a.ravel() for a in np.meshgrid(rho_nodes, theta_nodes, indexing="ij")
        )
        tolerance = self.tolerance * self.minor_radius
        rho_out = np.full(r.shape, np.nan)
        field_out = np.full((*r.shape, 3), np.nan)
        for j, plane in enumerate(plane_of.ravel()):
            splines = [
                RectBivariateSpline(
                    rho_nodes,
                    theta_padded,
                    np.concatenate(
                        [t[plane][:, -pad:], t[plane], t[plane][:, :pad]], 1
                    ),
                )
                for t in tables
            ]
            rho, theta, residual = _newton_in_plane(
                splines[0],
                splines[1],
                r[j],
                z[j],
                KDTree(
                    np.stack([tables[0][plane].ravel(), tables[1][plane].ravel()], -1)
                ),
                (rho_table, theta_table),
                tolerance,
                iterations,
            )
            # outside as in flux_coordinates: 1 + residual / minor radius, the
            # field of the boundary point Newton ends on
            inside = (residual < tolerance) & (rho < 1.0)
            rho_out[j] = np.where(inside, rho, 1.0 + residual / self.minor_radius)
            for k, spline in enumerate(splines[2:]):
                field_out[j, :, k] = spline.ev(rho, theta)
        return rho_out, field_out

    def _shape(self, rho, theta, zeta):
        r"""Return the surface shape and the field at flux coordinates.

        $R$, $\partial_\rho R$, $\partial_\theta R$, $\partial_\rho Z$, $\partial_\theta Z$,
        $|B|$ and $\iota$.
        """
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
        """Return the shells, see `FluxSurfaces.from_flux_coordinates`."""
        return FluxSurfaces.from_flux_coordinates(
            self._shape, self.nfp, rho_edges, **kwargs
        )
