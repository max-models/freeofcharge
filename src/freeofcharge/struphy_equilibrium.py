r"""MHD equilibria of Struphy (github.com/struphy-hub/struphy), the optional extra ``struphy``.

A Struphy equilibrium lives on a mapped domain $F: (\eta_1, \eta_2, \eta_3) \in
[0, 1]^3 \mapsto \mathbf{x}$, whose Jacobian $DF$ and field $B^{\eta_i}$
(contravariant components) Struphy evaluates in compiled kernels. Its 3D
stellarator equilibria come from DESC (``DESCequilibrium`` on the domain
``DESCunit``) and GVEC (``GVECequilibrium`` on ``GVECunit``); there $F$ is a
spline interpolant of the code's map and

$$
\rho = \rho_0 + (1 - \rho_0)\,\eta_1, \qquad \theta = 2\pi\eta_2, \qquad
\zeta = 2\pi\eta_3 / N_\mathrm{fp},
$$

with $\rho$ the code's radial coordinate (the square root of the normalised
toroidal flux in both) and $\rho_0$ the radius of the hole Struphy cuts around the
axis (``rmin``). The inverse map is a damped 3D Newton iteration on $F(\eta) =
\mathbf{x}$ with Struphy's $DF$, started from the nearest node of a cached grid;
the field is pushed forward, $\mathbf{B} = DF\,(B^{\eta_1}, B^{\eta_2},
B^{\eta_3})$.

Struphy 3.6.1 evaluates GVEC's field at $\zeta = 2\pi\eta_3$ instead of
$2\pi\eta_3 / N_\mathrm{fp}$, so ``field_at`` of a GVEC equilibrium with more
than one field period is wrong until Struphy fixes it; $\rho$ is not affected.
"""

from __future__ import annotations

import cunumpy as xp
import numpy as np
from methodcard import method
from scipy.spatial import KDTree

from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["StruphyEquilibrium"]


def _require_struphy():
    try:
        import struphy  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "Struphy equilibria need the optional extra: pip install freeofcharge[struphy]"
        ) from error
    return struphy


def _rotate(vectors, angle):
    """Return ``vectors`` ``(n, 3)`` rotated about $z$ by ``angle`` ``(n,)``."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x, y = vectors[:, 0], vectors[:, 1]
    return np.stack([cos_a * x - sin_a * y, sin_a * x + cos_a * y, vectors[:, 2]], -1)


@method(
    equations={
        "radius": r"\rho = \rho_0 + (1 - \rho_0)\,\eta_1",
        "inverse": r"\boldsymbol\eta \leftarrow \boldsymbol\eta - \lambda\, DF(\boldsymbol\eta)^{-1}\,\bigl(F(\boldsymbol\eta) - R_z(-\alpha)\,\mathbf{x}\bigr), \qquad F(\eta_1, \eta_2, \eta_3 + 1) = R_z(\Delta\varphi)\,F(\eta_1, \eta_2, \eta_3)",
        "field": r"\mathbf{B}(\mathbf{x}) = R_z(\alpha)\, DF\,(B^{\eta_1}, B^{\eta_2}, B^{\eta_3})^T",
        "iota": r"\iota = N_\mathrm{fp}\,\frac{\langle \sqrt g\, B^{\eta_2} \rangle}{\langle \sqrt g\, B^{\eta_3} \rangle}",
    },
    assumptions=(
        r"a toroidal domain: one field period $\Delta\varphi = \pm 2\pi / N_\mathrm{fp}$ per unit of $\eta_3$, found from the map",
        r"$\eta_1$ labels the flux surfaces, $\rho$ linear in it; points within the hole $\rho < \rho_0$ get $\rho$ from the map taken as affine there: its edge $F(0, \eta_2, \eta_3)$ an ellipse",
        r"the map is Struphy's spline interpolant of DESC's or GVEC's: $\rho$ and $\mathbf{B}$ carry its interpolation error",
        r"Struphy's coordinates in metres ($x$ unit 1 m); $\mathbf{B}$ times its $B$ unit; no first wall",
        r"$\iota$ the code's profile for DESC and GVEC, else from the uniform $(\eta_2, \eta_3)$ grid of each surface; the flux surfaces at $\zeta = 2\pi\eta_3/N_\mathrm{fp}$ taken as poloidal planes",
    ),
    references=("[Struphy](https://github.com/struphy-hub/struphy)",),
)
class StruphyEquilibrium:
    r"""A Struphy MHD equilibrium as a `freeofcharge.MagneticEquilibrium`.

    Args:
        equilibrium: A Struphy MHD equilibrium whose ``domain`` is set (the
            numerical ones, ``DESCequilibrium`` and ``GVECequilibrium``, set their
            own).
        rho_inner: $\rho_0$, the radius at $\eta_1 = 0$; by default the
            equilibrium's ``rmin`` for DESC and GVEC, else 0 ($\rho = \eta_1$).
        grid: Nodes in $(\eta_1, \eta_2, \eta_3)$ of the cached starting grid.
        tolerance: Newton tolerance on $\mathbf{x}$ relative to the minor radius.
        iterations: Most Newton iterations.
    """

    has_wall = False

    # ------------------------------------------------------------------
    # Struphy's domain and field at flat logical points (n, 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _markers(eta):
        """Return ``eta`` as Struphy's markers ``(3, n)``, at least two columns.

        Pyccel takes a single column, contiguous in both orders, for Fortran order.
        """
        if eta.shape[0] == 1:
            eta = np.concatenate([eta, eta])
        return np.ascontiguousarray(eta.T)

    def _map(self, eta):
        r"""Return $F(\eta)$, shape ``(n, 3)``."""
        out = self.domain(self._markers(eta), change_out_order=True)
        return np.asarray(xp.to_numpy(out), dtype=float)[: len(eta)]

    def __init__(
        self,
        equilibrium,
        *,
        rho_inner: float | None = None,
        grid: tuple[int, int, int] = (12, 32, 16),
        tolerance: float = 1.0e-10,
        iterations: int = 40,
    ) -> None:
        """Find the field period, cache the starting grid and the boundary box."""
        _require_struphy()
        self.equilibrium = equilibrium
        self.domain = equilibrium.domain
        numerical = type(equilibrium).__name__ in (
            "DESCequilibrium",
            "GVECequilibrium",
        )
        # DESC's and GVEC's radius is the root of the normalised toroidal flux; an
        # analytic equilibrium's eta1 is a geometric radius
        self.radial_coordinate = "toroidal" if numerical else "unspecified"
        if rho_inner is None:
            rho_inner = float(equilibrium.params["rmin"]) if numerical else 0.0
        self.rho_inner = rho_inner
        self.tolerance = tolerance
        self.iterations = iterations
        units = getattr(equilibrium, "units", None)
        self._b_unit = 1.0 if units is None else float(units.B)

        # the field period: the toroidal angle swept by eta3 from 0 to 1
        steps = np.linspace(0.0, 1.0, 17)
        probe = self._map(np.stack([np.full(17, 0.5), np.zeros(17), steps], -1))
        swept = np.unwrap(np.arctan2(probe[:, 1], probe[:, 0]))
        self.period = float(swept[-1] - swept[0])
        if abs(self.period) < 1.0e-3:
            raise ValueError("the domain is not toroidal: eta3 sweeps no angle")
        self.nfp = round(2.0 * np.pi / abs(self.period))
        self._phi0 = float(swept[0])

        n1, n2, n3 = grid
        e1, e2, e3 = np.meshgrid(
            np.linspace(0.0, 1.0, n1),
            np.linspace(0.0, 1.0, n2, endpoint=False),
            np.linspace(0.0, 1.0, n3 + 1),
            indexing="ij",
        )
        self._nodes = np.stack([e1.ravel(), e2.ravel(), e3.ravel()], -1)
        nodes_x = self._map(self._nodes)
        self._tree = KDTree(nodes_x)
        edge = nodes_x.reshape(n1, n2, n3 + 1, 3)[-1]
        edge_r = np.hypot(edge[..., 0], edge[..., 1])
        self.minor_radius = 0.5 * float(np.mean(edge_r.max(0) - edge_r.min(0)))
        r_max = float(edge_r.max())
        self._bounds = (
            np.array([-r_max, -r_max, float(edge[..., 2].min())]),
            np.array([r_max, r_max, float(edge[..., 2].max())]),
        )

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box holding the last closed surface [m]."""
        return self._bounds

    def _jacobian(self, eta):
        r"""Return $DF(\eta)$, shape ``(n, 3, 3)``, $\partial x_i / \partial\eta_j$."""
        out = self.domain.jacobian(self._markers(eta), change_out_order=True)
        return np.asarray(xp.to_numpy(out), dtype=float)[: len(eta)]

    def _contravariant(self, eta):
        r"""Return $B^{\eta_i}$ [T/m], shape ``(n, 3)``."""
        out = self.equilibrium.bv(self._markers(eta))
        # Struphy's DESC and GVEC equilibria cache every grid they were asked for
        cache = getattr(self.equilibrium, "_cache", None)
        if isinstance(cache, dict):
            for entry in cache.values():
                entry["grids"].clear()
                entry["outs"].clear()
        out = np.stack([np.asarray(xp.to_numpy(c), dtype=float) for c in out], -1)
        return self._b_unit * out[: len(eta)]

    def _wrap(self, eta, index, alpha, target):
        r"""Return ``eta`` in the cube; across $\eta_3$ = 0 or 1 rotate ``target``."""
        eta[:, 1] = np.mod(eta[:, 1], 1.0)
        turns = np.floor(eta[:, 2])
        eta[:, 2] -= turns
        if np.any(turns):
            alpha[index] += turns * self.period
            target[index] = _rotate(target[index], -turns * self.period)
        return eta

    def _project(self, eta, index, alpha, target, iterations: int = 20):
        r"""Move ``eta[index]`` to the nearest point of their surface $\eta_1$.

        Gauss-Newton in $(\eta_2, \eta_3)$ on $|F(\eta) - \mathbf{x}|^2$; returns
        the distances.
        """
        for _ in range(iterations):
            ea = eta[index]
            f = self._map(ea) - target[index]
            tangent = self._jacobian(ea)[:, :, 1:]
            normal = np.einsum("nki,nkj->nij", tangent, tangent)
            step = np.linalg.solve(
                normal, np.einsum("nki,nk->ni", tangent, f)[..., None]
            )
            ea[:, 1:] -= np.clip(step[..., 0], -0.25, 0.25)
            eta[index] = self._wrap(ea, index, alpha, target)
            if np.abs(step).max() < 1.0e-12:
                break
        return np.linalg.norm(self._map(eta[index]) - target[index], axis=-1)

    def _rho_in_hole(self, eta, target):
        r"""Return $\rho$ of points in the hole around the axis.

        The edge of the hole, $\mathbf{e}(\eta_2) = F(0, \eta_2, \eta_3)$, taken as
        an ellipse $\mathbf{c} + \mathbf{a}\cos 2\pi\eta_2 + \mathbf{b}\sin 2\pi\eta_2$
        (the map affine near the axis): $\rho = \rho_0 |(s, t)|$ for
        $\mathbf{x} - \mathbf{c} = s\,(\mathbf{e}(\eta_2) - \mathbf{c}) +
        t\,(\mathbf{e}(\eta_2 + 1/4) - \mathbf{c})$ in the least-squares sense.
        """
        edges = []
        for shift in (0.0, 0.25, 0.5, 0.75):
            moved = eta.copy()
            moved[:, 1] = np.mod(moved[:, 1] + shift, 1.0)
            edges.append(self._map(moved))
        centre = 0.25 * sum(edges)
        basis = np.stack([edges[0] - centre, edges[1] - centre], -1)
        normal = np.einsum("nki,nkj->nij", basis, basis)
        right = np.einsum("nki,nk->ni", basis, target - centre)
        coefficients = np.linalg.solve(normal, right[..., None])[..., 0]
        return self.rho_inner * np.linalg.norm(coefficients, axis=-1)

    # ------------------------------------------------------------------
    # The inverse map
    # ------------------------------------------------------------------

    def logical_coordinates(self, points):
        r"""Return $\eta$ ``(n, 3)``, the rotation $\alpha$ and $\rho$ of ``points``.

        ``points`` are Cartesian, ``(..., 3)``; $F(\eta) = R_z(-\alpha)\,\mathbf{x}$.
        $\rho$ is 1 plus the residual of the inverse map over the minor radius
        for points outside the last closed surface.
        """
        flat = xp.to_numpy(xp.asarray(points, dtype=float)).reshape(-1, 3)
        # into the first field period, then the nearest node
        sweep = np.sign(self.period) * (np.arctan2(flat[:, 1], flat[:, 0]) - self._phi0)
        alpha = self.period * np.floor(np.mod(sweep, 2.0 * np.pi) / abs(self.period))
        target = _rotate(flat, -alpha)
        eta = self._nodes[self._tree.query(target)[1]].copy()

        scale = self.tolerance * self.minor_radius
        residual = np.full(flat.shape[0], np.inf)
        active = np.arange(flat.shape[0])
        for _ in range(self.iterations):
            if active.size == 0:
                break
            ea = eta[active]
            f = self._map(ea) - target[active]
            previous = residual[active]
            residual[active] = np.linalg.norm(f, axis=-1)
            step = np.linalg.solve(self._jacobian(ea), f[..., None])[..., 0]
            # damp steps longer than a quarter of the logical cube
            damping = np.minimum(1.0, 0.25 / np.maximum(np.abs(step).max(-1), 1e-300))
            ea = ea - damping[:, None] * step
            ea[:, 0] = np.clip(ea[:, 0], 0.0, 1.0)
            eta[active] = self._wrap(ea, active, alpha, target)
            on_boundary = (ea[:, 0] <= 0.0) | (ea[:, 0] >= 1.0)
            stuck = on_boundary & (residual[active] > 0.9 * previous)
            active = active[(residual[active] >= scale) & ~stuck]
        missed = residual >= max(scale, 1.0e-6 * self.minor_radius)
        if np.any(missed):
            residual[missed] = self._project(eta, np.flatnonzero(missed), alpha, target)
        rho = self.rho_inner + (1.0 - self.rho_inner) * eta[:, 0]
        rho[missed] = 1.0 + residual[missed] / self.minor_radius
        hole = missed & (eta[:, 0] <= 0.0) & (self.rho_inner > 0.0)
        if np.any(hole):
            rho[hole] = self._rho_in_hole(eta[hole], target[hole])
        return eta, alpha, rho

    # ------------------------------------------------------------------
    # MagneticEquilibrium
    # ------------------------------------------------------------------

    def rho_at(self, points):
        r"""Return $\rho$ at Cartesian ``points`` ``(..., 3)``."""
        shape = xp.asarray(points).shape[:-1]
        return xp.asarray(self.logical_coordinates(points)[2].reshape(shape))

    def inside_at(self, points):
        """Return whether Cartesian ``points`` are inside the last closed surface."""
        return self.rho_at(points) < 1.0

    def inside_wall_at(self, points):
        """Return True everywhere: Struphy's equilibria carry no wall."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def field_at(self, points):
        r"""Return $\mathbf{B}$ [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``.

        Points outside the last closed surface get the field on it, points in the
        hole around the axis the field on its edge.
        """
        eta, alpha, _ = self.logical_coordinates(points)
        b = np.einsum("nij,nj->ni", self._jacobian(eta), self._contravariant(eta))
        field = _rotate(b, alpha)
        return xp.asarray(field.reshape(*xp.asarray(points).shape[:-1], 3))

    def _iota_profile(self, rho):
        r"""Return $\iota(\rho)$ of DESC's or GVEC's profile, None for others."""
        name = type(self.equilibrium).__name__
        if name == "DESCequilibrium":
            from desc.grid import LinearGrid  # pyright: ignore[reportMissingImports]

            eq = self.equilibrium.eq
            grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            return np.asarray(grid.compress(eq.compute("iota", grid=grid)["iota"]))
        if name == "GVECequilibrium":  # pragma: no cover - GVEC builds from source
            return np.asarray(self.equilibrium.state.evaluate_profile("iota", rho))
        return None

    def _shape(self, rho, theta, zeta):
        r"""Return the surface shape and the field at flux coordinates.

        $R$, $\partial_\rho R$, $\partial_\theta R$, $\partial_\rho Z$, $\partial_\theta Z$,
        $|B|$ and $\iota$; $\rho < \rho_0$ is taken on the edge of the hole. $\iota$
        is the code's profile for DESC and GVEC, else it comes from all points of
        the same $\rho$, a uniform $(\theta, \zeta)$ grid (exact in 2D; in 3D
        $\partial_\zeta\lambda$ averages out only on a fine grid in $\zeta$).
        """
        eta = np.stack(
            [
                np.clip((rho - self.rho_inner) / (1.0 - self.rho_inner), 0.0, 1.0),
                np.mod(theta / (2.0 * np.pi), 1.0),
                np.mod(zeta * self.nfp / (2.0 * np.pi), 1.0),
            ],
            -1,
        )
        x = self._map(eta)
        jacobian = self._jacobian(eta)
        contravariant = self._contravariant(eta)
        r = np.hypot(x[:, 0], x[:, 1])
        r_eta = (x[:, 0, None] * jacobian[:, 0] + x[:, 1, None] * jacobian[:, 1]) / r[
            :, None
        ]
        z_eta = jacobian[:, 2]
        d_rho, d_theta = 1.0 - self.rho_inner, 2.0 * np.pi
        strength = np.linalg.norm(
            np.einsum("nij,nj->ni", jacobian, contravariant), axis=-1
        )
        radii, surface = np.unique(rho, return_inverse=True)
        profile = self._iota_profile(np.maximum(radii, 1.0e-8))
        if profile is None:
            # iota = N_fp <sqrt(g) B^eta2> / <sqrt(g) B^eta3> on each surface
            flux = np.linalg.det(jacobian)[:, None] * contravariant
            poloidal = np.bincount(surface, flux[:, 1])
            profile = self.nfp * poloidal / np.bincount(surface, flux[:, 2])
        iota = profile[surface]
        return (
            r,
            r_eta[:, 0] / d_rho,
            r_eta[:, 1] / d_theta,
            z_eta[:, 0] / d_rho,
            z_eta[:, 1] / d_theta,
            strength,
            iota,
        )

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the shells, see `FluxSurfaces.from_flux_coordinates`."""
        return FluxSurfaces.from_flux_coordinates(
            self._shape, self.nfp, rho_edges, **kwargs
        )
