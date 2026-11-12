r"""3D equilibria from VMEC (``wout`` files or VMEC++ runs), the optional extra ``vmec``.

VMEC writes $R$, $Z$ on the full radial mesh $s_j = j / (n_s - 1)$ and the
contravariant field $B^u$, $B^v$ on the half mesh $s_{j - 1/2}$, as Fourier
series in the poloidal angle $u$ and the cylindrical angle $v = \varphi$:

$$
R = \sum_{mn} R_{mn}(s) \cos(m u - n v), \qquad
Z = \sum_{mn} Z_{mn}(s) \sin(m u - n v),
$$

$n$ a multiple of the number of field periods (plus the sine and cosine series
of non-symmetric equilibria). $s$ is the normalized toroidal flux and
$\rho = \sqrt{s}$ as for GVEC and DESC. The coefficients are cubic splines in
$s$, those of odd $m$ divided by $\rho$ first, which makes them smooth on the
axis. A Cartesian point fixes $v$, and the inverse map is a 2D Newton iteration
for $(R, Z)$ in $(\rho\cos u, \rho\sin u)$ as in `GVECEquilibrium`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cunumpy as xp
import numpy as np
from methodcard import method
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree

from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["VMECEquilibrium"]

# the wout variables read from a netCDF file
_WOUT_NAMES = (
    "nfp",
    "ns",
    "lasym__logical__",
    "xm",
    "xn",
    "xm_nyq",
    "xn_nyq",
    "rmnc",
    "zmns",
    "rmns",
    "zmnc",
    "bsupumnc",
    "bsupvmnc",
    "bsupumns",
    "bsupvmns",
    "bmnc",
    "bmns",
    "iotaf",
    "phi",
)

# points per block of a Fourier sum, to bound the (points, modes) arrays
_BLOCK = 2048


def _require_vmecpp():
    try:
        import vmecpp  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "VMEC++ runs need the optional extra: pip install freeofcharge[vmec]"
        ) from error
    return vmecpp


def _read_wout(path: str | Path) -> dict:
    """Return the variables of a ``wout`` netCDF file, 2D ones as ``(modes, radius)``."""
    try:
        import netCDF4  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise ImportError(
            "wout files need the optional extra: pip install freeofcharge[vmec]"
        ) from error
    with netCDF4.Dataset(str(path), "r") as data:
        data.set_auto_mask(False)
        return {
            name: np.asarray(data.variables[name][()]).T
            for name in _WOUT_NAMES
            if name in data.variables
        }


def _get(wout, name: str) -> Any:
    """Return ``wout``'s ``name`` (a mapping or attributes), None if missing."""
    if isinstance(wout, dict):
        return wout.get(name)
    return getattr(wout, name, None)


class _Radial:
    r"""Cubic splines in $s$ of Fourier coefficients ``(modes, len(s))``.

    Odd-$m$ coefficients behave as $\rho$ times a smooth function of $s$ near
    the axis: those are divided by $\rho$ first and fitted on $s > 0$ only.
    """

    def __init__(self, s, coefficients, m) -> None:
        """Fit the splines; ``coefficients`` ``(modes, len(s))`` at radii ``s``."""
        s = np.asarray(s, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)
        self.odd = np.asarray(m, dtype=int) % 2 == 1
        self._even = CubicSpline(s, coefficients[~self.odd], axis=1)
        positive = s > 0.0
        self._odd = CubicSpline(
            s[positive],
            coefficients[self.odd][:, positive] / np.sqrt(s[positive]),
            axis=1,
        )

    def __call__(self, rho):
        r"""Return the coefficients and their $\rho$ derivatives, ``(len(rho), modes)``."""
        rho = np.asarray(rho, dtype=float)
        s = rho**2
        value = np.empty((rho.size, self.odd.size))
        slope = np.empty_like(value)
        value[:, ~self.odd] = self._even(s).T
        slope[:, ~self.odd] = (2.0 * rho * self._even(s, 1)).T
        g, g_s = self._odd(s).T, self._odd(s, 1).T
        value[:, self.odd] = rho[:, None] * g
        slope[:, self.odd] = g + 2.0 * s[:, None] * g_s
        return value, slope


class _Series:
    r"""A Fourier series $\sum_{mn} a_{mn}(s) \cos(m u - n v) + b_{mn}(s) \sin(m u - n v)$."""

    def __init__(self, s, m, n, cos=None, sin=None) -> None:
        """Fit the radial splines of the cosine and sine coefficients (either may be None)."""
        self.m = np.asarray(m, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.cos = None if cos is None else _Radial(s, cos, self.m)
        self.sin = None if sin is None else _Radial(s, sin, self.m)

    def __call__(self, rho, u, v):
        r"""Return $f$, $\partial_\rho f$, $\partial_u f$ and $\partial_v f$ at flat arrays."""
        rho, u, v = (np.asarray(a, dtype=float).ravel() for a in (rho, u, v))
        out = np.zeros((4, rho.size))
        for start in range(0, rho.size, _BLOCK):
            block = slice(start, start + _BLOCK)
            angle = self.m * u[block, None] - self.n * v[block, None]
            cos, sin = np.cos(angle), np.sin(angle)
            for radial, even, odd, sign in (
                (self.cos, cos, sin, -1.0),
                (self.sin, sin, cos, 1.0),
            ):
                if radial is None:
                    continue
                c, c_rho = radial(rho[block])
                out[0, block] += np.sum(c * even, axis=1)
                out[1, block] += np.sum(c_rho * even, axis=1)
                out[2, block] += sign * np.sum(c * self.m * odd, axis=1)
                out[3, block] -= sign * np.sum(c * self.n * odd, axis=1)
        return out


def _defaults() -> dict:
    """Return VMEC++'s settings that a fixed-boundary run leaves at their usual values."""
    return {
        "gamma": 0.0,
        "spres_ped": 1.0,
        "piota_type": "power_series",
        "ai": np.zeros(1),
        "bloat": 1.0,
        "mgrid_file": "",
        "extcur": np.zeros(0),
        "nvacskip": 1,
        "nstep": 200,
        "aphi": np.ones(1),
        "delt": 0.9,
        "tcon0": 1.0,
        "lforbal": False,
    }


@method(
    equations={
        "map": r"R = \sum_{mn} R_{mn}(s)\cos(mu - nv), \quad Z = \sum_{mn} Z_{mn}(s)\sin(mu - nv), \quad \mathbf{x} = (R\cos v, R\sin v, Z), \quad \rho = \sqrt{s}",
        "field": r"\mathbf{B} = B^u\,\partial_u\mathbf{x} + B^v\,\partial_v\mathbf{x}, \qquad B^u = \sum_{mn} B^u_{mn}(s)\cos(mu - nv), \quad B^v = \sum_{mn} B^v_{mn}(s)\cos(mu - nv)",
        "radial": r"c_{mn}(s) = \rho^{m \bmod 2}\, g_{mn}(s), \qquad g_{mn} \text{ a cubic spline in } s",
    },
    assumptions=(
        r"$R$, $Z$ on VMEC's full radial mesh, $B^u$, $B^v$ and $|B|$ on its half mesh, extrapolated by the end splines to the axis and to $s = 1$",
        r"$v$ is the cylindrical angle $\varphi$; $s$ the normalized toroidal flux",
        r"inverse map by a damped 2D Newton iteration in $(\rho\cos u, \rho\sin u)$; no first wall",
    ),
    references=(
        "S. P. Hirshman and J. C. Whitson, Phys. Fluids 26, 3553 (1983)",
        "[VMEC++](https://github.com/proximafusion/vmecpp)",
    ),
)
class VMECEquilibrium:
    r"""A VMEC equilibrium as a `freeofcharge.MagneticEquilibrium`.

    Args:
        wout: VMEC's output: a ``vmecpp.VmecOutput``, its ``wout``
            (``vmecpp.VmecWOut``), or a dict of ``wout`` variables with 2D arrays
            as ``(modes, radius)``.
        grid: Nodes $(\rho, u, v)$ of the cached starting grid, $v$ over one field
            period.
        tolerance: Newton tolerance on $(R, Z)$ relative to the minor radius.
        iterations: Most Newton iterations.
    """

    has_wall = False

    def __init__(
        self,
        wout,
        *,
        grid: tuple[int, int, int] = (16, 32, 16),
        tolerance: float = 1.0e-10,
        iterations: int = 30,
    ) -> None:
        """Fit the radial splines, cache the starting grid and the boundary box."""
        wout = getattr(wout, "wout", wout)
        self.wout = wout
        self.nfp = int(_get(wout, "nfp"))
        self.tolerance = tolerance
        self.iterations = iterations
        ns = int(_get(wout, "ns"))
        lasym = bool(_get(wout, "lasym__logical__") or _get(wout, "lasym"))
        full = np.linspace(0.0, 1.0, ns)
        half = (np.arange(1, ns) - 0.5) / (ns - 1)
        m, n = _get(wout, "xm"), _get(wout, "xn")
        m_nyq, n_nyq = _get(wout, "xm_nyq"), _get(wout, "xn_nyq")

        def series(s, mm, nn, cos, sin, part=slice(None)):
            """Return the series of ``wout``'s arrays ``cos`` and ``sin`` (None: absent)."""
            arrays = [
                None if name is None else np.asarray(_get(wout, name))[:, part]
                for name in (cos, sin)
            ]
            return _Series(s, mm, nn, cos=arrays[0], sin=arrays[1])

        # the second series of each pair only for non-symmetric equilibria
        self._r = series(full, m, n, "rmnc", "rmns" if lasym else None)
        self._z = series(full, m, n, "zmnc" if lasym else None, "zmns")
        # the half mesh: column 0 of the half-mesh arrays is unused
        nyq = (half, m_nyq, n_nyq)
        part = slice(1, None)
        self._b_u = series(*nyq, "bsupumnc", "bsupumns" if lasym else None, part)
        self._b_v = series(*nyq, "bsupvmnc", "bsupvmns" if lasym else None, part)
        self._b = series(*nyq, "bmnc", "bmns" if lasym else None, part)
        self._iota = CubicSpline(full, np.asarray(_get(wout, "iotaf"), dtype=float))

        n_rho, n_theta, n_zeta = grid
        rho = np.linspace(0.0, 1.0, n_rho)
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        self._zeta = np.linspace(0.0, 2.0 * np.pi / self.nfp, n_zeta, endpoint=False)
        rr, tt, zz = np.meshgrid(rho, theta, self._zeta, indexing="ij")
        r_grid = self._r(rr, tt, zz)[0].reshape(rr.shape)
        z_grid = self._z(rr, tt, zz)[0].reshape(rr.shape)
        self._trees = [
            KDTree(np.stack([r_grid[..., k].ravel(), z_grid[..., k].ravel()], -1))
            for k in range(n_zeta)
        ]
        # off the axis, where the angle is undefined
        starts = np.maximum(rr[..., 0], 0.5 * rho[1])
        self._starts = np.stack([starts.ravel(), tt[..., 0].ravel()], axis=-1)
        edge_r, edge_z = r_grid[-1], z_grid[-1]
        self.minor_radius = 0.5 * float(np.mean(edge_r.max(0) - edge_r.min(0)))
        r_max = float(edge_r.max())
        self._bounds = (
            np.array([-r_max, -r_max, float(edge_z.min())]),
            np.array([r_max, r_max, float(edge_z.max())]),
        )

    @classmethod
    def from_wout(cls, path: str | Path, **kwargs) -> VMECEquilibrium:
        """Return the equilibrium of a ``wout_*.nc`` file (VMEC, VMEC++, DESC's export)."""
        return cls(_read_wout(path), **kwargs)

    @classmethod
    def from_input(
        cls,
        vmec_input,
        *,
        max_threads: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> VMECEquilibrium:
        """Run VMEC++ on ``vmec_input`` and return its equilibrium.

        Args:
            vmec_input: A ``vmecpp.VmecInput``, or the path of a VMEC++ JSON or
                Fortran VMEC ``input.*`` file.
            max_threads: VMEC++'s threads.
            verbose: Whether VMEC++ prints its iterations.
            **kwargs: Passed on to `VMECEquilibrium`.
        """
        vmecpp = _require_vmecpp()
        if isinstance(vmec_input, (str, Path)):
            vmec_input = vmecpp.VmecInput.from_file(vmec_input)
        output = vmecpp.run(vmec_input, max_threads=max_threads, verbose=verbose)
        return cls(output, **kwargs)

    @staticmethod
    def vacuum_input(
        boundary,
        *,
        field_strength: float = 1.0,
        mpol: int | None = None,
        ntor: int | None = None,
        ns: tuple[int, ...] = (9, 17),
        ftol: float = 1.0e-12,
        niter: int = 5000,
    ):
        r"""Return a fixed-boundary ``vmecpp.VmecInput`` without pressure or current ($\beta = 0$).

        Args:
            boundary: ConStellaration's boundary: a dict (or its JSON string) with
                ``r_cos``, ``z_sin`` ``(M + 1, 2N + 1)`` and ``n_field_periods``
                (``r_sin``, ``z_cos`` for non-symmetric ones).
            field_strength: Mean $|B|$ [T], roughly: the edge flux is this times the
                mean cross-section area.
            mpol: Poloidal modes $m < $ ``mpol``; default those of the boundary. The
                equilibrium needs more than its boundary: on the CoilStellaration
                sample (``mpol=5, ntor=4``), $\mathbf{B}$ differs from that at
                ``mpol=12, ntor=11`` by 1 % on average, at ``mpol=8, ntor=7`` by 0.1 %.
            ntor: Toroidal modes $|n| \le$ ``ntor``; default those of the boundary.
            ns: Radial surfaces of VMEC's multigrid steps.
            ftol: Force tolerance of the last step (the others 100 times larger).
            niter: Most iterations per step.
        """
        from freeofcharge.coils import PlasmaBoundary

        vmecpp = _require_vmecpp()
        surface = PlasmaBoundary.from_boundary(boundary)
        r_cos, z_sin = surface.r_cos, surface.z_sin
        n_tor = (r_cos.shape[1] - 1) // 2
        n_pol = r_cos.shape[0] if mpol is None else int(mpol)
        n_out = n_tor if ntor is None else int(ntor)
        lasym = surface.r_sin is not None or surface.z_cos is not None

        def resize(coefficients):
            out = np.zeros((n_pol, 2 * n_out + 1))
            if coefficients is None:
                return out
            m = min(n_pol, coefficients.shape[0])
            n = min(n_out, n_tor)
            out[:m, n_out - n : n_out + n + 1] = coefficients[
                :m, n_tor - n : n_tor + n + 1
            ]
            return out

        # the axis guess: the boundary's m = 0 terms
        rbc, zbs = resize(r_cos), resize(z_sin)
        raxis = np.concatenate(
            [[rbc[0, n_out]], rbc[0, n_out + 1 :] + rbc[0, n_out - 1 :: -1]]
        )
        zaxis = np.concatenate([[0.0], zbs[0, n_out + 1 :] - zbs[0, n_out - 1 :: -1]])
        # the mean cross-section area, from a sampled boundary
        theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        phi = np.linspace(0.0, 2.0 * np.pi / surface.nfp, 16, endpoint=False)
        r, z = surface.rz(theta[:, None], phi[None, :])
        area = np.mean(
            np.abs(0.5 * np.sum(r * np.roll(z, -1, 0) - np.roll(r, -1, 0) * z, 0))
        )
        steps = len(ns)
        settings = {
            "lasym": lasym,
            "nfp": surface.nfp,
            "mpol": n_pol,
            "ntor": n_out,
            "ntheta": 0,
            "nzeta": 0,
            "ns_array": np.asarray(ns, dtype=int),
            "ftol_array": np.array([100.0 * ftol] * (steps - 1) + [ftol]),
            "niter_array": np.full(steps, niter, dtype=int),
            "phiedge": field_strength * area,
            "ncurr": 1,
            "pmass_type": "power_series",
            "am": np.zeros(1),
            "pres_scale": 0.0,
            "pcurr_type": "power_series",
            "ac": np.zeros(1),
            "curtor": 0.0,
            "lfreeb": False,
            "raxis_c": raxis,
            "zaxis_s": zaxis,
            "rbc": rbc,
            "zbs": zbs,
        }
        if lasym:
            settings.update(
                raxis_s=np.zeros(n_out + 1),
                zaxis_c=np.zeros(n_out + 1),
                rbs=resize(surface.r_sin),
                zbc=resize(surface.z_cos),
            )
        return vmecpp.VmecInput(**_defaults(), **settings)

    @classmethod
    def from_boundary(
        cls,
        boundary,
        *,
        field_strength: float = 1.0,
        mpol: int | None = None,
        ntor: int | None = None,
        ns: tuple[int, ...] = (9, 17),
        max_threads: int = 1,
        **kwargs,
    ) -> VMECEquilibrium:
        r"""Return the vacuum ($\beta = 0$, no current) equilibrium inside a ConStellaration boundary.

        See `vacuum_input` for the arguments; ``kwargs`` go to `VMECEquilibrium`.
        """
        vmec_input = cls.vacuum_input(
            boundary, field_strength=field_strength, mpol=mpol, ntor=ntor, ns=ns
        )
        return cls.from_input(vmec_input, max_threads=max_threads, **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box holding the last closed surface [m]."""
        return self._bounds

    # ------------------------------------------------------------------
    # The inverse map
    # ------------------------------------------------------------------

    def flux_coordinates(self, points):
        r"""Return $(\rho, u, v)$ of Cartesian ``points`` ``(..., 3)``.

        $\rho$ is 1 plus the residual of the inverse map over the minor radius
        for points outside the last closed surface.
        """
        points = xp.to_numpy(xp.asarray(points, dtype=float))
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        r_target = np.hypot(flat[:, 0], flat[:, 1])
        z_target = flat[:, 2]
        zeta = np.mod(np.arctan2(flat[:, 1], flat[:, 0]), 2.0 * np.pi)

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
            x1, x1_r, x1_t, _ = self._r(rho, theta, za)
            x2, x2_r, x2_t, _ = self._z(rho, theta, za)
            f1, f2 = x1 - r_target[active], x2 - z_target[active]
            previous = residual[active]
            residual[active] = np.hypot(f1, f2)
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
        r"""Return $\rho$ at Cartesian ``points`` ``(..., 3)``."""
        return xp.asarray(self.flux_coordinates(points)[0])

    def inside_at(self, points):
        """Return whether Cartesian ``points`` are inside the last closed surface."""
        return self.rho_at(points) < 1.0

    def inside_wall_at(self, points):
        """Return True everywhere: VMEC equilibria carry no wall."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def _field(self, rho, theta, zeta):
        r"""Return $\mathbf{B}$, shape ``(n, 3)``, at flux coordinates."""
        r, _, r_u, r_v = self._r(rho, theta, zeta)
        _, _, z_u, z_v = self._z(rho, theta, zeta)
        b_u = self._b_u(rho, theta, zeta)[0]
        b_v = self._b_v(rho, theta, zeta)[0]
        cos_p, sin_p = np.cos(zeta), np.sin(zeta)
        e_u = np.stack([r_u * cos_p, r_u * sin_p, z_u], axis=-1)
        e_v = np.stack([r_v * cos_p - r * sin_p, r_v * sin_p + r * cos_p, z_v], axis=-1)
        return b_u[:, None] * e_u + b_v[:, None] * e_v

    def field_at(self, points):
        r"""Return $\mathbf{B}$ [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``.

        Points outside the last closed surface get the field on it.
        """
        rho, theta, zeta = (c.ravel() for c in self.flux_coordinates(points))
        field = self._field(np.minimum(rho, 1.0), theta, zeta)
        return xp.asarray(field.reshape(*xp.asarray(points).shape[:-1], 3))

    def iota(self, rho):
        r"""Return the rotational transform $\iota$ at ``rho``."""
        return self._iota(np.asarray(rho, dtype=float) ** 2)

    def _shape(self, rho, theta, zeta):
        r"""Return the surface shape and the field at flux coordinates.

        $R$, $\partial_\rho R$, $\partial_\theta R$, $\partial_\rho Z$, $\partial_\theta Z$,
        $|B|$ and $\iota$.
        """
        r, r_r, r_t, _ = self._r(rho, theta, zeta)
        _, z_r, z_t, _ = self._z(rho, theta, zeta)
        strength = self._b(rho, theta, zeta)[0]
        return r, r_r, r_t, z_r, z_t, strength, self.iota(rho)

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the shells, see `FluxSurfaces.from_flux_coordinates`."""
        return FluxSurfaces.from_flux_coordinates(
            self._shape, self.nfp, rho_edges, **kwargs
        )
