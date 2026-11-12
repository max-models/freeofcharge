r"""A 3D equilibrium sampled once on a grid per field period, then interpolated.

The 3D adapters invert their maps point by point: GVEC about 30 µs per point,
DESC 1–3 ms. `GriddedEquilibrium` wraps any `MagneticEquilibrium`, samples
$s = \rho^2$ and the cylindrical components $(B_R, B_\varphi, B_Z)$ once on a
uniform grid in $(R, \varphi, Z)$ over one of the $N_\mathrm{fp}$ field periods,
and answers by cubic B-spline interpolation (``scipy.ndimage`` through cunumpy,
so on the GPU too). A Cartesian point is reduced to the period,

$$
\varphi' = \operatorname{atan2}(y, x) \bmod \frac{2\pi}{N_\mathrm{fp}},
$$

where the cylindrical components are the same, and the interpolated field is
rotated back with the point's own $\varphi$. With stellarator symmetry,

$$
s(R, -\varphi, -Z) = s(R, \varphi, Z), \quad
B_R(R, -\varphi, -Z) = -B_R(R, \varphi, Z), \quad
(B_\varphi, B_Z)(R, -\varphi, -Z) = (B_\varphi, B_Z)(R, \varphi, Z),
$$

only half a period is sampled and the other half filled in. An adapter with
``rho_and_field_in_planes`` (DESC's) is sampled plane by plane through it, much
faster than point by point.

Accuracy: $s$ and $\mathbf{B}$ are smooth inside the plasma, where the error
falls as $h^4$ with the cell size $h$; $\rho^2$ rather than $\rho$ because
$\rho$ has a cone on the magnetic axis. At $\rho = 1$ the adapters continue
$\rho$ and $\mathbf{B}$ outside with a kink (GVEC and DESC: $\rho = 1 + d/a$,
the field of the nearest boundary point), so within about two cells of the
last closed surface the error is first order in $h$, and the surface found as
$\rho = 1$ of the interpolant is off by a fraction of a cell. Outside the grid
$\rho$ and $\mathbf{B}$ are NaN (so `inside_at` is False there).
"""

from __future__ import annotations

from pathlib import Path

import cunumpy as xp
import numpy as np
import xarray as xr
from methodcard import ConvergenceProblem, method
from scipy import ndimage

from freeofcharge.equilibrium import (
    FluxSurfaces,
    MagneticEquilibrium,
    radial_coordinate_of,
)

__all__ = ["GriddedEquilibrium"]

_COMPONENTS = ("rho_squared", "b_r", "b_phi", "b_z")


class _HelicalTestField:
    r"""A smooth, stellarator-symmetric 3D test field with closed forms.

    $\rho^2 = (u^2 + v^2) / a^2$ with $u = (R^2 - R_0^2) / (2R_0) - \delta\cos N\varphi$
    and $v = Z - \delta\sin N\varphi$; $B_R = c\,v$, $B_\varphi = B_0 R_0 / R$,
    $B_Z = -c\,u$ (not divergence free; a test of the interpolation only).
    """

    has_wall = False

    def __init__(self, nfp: int = 5, major_radius: float = 3.0) -> None:
        """Set the shape."""
        self.nfp = nfp
        self.stellarator_symmetric = True
        self.r0, self.a, self.delta = major_radius, 0.5, 0.1

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box around the plasma [m]."""
        r_max = self.r0 + self.a + self.delta + 0.1
        z_max = self.a + self.delta
        return np.array([-r_max, -r_max, -z_max]), np.array([r_max, r_max, z_max])

    def _uv(self, points):
        points = xp.asarray(points, dtype=float)
        r = xp.hypot(points[..., 0], points[..., 1])
        phi = xp.arctan2(points[..., 1], points[..., 0])
        u = (r**2 - self.r0**2) / (2.0 * self.r0) - self.delta * xp.cos(self.nfp * phi)
        v = points[..., 2] - self.delta * xp.sin(self.nfp * phi)
        return r, phi, u, v

    def rho_at(self, points):
        """Return rho."""
        _, _, u, v = self._uv(points)
        return xp.sqrt(u**2 + v**2) / self.a

    def inside_at(self, points):
        """Return rho < 1."""
        return self.rho_at(points) < 1.0

    def inside_wall_at(self, points):
        """Return True everywhere."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def field_at(self, points):
        """Return B."""
        r, phi, u, v = self._uv(points)
        b_r, b_phi, b_z = 0.3 * v, 2.0 * self.r0 / r, -0.3 * u
        cos_p, sin_p = xp.cos(phi), xp.sin(phi)
        return xp.stack(
            [b_r * cos_p - b_phi * sin_p, b_r * sin_p + b_phi * cos_p, b_z], axis=-1
        )

    def _shape(self, rho, theta, zeta):
        r"""Return $R$, its and $Z$'s derivatives, $|B|$ and $\iota$ at flux coordinates.

        $u = a\rho\cos\theta$, $v = a\rho\sin\theta$, so the Jacobian
        $R\,(R_\rho Z_\theta - R_\theta Z_\rho) = R_0 a^2 \rho$ and the shell
        volumes are $2\pi^2 R_0 a^2 (\rho_{k+1}^2 - \rho_k^2)$.
        """
        u = self.a * rho * np.cos(theta) + self.delta * np.cos(self.nfp * zeta)
        z = self.a * rho * np.sin(theta) + self.delta * np.sin(self.nfp * zeta)
        r = np.sqrt(self.r0**2 + 2.0 * self.r0 * u)
        r_rho = self.r0 * self.a * np.cos(theta) / r
        r_theta = -self.r0 * self.a * rho * np.sin(theta) / r
        z_rho, z_theta = self.a * np.sin(theta), self.a * rho * np.cos(theta)
        points = np.stack([r * np.cos(zeta), r * np.sin(zeta), z], axis=-1)
        strength = np.linalg.norm(xp.to_numpy(self.field_at(points)), axis=-1)
        return r, r_rho, r_theta, z_rho, z_theta, strength, np.full(r.shape, 0.4)

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the shells, see `FluxSurfaces.from_flux_coordinates`."""
        return FluxSurfaces.from_flux_coordinates(
            self._shape, self.nfp, rho_edges, **kwargs
        )


def _interpolation_problem() -> ConvergenceProblem:
    """The gridded test field against its closed form, at points inside."""
    exact = _HelicalTestField()
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.2, 0.9, 400)
    theta = rng.uniform(0.0, 2.0 * np.pi, 400)
    phi = rng.uniform(0.0, 2.0 * np.pi, 400)
    # points at the radius rho of the test field, through its (u, v)
    u = exact.a * rho * np.cos(theta) + exact.delta * np.cos(exact.nfp * phi)
    v = exact.a * rho * np.sin(theta) + exact.delta * np.sin(exact.nfp * phi)
    r = np.sqrt(exact.r0**2 + 2.0 * exact.r0 * u)
    points = np.stack([r * np.cos(phi), r * np.sin(phi), v], axis=-1)
    field = xp.to_numpy(exact.field_at(points))

    def error(step: float) -> dict[str, float]:
        n = round(1.0 / step)
        grid = GriddedEquilibrium(exact, resolution=(n, n, n), margin=0.05)
        b = xp.to_numpy(grid.field_at(points))
        return {
            "rho": float(
                np.sqrt(np.mean((xp.to_numpy(grid.rho_at(points)) - rho) ** 2))
            ),
            "field": float(np.sqrt(np.mean((b - field) ** 2))),
        }

    return ConvergenceProblem(error, steps=[1 / 48, 1 / 64, 1 / 96], tolerance=0.3)


def _fill_non_finite(values: dict, dr: float, dz: float) -> None:
    r"""Fill non-finite samples in place from the nearest finite one in their plane.

    $\rho$ becomes $\max(\rho_\mathrm{nearest}, 1) + d / a$, $d$ the distance to
    it and $a$ half the width in $R$ of the samples inside; the field the
    nearest one's.
    """
    rho = values["rho"]
    inside = np.isfinite(rho) & (rho < 1.0)
    widths = [
        np.ptp(np.nonzero(inside[:, j].any(axis=1))[0])
        for j in range(rho.shape[1])
        if inside[:, j].any()
    ]
    a = 0.5 * dr * max(max(widths, default=1), 1)
    field_bad = ~np.isfinite(values["b_r"])
    for name in ("b_phi", "b_z"):
        field_bad |= ~np.isfinite(values[name])
    for bad, names in (
        (~np.isfinite(rho), ("rho",)),
        (field_bad, ("b_r", "b_phi", "b_z")),
    ):
        for j in range(rho.shape[1]):
            plane = bad[:, j]
            if not plane.any():
                continue
            if plane.all():
                raise ValueError("a toroidal plane of the grid has no finite sample")
            result = ndimage.distance_transform_edt(  # (distances, indices)
                plane, sampling=(dr, dz), return_indices=True
            )
            distance, (ni, nk) = result  # pyright: ignore[reportGeneralTypeIssues,reportOptionalIterable]
            for name in names:
                nearest = values[name][:, j][ni, nk][plane]
                if name == "rho":
                    nearest = np.maximum(nearest, 1.0) + distance[plane] / a
                values[name][:, j][plane] = nearest


@method(
    equations={
        "period": r"\varphi' = \operatorname{atan2}(y, x) \bmod \frac{2\pi}{N_\mathrm{fp}}",
        "symmetry": r"s(R, -\varphi, -Z) = s(R, \varphi, Z), \quad B_R(R, -\varphi, -Z) = -B_R(R, \varphi, Z), \quad (B_\varphi, B_Z)(R, -\varphi, -Z) = (B_\varphi, B_Z)(R, \varphi, Z)",
        "interpolation": r"f(R, \varphi', Z) = \sum_{ijk} c_{ijk}\, \beta_3\!\left(\frac{R - R_i}{h_R}\right) \beta_3\!\left(\frac{\varphi' - \varphi_j}{h_\varphi}\right) \beta_3\!\left(\frac{Z - Z_k}{h_Z}\right), \quad f \in \{\rho^2, B_R, B_\varphi, B_Z\}",
        "field": r"\mathbf{B} = (B_R\cos\varphi - B_\varphi\sin\varphi,\; B_R\sin\varphi + B_\varphi\cos\varphi,\; B_Z)",
    },
    order={"rho": 4, "field": 4},
    assumptions=(
        r"$\rho^2$ and the cylindrical components of $\mathbf{B}$ interpolated (cubic B-splines, periodic in $\varphi$), $\rho = \sqrt{\max(s, 0)}$",
        r"fourth order where they are smooth; first order within about two cells of $\rho = 1$, where the adapters continue them with a kink",
        r"the grid is padded in $R$ and $Z$ by odd reflection, so the spline's end conditions act outside it; NaN outside the grid",
        r"non-finite samples (outside a map's reach) filled from the nearest finite one, $\rho$ growing as $1 + d/a$",
    ),
    problem=_interpolation_problem,
)
class GriddedEquilibrium:
    r"""A `freeofcharge.MagneticEquilibrium` sampled on a grid per field period.

    $\rho$, $\mathbf{B}$ and `inside_at` come from the grid; `has_wall`,
    `inside_wall_at` and `flux_surfaces` from the wrapped equilibrium.

    Args:
        equilibrium: The equilibrium to sample (None only for `load`).
        resolution: Grid nodes $(R, \varphi, Z)$, $\varphi$ over one field period
            (an even number with stellarator symmetry, of which half plus one
            are sampled).
        nfp: Field periods; by default the equilibrium's ``nfp`` (1 if it has none).
        symmetric: Stellarator symmetry; by default the equilibrium's
            ``stellarator_symmetric`` (False if it has none).
        r_range: $(R_\min, R_\max)$ of the plasma [m]; by default found on a
            coarse probe of `inside_at` (``probe`` nodes in $R, \varphi, Z$).
        z_range: $(Z_\min, Z_\max)$ of the plasma [m]; by default from ``bounds``;
            made symmetric about $Z = 0$ with stellarator symmetry.
        margin: Space added around the plasma on each side, a fraction of its
            extent in $R$ and $Z$.
        order: Order of the B-splines (1 trilinear, 3 cubic).
        chunk: Points per call of the equilibrium while sampling.
        probe: Nodes $(R, \varphi, Z)$ of the probe for ``r_range``.
    """

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _plasma_box(
        equilibrium, nfp: int, r_range, z_range, probe
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        r"""Return $(R_\min, R_\max)$ and $(Z_\min, Z_\max)$ of the plasma."""
        lower, upper = (np.asarray(b, dtype=float) for b in equilibrium.bounds)
        if z_range is None:
            z_range = (float(lower[2]), float(upper[2]))
        if r_range is None:
            r_max = float(max(np.abs(lower[:2]).max(), np.abs(upper[:2]).max()))
            n_r, n_phi, n_z = probe
            r = np.linspace(0.0, r_max, n_r)
            phi = np.linspace(0.0, 2.0 * np.pi / nfp, n_phi, endpoint=False)
            z = np.linspace(z_range[0], z_range[1], n_z)
            rr, pp, zz = np.meshgrid(r, phi, z, indexing="ij")
            planes = getattr(equilibrium, "rho_and_field_in_planes", None)
            if planes is None:
                points = np.stack([rr * np.cos(pp), rr * np.sin(pp), zz], axis=-1)
                inside = xp.to_numpy(equilibrium.inside_at(points)).astype(bool)
            else:
                rho, _ = planes(
                    phi,
                    np.moveaxis(rr, 1, 0).reshape(n_phi, -1),
                    np.moveaxis(zz, 1, 0).reshape(n_phi, -1),
                )
                rho = np.moveaxis(np.asarray(rho).reshape(n_phi, n_r, n_z), 0, 1)
                inside = np.isfinite(rho) & (rho < 1.0)
            if not inside.any():
                raise ValueError("the probe found no point inside the plasma")
            r_inside = rr[inside]
            cell = r[1] - r[0]
            r_range = (
                max(float(r_inside.min()) - cell, 0.0),
                min(float(r_inside.max()) + cell, r_max),
            )
        return (float(r_range[0]), float(r_range[1])), (
            float(z_range[0]),
            float(z_range[1]),
        )

    @staticmethod
    def _sample_points(equilibrium, rr, pp, zz, chunk: int) -> dict:
        r"""Return $\rho$ and the cylindrical field at the nodes, in chunks of points."""
        points = np.stack([rr * np.cos(pp), rr * np.sin(pp), zz], axis=-1).reshape(
            -1, 3
        )
        rho = np.empty(points.shape[0])
        field = np.empty((points.shape[0], 3))
        # one inversion for both where the adapter offers it
        both = getattr(equilibrium, "rho_and_field_at", None)
        for start in range(0, points.shape[0], chunk):
            part = slice(start, start + chunk)
            if both is None:
                sample = (
                    equilibrium.rho_at(points[part]),
                    equilibrium.field_at(points[part]),
                )
            else:
                sample = both(points[part])
            rho[part] = xp.to_numpy(sample[0]).reshape(-1)
            field[part] = xp.to_numpy(sample[1]).reshape(-1, 3)
        cos_p, sin_p = np.cos(pp).ravel(), np.sin(pp).ravel()
        values = {
            "rho": rho,
            "b_r": field[:, 0] * cos_p + field[:, 1] * sin_p,
            "b_phi": -field[:, 0] * sin_p + field[:, 1] * cos_p,
            "b_z": field[:, 2],
        }
        return {k: v.reshape(rr.shape) for k, v in values.items()}

    @classmethod
    def _sample(
        cls,
        equilibrium,
        resolution,
        *,
        nfp: int,
        symmetric: bool,
        r_range,
        z_range,
        margin: float,
        chunk: int,
        probe,
    ) -> xr.Dataset:
        """Return the samples on the grid of one field period."""
        n_r, n_phi, n_z = (int(n) for n in resolution)
        if symmetric and n_phi % 2:
            raise ValueError(
                "with stellarator symmetry the toroidal nodes must be even"
            )
        (r_lo, r_hi), (z_lo, z_hi) = cls._plasma_box(
            equilibrium, nfp, r_range, z_range, probe
        )
        if symmetric:
            z_hi = max(abs(z_lo), abs(z_hi))
            z_lo = -z_hi
        dr, dz = margin * (r_hi - r_lo), margin * (z_hi - z_lo)
        r = np.linspace(max(r_lo - dr, 0.0), r_hi + dr, n_r)
        z = np.linspace(z_lo - dz, z_hi + dz, n_z)
        period = 2.0 * np.pi / nfp
        phi = np.arange(n_phi) * period / n_phi
        sampled = phi[: n_phi // 2 + 1] if symmetric else phi
        rr, pp, zz = np.meshgrid(r, sampled, z, indexing="ij")
        planes = getattr(equilibrium, "rho_and_field_in_planes", None)
        if planes is not None:
            # the adapter's fast path for points in a few planes (DESC)
            plane_r = np.moveaxis(rr, 1, 0).reshape(len(sampled), -1)
            plane_z = np.moveaxis(zz, 1, 0).reshape(len(sampled), -1)
            rho, field = planes(sampled, plane_r, plane_z)
            values = {
                "rho": np.asarray(rho),
                "b_r": np.asarray(field)[..., 0],
                "b_phi": np.asarray(field)[..., 1],
                "b_z": np.asarray(field)[..., 2],
            }
            values = {
                k: np.moveaxis(v.reshape(len(sampled), len(r), len(z)), 0, 1)
                for k, v in values.items()
            }
        else:
            values = cls._sample_points(equilibrium, rr, pp, zz, chunk)
        _fill_non_finite(values, r[1] - r[0], z[1] - z[0])
        if symmetric:
            # (R, -phi, -Z): the planes n_phi - j from the planes j
            mirrored = slice(n_phi // 2 - 1, 0, -1)
            for name, sign in (("rho", 1), ("b_r", -1), ("b_phi", 1), ("b_z", 1)):
                values[name] = np.concatenate(
                    [values[name], sign * values[name][:, mirrored, ::-1]], axis=1
                )
        dims = ("R", "phi", "Z")
        return xr.Dataset(
            {
                "rho_squared": (dims, values["rho"] ** 2),
                "b_r": (dims, values["b_r"]),
                "b_phi": (dims, values["b_phi"]),
                "b_z": (dims, values["b_z"]),
            },
            coords={"R": r, "phi": phi, "Z": z},
            attrs={
                "nfp": nfp,
                "symmetric": int(symmetric),
                "radial_coordinate": radial_coordinate_of(equilibrium),
                "bounds_lower": np.asarray(equilibrium.bounds[0], dtype=float),
                "bounds_upper": np.asarray(equilibrium.bounds[1], dtype=float),
            },
        )

    def _prefilter(self, values: np.ndarray) -> np.ndarray:
        """Return the B-spline coefficients, padded for the evaluation."""
        p = self._pad
        # odd reflection in R and Z keeps linear trends, so the spline's end
        # conditions act on the padding; periodic in phi
        values = np.pad(
            values, ((p, p), (0, 0), (p, p)), mode="reflect", reflect_type="odd"
        )
        if self.order > 1:
            for axis, mode in ((0, "mirror"), (1, "grid-wrap"), (2, "mirror")):
                values = ndimage.spline_filter1d(
                    values, self.order, axis=axis, mode=mode
                )
        return np.pad(values, ((0, 0), (p, p), (0, 0)), mode="wrap")

    def __init__(
        self,
        equilibrium: MagneticEquilibrium | None,
        resolution: int | tuple[int, int, int] = (64, 32, 64),
        *,
        nfp: int | None = None,
        symmetric: bool | None = None,
        r_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        margin: float = 0.1,
        order: int = 3,
        chunk: int = 65536,
        probe: tuple[int, int, int] = (32, 8, 16),
        _samples: xr.Dataset | None = None,
    ) -> None:
        """Sample the equilibrium and prepare the splines."""
        self.equilibrium = equilibrium
        self.order = int(order)
        if not 1 <= self.order <= 5:
            raise ValueError(f"order must be 1 to 5, got {order}")
        if _samples is None:
            if equilibrium is None:
                raise ValueError("an equilibrium is needed to sample")
            _samples = self._sample(
                equilibrium,
                (resolution,) * 3 if isinstance(resolution, int) else resolution,
                nfp=int(getattr(equilibrium, "nfp", 1)) if nfp is None else nfp,
                symmetric=bool(getattr(equilibrium, "stellarator_symmetric", False))
                if symmetric is None
                else symmetric,
                r_range=r_range,
                z_range=z_range,
                margin=margin,
                chunk=chunk,
                probe=probe,
            )
        self.samples = _samples
        self.nfp = int(_samples.attrs["nfp"])
        self.stellarator_symmetric = bool(_samples.attrs["symmetric"])
        self.period = 2.0 * np.pi / self.nfp
        r, phi, z = (np.asarray(_samples[c]) for c in ("R", "phi", "Z"))
        self._lower = (float(r[0]), float(z[0]))
        self._step = (r[1] - r[0], self.period / len(phi), z[1] - z[0])
        self._shape = (len(r), len(phi), len(z))
        self._pad = self.order // 2 + 2
        self._coefficients = [
            xp.asarray(self._prefilter(np.asarray(_samples[name])))
            for name in _COMPONENTS
        ]

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the samples to a netCDF file."""
        self.samples.to_netcdf(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        equilibrium: MagneticEquilibrium | None = None,
        *,
        order: int = 3,
    ) -> GriddedEquilibrium:
        """Return the gridded equilibrium saved in ``path``.

        Without ``equilibrium`` there is no wall and no `flux_surfaces`.
        """
        with xr.open_dataset(path) as dataset:
            samples = dataset.load()
        return cls(equilibrium, order=order, _samples=samples)

    @classmethod
    def cached(
        cls,
        path: str | Path,
        equilibrium: MagneticEquilibrium,
        resolution: int | tuple[int, int, int] = (64, 32, 64),
        **kwargs,
    ) -> GriddedEquilibrium:
        """Return the samples saved in ``path`` if they have ``resolution``, else sample and save.

        A cache: sample a 3D equilibrium once (DESC's W7-X at (128, 64, 128)
        takes about half a minute), then load it in a second. ``kwargs`` go to
        the constructor; only the resolution is checked against the file.
        """
        path = Path(path)
        n_r, n_phi, n_z = (
            (resolution,) * 3 if isinstance(resolution, int) else resolution
        )
        shape = (int(n_r), int(n_phi), int(n_z))
        if path.exists():
            loaded = cls.load(path, equilibrium, order=kwargs.get("order", 3))
            if loaded._shape == shape:
                return loaded
        gridded = cls(equilibrium, shape, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        gridded.save(path)
        return gridded

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _interpolate(self, points, names):
        r"""Return the components ``names`` and $\varphi$ at Cartesian ``points``."""
        points = xp.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        r = xp.hypot(flat[:, 0], flat[:, 1])
        phi = xp.arctan2(flat[:, 1], flat[:, 0])
        p = self._pad
        i = (r - self._lower[0]) / self._step[0]
        j = xp.mod(phi, self.period) / self._step[1]
        k = (flat[:, 2] - self._lower[1]) / self._step[2]
        # written so that NaN coordinates count as outside
        inside = (i >= 0) & (i <= self._shape[0] - 1) & (k >= 0)
        inside &= (k <= self._shape[2] - 1) & xp.isfinite(j)
        # only the points on the grid: a beam line is mostly off it
        index = xp.nonzero(inside)[0]
        coordinates = xp.stack([i[index] + p, j[index] + p, k[index] + p])
        results = []
        for name in names:
            value = xp.full(flat.shape[0], xp.nan)
            value[index] = xp.scipy.ndimage.map_coordinates(
                self._coefficients[_COMPONENTS.index(name)],
                coordinates,
                order=self.order,
                mode="nearest",
                prefilter=False,
            )
            results.append(value.reshape(shape))
        return results, phi.reshape(shape)

    def rho_at(self, points):
        r"""Return $\rho$ at Cartesian ``points`` ``(..., 3)`` (NaN outside the grid)."""
        (s,), _ = self._interpolate(points, ("rho_squared",))
        return xp.sqrt(xp.maximum(s, 0.0))

    def inside_at(self, points):
        r"""Return whether the interpolated $\rho$ is below 1."""
        return self.rho_at(points) < 1.0

    def rho_and_inside_at(self, points):
        r"""Return $\rho$ and `inside_at` from one interpolation."""
        rho = self.rho_at(points)
        return rho, rho < 1.0

    def field_at(self, points):
        r"""Return $\mathbf{B}$ [T] at Cartesian ``points`` ``(..., 3)``, shape ``(..., 3)``."""
        (b_r, b_phi, b_z), phi = self._interpolate(points, ("b_r", "b_phi", "b_z"))
        cos_p, sin_p = xp.cos(phi), xp.sin(phi)
        return xp.stack(
            [b_r * cos_p - b_phi * sin_p, b_r * sin_p + b_phi * cos_p, b_z], axis=-1
        )

    # ------------------------------------------------------------------
    # Delegated to the equilibrium
    # ------------------------------------------------------------------

    @property
    def radial_coordinate(self) -> str:
        """Return the wrapped equilibrium's radial coordinate (saved with the samples)."""
        if "radial_coordinate" in self.samples.attrs:
            return str(self.samples.attrs["radial_coordinate"])
        return radial_coordinate_of(self.equilibrium)

    @property
    def has_wall(self) -> bool:
        """Return whether the equilibrium has a first wall."""
        return self.equilibrium is not None and self.equilibrium.has_wall

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the equilibrium's box [m]."""
        return (
            np.asarray(self.samples.attrs["bounds_lower"], dtype=float),
            np.asarray(self.samples.attrs["bounds_upper"], dtype=float),
        )

    def inside_wall_at(self, points):
        """Return whether the points are inside the equilibrium's first wall."""
        if self.equilibrium is None:
            return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)
        return self.equilibrium.inside_wall_at(points)

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the equilibrium's shells."""
        if self.equilibrium is None:
            raise ValueError("flux_surfaces needs the equilibrium: pass it to load")
        return self.equilibrium.flux_surfaces(rho_edges, **kwargs)
