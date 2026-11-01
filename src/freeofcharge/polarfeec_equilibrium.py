r"""3D equilibria through PolarFEEC.jl (the Python package ``polarfeec``).

PolarFEEC.jl reads GVEC and DESC equilibria in Julia and inverts their maps
robustly, across the magnetic axis included; ``polarfeec`` is its Python
wrapper (``python/`` of PolarFEEC.jl, installed from a checkout:
``pip install -e <PolarFEEC.jl>/python``). Its logical coordinates are
``(rho, y, z)`` in ``[0, 1]^3``: the flux label ``rho``, the poloidal angle
``theta = 2 pi y`` and the toroidal angle ``zeta = 2 pi z`` over the whole
torus. The flux surfaces need poloidal sections at fixed ``zeta`` in a
cylindrical ``(R, Z)`` plane (GVEC with ``which_hmap = 1``, DESC); the
derivatives of ``R`` and ``Z`` are central differences of the map.
"""

from __future__ import annotations

from pathlib import Path

import cunumpy as xp
import numpy as np

from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["PolarFEECEquilibrium"]


def _require_polarfeec():
    try:
        import polarfeec  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "PolarFEEC equilibria need the polarfeec package: pip install -e "
            "<a checkout of PolarFEEC.jl>/python"
        ) from error
    return polarfeec


class PolarFEECEquilibrium:
    """A PolarFEEC.jl equilibrium as a :class:`~freeofcharge.MagneticEquilibrium`.

    Points outside the last closed surface, and points PolarFEEC.jl could not
    invert, have ``rho = inf`` and the field ``NaN``.

    Args:
        equilibrium: A ``polarfeec.Equilibrium``.
        boundary_grid: Points ``(theta, zeta)`` on the last closed surface over the
            whole torus for the bounding box.
        step: Step of the central differences of ``R`` and ``Z`` in ``rho`` and
            ``theta``.
    """

    has_wall = False

    def __init__(
        self,
        equilibrium,
        *,
        boundary_grid: tuple[int, int] = (64, 128),
        step: float = 1.0e-6,
    ) -> None:
        """Cache the bounding box of the last closed surface."""
        self.equilibrium = equilibrium
        self.nfp = int(equilibrium.nfp)
        self.step = step
        n_theta, n_zeta = boundary_grid
        y, z = np.meshgrid(
            np.linspace(0.0, 1.0, n_theta, endpoint=False),
            np.linspace(0.0, 1.0, n_zeta, endpoint=False),
            indexing="ij",
        )
        edge = equilibrium.to_cartesian(
            np.stack([np.ones(y.size), y.ravel(), z.ravel()], axis=-1)
        )
        r_max = float(np.hypot(edge[:, 0], edge[:, 1]).max())
        self._bounds = (
            np.array([-r_max, -r_max, float(edge[:, 2].min())]),
            np.array([r_max, r_max, float(edge[:, 2].max())]),
        )

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> PolarFEECEquilibrium:
        """Return the equilibrium of a GVEC run directory or a DESC HDF5 file.

        ``representation`` and ``regularity`` go to ``polarfeec.Equilibrium``, the
        rest to :class:`PolarFEECEquilibrium`.
        """
        polarfeec = _require_polarfeec()
        options = {
            k: kwargs.pop(k) for k in ("representation", "regularity") if k in kwargs
        }
        return cls(polarfeec.Equilibrium(str(path), **options), **kwargs)

    @classmethod
    def from_gvec_files(
        cls, parameterfile: str | Path, statefile: str | Path, **kwargs
    ) -> PolarFEECEquilibrium:
        """Return the equilibrium of a GVEC parameter file and state file."""
        polarfeec = _require_polarfeec()
        options = {
            k: kwargs.pop(k) for k in ("representation", "regularity") if k in kwargs
        }
        equilibrium = polarfeec.Equilibrium(
            (str(parameterfile), str(statefile)), **options
        )
        return cls(equilibrium, **kwargs)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a box holding the last closed surface [m]."""
        return self._bounds

    # ------------------------------------------------------------------
    # MagneticEquilibrium
    # ------------------------------------------------------------------

    def logical_at(self, points):
        """Return the logical points ``(rho, y, z)`` of Cartesian ``points`` ``(..., 3)``.

        ``NaN`` outside the last closed surface and where the inversion failed.
        """
        points = xp.to_numpy(xp.asarray(points, dtype=float))
        flat = points.reshape(-1, 3)
        logical = self.equilibrium.to_logical(flat) if flat.shape[0] else flat.copy()
        return logical.reshape(points.shape)

    def rho_at(self, points):
        """Return ``rho`` at Cartesian ``points`` ``(..., 3)``, ``inf`` outside."""
        rho = self.logical_at(points)[..., 0]
        return xp.asarray(np.where(np.isnan(rho), np.inf, rho))

    def inside_at(self, points):
        """Return whether Cartesian ``points`` are inside the last closed surface."""
        return self.rho_at(points) < 1.0

    def inside_wall_at(self, points):
        """Return True everywhere: these equilibria carry no wall."""
        return xp.ones(xp.asarray(points).shape[:-1], dtype=bool)

    def field_at(self, points):
        """Return ``B`` [T] at Cartesian ``points`` ``(..., 3)``, ``NaN`` outside."""
        logical = self.logical_at(points)
        flat = logical.reshape(-1, 3)
        field = self.equilibrium.field(flat) if flat.shape[0] else flat.copy()
        return xp.asarray(field.reshape(logical.shape))

    def _shape(self, rho, theta, zeta):
        """Return ``R``, ``dR/drho``, ``dR/dtheta``, ``dZ/drho``, ``dZ/dtheta``, |B|, iota."""
        rho, theta, zeta = (np.asarray(c, dtype=float) for c in (rho, theta, zeta))
        y, z = theta / (2.0 * np.pi), zeta / (2.0 * np.pi)
        h = self.step
        # central differences, one-sided at rho = 0 and 1
        upper, lower = np.minimum(rho + h, 1.0), np.maximum(rho - h, 0.0)
        dy = h / (2.0 * np.pi)
        logical = np.concatenate(
            [
                np.stack([rho, y, z], -1),
                np.stack([upper, y, z], -1),
                np.stack([lower, y, z], -1),
                np.stack([rho, y + dy, z], -1),
                np.stack([rho, y - dy, z], -1),
            ]
        )
        x = self.equilibrium.to_cartesian(logical).reshape(5, rho.size, 3)
        r, z_coord = np.hypot(x[..., 0], x[..., 1]), x[..., 2]
        width = upper - lower
        r_r = (r[1] - r[2]) / width
        z_r = (z_coord[1] - z_coord[2]) / width
        r_t = (r[3] - r[4]) / (2.0 * h)
        z_t = (z_coord[3] - z_coord[4]) / (2.0 * h)
        strength = np.linalg.norm(self.equilibrium.field(logical[: rho.size]), axis=-1)
        iota = self.equilibrium.rotational_transform(rho)
        return r[0], r_r, r_t, z_r, z_t, strength, iota

    def flux_surfaces(self, rho_edges, **kwargs) -> FluxSurfaces:
        """Return the shells, see :meth:`FluxSurfaces.from_flux_coordinates`."""
        return FluxSurfaces.from_flux_coordinates(
            self._shape, self.nfp, rho_edges, **kwargs
        )
