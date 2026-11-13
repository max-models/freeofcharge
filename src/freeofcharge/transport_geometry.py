r"""Flux-surface averages for 1.5D transport codes.

A transport code solves on the radius $\rho$ alone; the shape of the surfaces
enters through a few averages per surface. With the Jacobian $\sqrt g$ of flux
coordinates $(\rho, \theta, \zeta)$, the flux-surface average and the
differential volume are

$$
\langle f \rangle = \frac{\oint f\,\sqrt g\, d\theta\, d\zeta}{\oint \sqrt g\, d\theta\, d\zeta},
\qquad
V'(\rho) = \frac{dV}{d\rho} = \oint |\sqrt g|\, d\theta\, d\zeta,
$$

and the gradient of the radius follows from the covariant metric,

$$
|\nabla\rho|^2 = \frac{g_{\theta\theta} g_{\zeta\zeta} - g_{\theta\zeta}^2}{g}.
$$

`surface_geometry` computes them from these quantities on a $(\theta, \zeta)$
grid of each surface, whatever the equilibrium; `GVECEquilibrium.transport_geometry`
gives it GVEC's metric. The same averages hold in a tokamak and in a
stellarator.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from methodcard import method

from freeofcharge.equilibrium import effective_trapped_fraction, rho_coordinate

__all__ = ["surface_geometry"]


def _shape_parameters(r, z):
    r"""Elongation and triangularity of each surface, averaged over the slices.

    ``r`` and ``z`` have the shape ``(surfaces, theta, zeta)``; each fixed
    $\zeta$ is one poloidal slice.
    """
    r_max, r_min = r.max(axis=1), r.min(axis=1)
    major = 0.5 * (r_max + r_min)
    minor = 0.5 * (r_max - r_min)
    upper = np.take_along_axis(r, z.argmax(axis=1)[:, None, :], axis=1)[:, 0]
    lower = np.take_along_axis(r, z.argmin(axis=1)[:, None, :], axis=1)[:, 0]
    elongation = (z.max(axis=1) - z.min(axis=1)) / (2.0 * minor)
    upper_triangularity = (major - upper) / minor
    lower_triangularity = (major - lower) / minor
    return (
        r_min.mean(axis=1),
        r_max.mean(axis=1),
        elongation.mean(axis=1),
        upper_triangularity.mean(axis=1),
        lower_triangularity.mean(axis=1),
    )


@method(
    equations={
        "average": r"\langle f\rangle = \frac{\sum f\,|\sqrt g|}{\sum |\sqrt g|}",
        "volume_derivative": r"V' = N_\mathrm{fp}\,\frac{2\pi}{n_\theta}\frac{2\pi/N_\mathrm{fp}}{n_\zeta}\sum |\sqrt g|",
        "gradient": r"|\nabla\rho|^2 = \frac{g_{\theta\theta}g_{\zeta\zeta} - g_{\theta\zeta}^2}{g}",
    },
    assumptions=(
        r"a uniform $(\theta, \zeta)$ grid over one field period, endpoints excluded: the periodic trapezoidal rule",
        r"$\rho > 0$: the averages are 0/0 on the axis",
        r"elongation, triangularity, $R_\mathrm{in}$ and $R_\mathrm{out}$ per fixed-$\zeta$ slice, then averaged over the slices",
    ),
)
def surface_geometry(
    rho,
    jacobian,
    grad_rho2,
    major_radius,
    height,
    field_strength,
    nfp: int = 1,
    *,
    rotational_transform=None,
    toroidal_current=None,
    toroidal_flux_edge: float | None = None,
    radial_coordinate: str = "toroidal",
) -> xr.Dataset:
    r"""Return the flux-surface averages that a 1.5D transport code needs.

    The arrays have the shape ``(surfaces, theta, zeta)``, on a uniform grid of
    $\theta \in [0, 2\pi)$ and $\zeta \in [0, 2\pi/N_\mathrm{fp})$ (endpoints
    excluded) of each surface $\rho$.

    Args:
        rho: Radii of the surfaces, all above zero.
        jacobian: $\sqrt g$ of the flux coordinates [$\mathrm{m^3}$].
        grad_rho2: $|\nabla\rho|^2$ [$\mathrm{m^{-2}}$].
        major_radius: $R$ [m].
        height: $Z$ [m].
        field_strength: $|B|$ [T].
        nfp: Number of field periods.
        rotational_transform: $\iota$ per surface, optional.
        toroidal_current: Toroidal current enclosed by each surface [A], optional.
        toroidal_flux_edge: Toroidal flux through the last surface [Wb], optional.
        radial_coordinate: Which flux $\rho$ is the root of (`rho_coordinate`).

    Returns:
        The dataset over ``rho``: ``volume_derivative`` $V'$, ``grad_rho``
        $\langle|\nabla\rho|\rangle$, ``grad_rho2`` $\langle|\nabla\rho|^2\rangle$,
        ``grad_rho2_over_R2`` $\langle|\nabla\rho|^2/R^2\rangle$, ``inv_R``
        $\langle 1/R\rangle$, ``inv_R2`` $\langle 1/R^2\rangle$, ``B2``
        $\langle B^2\rangle$, ``inv_B2`` $\langle 1/B^2\rangle$,
        ``trapped_fraction`` (effective), ``R_in``, ``R_out``, ``elongation``,
        ``triangularity_upper``, ``triangularity_lower``, and where given
        ``rotational_transform`` and ``toroidal_current``.
    """
    rho = np.asarray(rho, dtype=float)
    if np.any(rho <= 0.0):
        raise ValueError("the surfaces need rho > 0; the averages are 0/0 on the axis")
    jacobian = np.abs(np.asarray(jacobian, dtype=float))
    grad_rho2 = np.asarray(grad_rho2, dtype=float)
    r = np.asarray(major_radius, dtype=float)
    z = np.asarray(height, dtype=float)
    b = np.asarray(field_strength, dtype=float)
    shape = (rho.size, -1)
    n_theta, n_zeta = jacobian.shape[1:]
    cell = (2.0 * np.pi / n_theta) * (2.0 * np.pi / nfp / n_zeta)
    total = jacobian.reshape(shape).sum(axis=1)
    weights = jacobian.reshape(shape) / total[:, None]

    def average(values):
        return np.sum(weights * np.asarray(values).reshape(shape), axis=1)

    r_in, r_out, elongation, delta_upper, delta_lower = _shape_parameters(r, z)
    coords = {"rho": ("rho", rho, rho_coordinate(radial_coordinate))}
    variables = {
        "volume_derivative": (nfp * cell * total, "dV/drho", "m^3"),
        "grad_rho": (average(np.sqrt(grad_rho2)), r"<|grad rho|>", "1/m"),
        "grad_rho2": (average(grad_rho2), r"<|grad rho|^2>", "1/m^2"),
        "grad_rho2_over_R2": (
            average(grad_rho2 / r**2),
            r"<|grad rho|^2/R^2>",
            "1/m^4",
        ),
        "inv_R": (average(1.0 / r), "<1/R>", "1/m"),
        "inv_R2": (average(1.0 / r**2), "<1/R^2>", "1/m^2"),
        "B2": (average(b**2), "<B^2>", "T^2"),
        "inv_B2": (average(1.0 / b**2), "<1/B^2>", "1/T^2"),
        "trapped_fraction": (
            effective_trapped_fraction(b.reshape(shape), weights),
            "effective trapped fraction",
            "",
        ),
        "R_in": (r_in, "inboard radius", "m"),
        "R_out": (r_out, "outboard radius", "m"),
        "elongation": (elongation, "elongation", ""),
        "triangularity_upper": (delta_upper, "upper triangularity", ""),
        "triangularity_lower": (delta_lower, "lower triangularity", ""),
    }
    if rotational_transform is not None:
        variables["rotational_transform"] = (
            np.broadcast_to(np.asarray(rotational_transform, dtype=float), rho.shape),
            "rotational transform",
            "",
        )
    if toroidal_current is not None:
        variables["toroidal_current"] = (
            np.broadcast_to(np.asarray(toroidal_current, dtype=float), rho.shape),
            "enclosed toroidal current",
            "A",
        )
    attrs = {"nfp": int(nfp), "radial_coordinate": radial_coordinate}
    if toroidal_flux_edge is not None:
        attrs["toroidal_flux_edge"] = float(toroidal_flux_edge)
    return xr.Dataset(
        {
            name: ("rho", np.asarray(values), {"long_name": long, "units": units})
            for name, (values, long, units) in variables.items()
        },
        coords=coords,
        attrs=attrs,
    )
