r"""The steady-state fast-ion distribution $f(\rho, E, \xi)$, local and from the orbits.

A fast ion born at speed $v_0$ and pitch $\xi_0 = v_\parallel / v$ slows down as
$dv/dt = -a_e v - a_i / v^2$ and its pitch diffuses by the Lorentz operator
$\frac{\nu_D}{2}\partial_\xi (1 - \xi^2) \partial_\xi$ (`freeofcharge.slowing_down`,
`collision_coefficients`). A steady source of $w$ ions per second then holds
$w\,dt/dv\,dv$ ions between $v$ and $v + dv$, spread in pitch by the Green's
function of the Lorentz operator on the sphere,

$$
P(\xi \mid \xi_0, s) = \sum_{l \ge 0} \frac{2l + 1}{2} P_l(\xi_0) P_l(\xi)\,
e^{-l(l+1)s/2}, \qquad s(v) = \int_v^{v_0} \nu_D \frac{dt}{dv'}\,dv',
$$

with the Legendre polynomials $P_l$. With constant Coulomb logarithms
$dt/dv = \tau_s v^2 / (v^3 + v_c^3)$ and
$e^{-l(l+1)s/2} = [v^3 (v_0^3 + v_c^3) / (v_0^3 (v^3 + v_c^3))]^{l(l+1)\beta/6}$, so

$$
f(v, \xi) = \frac{w\,\tau_s}{2\pi V (v^3 + v_c^3)}
\sum_l \frac{2l + 1}{2} P_l(\xi_0) P_l(\xi)
\left[\frac{v^3 (v_0^3 + v_c^3)}{v_0^3 (v^3 + v_c^3)}\right]^{l(l+1)\beta/6}
$$

per unit volume of velocity space, $d^3v = 2\pi v^2\,dv\,d\xi$, for
$v_\mathrm{end} < v < v_0$: the classical slowing-down distribution
$f \propto S \tau_s / (v^3 + v_c^3)$ (Stix, Plasma Phys. 14, 367, 1972) with the
Legendre series of the pitch-angle scattering (Gaffey, J. Plasma Phys. 16, 149, 1976,
without its energy diffusion), the eigenfunctions of the Lorentz operator being the
$P_l$ with eigenvalues $l(l+1)/2$ (Helander and Sigmar, *Collisional Transport in
Magnetized Plasmas*, 2002). The $l = 1$ term
alone is the mean pitch of `freeofcharge.slowing_down.pitch_at_speed`.

`slowing_down_distribution` sums this over the births of a deposition, shell by
shell, with each species' own Coulomb logarithm as `freeofcharge.slow_down` does
(the drags and $\nu_D$ at each speed, $s$ integrated numerically). It returns the
distribution in energy and pitch, $f = dn / (dE\,d\xi)$ [$\mathrm{1/(m^3\,eV)}$],
averaged over bins: the time spent in an energy bin by Gauss-Legendre quadrature in
$v$, and the series integrated exactly over each pitch bin. Its moments
(`distribution_moments`) are those of `slow_down` with ``orbit_average=False``: the
pitch is the local one, scattered as in a uniform field (no mirror force, no
trapping), and every ion stays on its birth shell.

`orbit_distribution` turns the histogram that `freeofcharge.follow_orbits` records
with ``histogram=(energy_edges, pitch_edges)`` into the same layout: the time each
marker spends in each $(\rho, E, \xi)$ bin times its weight, over the shell volume
and the bin widths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from maybempi import MPI
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import to_unit

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.deposition import allreduce
from freeofcharge.equilibrium import rho_coordinate
from freeofcharge.slowing_down import collision_coefficients

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces
    from freeofcharge.orbits import OrbitFollowing
    from freeofcharge.profiles import Plasma

__all__ = [
    "distribution_at",
    "distribution_moments",
    "energy_spectrum",
    "legendre_bin_integrals",
    "legendre_polynomials",
    "orbit_distribution",
    "pitch_distribution",
    "slowing_down_distribution",
]

_E = ELEMENTARY_CHARGE


def legendre_polynomials(x, count: int) -> np.ndarray:
    r"""Return $P_0(x), \dots, P_{count-1}(x)$, shape ``(*x.shape, count)``.

    Bonnet's recursion $(l + 1) P_{l+1} = (2l + 1) x P_l - l P_{l-1}$.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty((*x.shape, max(count, 2)))
    out[..., 0] = 1.0
    out[..., 1] = x
    for degree in range(1, count - 1):
        out[..., degree + 1] = (
            (2 * degree + 1) * x * out[..., degree] - degree * out[..., degree - 1]
        ) / (degree + 1)
    return out[..., :count]


def legendre_bin_integrals(edges, count: int) -> np.ndarray:
    r"""Return $\int P_l\,d\xi$ over each bin of ``edges``, shape ``(bins, count)``.

    $\int_a^b P_l\,d\xi = [P_{l+1} - P_{l-1}]_a^b / (2l + 1)$ for $l \ge 1$, $b - a$
    for $l = 0$.
    """
    edges = np.asarray(edges, dtype=float)
    p = legendre_polynomials(edges, count + 1)
    antiderivative = np.empty((len(edges), count))
    antiderivative[:, 0] = edges
    degree = np.arange(1, count)
    antiderivative[:, 1:] = (p[:, 2 : count + 1] - p[:, : count - 1]) / (2 * degree + 1)
    return np.diff(antiderivative, axis=0)


def _gauss_nodes(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _column(values) -> np.ndarray:
    return np.asarray(xp.to_numpy(xp.asarray(values)), dtype=float)[:, None]


def _coefficients(speed, mass, plasma_here, plasma: Plasma, ln_lambda):
    """`collision_coefficients` at ``speed`` ``(n, G)`` with this chunk's plasma."""
    ne, te, ti, ions, ln = plasma_here
    a_e, a_i, deflection = collision_coefficients(
        speed,
        mass,
        1,
        ne,
        te,
        ions,
        plasma.ion_masses,
        plasma.ion_charges,
        ti,
        ln,
    )
    shape = np.shape(speed)
    return tuple(
        np.broadcast_to(np.asarray(xp.to_numpy(xp.asarray(c)), dtype=float), shape)
        for c in (a_e, a_i, deflection)
    )


def _dataset(
    counts, volume, energy_edges, pitch_edges, rho_edges, radial_coordinate, attrs
) -> xr.Dataset:
    """The distribution of ``counts`` (ions per bin), in the layout of this module."""
    widths = np.diff(energy_edges)[:, None] * np.diff(pitch_edges)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = counts / (np.asarray(volume, dtype=float)[:, None, None] * widths)
    rho = 0.5 * (rho_edges[1:] + rho_edges[:-1])
    return xr.Dataset(
        {
            "f": (
                ("rho", "energy", "pitch"),
                f,
                {"long_name": "fast-ion distribution", "units": "1/(m^3 eV)"},
            ),
            "volume": ("rho", np.asarray(volume, dtype=float), {"units": "m^3"}),
        },
        coords={
            "rho": ("rho", rho, rho_coordinate(radial_coordinate)),
            "energy": (
                "energy",
                0.5 * (energy_edges[1:] + energy_edges[:-1]),
                {"long_name": "$E$", "units": "eV"},
            ),
            "pitch": (
                "pitch",
                0.5 * (pitch_edges[1:] + pitch_edges[:-1]),
                {"long_name": r"$\xi = v_\parallel / v$"},
            ),
            "rho_edge": ("rho_edge", np.asarray(rho_edges, dtype=float)),
            "energy_edge": (
                "energy_edge",
                np.asarray(energy_edges, dtype=float),
                {"units": "eV"},
            ),
            "pitch_edge": ("pitch_edge", np.asarray(pitch_edges, dtype=float)),
        },
        attrs={"radial_coordinate": radial_coordinate, **attrs},
    )


def _slowing_down_distribution(
    deposition,
    plasma,
    surfaces,
    energy_edges,
    energy_bins,
    pitch_bins,
    legendre_modes,
    ln_lambda,
    thermal_factor,
    minimum_energy,
    speed_points,
    quadrature,
    chunk,
    comm,
) -> xr.Dataset:
    n_shells = len(surfaces.rho_edges) - 1
    mass = deposition.mass_number * ATOMIC_MASS
    births = deposition.births
    rho = np.minimum(np.asarray(xp.to_numpy(deposition.birth_rho), dtype=float), 1.0)
    velocities = np.asarray(xp.to_numpy(births.velocities), dtype=float)
    weights = np.asarray(xp.to_numpy(births.weights), dtype=float)
    pitch0 = np.asarray(xp.to_numpy(deposition.birth_pitch), dtype=float)
    speed = np.sqrt(np.sum(velocities**2, axis=1))
    energy = 0.5 * mass * speed**2  # J
    if energy_edges is None:
        top = float(energy.max()) / _E if len(energy) else 0.0
        if comm is not None and comm.Get_size() > 1:
            top = float(comm.allreduce(top, op=MPI.MAX))
        energy_edges = np.linspace(0.0, top, energy_bins + 1)
    energy_edges = np.asarray(energy_edges, dtype=float)
    pitch_edges = np.linspace(-1.0, 1.0, pitch_bins + 1)
    n_energy = len(energy_edges) - 1
    ti = np.asarray(xp.to_numpy(plasma.ion_temperature(rho)), dtype=float)
    end_energy = np.minimum(
        np.maximum(thermal_factor * ti, minimum_energy) * _E, energy
    )
    ratio = np.maximum(np.sqrt(end_energy / energy), 1.0e-4)
    shell = np.clip(
        np.asarray(
            xp.to_numpy(
                surfaces.shell_index(np.minimum(rho, 0.999999 * surfaces.rho_edges[-1]))
            )
        ),
        0,
        n_shells - 1,
    )
    lns = None
    if ln_lambda is not None:
        lns = np.broadcast_to(np.asarray(ln_lambda, dtype=float), speed.shape)
    v_edges = np.sqrt(2.0 * energy_edges * _E / mass)
    nodes, gauss = _gauss_nodes(quadrature)
    grid = np.linspace(0.0, 1.0, speed_points)
    degree = np.arange(legendre_modes)
    kappa = 0.5 * degree * (degree + 1)
    # sum_b w_b P_l(xi_0) int_bin e^{-kappa_l s} dt per shell, energy bin and l
    amplitudes = np.zeros((n_shells, n_energy, legendre_modes))
    for lo in range(0, len(speed), chunk):
        part = slice(lo, lo + chunk)
        v0 = speed[part]
        n = len(v0)
        here = (
            _column(plasma.electron_density(rho[part])),
            _column(plasma.electron_temperature(rho[part])),
            ti[part, None],
            [_column(d) for d in plasma.ion_densities(rho[part])],
            None if lns is None else lns[part, None],
        )
        # s on speeds equal in ln v, from the birth speed down to the end speed
        log_span = -np.log(ratio[part])
        fine = v0[:, None] * ratio[part, None] ** grid
        a_e, a_i, deflection = _coefficients(fine, mass, here, plasma, ln_lambda)
        rate = deflection / (a_e * fine**3 + a_i)  # nu_D dt / d(ln v)
        h = (log_span / (speed_points - 1))[:, None]
        s_fine = np.concatenate(
            [np.zeros((n, 1)), np.cumsum(0.5 * h * (rate[:, 1:] + rate[:, :-1]), 1)],
            axis=1,
        )
        # Gauss-Legendre nodes of each energy bin within [v_end, v_0]
        v_end = (v0 * ratio[part])[:, None]
        low = np.clip(v_edges[None, :-1], v_end, v0[:, None])
        high = np.clip(v_edges[None, 1:], v_end, v0[:, None])
        v = low[..., None] + (high - low)[..., None] * nodes  # (n, bins, q)
        a_e, a_i, _ = _coefficients(v.reshape(n, -1), mass, here, plasma, ln_lambda)
        dt = (
            (high - low)[..., None]
            * gauss
            / (a_e.reshape(v.shape) * v + a_i.reshape(v.shape) / v**2)
        )
        position = np.log(v0[:, None, None] / v) / h[:, :, None]
        position = np.clip(position, 0.0, speed_points - 1.0)
        index = np.minimum(position.astype(np.int64), speed_points - 2)
        frac = position - index
        flat = index.reshape(n, -1)
        s = np.take_along_axis(s_fine, flat, 1) * (1.0 - frac.reshape(n, -1))
        s += np.take_along_axis(s_fine, flat + 1, 1) * frac.reshape(n, -1)
        decay = np.exp(-s.reshape(v.shape)[..., None] * kappa)  # (n, bins, q, l)
        times = np.einsum("nbq,nbql->nbl", dt, decay)
        source = weights[part, None] * legendre_polynomials(
            pitch0[part], legendre_modes
        )
        contribution = (times * source[:, None, :]).reshape(n, -1)
        owner = np.zeros((n_shells, n))
        owner[shell[part], np.arange(n)] = 1.0
        amplitudes += (owner @ contribution).reshape(amplitudes.shape)
    amplitudes = allreduce(amplitudes, comm)
    series = (
        (2 * degree + 1) / 2.0 * legendre_bin_integrals(pitch_edges, legendre_modes)
    )
    counts = np.einsum("sel,pl->sep", amplitudes, series)
    return _dataset(
        counts,
        surfaces.volume,
        energy_edges,
        pitch_edges,
        np.asarray(surfaces.rho_edges, dtype=float),
        surfaces.radial_coordinate,
        {
            "mass_number": int(deposition.mass_number),
            "charge_number": 1,
            "legendre_modes": int(legendre_modes),
            "source": "slowing_down_distribution",
        },
    )


@method(
    equations={
        "distribution": r"f(v, \xi) = \frac{w}{2\pi V v^2} \frac{dt}{dv} \sum_l \frac{2l + 1}{2} P_l(\xi_0) P_l(\xi)\, e^{-l(l+1) s / 2}, \qquad \frac{dt}{dv} = \frac{1}{a_e v + a_i / v^2}, \qquad s = \int_v^{v_0} \nu_D \frac{dt}{dv'} dv'",
        "classical": r"\frac{dt}{dv} = \frac{\tau_s v^2}{v^3 + v_c^3}, \qquad e^{-l(l+1)s/2} = \left[\frac{v^3 (v_0^3 + v_c^3)}{v_0^3 (v^3 + v_c^3)}\right]^{l(l+1)\beta/6} \quad \text{(constant Coulomb logarithms)}",
        "bins": r"N_{ij} = \sum_b w_b \sum_l \frac{2l + 1}{2} P_l(\xi_{0,b}) \int_{E_i} e^{-l(l+1) s / 2}\, dt \int_{\xi_j} P_l\,d\xi, \qquad f_{ij} = \frac{N_{ij}}{V \Delta E_i \Delta\xi_j}",
    },
    assumptions=(
        r"the local slowing down of `slow_down` with `orbit_average=False`: every ion on its birth shell, the local pitch scattered as in a uniform field (no mirror force, no trapping)",
        r"no energy diffusion: nothing above the birth energy or below the end energy",
        r"the time in an energy bin by Gauss-Legendre quadrature in $v$; $s$ by the trapezoidal rule on `speed_points` speeds equal in $\ln v$, interpolated linearly",
        r"the Legendre series truncated after `legendre_modes` terms: near the birth energy, where the pitch is still narrow, the bins can ring (slightly negative); the moments up to $\xi^2$ are exact",
    ),
    references=(
        "[Stix, Plasma Phys. 14, 367 (1972)](https://doi.org/10.1088/0032-1028/14/4/002)",
        "[Gaffey, J. Plasma Phys. 16, 149 (1976)](https://doi.org/10.1017/S0022377800020134)",
        "Helander and Sigmar, Collisional Transport in Magnetized Plasmas (2002)",
    ),
    components=(collision_coefficients,),
)
def slowing_down_distribution(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    *,
    energy_edges=None,
    energy_bins: int = 120,
    pitch_bins: int = 40,
    legendre_modes: int = 48,
    ln_lambda=None,
    thermal_factor: float = 0.0,
    minimum_energy: float = 0.0,
    speed_points: int = 201,
    quadrature: int = 4,
    chunk: int = 256,
    comm: MPI.Comm | None = None,
) -> xr.Dataset:
    r"""Return the steady-state distribution $f(\rho, E, \xi)$ of the births of ``deposition``.

    Each birth slows down on its shell (as `freeofcharge.slow_down`), its pitch
    spread by the Legendre series of the Lorentz operator.

    Args:
        deposition: Births (this rank's) and their shells.
        plasma: Profiles.
        surfaces: Volumes of the shells (those of the deposition).
        energy_edges: Energy bins [eV]; ``energy_bins`` equal ones from 0 to the
            highest birth energy if None (a multiple of 6 puts the half and third
            energies on edges).
        energy_bins: Number of energy bins if ``energy_edges`` is None.
        pitch_bins: Equal pitch bins on $[-1, 1]$.
        legendre_modes: Legendre terms $l = 0, \dots$ of the pitch series.
        ln_lambda: One Coulomb logarithm for all species (a number or one per
            birth); each its own if None.
        thermal_factor: The ions are thermal below ``thermal_factor`` $\times\, T_i$ ...
        minimum_energy: ... or below this energy [eV], whichever is higher.
        speed_points: Speeds of the scattering integral $s$ (equal steps in $\ln v$).
        quadrature: Gauss-Legendre points per energy bin.
        chunk: Births handled at once (memory).
        comm: Communicator over which the births are split; serial if None.

    Returns:
        A Dataset with ``f`` [$\mathrm{1/(m^3\,eV)}$] over ``(rho, energy, pitch)``,
        the shell ``volume`` and the bin edges; attributes ``mass_number`` and
        ``charge_number`` for the moments.
    """
    minimum_energy = to_unit(minimum_energy, "eV", "minimum_energy")
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    with sp.region("freeofcharge.slowing_down_distribution", tags=PROFILE_TAGS):
        return _slowing_down_distribution(
            deposition,
            plasma,
            surfaces,
            energy_edges,
            energy_bins,
            pitch_bins,
            legendre_modes,
            ln_lambda,
            thermal_factor,
            minimum_energy,
            speed_points,
            quadrature,
            chunk,
            comm,
        )


@method(
    equations={
        "estimator": r"f_{kij} = \frac{1}{V_k \Delta E_i \Delta\xi_j} \sum_b w_b \int_{\rho \in k,\ E \in i,\ \xi \in j} dt",
    },
    assumptions=(
        r"the histogram of `follow_orbits(histogram=...)`: the time of a push interval in each shell exactly, spread over the energies between its start and end (linear in time), at the pitch of its end (a random phase of the orbit)",
        r"the local pitch, so trapped ions count at every pitch they pass",
    ),
)
def orbit_distribution(result: OrbitFollowing, *, mass_number: int) -> xr.Dataset:
    r"""Return the distribution $f(\rho, E, \xi)$ that `follow_orbits` recorded.

    Args:
        result: Of `freeofcharge.follow_orbits` with ``histogram`` and ``surfaces``.
        mass_number: Of the fast ions (for the moments).

    Returns:
        A Dataset in the layout of `slowing_down_distribution`.
    """
    if result.histogram is None or result.histogram_edges is None:
        raise ValueError("follow_orbits recorded no histogram: pass histogram=")
    energy_edges, pitch_edges = result.histogram_edges
    return _dataset(
        result.histogram,
        result.volume,
        np.asarray(energy_edges, dtype=float),
        np.asarray(pitch_edges, dtype=float),
        np.asarray(result.rho_edges, dtype=float),
        result.radial_coordinate,
        {
            "mass_number": int(mass_number),
            "charge_number": 1,
            "source": "orbit_distribution",
        },
    )


def _widths(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    return (
        xr.DataArray(np.diff(ds.energy_edge.values), dims="energy"),
        xr.DataArray(np.diff(ds.pitch_edge.values), dims="pitch"),
    )


def energy_spectrum(ds: xr.Dataset) -> xr.DataArray:
    r"""Return $dn/dE = \int f\,d\xi$ [$\mathrm{1/(m^3\,eV)}$] over ``(rho, energy)``."""
    _, d_pitch = _widths(ds)
    out = (ds.f * d_pitch).sum("pitch")
    out.attrs = {"long_name": r"$dn/dE$", "units": "1/(m^3 eV)"}
    return out


def pitch_distribution(ds: xr.Dataset) -> xr.DataArray:
    r"""Return $dn/d\xi = \int f\,dE$ [$\mathrm{1/m^3}$] over ``(rho, pitch)``."""
    d_energy, _ = _widths(ds)
    out = (ds.f * d_energy).sum("energy")
    out.attrs = {"long_name": r"$dn/d\xi$", "units": "1/m^3"}
    return out


def distribution_at(ds: xr.Dataset, rho: float) -> xr.DataArray:
    r"""Return $f(E, \xi)$ on the shell nearest to ``rho``."""
    return ds.f.sel(rho=rho, method="nearest")


@method(
    equations={
        "density": r"n = \sum_{ij} f_{ij}\,\Delta E_i \Delta\xi_j",
        "pressures": r"p_\parallel = \sum_{ij} 2 E_i \xi_j^2 f_{ij} \Delta E_i \Delta\xi_j, \qquad p_\perp = \sum_{ij} E_i (1 - \xi_j^2) f_{ij} \Delta E_i \Delta\xi_j, \qquad p = \frac{p_\parallel + 2 p_\perp}{3}",
        "current": r"j = Z_b e \sum_{ij} v_i \xi_j f_{ij} \Delta E_i \Delta\xi_j, \qquad v_i = \sqrt{2 E_i / m}",
    },
    assumptions=(r"the midpoints of the bins (second order in their widths)",),
)
def distribution_moments(ds: xr.Dataset) -> xr.Dataset:
    r"""Return the moments of a distribution per shell.

    ``fast_ion_density`` [$\mathrm{m^{-3}}$], ``fast_ion_pressure`` (two thirds of
    the energy density, as `freeofcharge.SlowingDown`), ``parallel_pressure`` and
    ``perpendicular_pressure`` [Pa], the ``anisotropy`` $p_\perp / p_\parallel$ and
    the ``fast_ion_current_density`` [$\mathrm{A/m^2}$], positive along $\mathbf{B}$.
    """
    d_energy, d_pitch = _widths(ds)
    dn = ds.f * d_energy * d_pitch
    energy = ds.energy * _E  # J
    xi = ds.pitch
    mass = ds.attrs["mass_number"] * ATOMIC_MASS
    speed = np.sqrt(2.0 * energy / mass)
    density = dn.sum(("energy", "pitch"))
    parallel = (2.0 * energy * xi**2 * dn).sum(("energy", "pitch"))
    perpendicular = (energy * (1.0 - xi**2) * dn).sum(("energy", "pitch"))
    current = (
        ds.attrs["charge_number"] * _E * (speed * xi * dn).sum(("energy", "pitch"))
    )

    def var(values: xr.DataArray, name: str, units: str) -> xr.DataArray:
        return values.assign_attrs(long_name=name, units=units)

    with np.errstate(invalid="ignore", divide="ignore"):
        anisotropy = perpendicular / parallel
    return xr.Dataset(
        {
            "fast_ion_density": var(density, "fast-ion density", "1/m^3"),
            "fast_ion_pressure": var(
                (parallel + 2.0 * perpendicular) / 3.0, "fast-ion pressure", "Pa"
            ),
            "parallel_pressure": var(parallel, r"$p_\parallel$", "Pa"),
            "perpendicular_pressure": var(perpendicular, r"$p_\perp$", "Pa"),
            "anisotropy": var(anisotropy, r"$p_\perp / p_\parallel$", "1"),
            "fast_ion_current_density": var(
                current, "fast-ion current density", "A/m^2"
            ),
        },
        attrs={"radial_coordinate": ds.attrs.get("radial_coordinate", "unspecified")},
    )
