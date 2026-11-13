r"""Toroidal torque of a neutral beam: collisional and $\mathbf{j}\times\mathbf{B}$, per shell.

A beam brings toroidal angular momentum into the plasma at the rate
$\dot L = \sum_b w_b\, m R\, \mathbf{v}_b \cdot \hat e_\varphi$ of its births (the
angular momentum about the machine axis $Z$, positive along $+\varphi$ with
$(R, \varphi, Z)$ right-handed). In an axisymmetric field the canonical toroidal
momentum $p_\varphi = m R v_\varphi + q\psi$ is conserved ($R A_\varphi = \psi$,
the poloidal flux per radian of `freeofcharge.Equilibrium`), and the ion's
angular momentum reaches the plasma in two ways, the collisional and the
$\mathbf{j}\times\mathbf{B}$ torque that NUBEAM reports (Pankin et al., Comput.
Phys. Commun. 159, 157, 2004; the radial current of the fast ions and its torque:
Honda et al., Nucl. Fusion 49, 035009, 2009):

1. $\mathbf{j}\times\mathbf{B}$. At ionisation the guiding centre is a Larmor radius
   away from the birth point: the gyration's share of the angular momentum,
   $m R\,(v_\varphi - v_\parallel b_\varphi)$, goes into a jump of the flux,
   $q\,(\psi_\mathrm{gc} - \psi_\mathrm{birth}) = m R\,(v_\varphi - v_\parallel b_\varphi)$
   from $p_\varphi$ conservation. The fast ions' charge so moves across the surfaces
   between $\psi_\mathrm{birth}$ and $\psi_\mathrm{gc}$, a radial current $I_\psi$
   (positive towards larger $\psi$), and the return current $-I_\psi$ of the
   thermal plasma feels the torque density $R\,(\mathbf{j}\times\mathbf{B})_\varphi
   = -\mathbf{j}\cdot\nabla\psi$ (since $\mathbf{B}_p = \nabla\psi\times\nabla\varphi$),
   i.e. $dT = I_\psi\, d\psi$. A shell $[\psi_k, \psi_{k+1}]$ gets
   $T_k = \sum_b w_b\, q\,[\mathrm{clip}(\psi_\mathrm{gc}) - \mathrm{clip}(\psi_\mathrm{birth})]$,
   both clipped to the shell, and the total is $\sum_b w_b\, q\,\Delta\psi$.
2. Collisions. The guiding centre carries $m v_\parallel R b_\varphi$, which the
   drags hand to the electrons and the ions as it slows down. With
   $d\langle\mathbf{v}\rangle/dt = -a_e \mathbf{v} - \sum_j (1 + m_b/m_j)\,\Gamma_j \mathbf{v}/v^3$
   for $v_{Ti} \ll v \ll v_{Te}$ (Trubnikov's cold-target limit, Helander and
   Sigmar, *Collisional Transport in Magnetized Plasmas*, 2002, ch. 3), the
   parallel momentum goes at the rate $m a_e v_\parallel$ to the electrons and
   $m (a_i + \sum_j\Gamma_j) v_\parallel / v^3$ to the ions: their energy drag
   plus the pitch-angle scattering, $\nu_D v_\parallel$, which only the ions do.
   These add up to $-m\,d\langle v_\parallel\rangle/dt$ of `freeofcharge.slowing_down`
   ($d\langle\xi v\rangle/dt$ with Gaffey's pitch decay), so in steady state a
   birth gives $m R b_\varphi (v_{\parallel 0} - v_{\parallel,\mathrm{end}})$,
   the rest thermalising with the ion.

Together $T_\mathrm{coll} + T_{j\times B} + \dot L_\mathrm{th} = \dot L$: all the
angular momentum of the births ends up in the plasma (`BeamTorque` reports the
balance). Per watt the torque falls as $\dot L / P = 2 R \xi_\varphi / v
\propto E^{-1/2}$.

Assumptions as in `freeofcharge.slow_down`: zero orbit width (the collisional
torque on the birth shell, $R b_\varphi$ frozen at the birth point), the local pitch
(no orbit average: the angular momentum of trapped ions, which the mirror force
trades with the field along the banana, is treated like that of passing ones), no
losses, no charge exchange, $Z_b = 1$. The $\mathbf{j}\times\mathbf{B}$ part is
only the prompt one, at birth, and needs the axisymmetric `Equilibrium`; the
collisional part holds on the shells of any equilibrium (the field from
``field_at``). The split and the $p_\varphi$ argument are derived here and checked
by the tests (balance, and the jump against the flux at the first-order guiding
centre), not symbol by symbol against the papers.
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
from freeofcharge.equilibrium import Equilibrium, rho_coordinate
from freeofcharge.slowing_down import collision_coefficients

if TYPE_CHECKING:
    from freeofcharge.deposition import Deposition
    from freeofcharge.equilibrium import FluxSurfaces, MagneticEquilibrium
    from freeofcharge.profiles import Plasma

__all__ = ["beam_torque", "parallel_momentum_moments"]

_E = ELEMENTARY_CHARGE

#: The components of the torque, in the order of ``shell_torque``.
COMPONENTS = ("collisional_electrons", "collisional_ions", "jxb")

_LONG_NAMES = {
    "collisional_electrons": "collisional torque on the electrons",
    "collisional_ions": "collisional torque on the ions",
    "jxb": r"$j \times B$ torque",
    "total": "beam torque",
}


@method(
    equations={
        "trajectory": r"\frac{dt}{d\ln v} = \frac{v}{a_e v + a_i / v^2}, \qquad \langle\xi\rangle = \xi_0\, e^{-\int \nu_D\, dt'}, \quad \nu_D = \frac{\sum_j\Gamma_j}{v^3}",
        "electrons": r"\frac{\Delta p_{\parallel,e}}{m \xi_0} = \int a_e\, v\, e^{-\int\nu_D dt'}\, dt",
        "ions": r"\frac{\Delta p_{\parallel,i}}{m \xi_0} = \int \frac{a_i + \sum_j \Gamma_j}{v^2}\, e^{-\int\nu_D dt'}\, dt",
        "balance": r"\frac{\Delta p_{\parallel,e} + \Delta p_{\parallel,i}}{m \xi_0} + v_\mathrm{end}\, e^{-\int\nu_D dt'} = v_0",
    },
    order=4,
    assumptions=(
        r"cold-target momentum drag $(1 + m_b/m_j)\,\Gamma_j / v^3$ on the ions, $a_e$ on the electrons",
        r"speeds equally spaced in $\ln v$, an odd number of them; Simpson's rule, the cumulative scattering by the trapezoidal rule",
    ),
    references=(
        "Helander and Sigmar, Collisional Transport in Magnetized Plasmas (2002), ch. 3",
        "[Gaffey, J. Plasma Phys. 16, 149 (1976)](https://doi.org/10.1017/S0022377800020134)",
    ),
)
def parallel_momentum_moments(speed, electron_drag, ion_drag, deflection):
    r"""Return the parallel momentum a slowing-down ion gives the electrons and the ions.

    Per unit mass and unit birth pitch $\xi_0$, over trajectories sampled as in
    `freeofcharge.slowing_down.slowing_down_moments` (rows of ``speed`` from the
    birth speed down, equal steps in $\ln v$, an odd number) with the drags and
    the deflection of `freeofcharge.slowing_down.collision_coefficients` on them.

    Returns:
        ``(electrons, ions, end)`` [m/s]: $\int a_e v\,e^{-\int\nu_D}\,dt$,
        $\int (a_i + \sum_j\Gamma_j)\,v^{-2}\, e^{-\int\nu_D}\,dt$ and the
        parallel velocity left at the end speed, $v_\mathrm{end}\,e^{-\int\nu_D}$;
        the three add up to the birth speed.
    """
    v = xp.asarray(speed, dtype=float)
    points = v.shape[1]
    if points % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of speeds")
    log_step = xp.log(v[:, 0] / v[:, -1]) / (points - 1)
    simpson = np.full(points, 2.0)
    simpson[1::2] = 4.0
    simpson[0] = simpson[-1] = 1.0
    weights = log_step[:, None] * xp.asarray(simpson / 3.0)
    a_e = electron_drag + xp.zeros_like(v)
    a_i = ion_drag + xp.zeros_like(v)
    gamma = deflection + xp.zeros_like(v)
    dt = v / (a_e * v + a_i / v**2)  # dt / d(ln v)
    rate = gamma / v**3 * dt
    half = 0.5 * log_step[:, None] * (rate[:, 1:] + rate[:, :-1])
    exponent = xp.concatenate(
        [xp.zeros((v.shape[0], 1)), xp.cumsum(half, axis=-1)], axis=-1
    )
    decay = xp.exp(-exponent)
    electrons = xp.sum(weights * a_e * v * decay * dt, axis=-1)
    ions = xp.sum(weights * (a_i + gamma) / v**2 * decay * dt, axis=-1)
    return electrons, ions, v[:, -1] * decay[:, -1]


def _shell_overlap(start, end, edges):
    r"""Return the signed length of $[\min, \max](\text{start}, \text{end})$ in each shell.

    ``start``, ``end`` ``(N,)``, ``edges`` ``(K + 1,)``: ``(N, K)``, positive where
    ``end`` > ``start``. Summed over the shells it is ``end - start`` clipped to the
    edges' range.
    """
    lower = xp.asarray(edges[:-1])[None, :]
    upper = xp.asarray(edges[1:])[None, :]

    def clip(values):
        return xp.minimum(xp.maximum(values[:, None], lower), upper)

    return clip(end) - clip(start)


@sp.profile("freeofcharge.beam_torque", tags=PROFILE_TAGS)
@method(
    equations={
        "injected": r"\dot L = \sum_b w_b\, m R\, v_\varphi, \qquad v_\varphi = \mathbf{v}\cdot\hat e_\varphi",
        "jump": r"q\,(\psi_\mathrm{gc} - \psi_\mathrm{birth}) = m R\,(v_\varphi - v_\parallel b_\varphi), \qquad p_\varphi = m R v_\varphi + q\psi",
        "jxb": r"T_{j\times B,k} = \sum_b w_b\, q \left[\mathrm{clip}_k(\psi_\mathrm{gc}) - \mathrm{clip}_k(\psi_\mathrm{birth})\right], \qquad R\,(\mathbf{j}\times\mathbf{B})_\varphi = -\mathbf{j}\cdot\nabla\psi",
        "collisional": r"T_{e,k} = \sum_{b \in k} w_b\, m R b_\varphi\, \xi_0 \int a_e v\, e^{-\int \nu_D}\, dt, \qquad T_{i,k} = \sum_{b \in k} w_b\, m R b_\varphi\,(v_{\parallel 0} - v_{\parallel,\mathrm{end}}) - T_{e,k}",
        "balance": r"\sum_k \left(T_{e,k} + T_{i,k} + T_{j\times B,k}\right) + T_\mathrm{outside} + \dot L_\mathrm{th} = \dot L",
    },
    assumptions=(
        r"zero orbit width: the collisional torque on the birth shell, $R b_\varphi$ at the birth point",
        r"the local pitch (Gaffey's decay, no orbit average); $v_{Ti} \ll v \ll v_{Te}$; $Z_b = 1$",
        r"$\mathbf{j}\times\mathbf{B}$: only the prompt jump at birth, axisymmetric equilibria only; the part beyond the outermost edge is $T_\mathrm{outside}$",
        r"the ions' share is the rest of the parallel momentum, so the balance is exact",
    ),
    references=(
        "Pankin et al., Comput. Phys. Commun. 159, 157 (2004)",
        "[Honda et al., Nucl. Fusion 49, 035009 (2009)](https://doi.org/10.1088/0029-5515/49/3/035009)",
        "Helander and Sigmar, Collisional Transport in Magnetized Plasmas (2002)",
    ),
    components=(collision_coefficients, parallel_momentum_moments),
)
def beam_torque(
    deposition: Deposition,
    plasma: Plasma,
    surfaces: FluxSurfaces,
    equilibrium: MagneticEquilibrium,
    *,
    jxb: bool = True,
    ln_lambda=None,
    thermal_factor: float = 0.0,
    minimum_energy: float = 0.0,
    speed_points: int = 201,
    comm: MPI.Comm | None = None,
) -> xr.Dataset:
    r"""Return the toroidal torque of the births of ``deposition``, shell by shell.

    Args:
        deposition: Births (this rank's) and their shells.
        plasma: Profiles.
        surfaces: Volumes of the shells of the deposition.
        equilibrium: The field at the births; for ``jxb`` the axisymmetric
            `freeofcharge.Equilibrium` (its flux $\psi$).
        jxb: Add the $\mathbf{j}\times\mathbf{B}$ torque; a ValueError on a 3D
            equilibrium, where pass False for the collisional torque alone.
        ln_lambda: One Coulomb logarithm for all species (a number or one per
            birth); each its own (`freeofcharge.slowing_down.coulomb_logarithms`)
            if None.
        thermal_factor: The ions are thermal below ``thermal_factor`` $\times\,T_i$ ...
        minimum_energy: ... or below this energy [eV], whichever is higher, as in
            `freeofcharge.slow_down`.
        speed_points: Points of the speed integrals (equal steps in $\ln v$), odd.
        comm: Communicator over which the births are split; serial if None.

    Returns:
        On the shell centres, the torque densities [N m/m^3] along $+\varphi$
        ``collisional_electrons``, ``collisional_ions``, ``jxb`` (if ``jxb``) and
        ``total``, the torque of each shell ``shell_torque`` [N m] by
        ``component``, and in the attributes the totals [N m] (``<name>_total``),
        the angular momentum rate of the births ``birth_angular_momentum``, what
        thermalises (``thermalised_angular_momentum``), the $\mathbf{j}\times\mathbf{B}$
        torque beyond the outermost edge (``jxb_outside``), the birth power [W]
        and ``balance``, the relative mismatch of the angular momentum (zero
        but for roundoff). Without ``jxb`` the balance is against the guiding
        centres' angular momentum $\sum_b w_b\, m R v_\parallel b_\varphi$.
    """
    minimum_energy = to_unit(minimum_energy, "eV", "minimum_energy")
    if jxb and not isinstance(equilibrium, Equilibrium):
        raise ValueError(
            "the j x B torque uses the poloidal flux of an axisymmetric Equilibrium "
            "(canonical toroidal momentum); on a 3D equilibrium pass jxb=False for "
            "the collisional torque alone"
        )
    if not np.array_equal(surfaces.rho_edges, deposition.rho_edges):
        raise ValueError("the surfaces and the deposition have different shells")
    plasma.check_radial_coordinate(surfaces.radial_coordinate)
    n_shells = len(surfaces.rho_edges) - 1
    mass = deposition.mass_number * ATOMIC_MASS
    charge = _E  # Z_b = 1
    births = deposition.births
    x = xp.asarray(births.positions, dtype=float)
    v = xp.asarray(births.velocities, dtype=float)
    w = xp.asarray(births.weights, dtype=float)
    radius = xp.hypot(x[:, 0], x[:, 1])
    e_phi = xp.stack([-x[:, 1] / radius, x[:, 0] / radius, xp.zeros_like(radius)], -1)
    field = xp.asarray(equilibrium.field_at(x), dtype=float)
    b_unit = field / xp.sqrt(xp.sum(field**2, axis=1))[:, None]
    speed = xp.sqrt(xp.sum(v**2, axis=1))
    v_par = xp.sum(v * b_unit, axis=1)
    pitch = v_par / speed
    b_phi = xp.sum(b_unit * e_phi, axis=1)
    birth_momentum = mass * radius * xp.sum(v * e_phi, axis=1)  # m R v_phi
    centre_momentum = mass * radius * v_par * b_phi  # m R v_par b_phi

    # collisional: the slowing down of slow_down, local pitch
    rho = xp.minimum(xp.asarray(deposition.birth_rho), 1.0)
    ti = plasma.ion_temperature(rho)
    energy = 0.5 * mass * speed**2
    end_energy = xp.minimum(
        xp.maximum(thermal_factor * ti, minimum_energy) * _E, energy
    )
    ratio = xp.maximum(xp.sqrt(end_energy / energy), 1.0e-4)
    speeds = speed[:, None] * ratio[:, None] ** xp.linspace(0.0, 1.0, speed_points)

    def column(values):
        return xp.asarray(values)[:, None]

    electron_drag, ion_drag, deflection = collision_coefficients(
        speeds,
        mass,
        1,
        column(plasma.electron_density(rho)),
        column(plasma.electron_temperature(rho)),
        [column(n) for n in plasma.ion_densities(rho)],
        plasma.ion_masses,
        plasma.ion_charges,
        column(ti),
        None if ln_lambda is None else xp.reshape(xp.asarray(ln_lambda), (-1, 1)),
    )
    to_electrons, _, end = parallel_momentum_moments(
        speeds, electron_drag, ion_drag, deflection
    )
    lever = mass * radius * b_phi * pitch  # m R b_phi xi_0
    torque_e = lever * to_electrons
    thermalised = lever * end
    torque_i = centre_momentum - thermalised - torque_e
    shell = xp.clip(
        surfaces.shell_index(xp.minimum(rho, 0.999999 * surfaces.rho_edges[-1])),
        0,
        n_shells - 1,
    )
    sums = xp.algorithms.segment_sum(
        xp.stack([w * torque_e, w * torque_i], axis=-1), shell, n_shells
    )
    totals = xp.stack(
        [
            xp.sum(w * birth_momentum),
            xp.sum(w * centre_momentum),
            xp.sum(w * thermalised),
            xp.sum(w * energy),
            xp.zeros(()),
        ]
    )
    shells = xp.zeros((n_shells, 3))
    shells[:, :2] = sums
    if jxb:
        assert isinstance(equilibrium, Equilibrium)
        span = equilibrium.psi_boundary - equilibrium.psi_axis
        psi_n = equilibrium.psi_normalized(radius, x[:, 2])
        # p_phi conservation: q (psi_gc - psi_birth) = m R (v_phi - v_par b_phi)
        jump = birth_momentum - centre_momentum
        psi_n_centre = psi_n + jump / (charge * span)
        overlap = _shell_overlap(psi_n, psi_n_centre, surfaces.rho_edges**2)
        shells[:, 2] = charge * span * xp.sum(w[:, None] * overlap, axis=0)
        totals[4] = xp.sum(w * jump) - xp.sum(shells[:, 2])
    shells = xp.to_numpy(allreduce(shells, comm))
    totals = xp.to_numpy(allreduce(totals, comm))
    birth_total, centre_total, thermal_total, power, outside = (
        float(t) for t in totals
    )
    components = COMPONENTS if jxb else COMPONENTS[:2]
    shell_torque = shells[:, : len(components)].T
    reference = birth_total if jxb else centre_total
    deposited = float(np.sum(shell_torque)) + outside + thermal_total
    volume = surfaces.volume
    variables: dict[str, tuple] = {
        name: (
            "rho",
            shell_torque[k] / volume,
            {"long_name": _LONG_NAMES[name], "units": "N m/m^3"},
        )
        for k, name in enumerate(components)
    }
    variables["total"] = (
        "rho",
        np.sum(shell_torque, axis=0) / volume,
        {"long_name": _LONG_NAMES["total"], "units": "N m/m^3"},
    )
    variables["shell_torque"] = (
        ("component", "rho"),
        shell_torque,
        {"long_name": "torque per shell", "units": "N m"},
    )
    attrs: dict[str, float | str] = {
        f"{name}_total": float(np.sum(shell_torque[k]))
        for k, name in enumerate(components)
    }
    attrs.update(
        total=float(np.sum(shell_torque)),
        birth_angular_momentum=birth_total,
        guiding_centre_angular_momentum=centre_total,
        thermalised_angular_momentum=thermal_total,
        jxb_outside=outside,
        birth_power=power,
        balance=deposited / reference - 1.0 if reference != 0.0 else 0.0,
        radial_coordinate=surfaces.radial_coordinate,
        units="N m",
    )
    rho_centres = 0.5 * (surfaces.rho_edges[1:] + surfaces.rho_edges[:-1])
    return xr.Dataset(
        variables,
        coords={
            "rho": ("rho", rho_centres, rho_coordinate(surfaces.radial_coordinate)),
            "component": list(components),
        },
        attrs=attrs,
    )
