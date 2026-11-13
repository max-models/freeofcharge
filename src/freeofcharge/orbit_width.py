r"""Finite orbit width of the births: the first-orbit (orbit-averaged) deposition.

`freeofcharge.deposit` and `freeofcharge.slow_down` put each fast ion on the flux
shell of its birth point (zero orbit width). A fast ion moves on its guiding-centre
orbit, which in a tokamak conserves the energy, the magnetic moment
$\mu = v_\perp^2 / (2B)$ and the canonical toroidal momentum

$$
p_\varphi = m R v_\parallel b_\varphi + q \psi,
$$

so it leaves its birth surface by
$\Delta\psi = -\frac{m}{q}\,\Delta(R v_\parallel b_\varphi)$: passing ions by the
drift-surface shift, $\sim q_s \rho_\parallel$ ($q_s$ the safety factor,
$\rho_\parallel = m v_\parallel / (qB)$), trapped ones over a banana of width
$\sim q_s \rho_L / \sqrt{\epsilon}$ (Helander and Sigmar, *Collisional Transport in
Magnetized Plasmas*, 2002, ch. 7). Codes that slow the ions down locally correct the
deposition for it by averaging each birth over its first orbit: NUBEAM's
"first-orbit" redistribution of the births (Pankin et al., Comput. Phys. Commun.
159, 157, 2004; Goldston et al., J. Comput. Phys. 43, 61, 1981) and RABBIT's
orbit-averaged birth profile (Weiland et al., Nucl. Fusion 58, 082032, 2018). Ions
whose first orbit leaves the plasma are prompt losses.

`orbit_average_births` follows each birth's collisionless guiding-centre orbit
(`freeofcharge.orbits`: `guiding_centres` and the compiled push) for one period. In
an axisymmetric field the orbit's projection on the poloidal plane $(R, Z)$ is a
closed curve, a transit of a passing orbit or a bounce of a trapped (banana or
potato) one: the period ends where the guiding centre crosses, in its starting
direction and near its start, the line through the starting point across its first
poloidal displacement (a Poincaré section). This also closes passing orbits near
the axis that do not encircle it. The time the orbit spends in each shell over that
period, $t_{bk}$, splits the birth's weight:

$$
w_{bk} = w_b \frac{t_{bk}}{\sum_j t_{bj}},
$$

one sub-birth per visited shell, with the invariants of the orbit: in the birth's
own shell the birth itself, in the others the orbit point closest to the shell's
centre (its guiding centre, $|B|$ and pitch $\xi = v_\parallel / v$, so
$\lambda = (1 - \xi^2) B_\mathrm{min} / B$ of `freeofcharge.trapping` is the orbit's).
A birth whose orbit crosses $\rho$ = ``rho_limit`` (or leaves the field's grid)
within the period is removed; its power is the prompt-loss power. The result is a
`freeofcharge.Deposition` that `slow_down` takes unchanged, so the local slowing
down (heating, fast-ion density, current) is done on the orbit-averaged births:
"first orbit, then local slowing down", without the radial transport by
pitch-angle scattering and slowing down that `follow_orbits` includes.

Tokamaks only: the period is that of the orbit's poloidal projection, closed in an
axisymmetric field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from methodcard import method
from plasmacoll import ParticleArrays
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.deposition import Deposition
from freeofcharge.orbits import _Pusher, guiding_centres, orbit_field

if TYPE_CHECKING:
    from freeofcharge.equilibrium import MagneticEquilibrium
    from freeofcharge.gridded_equilibrium import GriddedEquilibrium

__all__ = ["FirstOrbitDeposition", "orbit_average_births"]

_E = ELEMENTARY_CHARGE


def _kinetic_energy(deposition: Deposition) -> np.ndarray:
    """Kinetic energy of each birth [J]."""
    v = np.asarray(xp.to_numpy(deposition.births.velocities), dtype=float)
    return 0.5 * deposition.mass_number * ATOMIC_MASS * np.sum(v * v, axis=1)


def _shell_of(rho, rho_edges) -> np.ndarray:
    """Shell index of ``rho`` as `slow_down` takes it (clipped to the shells)."""
    rho = np.minimum(np.asarray(rho, dtype=float), 0.999999 * rho_edges[-1])
    index = np.searchsorted(rho_edges, rho, side="right") - 1
    return np.clip(index, 0, len(rho_edges) - 2)


@dataclass
class FirstOrbitDeposition(Deposition):
    r"""A `freeofcharge.Deposition` whose births are averaged over their first orbit.

    ``births``, ``birth_rho``, ``birth_pitch``, ``birth_field``, ``birth_rate``
    and ``power`` are those of the sub-births (prompt losses removed), so
    `freeofcharge.slow_down` takes it as it is; ``absorbed_power`` and the
    shine-through stay those of the beam. Per original birth (dimension of
    ``zero_width.births``): the shell fractions, whether it is a prompt loss, its
    orbit period and mean $\rho$.
    """

    zero_width: Deposition | None = field(default=None, repr=False)
    parent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    shell_fractions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    prompt_lost: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    unclosed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    trapped: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    orbit_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mean_rho: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prompt_loss_power: float = 0.0
    prompt_loss_rate: float = 0.0
    """The deposition before the redistribution."""
    """Index of the original birth of each sub-birth."""
    """``(n_births, n_shells)``: share of each birth's orbit period in each shell
    (zero rows for prompt losses)."""
    r"""Births whose first orbit leaves $\rho$ = ``rho_limit`` or the grid."""
    """Births whose orbit did not close within ``max_length`` (averaged over the
    time followed)."""
    r"""Births whose orbit has turning points ($v_\parallel$ changes sign)."""
    """Orbit period (transit or bounce) [s]; the time to the loss for prompt losses."""
    r"""Mean of the shell centres over the period, weighted by the time in each
    (NaN for prompt losses)."""
    """Energy rate of the prompt losses [W]."""
    """Ions per second lost promptly."""

    @property
    def confined_power(self) -> float:
        """Return the energy rate of the confined births [W]."""
        return float(np.sum(self.power))

    @property
    def prompt_loss_fraction(self) -> float:
        """Return the share of the birth power lost on the first orbit."""
        total = self.confined_power + self.prompt_loss_power
        return self.prompt_loss_power / total if total > 0.0 else 0.0

    def redistribution(self) -> np.ndarray:
        r"""Return $M_{jk}$, the share of the birth power of shell $j$ that the orbits put in shell $k$.

        Rows sum to one minus the prompt-loss share of shell $j$ (zero rows for
        shells without births).
        """
        source = self.zero_width
        if source is None:
            raise ValueError("no zero-width deposition to redistribute")
        n_shells = len(self.rho_edges) - 1
        weights = np.asarray(xp.to_numpy(source.births.weights), dtype=float)
        energy = _kinetic_energy(source)
        shell = _shell_of(np.asarray(xp.to_numpy(source.birth_rho)), self.rho_edges)
        matrix = np.zeros((n_shells, n_shells))
        np.add.at(matrix, shell, (weights * energy)[:, None] * self.shell_fractions)
        total = np.bincount(shell, weights * energy, minlength=n_shells)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total[:, None] > 0.0, matrix / total[:, None], 0.0)

    def to_dataset(self, surfaces=None) -> xr.Dataset:
        """Return the redistributed profiles, with those of the same births at zero width.

        ``zero_width_birth_rate`` and ``zero_width_power`` put the same births on
        their birth shells (not the noise-free tallies of `deposit`), so the
        difference is the orbit width alone.
        """
        dataset = super().to_dataset(surfaces)
        if self.zero_width is not None:
            n_shells = len(self.rho_edges) - 1
            source = self.zero_width
            weights = np.asarray(xp.to_numpy(source.births.weights), dtype=float)
            shell = _shell_of(np.asarray(xp.to_numpy(source.birth_rho)), self.rho_edges)
            rate = np.bincount(shell, weights, minlength=n_shells)
            power = np.bincount(
                shell, weights * _kinetic_energy(source), minlength=n_shells
            )
            dataset["zero_width_birth_rate"] = (
                "rho",
                rate,
                {"long_name": "birth rate at zero width", "units": "1/s"},
            )
            dataset["zero_width_power"] = (
                "rho",
                power,
                {"long_name": "deposited power at zero width", "units": "W"},
            )
            if surfaces is not None:
                dataset["zero_width_power_density"] = (
                    "rho",
                    power / surfaces.volume,
                    {
                        "long_name": "deposited power density at zero width",
                        "units": "W/m^3",
                    },
                )
        dataset.attrs.update(
            prompt_loss_power=self.prompt_loss_power,
            prompt_loss_rate=self.prompt_loss_rate,
            prompt_loss_fraction=self.prompt_loss_fraction,
            confined_power=self.confined_power,
        )
        return dataset


def _perpendicular(unit: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each row of ``unit``."""
    helper = np.where(
        (np.abs(unit[:, 2]) < 0.9)[:, None],
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    e1 = np.cross(unit, helper)
    return e1 / np.linalg.norm(e1, axis=1)[:, None]


@method(
    equations={
        "invariants": r"E = m\left(\tfrac12 v_\parallel^2 + \mu B\right), \qquad \mu = \frac{v_\perp^2}{2B}, \qquad p_\varphi = m R v_\parallel b_\varphi + q\psi",
        "period": r"T: \text{ the first return of } (R, Z) \text{ through } (\mathbf{X} - \mathbf{X}_0) \cdot \Delta\mathbf{X}_0 = 0 \text{ near } \mathbf{X}_0, \text{ in the direction of } \Delta\mathbf{X}_0",
        "split": r"w_{bk} = w_b \frac{t_{bk}}{\sum_j t_{bj}}, \qquad t_{bk} = \int_\mathrm{period} [\rho(t) \in k]\, dt",
        "losses": r"P_\mathrm{prompt} = \sum_{b \in \mathrm{lost}} w_b E_b, \qquad \sum_k P_k = \sum_b w_b E_b - P_\mathrm{prompt}",
    },
    assumptions=(
        r"collisionless guiding-centre orbits (`freeofcharge.orbits`) in an axisymmetric field, no electric field; the slowing down then local on each shell of the orbit",
        r"the section is the line through the starting point $\mathbf{X}_0$ across the first push interval's displacement $\Delta\mathbf{X}_0$ in $(R, Z)$; a crossing closes the orbit if it is closer to $\mathbf{X}_0$ than `closure` times the farthest point reached; the crossing is linear in time inside its push interval",
        r"orbits not closed within `max_length` averaged over the time followed",
        r"prompt loss: the orbit reaches $\rho$ = `rho_limit` (not a wall) or leaves the grid within the period",
        r"a sub-birth in another shell than the birth's takes the orbit point closest to that shell's centre (guiding centre, $|B|$, $\xi$); in the birth's shell the birth itself",
    ),
    references=(
        "[Pankin et al., Comput. Phys. Commun. 159, 157 (2004)](https://doi.org/10.1016/j.cpc.2003.11.002) (NUBEAM)",
        "[Goldston et al., J. Comput. Phys. 43, 61 (1981)](https://doi.org/10.1016/0021-9991(81)90111-X)",
        "[Weiland et al., Nucl. Fusion 58, 082032 (2018)](https://doi.org/10.1088/1741-4326/aabf0f) (RABBIT)",
        "Helander and Sigmar, Collisional Transport in Magnetized Plasmas (2002), ch. 7",
    ),
    components=(guiding_centres,),
)
@sp.profile("freeofcharge.orbit_average_births", tags=PROFILE_TAGS)
def orbit_average_births(
    deposition: Deposition,
    equilibrium: MagneticEquilibrium,
    *,
    field: GriddedEquilibrium | None = None,
    rho_limit: float = 1.0,
    step: float = 0.05,
    interval: float = 0.25,
    max_length: float = 3000.0,
    closure: float = 0.3,
) -> FirstOrbitDeposition:
    r"""Return ``deposition`` with each birth spread over the shells of its first orbit.

    Args:
        deposition: The births (zero orbit width), e.g. of `freeofcharge.deposit`.
        equilibrium: The axisymmetric equilibrium.
        field: Its cubic gridded field (`freeofcharge.orbits.orbit_field`); sampled
            from ``equilibrium`` if None.
        rho_limit: Prompt loss where the orbit reaches this $\rho$.
        step: Largest RK4 step along the orbit [m].
        interval: Path length [m] between the checks of the period (and the samples
            of the orbit); short against a transit.
        max_length: Longest path followed [m]; orbits not closed by then are
            averaged over it (``unclosed``).
        closure: A return through the section closes the orbit if it passes the
            starting point closer than this share of the largest distance from it
            reached so far.
    """
    field = orbit_field(equilibrium if field is None else field)
    if field.nfp != 1:
        raise ValueError(
            "orbit_average_births needs an axisymmetric field (one period); "
            "use freeofcharge.orbits.follow_orbits in 3D"
        )
    rho_edges = np.asarray(deposition.rho_edges, dtype=float)
    n_shells = len(rho_edges) - 1
    centres = 0.5 * (rho_edges[1:] + rho_edges[:-1])
    mass = deposition.mass_number * ATOMIC_MASS
    births = deposition.births
    positions = np.asarray(xp.to_numpy(births.positions), dtype=float)
    velocities = np.asarray(xp.to_numpy(births.velocities), dtype=float)
    weights = np.asarray(xp.to_numpy(births.weights), dtype=float)
    n = len(weights)
    markers = guiding_centres(positions, velocities, weights, field, mass)
    gc = markers.positions
    if n and not np.all(np.isfinite(xp.to_numpy(field.rho_at(gc)))):
        raise ValueError("births off the field's grid: sample the field over them")
    speed = np.sqrt(np.sum(velocities**2, axis=1))
    energy = 0.5 * mass * speed**2
    pusher = _Pusher(field, rho_edges, markers.charge_over_mass)

    state = np.stack([*gc.T, markers.v_parallel]).astype(float)
    mu = markers.mu.astype(float)

    def poloidal(points):
        return np.stack([np.hypot(points[:, 0], points[:, 1]), points[:, 2]], axis=-1)

    start = poloidal(gc)
    previous = start.copy()
    direction = np.zeros((n, 2))  # the first poloidal displacement
    side = np.zeros(n)  # (X - X_0) . direction at the last check
    farthest = np.zeros(n)
    v_prev = state[3].copy()
    crossings = np.zeros(n, dtype=np.int64)
    orbit_shells = np.zeros((n_shells, n))
    period = np.zeros(n)
    time = np.zeros(n)
    length = np.zeros(n)
    lost = np.zeros(n, dtype=bool)
    unclosed = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    # the orbit point closest to each shell's centre
    sample_gap = np.full((n_shells, n), np.inf)
    sample_rho = np.full((n_shells, n), np.nan)
    sample_point = np.full((n_shells, n, 4), np.nan)

    def keep_samples(index, points, vpar):
        rho = np.asarray(xp.to_numpy(field.rho_at(points)), dtype=float)
        ok = np.isfinite(rho) & (rho >= rho_edges[0]) & (rho < rho_edges[-1])
        index, points, vpar, rho = index[ok], points[ok], vpar[ok], rho[ok]
        shell = np.searchsorted(rho_edges, rho, side="right") - 1
        gap = np.abs(rho - centres[shell])
        better = gap < sample_gap[shell, index]
        s, k = shell[better], index[better]
        sample_gap[s, k] = gap[better]
        sample_rho[s, k] = rho[better]
        sample_point[s, k, :3] = points[better]
        sample_point[s, k, 3] = vpar[better]

    keep_samples(np.arange(n), gc, state[3])
    duration = interval / np.maximum(speed, 1e-30)
    max_step = step / np.maximum(speed, 1e-30)
    first = True
    while True:
        active = np.nonzero(~done)[0]
        if len(active) == 0:
            break
        new, _, elapsed, status, shell_time, *_ = pusher.push(
            state[:, active],
            mu[active],
            duration[active],
            max_step[active],
            rho_limit**2,
        )
        state[:, active] = new
        time[active] += elapsed
        length[active] += elapsed * speed[active]
        gone = status != 0
        lost[active[gone]] = True
        done[active[gone]] = True
        period[active[gone]] = time[active[gone]]
        on = ~gone
        index = active[on]
        st = shell_time[:, on]
        points = new[:3, on].T
        vpar = new[3, on]
        keep_samples(index, points, vpar)
        here = poloidal(points)
        if first:
            direction[index] = here - start[index]
            first = False
        offset = here - start[index]
        side_new = np.sum(offset * direction[index], axis=1)
        side_old = side[index]
        with np.errstate(invalid="ignore", divide="ignore"):
            cut = np.clip(side_old / (side_old - side_new), 0.0, 1.0)
        cut = np.where(np.isfinite(cut), cut, 1.0)
        crossing = previous[index] + cut[:, None] * (here - previous[index])
        gap = np.linalg.norm(crossing - start[index], axis=1)
        closes = (
            (side_old < 0.0) & (side_new >= 0.0) & (gap < closure * farthest[index])
        )
        share = np.where(closes, cut, 1.0)
        orbit_shells[:, index] += st * share
        period[index] = time[index] - (1.0 - share) * elapsed[on]
        crossings[index] += np.sign(vpar) * np.sign(v_prev[index]) < 0
        v_prev[index] = np.where(vpar != 0.0, vpar, v_prev[index])
        side[index] = side_new
        previous[index] = here
        farthest[index] = np.maximum(farthest[index], np.linalg.norm(offset, axis=1))
        too_long = ~closes & (length[index] >= max_length)
        unclosed[index[too_long]] = True
        done[index[closes | too_long]] = True

    trapped = crossings > 0
    total = orbit_shells.sum(axis=0)
    birth_rho = np.asarray(xp.to_numpy(deposition.birth_rho), dtype=float)
    birth_shell = _shell_of(birth_rho, rho_edges)
    # no time in the shells (an orbit outside them, confined): its birth shell
    empty = ~lost & (total <= 0.0)
    orbit_shells[birth_shell[empty], np.nonzero(empty)[0]] = 1.0
    total = np.where(empty, 1.0, total)
    fractions = np.where(
        lost[None, :], 0.0, orbit_shells / np.where(total > 0, total, 1.0)
    ).T
    rho_mid = np.where(lost, np.nan, fractions @ centres)

    # the sub-births: one per birth and visited shell
    birth_k, shell_k = np.nonzero(fractions > 0.0)
    own = shell_k == birth_shell[birth_k]
    moved = ~own
    sub_w = weights[birth_k] * fractions[birth_k, shell_k]
    sub_pos = positions[birth_k].copy()
    sub_vel = velocities[birth_k].copy()
    sub_rho = birth_rho[birth_k].copy()
    sub_pitch = np.asarray(xp.to_numpy(deposition.birth_pitch), dtype=float)[birth_k]
    sub_field = np.asarray(xp.to_numpy(deposition.birth_field), dtype=float)[birth_k]
    if np.any(moved):
        b, s = birth_k[moved], shell_k[moved]
        found = np.isfinite(sample_rho[s, b])
        # a shell the orbit only grazed between samples: its sample closest in rho
        if np.any(~found):
            nearest = np.nanargmin(
                np.abs(
                    np.where(
                        np.isfinite(sample_rho[:, b[~found]]),
                        sample_rho[:, b[~found]],
                        np.inf,
                    )
                    - centres[s[~found]][None, :]
                ),
                axis=0,
            )
            s_sample = s.copy()
            s_sample[~found] = nearest
        else:
            s_sample = s
        point = sample_point[s_sample, b]
        rho_here = np.clip(
            sample_rho[s_sample, b],
            rho_edges[s],
            np.nextafter(rho_edges[s + 1], -np.inf),
        )
        b_vec = np.asarray(xp.to_numpy(field.field_at(point[:, :3])), dtype=float)
        strength = np.linalg.norm(b_vec, axis=1)
        unit = b_vec / strength[:, None]
        v_par = point[:, 3]
        xi = np.clip(v_par / np.sqrt(v_par**2 + 2.0 * mu[b] * strength), -1.0, 1.0)
        v = speed[b]
        sub_pos[moved] = point[:, :3]
        sub_vel[moved] = v[:, None] * (
            xi[:, None] * unit + np.sqrt(1.0 - xi**2)[:, None] * _perpendicular(unit)
        )
        sub_rho[moved] = rho_here
        sub_pitch[moved] = xi
        sub_field[moved] = strength
    sub_energy = energy[birth_k]
    birth_rate = np.bincount(shell_k, sub_w, minlength=n_shells).astype(float)
    power = np.bincount(shell_k, sub_w * sub_energy, minlength=n_shells).astype(float)
    return FirstOrbitDeposition(
        births=ParticleArrays(
            xp.asarray(sub_pos), xp.asarray(sub_vel), xp.asarray(sub_w)
        ),
        birth_rho=xp.asarray(sub_rho),
        birth_pitch=xp.asarray(sub_pitch),
        birth_field=xp.asarray(sub_field),
        mass_number=deposition.mass_number,
        rho_edges=rho_edges,
        birth_rate=birth_rate,
        power=power,
        injected_power=deposition.injected_power,
        absorbed_power=deposition.absorbed_power,
        shine_through_power=deposition.shine_through_power,
        wall_hits=deposition.wall_hits,
        radial_coordinate=deposition.radial_coordinate,
        zero_width=deposition,
        parent=birth_k,
        shell_fractions=fractions,
        prompt_lost=lost,
        unclosed=unclosed,
        trapped=trapped,
        orbit_time=period,
        mean_rho=rho_mid,
        prompt_loss_power=float(np.sum(weights[lost] * energy[lost])),
        prompt_loss_rate=float(np.sum(weights[lost])),
    )
