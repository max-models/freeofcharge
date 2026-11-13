"""First-orbit redistribution of the births (`freeofcharge.orbit_width`)."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import ParticleArrays
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

import freeofcharge as foc
from freeofcharge import ascot5, orbits
from freeofcharge.deposition import Deposition
from freeofcharge.orbit_width import FirstOrbitDeposition, orbit_average_births

E = ELEMENTARY_CHARGE
MASS = 2 * ATOMIC_MASS


def _case(field_strength: float, markers: int = 200):
    """The ASCOT5 benchmark geometry (60 keV D, 44 % trapped) at ``field_strength``."""
    case = ascot5.benchmark_case(
        markers=markers, toroidal_field=field_strength, grid_points=65
    )
    deposition = foc.deposit(
        case.beam,
        case.equilibrium,
        case.plasma,
        step=0.01,
        births_per_marker=1,
        rho_edges=case.rho_edges,
        seed=2,
    )
    return case, deposition


@pytest.fixture(scope="module")
def benchmark():
    """The benchmark case at 2.5 T, its births and their first-orbit average."""
    case, deposition = _case(2.5)
    field = orbits.orbit_field(case.equilibrium, (96, 4, 96))
    return (
        case,
        deposition,
        field,
        orbit_average_births(deposition, case.equilibrium, field=field),
    )


def _deposition(field, positions, pitch, rho_edges, energy=60e3):
    """A `Deposition` of single births at ``positions`` with ``pitch``."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    b = np.asarray(field.field_at(positions))
    strength = np.linalg.norm(b, axis=1)
    unit = b / strength[:, None]
    across = np.cross(unit, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across, axis=1)[:, None]
    speed = np.sqrt(2 * energy * E / MASS)
    velocity = speed * (pitch[:, None] * unit + np.sqrt(1 - pitch**2)[:, None] * across)
    n_shells = len(rho_edges) - 1
    births = ParticleArrays(positions, velocity, np.ones(len(positions)))
    return Deposition(
        births=births,
        birth_rho=np.asarray(field.rho_at(positions)),
        birth_pitch=np.asarray(pitch, dtype=float),
        birth_field=strength,
        mass_number=2,
        rho_edges=np.asarray(rho_edges, dtype=float),
        birth_rate=np.zeros(n_shells),
        power=np.zeros(n_shells),
        injected_power=0.0,
        absorbed_power=0.0,
        shine_through_power=0.0,
        wall_hits=None,
    )


def test_power_is_conserved_minus_the_prompt_losses(benchmark):
    """Sum of the redistributed power = birth power - prompt losses, to roundoff."""
    case, deposition, _, first = benchmark
    w = np.asarray(deposition.births.weights)
    v = np.asarray(deposition.births.velocities)
    birth_power = float(np.sum(w * 0.5 * MASS * np.sum(v * v, axis=1)))
    assert isinstance(first, FirstOrbitDeposition)
    assert first.prompt_lost.any()  # a few edge births at 2.5 T
    assert first.prompt_loss_power > 0.0
    assert first.confined_power + first.prompt_loss_power == pytest.approx(
        birth_power, rel=1e-12
    )
    assert np.sum(first.birth_rate) + first.prompt_loss_rate == pytest.approx(
        np.sum(w), rel=1e-12
    )
    # every confined birth is split, with fractions summing to one
    rows = first.shell_fractions.sum(axis=1)
    np.testing.assert_allclose(rows[~first.prompt_lost], 1.0, rtol=1e-12)
    np.testing.assert_array_equal(rows[first.prompt_lost], 0.0)
    assert np.all(np.isnan(first.mean_rho[first.prompt_lost]))
    # the sub-births carry the redistributed power, each on its own shell
    sub_w = np.asarray(first.births.weights)
    sub_v = np.asarray(first.births.velocities)
    np.testing.assert_allclose(
        np.sum(sub_v**2, axis=1), np.sum(v**2, axis=1)[first.parent]
    )
    assert np.sum(sub_w * 0.5 * MASS * np.sum(sub_v**2, axis=1)) == pytest.approx(
        first.confined_power, rel=1e-12
    )
    shell = np.searchsorted(first.rho_edges, np.asarray(first.birth_rho), "right") - 1
    np.testing.assert_array_equal(
        np.bincount(shell, sub_w, minlength=len(first.rho_edges) - 1) > 0,
        first.birth_rate > 0,
    )
    # slow_down takes it unchanged and balances the confined power
    surfaces = case.equilibrium.flux_surfaces(case.rho_edges)
    steady = foc.slow_down(first, case.plasma, surfaces)
    assert steady.birth_power == pytest.approx(first.confined_power, rel=1e-12)
    assert abs(steady.power_balance) < 1e-10
    # the redistribution matrix: rows are the shells' confined shares
    matrix = first.redistribution()
    data = first.to_dataset(surfaces)
    np.testing.assert_allclose(
        data.zero_width_power.values @ matrix, first.power, rtol=1e-10, atol=1e-6
    )
    assert data.attrs["prompt_loss_power"] == first.prompt_loss_power
    assert 0.0 < data.attrs["prompt_loss_fraction"] < 0.05
    assert "zero_width_power_density" in data


def _case_equilibrium(strength):
    return ascot5.benchmark_case(
        markers=1, toroidal_field=strength, grid_points=65
    ).equilibrium


def test_high_field_is_the_identity():
    """Narrow orbits stay in their birth shell; the share that leaves falls as 1/B."""
    leaving = {}
    for strength in (10.0, 40.0):
        _, deposition = _case(strength, markers=120)
        first = orbit_average_births(deposition, _case_equilibrium(strength))
        zero_width = first.to_dataset().zero_width_power.values
        stay = np.sum(np.diag(first.redistribution()) * zero_width) / zero_width.sum()
        leaving[strength] = 1.0 - stay
        assert not first.prompt_lost.any()
        assert not first.unclosed.any()
    # the orbit width over a shell width of 0.05 in rho
    assert leaving[40.0] < 0.07
    assert leaving[10.0] / leaving[40.0] == pytest.approx(4.0, rel=0.2)


def test_one_period_returns_to_the_start(benchmark):
    """Pushed for its period, each orbit (passing and trapped) is back where it began."""
    case, deposition, field, first = benchmark
    pick = np.r_[
        np.nonzero(first.trapped & ~first.prompt_lost)[0][:3],
        np.nonzero(~first.trapped & ~first.prompt_lost)[0][:3],
    ]
    births = deposition.births
    markers = orbits.guiding_centres(
        np.asarray(births.positions)[pick],
        np.asarray(births.velocities)[pick],
        np.ones(len(pick)),
        field,
        MASS,
    )
    for k, j in enumerate(pick):
        one = orbits.GuidingCentres(
            markers.positions[k : k + 1],
            markers.v_parallel[k : k + 1],
            markers.mu[k : k + 1],
            np.ones(1),
            MASS,
        )
        run = orbits.trace_orbits(
            one, field, first.orbit_time[j], samples=401, step=0.01
        )
        r, z = run.R.values[:, 0], run.z.values[:, 0]
        size = np.max(np.hypot(r - r[0], z - z[0]))
        assert np.hypot(r[-1] - r[0], z[-1] - z[0]) < 0.02 * size
        # trapped orbits turn twice per bounce, passing ones never
        turns = np.sum(
            np.sign(run.v_parallel.values[1:, 0])
            != np.sign(run.v_parallel.values[:-1, 0])
        )
        assert turns == (2 if first.trapped[j] else 0)


def test_co_and_counter_shifts_follow_p_phi():
    """Births on the outboard midplane move as the conservation of p_phi predicts."""
    eq = foc.Equilibrium.solovev(1.65, 0.5, 1.6, 2.5, 1.0, wall_psi_n=1.3, points=65)
    field = orbits.orbit_field(eq, (96, 4, 96))
    r0, rb = 1.65, 1.9
    pitch = np.array([0.7, -0.7])
    edges = np.linspace(0.0, 1.0, 101)
    deposition = _deposition(field, [[rb, 0.0, 0.0]] * 2, pitch, edges)
    first = orbit_average_births(deposition, eq, field=field)
    assert not first.trapped.any() and not first.prompt_lost.any()
    births = deposition.births
    gc = orbits.guiding_centres(
        np.asarray(births.positions),
        np.asarray(births.velocities),
        np.ones(2),
        field,
        MASS,
    )
    r = np.hypot(gc.positions[:, 0], gc.positions[:, 1])
    z = gc.positions[:, 2]
    # p_phi = m R v_par b_phi + e psi, at the inboard midplane of the same surface
    r_in = np.sqrt(2 * r0**2 - r**2)  # the Solov'ev flux on the midplane
    b_in = np.asarray(field.field_at(np.stack([r_in, 0 * r_in, z], -1)))
    strength_in = np.linalg.norm(b_in, axis=1)
    speed = np.sqrt(2 * 60e3 * E / MASS)
    v_in = np.sign(gc.v_parallel) * np.sqrt(speed**2 - 2 * gc.mu * strength_in)
    b_0 = np.asarray(field.field_at(gc.positions))
    shift = -(MASS / E) * (
        r_in * v_in * b_in[:, 1] / strength_in
        - r * gc.v_parallel * b_0[:, 1] / np.linalg.norm(b_0, axis=1)
    )
    predicted = shift / (eq.psi_boundary - eq.psi_axis)
    centres = 0.5 * (edges[1:] + edges[:-1])
    mean = first.shell_fractions @ centres**2 - eq.psi_normalized(r, z)
    assert predicted[0] > 0 > predicted[1]  # co outward, counter inward
    assert np.all(np.sign(mean) == np.sign(predicted))
    # the orbit's mean is about half way to the inboard displacement
    assert np.all((mean / predicted > 0.3) & (mean / predicted < 0.8))


def test_co_and_counter_beams_shift_opposite():
    """A beam and its mirror image move the deposition in opposite directions."""
    eq = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 2.5, 1.2, wall_psi_n=1.3, points=65)
    plasma = foc.Plasma.parabolic(5e19, 3e3, ions={"H": (1, 1, 1.0)})
    edges = np.linspace(0.0, 1.0, 11)
    centres = 0.5 * (edges[1:] + edges[:-1])
    shifts = {}
    for sense in (1, -1):
        origin, direction = foc.tangential_injection(2.9, 6.0, sense=sense)
        beam = foc.NeutralBeam.gaussian(
            origin,
            direction,
            55e3,
            1e6,
            mass_number=1,
            width=0.1,
            divergence=0.005,
            markers=100,
            seed=1,
        )
        deposition = foc.deposit(
            beam, eq, plasma, step=0.01, births_per_marker=1, rho_edges=edges, seed=2
        )
        first = orbit_average_births(deposition, eq)
        w = np.asarray(deposition.births.weights)
        ok = ~first.prompt_lost
        shell = np.searchsorted(edges, np.asarray(deposition.birth_rho), "right") - 1
        shifts[sense] = (
            np.sum(w[ok] * (first.mean_rho[ok] - centres[shell[ok]])) / w[ok].sum()
        )
    assert shifts[1] > 0.005 and shifts[-1] < -0.005


def test_first_orbit_is_closer_to_orbit_following():
    """In the 2.5 T benchmark geometry, the first orbit is much closer to follow_orbits than zero width."""
    case = ascot5.benchmark_case(markers=60, grid_points=65)
    plasma = foc.Plasma.parabolic(1.5e20, 1.0e3, ions={"D": (2, 1, 1.0)})
    deposition = foc.deposit(
        case.beam,
        case.equilibrium,
        plasma,
        step=0.01,
        births_per_marker=2,
        rho_edges=case.rho_edges[::2],
        seed=5,
    )
    field = orbits.orbit_field(case.equilibrium, (96, 4, 96))
    surfaces = case.equilibrium.flux_surfaces(deposition.rho_edges)
    end = {"thermal_factor": 2.0, "minimum_energy": 1e3}
    first = orbit_average_births(deposition, case.equilibrium, field=field)
    zero = foc.slow_down(deposition, plasma, surfaces, **end)
    orbit = foc.slow_down(first, plasma, surfaces, **end)
    followed = orbits.follow_orbits(
        deposition,
        field,
        plasma,
        surfaces=surfaces,
        step=0.2,
        energy_diffusion=False,
        seed=3,
        **end,
    )

    def gap(a, b):
        return np.sum(np.abs(a / a.sum() - b / b.sum()))

    reference = followed.electron_heating + followed.ion_heating
    for model in (zero, orbit):
        assert model.birth_power > 0
    zero_gap = gap(zero.electron_heating + zero.ion_heating, reference)
    first_gap = gap(orbit.electron_heating + orbit.ion_heating, reference)
    assert first_gap < 0.6 * zero_gap
    density_zero = gap(zero.fast_ion_density * surfaces.volume, followed.particles)
    density_first = gap(orbit.fast_ion_density * surfaces.volume, followed.particles)
    assert density_first < 0.6 * density_zero


def test_unclosed_and_outside_and_errors(benchmark):
    """Orbits cut by max_length are averaged; 3D fields and births off the grid fail."""
    case, deposition, field, _ = benchmark
    cut = orbit_average_births(
        deposition, case.equilibrium, field=field, max_length=1.0
    )
    assert cut.unclosed[~cut.prompt_lost].all()
    np.testing.assert_allclose(cut.shell_fractions[~cut.prompt_lost].sum(axis=1), 1.0)
    # shells covering only the core: orbits outside them stay on their birth shell
    inner = _deposition(
        field, [[2.05, 0.0, 0.0]], np.array([0.9]), np.linspace(0, 0.5, 6)
    )
    outside = orbit_average_births(inner, case.equilibrium, field=field, rho_limit=1.5)
    np.testing.assert_allclose(outside.shell_fractions, [[0, 0, 0, 0, 1.0]])
    far = _deposition(field, [[1.0, 0.0, 5.0]], np.array([0.5]), case.rho_edges)
    with pytest.raises(ValueError, match="off the field's grid"):
        orbit_average_births(far, case.equilibrium, field=field)
    helical = foc.GriddedEquilibrium(
        foc.gridded_equilibrium._HelicalTestField(), (16, 8, 16), order=3
    )
    with pytest.raises(ValueError, match="axisymmetric"):
        orbit_average_births(deposition, helical)
    bare = FirstOrbitDeposition(**{**vars(deposition)})
    with pytest.raises(ValueError, match="zero-width"):
        bare.redistribution()
    assert bare.prompt_loss_fraction == 0.0
