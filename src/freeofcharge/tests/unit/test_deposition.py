import numpy as np
import pytest
from plasmacoll import ParticleArrays

from freeofcharge import (
    ConstantStopping,
    Equilibrium,
    IonSpecies,
    NeutralBeam,
    Plasma,
    deposit,
    slow_down,
    tangential_injection,
)

R0, A = 3.0, 0.9
R_INNER = np.sqrt(2 * R0**2 - (R0 + A) ** 2)


@pytest.fixture(scope="module")
def equilibrium():
    return Equilibrium.solovev(R0, A, 1.6, 2.5, 1.2, wall_psi_n=1.3)


def flat_plasma(ne=5e19, te=5e3):
    rho = np.array([0.0, 1.0])
    return Plasma(
        rho=rho,
        ne=np.full(2, ne),
        te=np.full(2, te),
        ti=np.full(2, te),
        ions={"D": IonSpecies(2, 1, np.full(2, ne))},
    )


def pencil(origin, direction, energy=100e3, rate=1.0e20, mass_number=2):
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    speed = np.sqrt(2 * energy * 1.602176634e-19 / (mass_number * 1.66053906660e-27))
    markers = ParticleArrays(
        np.array([origin], float), np.array([speed * direction]), np.array([rate])
    )
    return NeutralBeam(markers, mass_number)


def test_radial_pencil_matches_the_slab_attenuation(equilibrium):
    sigma, ne = 2.0e-20, 5e19
    beam = pencil([R0 + A + 0.3, 0.0, 0.0], [-1.0, 0.0, 0.0])
    result = deposit(
        beam, equilibrium, flat_plasma(ne), ConstantStopping(sigma), step=1e-3, seed=1
    )
    chord = (R0 + A) - R_INNER  # the line stops at the inboard wall
    tau = ne * sigma * chord
    assert -np.log(result.shine_through_fraction) == pytest.approx(tau, rel=2e-3)
    assert result.absorbed_power + result.shine_through_power == pytest.approx(
        result.injected_power, rel=1e-12
    )


def test_radial_pencil_hits_the_inboard_wall(equilibrium):
    beam = pencil([R0 + A + 0.3, 0.0, 0.0], [-1.0, 0.0, 0.0])
    result = deposit(
        beam, equilibrium, flat_plasma(), ConstantStopping(1e-20), step=1e-3
    )
    wall_inner = equilibrium.wall.vertices[:, 0].min()
    hit = np.asarray(result.wall_hits.positions)[0]
    assert np.hypot(hit[0], hit[1]) == pytest.approx(wall_inner, abs=2e-3)
    assert float(
        result.wall_hits.weights[0]
    ) * 100e3 * 1.602176634e-19 == pytest.approx(result.shine_through_power, rel=1e-12)


def test_without_a_wall_the_line_crosses_the_plasma_twice():
    equilibrium = Equilibrium.solovev(R0, A, 1.6, 2.5, 1.2, wall_psi_n=None)
    sigma, ne = 1.0e-20, 5e19
    beam = pencil([R0 + A + 0.3, 0.0, 0.0], [-1.0, 0.0, 0.0])
    result = deposit(
        beam, equilibrium, flat_plasma(ne), ConstantStopping(sigma), step=1e-3
    )
    assert result.wall_hits is None
    tau = ne * sigma * 2 * ((R0 + A) - R_INNER)
    assert -np.log(result.shine_through_fraction) == pytest.approx(tau, rel=2e-3)


def test_births_sample_the_deposition_profile(equilibrium):
    plasma = Plasma.parabolic(6e19, 4e3, ions={"D": (2, 1, 1.0)})
    origin, direction = tangential_injection(R0 - 0.2, R0 + 3 * A)
    beam = NeutralBeam.gaussian(
        origin, direction, 80e3, 2e6, width=0.1, divergence=0.01, markers=4000, seed=3
    )
    result = deposit(
        beam,
        equilibrium,
        plasma,
        births_per_marker=5,
        step=5e-3,
        seed=4,
        rho_edges=np.linspace(0, 1, 11),
    )
    weights = np.asarray(result.births.weights)
    assert weights.sum() == pytest.approx(result.birth_rate.sum(), rel=1e-10)
    sampled, _ = np.histogram(
        np.asarray(result.birth_rho), bins=result.rho_edges, weights=weights
    )
    share = result.birth_rate / result.birth_rate.sum()
    np.testing.assert_allclose(sampled / weights.sum(), share, atol=0.015)
    # a birth lies at most half a step beyond the boundary
    assert np.all(np.asarray(result.birth_rho) <= 1.01)


def test_power_balance_with_suzuki(equilibrium):
    plasma = Plasma.parabolic(
        5e19, 5e3, ions={"D": (2, 1, 0.9), "C": (12, 6, 1.0 / 60.0)}
    )
    origin, direction = tangential_injection(R0, R0 + 3 * A)
    beam = NeutralBeam.gaussian(
        origin, direction, 100e3, 5e6, width=0.1, divergence=0.01, markers=500, seed=5
    )
    result = deposit(beam, equilibrium, plasma, step=5e-3, seed=6)
    assert result.injected_power == pytest.approx(5e6, rel=1e-12)
    assert result.power.sum() == pytest.approx(result.absorbed_power, rel=1e-10)
    assert result.absorbed_power + result.shine_through_power == pytest.approx(
        5e6, rel=1e-12
    )
    assert 0.0 < result.shine_through_fraction < 0.5


def test_slowing_down_conserves_the_absorbed_power(equilibrium):
    plasma = Plasma.parabolic(5e19, 5e3, ions={"D": (2, 1, 1.0)})
    origin, direction = tangential_injection(R0, R0 + 3 * A)
    beam = NeutralBeam.gaussian(
        origin, direction, 100e3, 5e6, width=0.1, divergence=0.01, markers=1000, seed=7
    )
    result = deposit(
        beam, equilibrium, plasma, step=5e-3, seed=8, rho_edges=np.linspace(0, 1, 21)
    )
    surfaces = equilibrium.flux_surfaces(result.rho_edges)
    steady = slow_down(result, plasma, surfaces)
    heating = steady.electron_heating.sum() + steady.ion_heating.sum()
    assert heating == pytest.approx(result.absorbed_power, rel=1e-10)
    assert steady.driven_current > 0.0  # co-injection, along B
    assert np.all(
        np.abs(steady.current_density) <= np.abs(steady.fast_ion_current_density)
    )
    assert steady.stored_energy > 0.0


def test_slowing_down_needs_the_same_shells(equilibrium):
    plasma = Plasma.parabolic(5e19, 5e3, ions={"D": (2, 1, 1.0)})
    beam = pencil([R0 + A + 0.3, 0.0, 0.0], [-1.0, 0.0, 0.0])
    result = deposit(beam, equilibrium, plasma, step=5e-3)
    with pytest.raises(ValueError, match="shells"):
        slow_down(result, plasma, equilibrium.flux_surfaces(np.linspace(0, 1, 5)))
