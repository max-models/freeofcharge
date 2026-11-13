"""The toroidal torque of a beam: collisional split, j x B jump and balance."""

import numpy as np
import pytest
from plasmacoll import ParticleArrays
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from scipy.integrate import quad

from freeofcharge import (
    Equilibrium,
    GriddedEquilibrium,
    NeutralBeam,
    Plasma,
    deposit,
    tangential_injection,
)
from freeofcharge.gridded_equilibrium import _HelicalTestField
from freeofcharge.orbits import guiding_centres
from freeofcharge.slowing_down import collision_coefficients, pitch_at_speed
from freeofcharge.torque import beam_torque, parallel_momentum_moments

EDGES = np.linspace(0.0, 1.0, 11)
ENERGY = 1.0e6  # eV
MASS = 2 * ATOMIC_MASS


@pytest.fixture(scope="module")
def equilibrium():
    return Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3, points=97)


@pytest.fixture(scope="module")
def plasma():
    return Plasma.parabolic(1e20, 2e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)})


@pytest.fixture(scope="module")
def surfaces(equilibrium):
    return equilibrium.flux_surfaces(EDGES)


def _deposition(equilibrium, plasma, sense=1, tangency=5.3, width=0.3):
    origin, direction = tangential_injection(tangency, 12.0, sense=sense)
    beam = NeutralBeam.gaussian(
        origin,
        direction,
        ENERGY,
        16.5e6,
        width=width,
        divergence=0.0,
        markers=200,
        seed=1,
    )
    return deposit(
        beam,
        equilibrium,
        plasma,
        step=0.03,
        births_per_marker=2,
        rho_edges=EDGES,
        seed=2,
    )


@pytest.fixture(scope="module")
def co(equilibrium, plasma):
    return _deposition(equilibrium, plasma, 1)


@pytest.fixture(scope="module")
def counter(equilibrium, plasma):
    return _deposition(equilibrium, plasma, -1)


def test_momentum_moments_match_quadrature_with_constant_drags():
    # one Coulomb logarithm: a_i tau_s = v_c^3 and nu_D = beta v_c^3 / (tau_s v^3)
    v_c, tau_s, beta, v0 = 2.0e6, 0.3, 1.4, 7.0e6
    a_e, a_i = 1.0 / tau_s, v_c**3 / tau_s
    speed = v0 * 1.0e-6 ** np.linspace(0.0, 1.0, 401)[None, :]
    electrons, ions, end = parallel_momentum_moments(speed, a_e, a_i, beta * a_i)

    def pitch(v):
        return pitch_at_speed(v / v_c, v0 / v_c, 1.0, beta)

    # dt = tau_s v^2 dv / (v^3 + v_c^3)
    def dt(v):
        return tau_s * v**2 / (v**3 + v_c**3)

    exact_e = quad(lambda v: a_e * v * pitch(v) * dt(v), 0.0, v0, epsabs=0)[0]
    exact_i = quad(
        lambda v: (1 + beta) * a_i / v**2 * pitch(v) * dt(v), 0.0, v0, epsabs=0
    )[0]
    # the cumulative scattering is second order: 3e-5 with 401 speeds
    assert electrons[0] == pytest.approx(exact_e, rel=1e-4)
    assert ions[0] == pytest.approx(exact_i, rel=1e-4)
    # all the parallel momentum goes to the plasma: the three add up to v0
    assert electrons[0] + ions[0] + end[0] == pytest.approx(v0, rel=1e-4)
    assert end[0] < 1e-6 * v0


def test_momentum_moments_without_ions_go_to_the_electrons():
    # no ions: no scattering, the parallel momentum all to the electrons
    speed = 5.0e6 * 1.0e-4 ** np.linspace(0.0, 1.0, 201)[None, :]
    electrons, ions, end = parallel_momentum_moments(speed, 3.0, 0.0, 0.0)
    assert ions[0] == 0.0
    assert electrons[0] + end[0] == pytest.approx(5.0e6, rel=1e-7)


def test_momentum_moments_need_an_odd_number_of_speeds():
    with pytest.raises(ValueError, match="odd"):
        parallel_momentum_moments(np.ones((1, 4)), 1.0, 1.0, 1.0)


def test_ion_momentum_drag_includes_the_deflection(plasma):
    # at the critical speed the electron and ion energy drags are equal, but the
    # ions take more momentum: their drag plus the pitch-angle scattering
    a_e, a_i, gamma = collision_coefficients(
        3e6, MASS, 1, 1e20, 2e4, [0.96e20, 0.02e20], [MASS, 2 * MASS], [1, 2], 2e4
    )
    speed = 3e6 * 0.5 ** np.linspace(0.0, 1.0, 3)[None, :]
    electrons, ions, _ = parallel_momentum_moments(speed, a_e, a_i, gamma)
    ratio = ions[0] / electrons[0]
    assert ratio > float(a_i / (a_e * 3e6**3))


def test_all_birth_angular_momentum_ends_up_in_the_plasma(
    co, plasma, surfaces, equilibrium
):
    torque = beam_torque(co, plasma, surfaces, equilibrium)
    a = torque.attrs
    births = co.births
    x, v, w = births.positions, births.velocities, births.weights
    radius = np.hypot(x[:, 0], x[:, 1])
    v_phi = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / radius
    injected = float(np.sum(w * MASS * radius * v_phi))
    assert a["birth_angular_momentum"] == pytest.approx(injected, rel=1e-12)
    # a straight beam keeps R v_phi = v R_t: dL/dP = 2 R_t / v (times the absorbed share)
    speed = np.sqrt(2 * ENERGY * ELEMENTARY_CHARGE / MASS)
    assert injected / a["birth_power"] == pytest.approx(2 * 5.3 / speed, rel=0.01)
    # the components add up, and with the ions slowing down to rest nothing is left
    assert abs(a["balance"]) < 1e-12
    deposited = (
        a["collisional_electrons_total"]
        + a["collisional_ions_total"]
        + a["jxb_total"]
        + a["jxb_outside"]
    )
    assert deposited == pytest.approx(injected, rel=1e-4)
    assert abs(a["thermalised_angular_momentum"]) < 1e-4 * injected
    assert a["total"] == pytest.approx(float(torque.shell_torque.sum()))
    # densities times volumes are the shell torques
    np.testing.assert_allclose(
        torque.total * surfaces.volume, torque.shell_torque.sum("component")
    )
    # tangential co-injection: all components along the beam, the j x B part small
    assert a["collisional_electrons_total"] > 0
    assert a["collisional_ions_total"] > 0
    assert 0 < a["jxb_total"] < 0.2 * injected
    assert torque.total.attrs["units"] == "N m/m^3"
    assert list(torque.component.values) == [
        "collisional_electrons",
        "collisional_ions",
        "jxb",
    ]


def test_counter_injection_reverses_the_torque(
    co, counter, plasma, surfaces, equilibrium
):
    forward = beam_torque(co, plasma, surfaces, equilibrium).attrs
    backward = beam_torque(counter, plasma, surfaces, equilibrium).attrs
    for name in ("collisional_electrons_total", "collisional_ions_total", "jxb_total"):
        assert backward[name] < 0
        assert -backward[name] == pytest.approx(forward[name], rel=0.05)


def test_jxb_jump_matches_the_flux_at_the_guiding_centre(co, equilibrium):
    # p_phi conservation gives q (psi_gc - psi_birth) = m R (v_phi - v_par b_phi);
    # the first-order guiding centre x - b x v / Omega gives the same jump to O(rho_L / L)
    births = co.births
    centres = guiding_centres(
        births.positions, births.velocities, births.weights, equilibrium, MASS
    )
    x, v = births.positions, births.velocities
    radius = np.hypot(x[:, 0], x[:, 1])
    e_phi = np.stack([-x[:, 1], x[:, 0], np.zeros_like(radius)], -1) / radius[:, None]
    b = equilibrium.field_at(x)
    b_unit = b / np.linalg.norm(b, axis=1)[:, None]
    v_par = np.sum(v * b_unit, axis=1)
    jump = MASS * radius * (np.sum(v * e_phi, 1) - v_par * np.sum(b_unit * e_phi, 1))
    r_c = np.hypot(centres.positions[:, 0], centres.positions[:, 1])
    geometric = ELEMENTARY_CHARGE * (
        equilibrium.psi(r_c, centres.positions[:, 2]) - equilibrium.psi(radius, x[:, 2])
    )
    scale = np.max(np.abs(jump))
    assert np.max(np.abs(geometric - jump)) < 0.03 * scale  # 1.1 %
    assert np.sum(geometric) == pytest.approx(np.sum(jump), rel=0.03)  # 1.7 %


def _single(births, equilibrium, plasma):
    from freeofcharge.deposition import Deposition

    x = births.positions
    rho = equilibrium.rho_at(x)
    return Deposition(
        births=births,
        birth_rho=rho,
        birth_pitch=np.zeros(1),
        birth_field=np.linalg.norm(equilibrium.field_at(x), axis=1),
        mass_number=2,
        rho_edges=EDGES,
        birth_rate=np.zeros(len(EDGES) - 1),
        power=np.zeros(len(EDGES) - 1),
        injected_power=0.0,
        absorbed_power=0.0,
        shine_through_power=0.0,
        wall_hits=None,
    )


def test_jxb_torque_lies_between_birth_and_guiding_centre(
    equilibrium, plasma, surfaces
):
    # one birth on the midplane, its guiding centre one shell further in
    span = equilibrium.psi_boundary - equilibrium.psi_axis
    radii = np.linspace(6.2, 8.0, 2000)
    r_birth = float(
        np.interp(
            0.55**2, equilibrium.psi_normalized(radii, np.zeros_like(radii)), radii
        )
    )
    b = equilibrium.field_at(np.array([[r_birth, 0.0, 0.0]]))[0]
    # a velocity across B, toroidal component chosen for a jump of 0.2 in psi_n
    jump = (0.45**2 - 0.55**2) * span * ELEMENTARY_CHARGE
    v_phi = jump / (MASS * r_birth)
    b_unit = b / np.linalg.norm(b)
    velocity = np.array([0.0, v_phi, 0.0])
    velocity -= np.dot(velocity, b_unit) * b_unit  # perpendicular: v_par = 0
    scale = v_phi / velocity[1]
    velocity *= scale
    births = ParticleArrays(
        np.array([[r_birth, 0.0, 0.0]]), velocity[None, :], np.array([1.0e20])
    )
    deposition = _single(births, equilibrium, plasma)
    torque = beam_torque(deposition, plasma, surfaces, equilibrium)
    jxb = torque.shell_torque.sel(component="jxb").values
    # psi_n from 0.55^2 down to 0.45^2: shells [0.4, 0.5] (psi_n 0.16-0.25) and [0.5, 0.6]
    start = float(equilibrium.psi_normalized(r_birth, 0.0))
    expected = np.zeros(len(EDGES) - 1)
    expected[4] = 0.2025 - 0.25
    expected[5] = 0.25 - start
    expected *= 1.0e20 * ELEMENTARY_CHARGE * span
    np.testing.assert_allclose(
        jxb, expected, rtol=0.02, atol=1e-9 * np.abs(expected).max()
    )
    # no parallel velocity: no collisional torque
    assert torque.attrs["collisional_electrons_total"] == pytest.approx(0.0, abs=1e-12)
    assert torque.attrs["jxb_total"] == pytest.approx(1.0e20 * jump, rel=1e-3)


def test_shells_must_match(co, plasma, equilibrium):
    with pytest.raises(ValueError, match="different shells"):
        beam_torque(
            co, plasma, equilibrium.flux_surfaces(np.linspace(0, 1, 6)), equilibrium
        )


def test_thermal_factor_leaves_angular_momentum_to_thermalise(
    co, plasma, surfaces, equilibrium
):
    torque = beam_torque(
        co, plasma, surfaces, equilibrium, thermal_factor=1.5, ln_lambda=17.0
    )
    a = torque.attrs
    assert a["thermalised_angular_momentum"] > 0
    assert abs(a["balance"]) < 1e-12


def test_collisional_torque_on_3d_shells():
    helical = _HelicalTestField()
    gridded = GriddedEquilibrium(helical, (48, 32, 48))
    edges = np.linspace(0.0, 1.0, 6)
    surfaces = helical.flux_surfaces(edges, field_points=(32, 16))
    plasma = Plasma.parabolic(5e19, 3e3, ions={"H": (1, 1, 1.0)})
    beam = NeutralBeam.gaussian(
        [4.0, -2.5, 0.0],
        [-0.55, 1.0, 0.0],
        55e3,
        1.8e6,
        mass_number=1,
        width=0.05,
        divergence=0.01,
        markers=100,
        seed=1,
    )
    deposition = deposit(beam, gridded, plasma, step=0.01, rho_edges=edges, seed=2)
    with pytest.raises(ValueError, match="jxb=False"):
        beam_torque(deposition, plasma, surfaces, gridded)
    torque = beam_torque(deposition, plasma, surfaces, gridded, jxb=False)
    assert "jxb" not in torque
    assert list(torque.component.values) == [
        "collisional_electrons",
        "collisional_ions",
    ]
    a = torque.attrs
    assert abs(a["balance"]) < 1e-12
    assert a["total"] == pytest.approx(a["guiding_centre_angular_momentum"], rel=1e-3)
    assert torque.rho.attrs["radial_coordinate"] == "toroidal"
