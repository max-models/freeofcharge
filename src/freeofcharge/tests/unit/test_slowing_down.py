import numpy as np
import pytest
from scipy import constants
from scipy.integrate import quad, solve_ivp

from freeofcharge.slowing_down import (
    collision_coefficients,
    coulomb_logarithm,
    coulomb_logarithms,
    critical_velocity,
    electron_collisionality,
    ion_heating_fraction,
    l31_coefficient,
    pitch_angle_exponent,
    pitch_at_speed,
    slowing_down_integrals,
    slowing_down_moments,
    slowing_down_time,
    start_cordey_fit,
)

AMU = constants.atomic_mass


def test_slowing_down_time_matches_the_engineering_formula():
    # tau_s = 6.27e8 A T_e^(3/2) / (Z^2 n_e[cm^-3] lnL) s, T_e in eV
    ne, te, ln_lambda = 5e19, 8e3, 17.0
    tau = slowing_down_time(2 * AMU, 1, ne, te, ln_lambda)
    assert tau == pytest.approx(
        6.27e8 * 2 * te**1.5 / (ne * 1e-6 * ln_lambda), rel=3e-3
    )


def test_coulomb_logarithm_is_typical():
    assert 15.0 < coulomb_logarithm(1e20, 1e4) < 20.0


def test_critical_energy_matches_the_engineering_formula():
    # E_c = 14.8 A_b T_e [sum n_j Z_j^2 / (n_e A_j)]^(2/3)
    ne, te = 1e20, 1.2e4
    densities, masses, charges = [0.9 * ne, 0.05 * ne], [2, 4], [1, 2]
    v_c = critical_velocity(ne, te, densities, [m * AMU for m in masses], charges)
    e_c = 0.5 * 2 * AMU * v_c**2 / constants.e
    bracket = sum(
        n * z**2 / (ne * a) for n, a, z in zip(densities, masses, charges, strict=True)
    )
    assert e_c == pytest.approx(14.8 * 2 * te * bracket ** (2 / 3), rel=3e-3)


@pytest.mark.parametrize("x", [0.05, 1.0, 3.0, 40.0])
def test_ion_heating_fraction_matches_quadrature(x):
    integral, _ = quad(lambda y: 1.0 / (1.0 + y**1.5), 0.0, x)
    assert ion_heating_fraction(x) == pytest.approx(integral / x, rel=1e-9)


def test_ion_heating_fraction_limits():
    assert ion_heating_fraction(1e-6) == pytest.approx(1.0, abs=1e-8)
    assert ion_heating_fraction(1e6) < 0.01


def test_pitch_angle_exponent_is_one_in_a_pure_plasma_of_the_beam_isotope():
    beta = pitch_angle_exponent(2 * AMU, 1e20, [1e20], [2 * AMU], [1])
    assert beta == pytest.approx(1.0)


def test_slowing_down_integrals_match_the_equations_of_motion():
    v_c, tau_s, beta, xi0 = 2.0e6, 0.3, 1.4, 0.7
    v0 = 3.5 * v_c

    def rhs(t, y):
        v, xi, kinetic, parallel = y
        dv = -(v**3 + v_c**3) / (tau_s * v**2)
        dxi = -beta * v_c**3 * xi / (tau_s * v**3)
        return [dv, dxi, v**2, v * xi]

    stop = lambda t, y: y[0] - 1e-3 * v_c  # noqa: E731
    stop.terminal = True
    solution = solve_ivp(
        rhs, (0.0, 10.0), [v0, xi0, 0.0, 0.0], events=stop, rtol=1e-11, atol=1e-12
    )
    v_end, xi_end, kinetic, parallel = solution.y[:, -1]
    time, kinetic_closed, parallel_closed = slowing_down_integrals(
        np.array([v0]), np.array([xi0]), v_c, tau_s, beta
    )
    assert time[0] == pytest.approx(solution.t[-1], rel=1e-6)
    assert kinetic_closed[0] == pytest.approx(kinetic, rel=1e-6)
    assert parallel_closed[0] == pytest.approx(parallel, rel=1e-6)
    assert pitch_at_speed(v_end / v_c, v0 / v_c, xi0, beta) == pytest.approx(
        xi_end, rel=1e-6
    )


def test_energy_integral_is_closed_form():
    v_c, tau_s = 1.0e6, 0.2
    v0 = 4.0 * v_c
    _, kinetic, _ = slowing_down_integrals(
        np.array([v0]), np.array([1.0]), v_c, tau_s, 1.0
    )
    expected, _ = quad(lambda v: v**2 * tau_s * v**2 / (v**3 + v_c**3), 0.0, v0)
    assert kinetic[0] == pytest.approx(expected, rel=1e-12)


def test_coulomb_logarithms_of_a_60_kev_deuteron():
    # the centre of the ASCOT5 benchmark: ASCOT5 mccc_coefs_clog evaluated by hand
    ne, t = 5e19, 2.5e3
    ln_e, (ln_d, ln_c) = coulomb_logarithms(
        np.sqrt(2 * 60e3 * constants.e / (2 * AMU)),
        2 * AMU,
        1,
        ne,
        t,
        [0.94 * ne, 0.01 * ne],
        [2 * AMU, 12 * AMU],
        [1, 6],
        t,
    )
    assert ln_e == pytest.approx(16.70, abs=0.02)
    assert ln_d == pytest.approx(21.13, abs=0.02)
    assert ln_c == pytest.approx(19.85, abs=0.02)
    # the electron one is about the NRL one
    assert ln_e == pytest.approx(coulomb_logarithm(ne, t), abs=1.0)


def test_ion_coulomb_logarithm_grows_with_the_speed():
    args = (2 * AMU, 1, 5e19, 2.5e3, [5e19], [2 * AMU], [1], 2.5e3)
    slow = coulomb_logarithms(5e5, *args)
    fast = coulomb_logarithms(5e6, *args)
    assert fast[1][0] > slow[1][0] + 2.0
    assert fast[0] == pytest.approx(slow[0], abs=0.05)


def test_collision_coefficients_reduce_to_stix_with_one_logarithm():
    ne, te, ln = 6e19, 4e3, 17.0
    densities, masses, charges = [0.9 * ne, 0.02 * ne], [2 * AMU, 9 * AMU], [1, 4]
    a_e, a_i, deflection = collision_coefficients(
        1e6, 2 * AMU, 1, ne, te, densities, masses, charges, 3e3, ln_lambda=ln
    )
    tau = slowing_down_time(2 * AMU, 1, ne, te, ln)
    assert a_e * tau == pytest.approx(1.0)
    v_c = critical_velocity(ne, te, densities, masses, charges)
    assert a_i * tau == pytest.approx(v_c**3, rel=1e-12)
    beta = pitch_angle_exponent(2 * AMU, ne, densities, masses, charges)
    assert deflection / a_i == pytest.approx(beta, rel=1e-12)


def test_slowing_down_moments_match_the_closed_forms():
    v_c, tau_s, beta, xi0 = 2.0e6, 0.3, 1.4, 0.7
    v0 = np.array([3.5 * v_c, 0.8 * v_c])
    speeds = v0[:, None] * 1e-4 ** np.linspace(0.0, 1.0, 801)
    time, kinetic, ion_loss, parallel = slowing_down_moments(
        speeds, 1.0 / tau_s, v_c**3 / tau_s, beta * v_c**3 / tau_s, np.ones((2, 1))
    )
    closed = slowing_down_integrals(v0, np.full(2, xi0), v_c, tau_s, beta)
    assert time == pytest.approx(closed[0], rel=1e-4)
    assert kinetic == pytest.approx(closed[1], rel=1e-4)
    assert xi0 * parallel[:, 0] == pytest.approx(closed[2], rel=1e-4)
    share = ion_heating_fraction((v0 / v_c) ** 2)
    assert ion_loss == pytest.approx(share * 0.5 * v0**2, rel=1e-4)


def test_gaffey_higher_legendre_terms_decay_with_l_l_plus_one():
    """Gaffey's l-th term: exponent l (l + 1) beta / 6, rate l (l + 1) / 2."""
    v_c, tau_s, beta = 1.0e6, 0.2, 1.2
    v0 = np.array([3.0 * v_c])
    speeds = v0[:, None] * np.geomspace(1.0, 0.5, 401)[None, :]
    rates = np.array([[1.0, 3.0, 6.0]])
    *_, parallel = slowing_down_moments(
        speeds, 1.0 / tau_s, v_c**3 / tau_s, beta * v_c**3 / tau_s, rates
    )

    def term(order):
        exponent = order * (order + 1) / 2

        def integrand(v):
            pitch = pitch_at_speed(v / v_c, v0[0] / v_c, 1.0, beta)
            return v * pitch**exponent * tau_s * v**2 / (v**3 + v_c**3)

        return quad(integrand, 0.5 * v0[0], v0[0])[0]

    assert parallel[0] == pytest.approx([term(1), term(2), term(3)], rel=1e-4)


def test_start_cordey_fit_agrees_with_collisionless_l31():
    """Honda et al. (2012) find the two within a few percent at large aspect ratio."""
    epsilon = np.array([0.02, 0.05, 0.1, 0.2, 0.3])
    trapped = 1.46 * np.sqrt(epsilon) - 0.46 * epsilon**1.5
    for zeff in (1.0, 1.5, 2.0, 3.0):
        fit = start_cordey_fit(zeff, epsilon)
        l31 = l31_coefficient(zeff, trapped)
        assert fit == pytest.approx(l31, rel=0.07)


def test_collisions_reduce_the_trapped_electron_correction():
    nu = electron_collisionality(5e19, 2e3, 1.5, 1.5, 1.65, 0.1)
    # Sauter's nu_e* of a medium-size tokamak core: well below one
    assert 0.01 < nu < 1.0
    assert l31_coefficient(1.5, 0.4, nu) < l31_coefficient(1.5, 0.4)
    assert l31_coefficient(1.5, 0.4, 1e3) < 0.05
