import numpy as np
import pytest
from scipy import constants
from scipy.integrate import quad, solve_ivp

from freeofcharge.slowing_down import (
    coulomb_logarithm,
    critical_velocity,
    ion_heating_fraction,
    pitch_angle_exponent,
    pitch_at_speed,
    slowing_down_integrals,
    slowing_down_time,
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
