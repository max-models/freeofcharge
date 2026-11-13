"""Charge-exchange losses of the fast ions along their slowing down."""

import math

import numpy as np
import pytest
from maybempi import SerialComm
from plasmacoll.charge_exchange import proton_hydrogen_charge_exchange
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from scipy.integrate import solve_ivp

import freeofcharge.charge_exchange as cx
from freeofcharge import (
    Equilibrium,
    NeutralBeam,
    Plasma,
    deposit,
    slow_down,
    tangential_injection,
)
from freeofcharge.charge_exchange import (
    charge_exchange_moments,
    charge_exchange_rate,
    exponential_neutral_density,
    slow_down_with_charge_exchange,
)
from freeofcharge.slowing_down import collision_coefficients, slowing_down_moments

R0, A = 3.0, 0.9
EDGES = np.linspace(0.0, 1.0, 11)
MASS = 2 * ATOMIC_MASS


@pytest.fixture(scope="module")
def setup():
    equilibrium = Equilibrium.solovev(R0, A, 1.6, 2.5, 1.2, wall_psi_n=1.3)
    plasma = Plasma.parabolic(1e19, 2e3, ions={"D": (2, 1, 1.0)})
    origin, direction = tangential_injection(R0, R0 + 3 * A)
    beam = NeutralBeam.gaussian(
        origin, direction, 60e3, 2e6, width=0.1, divergence=0.01, markers=200, seed=7
    )
    deposition = deposit(beam, equilibrium, plasma, step=5e-3, seed=8, rho_edges=EDGES)
    surfaces = equilibrium.flux_surfaces(EDGES)
    return deposition, plasma, surfaces


def test_janev_cross_section_at_10_kev_per_nucleon():
    # Janev and Smith (1993): 3.2345 ln(235.88/10 + 2.3713) / (1 + 0.38371 + ...) = 7.55e-16 cm^2
    speed = math.sqrt(2 * 10e3 * ELEMENTARY_CHARGE / ATOMIC_MASS)
    rate = float(charge_exchange_rate(speed, 1e16))
    assert rate / (1e16 * speed) == pytest.approx(7.546e-20, rel=1e-3)
    assert rate == pytest.approx(1e16 * proton_hydrogen_charge_exchange(1e4) * speed)
    # a 20 keV deuteron has the speed of a 10 keV proton
    deuteron = math.sqrt(2 * 20e3 * ELEMENTARY_CHARGE / MASS)
    assert deuteron == pytest.approx(speed)


def test_exponential_neutral_density():
    density = exponential_neutral_density(1e17, 0.1)
    assert float(density(1.0)) == pytest.approx(1e17)
    assert float(density(0.9)) == pytest.approx(1e17 / math.e)
    assert float(density(1.2)) == pytest.approx(1e17)  # outside: the edge value
    with pytest.raises(ValueError, match="decay_length"):
        exponential_neutral_density(1e17, 0.0)


def test_constant_rate_without_ion_drag_matches_the_closed_form():
    # dv/dt = -a_e v, nu constant: S = (v/v0)^(nu/a_e),
    # P_cx / E_0 = nu / (nu + 2 a_e) [1 - (v/v0)^(2 + nu/a_e)]
    a_e, nu, v0, v1 = 2.0, 3.0, 1.0e6, 1.0e5
    speed = np.geomspace(v0, v1, 401)[None, :]
    time, kinetic, ion_loss, parallel, electrons, lost, survival = (
        charge_exchange_moments(speed, a_e, 0.0, 0.0, np.zeros((1, 1)), nu)
    )
    ratio = v1 / v0
    assert survival[0, -1] == pytest.approx(ratio ** (nu / a_e), rel=1e-12)
    expected = nu / (nu + 2 * a_e) * (1 - ratio ** (2 + nu / a_e))
    assert lost[0] / (0.5 * v0**2) == pytest.approx(expected, rel=1e-6)
    # the time and the survival together: int S dt = (1 - S_end) / nu
    assert time[0] == pytest.approx((1 - survival[0, -1]) / nu, rel=1e-6)
    # energy: E_0 = P_e + P_cx + S_end E_end
    balance = electrons[0] + lost[0] + survival[0, -1] * 0.5 * v1**2
    assert balance == pytest.approx(0.5 * v0**2, rel=1e-6)
    assert ion_loss[0] == 0.0


def test_without_charge_exchange_the_moments_are_those_of_slowing_down():
    speed = np.geomspace(3.0, 0.5, 33)[None, :]
    rates = np.array([[1.0, 3.0]])
    ref = slowing_down_moments(speed, 1.0, 2.0, 0.5, rates)
    got = charge_exchange_moments(speed, 1.0, 2.0, 0.5, rates, 0.0)
    for a, b in zip(ref, got[:4], strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-14)
    assert got[6][0, -1] == 1.0
    with pytest.raises(ValueError, match="odd"):
        charge_exchange_moments(speed[:, :4], 1.0, 2.0, 0.5, rates, 0.0)


def test_survival_matches_the_equations_of_motion():
    # a 60 keV deuteron in 2 keV plasma with neutrals: integrate v, ln S and the
    # lost energy in time and compare with the speed-grid moments
    ne, te, n0 = 1e19, 2e3, 1e14

    def coefficients(v):
        return collision_coefficients(v, MASS, 1, ne, te, [ne], [MASS], [1], te)

    def rhs(t, y):
        v = y[0]
        a_e, a_i, _ = coefficients(v)
        nu = float(charge_exchange_rate(v, n0))
        survival = math.exp(-y[1])
        return [-(a_e * v + a_i / v**2), nu, survival * nu * 0.5 * MASS * v**2]

    v0 = math.sqrt(2 * 60e3 * ELEMENTARY_CHARGE / MASS)
    v1 = 0.05 * v0

    def reach(t, y):
        return y[0] - v1

    reach.terminal = True
    solution = solve_ivp(
        rhs, (0, 10), [v0, 0.0, 0.0], events=reach, rtol=1e-11, atol=1e-14
    )
    speed = np.geomspace(v0, v1, 201)[None, :]
    a_e, a_i, nu_d = coefficients(speed)
    rate = charge_exchange_rate(speed, n0)
    _, _, _, _, _, lost, survival = charge_exchange_moments(
        speed, a_e, a_i, nu_d, np.zeros((1, 1)), rate
    )
    assert survival[0, -1] == pytest.approx(math.exp(-solution.y[1, -1]), rel=1e-7)
    assert MASS * lost[0] == pytest.approx(solution.y[2, -1], rel=1e-7)
    assert 0.1 < survival[0, -1] < 0.9  # a test that sees the losses


def test_no_neutrals_reproduces_slow_down(setup):
    deposition, plasma, surfaces = setup
    for kwargs in ({}, {"orbit_average": False, "thermal_factor": 1.5}):
        ref = slow_down(deposition, plasma, surfaces, **kwargs)
        got = slow_down_with_charge_exchange(
            deposition, plasma, surfaces, 0.0, **kwargs
        )
        for field in (
            "electron_heating",
            "ion_heating",
            "fast_ion_density",
            "fast_ion_pressure",
            "fast_ion_current_density",
            "current_density",
        ):
            np.testing.assert_allclose(
                getattr(got.slowing_down, field),
                getattr(ref, field),
                rtol=1e-12,
                atol=0,
            )
        for field in (
            "driven_current",
            "stored_energy",
            "thermalised_power",
            "birth_power",
        ):
            assert getattr(got.slowing_down, field) == pytest.approx(
                getattr(ref, field), rel=1e-12
            )
        assert got.cx_power == 0.0
        assert np.all(got.fast_neutral_spectrum == 0.0)


def test_power_balance_and_the_explicit_electron_heating(setup):
    deposition, plasma, surfaces = setup
    neutrals = exponential_neutral_density(1e17, 0.1)
    result = slow_down_with_charge_exchange(
        deposition, plasma, surfaces, neutrals, thermal_factor=1.5
    )
    assert abs(result.power_balance) < 1e-12
    assert 0.1 < result.lost_fraction < 0.9
    # the electron power as the rest of the balance against its direct integral,
    # on the grid the function chooses (cut and refined where S falls steeply)
    rho = np.minimum(deposition.birth_rho, 1.0)
    speed = np.linalg.norm(deposition.births.velocities, axis=1)
    energy = 0.5 * MASS * speed**2
    ne, te = plasma.electron_density(rho), plasma.electron_temperature(rho)
    ti = plasma.ion_temperature(rho)
    end = np.minimum(1.5 * ti * ELEMENTARY_CHARGE, energy)
    ions = plasma.ion_densities(rho)

    def coefficients(v):
        return collision_coefficients(
            v,
            MASS,
            1,
            ne[:, None],
            te[:, None],
            [ions[0][:, None]],
            [MASS],
            [1],
            ti[:, None],
        )

    speeds, drags, rate, truncated = cx._speed_grid(
        speed,
        np.log(np.sqrt(end / energy)),
        201,
        coefficients,
        lambda v: charge_exchange_rate(v, neutrals(rho)[:, None]),
    )
    assert np.any(truncated) and speeds.shape[1] > 201
    _, _, ion_loss, _, electrons, lost, survival = charge_exchange_moments(
        speeds, *drags, np.zeros((len(speed), 1)), rate
    )
    final = np.where(truncated, 0.5 * MASS * speeds[:, -1] ** 2, end)
    rest = energy - survival[:, -1] * final - MASS * (ion_loss + lost)
    w = deposition.births.weights
    assert np.sum(w * MASS * electrons) == pytest.approx(np.sum(w * rest), rel=1e-6)
    heating = result.slowing_down.electron_heating.sum()
    assert heating == pytest.approx(np.sum(w * rest), rel=1e-9)


def test_losses_grow_with_the_neutral_density(setup):
    deposition, plasma, surfaces = setup
    ref = slow_down(deposition, plasma, surfaces)
    lost, current, energy = [], [], []
    for n0 in (1e15, 1e16, 1e17):
        result = slow_down_with_charge_exchange(
            deposition, plasma, surfaces, exponential_neutral_density(n0, 0.15)
        )
        lost.append(result.lost_fraction)
        current.append(result.slowing_down.driven_current)
        energy.append(result.slowing_down.stored_energy)
    assert np.all(np.diff(lost) > 0)
    assert np.all(np.diff(current) < 0) and current[0] < ref.driven_current
    assert np.all(np.diff(energy) < 0) and energy[0] < ref.stored_energy


def test_neutral_density_as_a_number_per_shell_or_a_function(setup):
    deposition, plasma, surfaces = setup
    centres = 0.5 * (EDGES[1:] + EDGES[:-1])
    per_shell = 1e16 * (1 + centres)
    by_shell = slow_down_with_charge_exchange(deposition, plasma, surfaces, per_shell)
    np.testing.assert_allclose(by_shell.neutral_density, per_shell)
    uniform = slow_down_with_charge_exchange(deposition, plasma, surfaces, 1e16)
    np.testing.assert_allclose(uniform.neutral_density, 1e16)
    function = slow_down_with_charge_exchange(
        deposition, plasma, surfaces, lambda rho: np.full_like(rho, 1e16)
    )
    np.testing.assert_allclose(function.cx_loss, uniform.cx_loss, rtol=1e-12)
    assert uniform.cx_power < by_shell.cx_power
    with pytest.raises(ValueError, match="one value per shell"):
        slow_down_with_charge_exchange(deposition, plasma, surfaces, np.ones(3))


def test_fast_neutral_spectrum_and_dataset(setup):
    deposition, plasma, surfaces = setup
    edges = np.linspace(0.0, 70e3, 71)
    result = slow_down_with_charge_exchange(
        deposition, plasma, surfaces, 1e16, spectrum_edges=edges, comm=SerialComm()
    )
    neutrals_per_second = np.sum(result.fast_neutral_spectrum * np.diff(edges))
    assert neutrals_per_second == pytest.approx(result.cx_rate.sum(), rel=1e-12)
    # no fast neutral above the birth energy (60 keV, a narrow spread)
    above = result.fast_neutral_spectrum[:, edges[:-1] >= 61e3]
    assert np.all(above == 0.0)
    data = result.to_dataset(surfaces)
    assert data.cx_loss.attrs["units"] == "W"
    assert data.fast_neutral_spectrum.attrs["units"] == "1/(s eV)"
    assert data.energy.size == 70
    np.testing.assert_allclose(data.cx_loss_density, result.cx_loss / surfaces.volume)
    assert data.attrs["cx_power"] == pytest.approx(result.cx_power)
    assert np.all((0 <= data.cx_probability) & (data.cx_probability <= 1))
    assert "fast_neutral_spectrum_by_shell" in data
    assert "cx_loss_density" not in result.to_dataset()


def test_cross_section_argument_and_default_bins(setup):
    deposition, plasma, surfaces = setup
    # a cross section falling as 1/v: a constant rate
    result = slow_down_with_charge_exchange(
        deposition,
        plasma,
        surfaces,
        1e16,
        cross_section=lambda e: 1e-19 / np.sqrt(e),
        orbit_average=False,
    )
    assert result.spectrum_edges.size == 101
    assert result.spectrum_edges[-1] == pytest.approx(60e3, rel=0.05)
    assert result.lost_fraction > 0.0


def test_unresolved_grid_warns(setup, monkeypatch):
    deposition, plasma, surfaces = setup
    monkeypatch.setattr(cx, "_MAX_POINTS", 21)
    with pytest.warns(UserWarning, match="does not resolve"):
        slow_down_with_charge_exchange(
            deposition, plasma, surfaces, 1e18, speed_points=21
        )


def test_mismatched_inputs(setup):
    deposition, plasma, surfaces = setup
    other = Equilibrium.solovev(R0, A, 1.6, 2.5, 1.2).flux_surfaces(
        np.linspace(0, 1, 5)
    )
    with pytest.raises(ValueError, match="shells"):
        slow_down_with_charge_exchange(deposition, plasma, other, 0.0)


def test_helical_shells_in_3d():
    from freeofcharge import GriddedEquilibrium
    from freeofcharge.gridded_equilibrium import _HelicalTestField

    helical = _HelicalTestField()
    edges = np.linspace(0.0, 1.0, 6)
    surfaces = helical.flux_surfaces(edges, field_points=(16, 8))
    plasma = Plasma.parabolic(2e19, 2e3, ions={"H": (1, 1, 1.0)})
    beam = NeutralBeam.gaussian(
        [4.0, -2.5, 0.0], [-0.55, 1.0, 0.0], 55e3, 1.8e6, mass_number=1,
        width=0.05, divergence=0.01, markers=100, seed=1,
    )  # fmt: skip
    deposition = deposit(
        beam, GriddedEquilibrium(helical, (32, 24, 32)), plasma, step=0.01,
        rho_edges=edges, seed=2,
    )  # fmt: skip
    neutrals = exponential_neutral_density(1e16, 0.2)
    with pytest.raises(ValueError, match="orbit_average=False"):
        slow_down_with_charge_exchange(deposition, plasma, surfaces, neutrals)
    with pytest.warns(UserWarning, match="axisymmetry"):
        result = slow_down_with_charge_exchange(
            deposition, plasma, surfaces, neutrals, orbit_average=False
        )
    assert abs(result.power_balance) < 1e-12
    assert result.cx_power > 0.0
    assert np.isnan(result.slowing_down.driven_current)
