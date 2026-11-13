import dataclasses

import numpy as np
import pytest
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

from freeofcharge import (
    Equilibrium,
    NeutralBeam,
    Plasma,
    deposit,
    slow_down,
    tangential_injection,
)
from freeofcharge.slowing_down import (
    critical_velocity,
    pitch_angle_exponent,
    pitch_at_speed,
    slowing_down_integrals,
    slowing_down_time,
    start_cordey_trapped_electron_correction,
)
from freeofcharge.time_dependent import (
    slowing_down_response,
    time_dependent_slow_down,
)

EDGES = np.linspace(0.0, 1.0, 9)
LN_LAMBDA = 17.0


@pytest.fixture(scope="module")
def setup():
    equilibrium = Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
    plasma = Plasma.parabolic(1e20, 2e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)})
    surfaces = equilibrium.flux_surfaces(EDGES)
    origin, direction = tangential_injection(5.3, 12.0, sense=1)
    beam = NeutralBeam.gaussian(
        origin,
        direction,
        1.0e6,
        16.5e6,
        width=0.3,
        divergence=0.005,
        markers=150,
        seed=1,
    )
    deposition = deposit(beam, equilibrium, plasma, step=0.03, rho_edges=EDGES, seed=2)
    return deposition, plasma, surfaces


@pytest.fixture(scope="module")
def response(setup):
    return slowing_down_response(*setup, thermal_factor=3.0)


def test_the_steady_state_is_that_of_slow_down(setup, response):
    deposition, plasma, surfaces = setup
    steady = slow_down(deposition, plasma, surfaces, thermal_factor=3.0)
    count, energy, to_e, to_i, parallel, thermalised = response.steady_state
    np.testing.assert_allclose(to_e, steady.electron_heating, rtol=1e-6)
    np.testing.assert_allclose(to_i, steady.ion_heating, rtol=1e-6)
    np.testing.assert_allclose(
        count / surfaces.volume, steady.fast_ion_density, rtol=1e-6
    )
    np.testing.assert_allclose(
        ELEMENTARY_CHARGE * parallel / surfaces.volume,
        steady.fast_ion_current_density,
        rtol=1e-6,
        atol=1e-9 * np.abs(steady.fast_ion_current_density).max(),
    )
    assert energy.sum() == pytest.approx(steady.stored_energy, rel=1e-6)
    assert thermalised.sum() == pytest.approx(steady.thermalised_power, rel=1e-6)
    assert response.birth_power == pytest.approx(steady.birth_power, rel=1e-12)


def test_switch_on_approaches_the_steady_state(setup, response):
    deposition, plasma, surfaces = setup
    steady = slow_down(deposition, plasma, surfaces, thermal_factor=3.0)
    time = np.linspace(0.0, 1.1 * response.age[-1], 801)
    data = response.evolve(time, response.birth_power)
    assert abs(float(data.stored_energy[0])) < 1e-9 * steady.stored_energy
    late = data.isel(time=-1)
    assert float(late.stored_energy) == pytest.approx(steady.stored_energy, rel=5e-3)
    assert float(late.driven_current) == pytest.approx(steady.driven_current, rel=5e-3)
    heating = late.heating.sum("rho").values
    np.testing.assert_allclose(
        heating, [steady.electron_heating.sum(), steady.ion_heating.sum()], rtol=5e-3
    )
    np.testing.assert_allclose(
        late.fast_ion_density, steady.fast_ion_density, rtol=5e-3
    )
    # the fast-ion content rises monotonically
    assert np.all(np.diff(data.stored_energy.values) >= -1e-9 * steady.stored_energy)
    assert data.heating.attrs["units"] == "W"


def test_switch_off_follows_the_classical_slowing_down(setup):
    deposition, plasma, surfaces = setup
    response = slowing_down_response(
        deposition, plasma, surfaces, ln_lambda=LN_LAMBDA, orbit_average=False
    )
    time = np.linspace(0.0, 0.6, 61)
    power = response.birth_power
    data = response.evolve(time, 0.0, initial_power=power)
    # each birth slows down along v^3 = (v0^3 + vc^3) exp(-3t/tau_s) - vc^3
    mass = deposition.mass_number * ATOMIC_MASS
    rho = np.minimum(deposition.birth_rho, 1.0)
    ne, te = plasma.electron_density(rho), plasma.electron_temperature(rho)
    ions = plasma.ion_densities(rho)
    args = (ions, plasma.ion_masses, plasma.ion_charges)
    v_c = critical_velocity(ne, te, *args)
    tau_s = slowing_down_time(mass, 1, ne, te, LN_LAMBDA)
    beta = pitch_angle_exponent(mass, ne, *args)
    v0 = np.linalg.norm(deposition.births.velocities, axis=1)
    w = deposition.births.weights
    xi0 = deposition.birth_pitch
    energy, parallel = [], []
    for t in time:
        cube = (v0**3 + v_c**3) * np.exp(-3.0 * t / tau_s) - v_c**3
        v = np.cbrt(np.maximum(cube, 1e-30 * v_c**3))
        xi = pitch_at_speed(v / v_c, v0 / v_c, xi0, beta)
        _, kinetic, par = slowing_down_integrals(v, xi, v_c, tau_s, beta)
        energy.append(np.sum(w * 0.5 * mass * kinetic))
        parallel.append(np.sum(w * par))
    np.testing.assert_allclose(data.stored_energy, energy, rtol=2e-3)
    current = np.sum(data.fast_ion_current_density * surfaces.volume, axis=-1)
    np.testing.assert_allclose(
        current, ELEMENTARY_CHARGE * np.asarray(parallel), rtol=2e-3
    )
    # the thermalisation time per shell is tau_s/3 ln(1 + (v0/vc)^3)
    lifetime = tau_s / 3.0 * np.log1p((v0 / v_c) ** 3)
    shell = np.clip(np.searchsorted(EDGES, rho, side="right") - 1, 0, len(EDGES) - 2)
    rate = np.bincount(shell, w, len(EDGES) - 1)
    expected = np.bincount(shell, w * lifetime, len(EDGES) - 1) / rate
    np.testing.assert_allclose(response.thermalisation_time, expected, rtol=1e-3)
    expected_tau = np.bincount(shell, w * tau_s, len(EDGES) - 1) / rate
    np.testing.assert_allclose(response.slowing_down_time, expected_tau, rtol=1e-10)


def test_energy_is_conserved_through_a_blip(response):
    dt = 1e-3
    time = np.arange(0.0, 1.5 * response.age[-1], dt)
    power = response.birth_power

    def blip(t):
        return np.where((t >= 0.1) & (t < 0.11), power, 0.0)

    data = response.evolve(time, blip)
    injected = np.sum(data.birth_power.values) * dt
    assert injected == pytest.approx(0.01 * power, rel=1e-9)
    out = data.total_heating.sum("species") + data.thermalised_power.sum("rho")
    assert float(out.sum()) * dt == pytest.approx(injected, rel=1e-6)
    assert float(data.stored_energy[-1]) == pytest.approx(0.0, abs=1e-9 * injected)
    # and at every time: W(t) = int (P_birth - P_e - P_i - P_th) dt, to the
    # trapezoidal rule of the heating over the 1 ms steps at the blip edges
    given = np.cumsum(data.birth_power.values) * dt
    lost = np.concatenate([[0.0], np.cumsum(0.5 * (out.values[1:] + out.values[:-1]))])
    balance = np.concatenate([[0.0], given[:-1]]) - lost * dt
    assert np.max(np.abs(data.stored_energy.values - balance)) < 3e-3 * injected
    assert float(data.thermalised_power.sum()) > 0.0


def test_the_response_is_linear_and_time_invariant(response):
    time = np.linspace(0.0, 0.8, 401)
    power = response.birth_power
    first = np.where(time < 0.1, power, 0.0)
    second = np.where((time >= 0.3) & (time < 0.4), 0.5 * power, 0.0)
    both = response.evolve(time, first + second)
    one = response.evolve(time, first)
    two = response.evolve(time, second)
    for name in ("stored_energy", "heating", "current_density", "fast_ion_density"):
        np.testing.assert_allclose(
            both[name],
            one[name] + two[name],
            rtol=1e-9,
            atol=1e-9 * abs(both[name]).max(),
        )
    # shifting the source by 0.3 s shifts the response
    shifted = response.evolve(time, np.roll(first, 150) * (time >= 0.3))
    np.testing.assert_allclose(
        shifted.stored_energy[150:],
        one.stored_energy[: time.size - 150],
        rtol=1e-9,
        atol=1e-9 * float(one.stored_energy.max()),
    )


def test_transfer_function_matches_a_modulated_source(response):
    frequency = 5.0
    dt = 2e-4
    time = np.arange(0.0, 4.0, dt)
    power = response.birth_power
    data = response.evolve(
        time,
        lambda t: power * (1.0 + 0.5 * np.sin(2 * np.pi * frequency * t)),
        initial_power=power,
    )
    late = time >= 2.0
    phase = np.exp(-2j * np.pi * frequency * time[late])
    modulation = 2.0 * np.mean(data.stored_energy.values[late] * phase)
    steady = response.steady_state[1].sum()
    # the source's modulation is 0.5 sin = 0.5 exp(i(wt - pi/2))
    measured = modulation / (0.5 * steady * np.exp(-0.5j * np.pi))
    gain = response.transfer_function([0.0, frequency])
    expected = complex(gain.stored_energy.sel(frequency=frequency))
    # the source is sampled in the middle of each step
    expected *= np.exp(-1j * np.pi * frequency * dt)
    assert abs(measured - expected) < 2e-3
    assert complex(gain.stored_energy.sel(frequency=0.0)) == pytest.approx(1.0)
    assert complex(gain.driven_current.sel(frequency=0.0)) == pytest.approx(1.0)
    # slower moments lag more: the energy lags the electron heating
    delay = -np.angle(gain.total_heating.sel(frequency=frequency))
    assert 0.0 < delay[0] < -np.angle(expected)
    assert abs(gain.stored_energy.sel(frequency=frequency)) < 1.0


def test_convenience_function_and_power_forms(setup, response):
    time = np.linspace(0.0, 0.2, 21)
    power = response.birth_power
    data = time_dependent_slow_down(*setup, time, power, thermal_factor=3.0)
    other = response.evolve(time, lambda t: power)
    np.testing.assert_allclose(data.stored_energy, other.stored_energy, rtol=1e-12)
    # an injected power as reference
    half = response.evolve(time, power, reference_power=2.0 * power)
    np.testing.assert_allclose(
        half.stored_energy, 0.5 * other.stored_energy, rtol=1e-12
    )
    assert half.attrs["reference_power"] == 2.0 * power


def test_bad_arguments_are_refused(setup, response):
    deposition, plasma, surfaces = setup
    with pytest.raises(ValueError, match="equally spaced"):
        response.evolve([0.0, 0.1, 0.3], 1.0)
    with pytest.raises(ValueError, match="at least two"):
        response.evolve([0.0], 1.0)
    with pytest.raises(ValueError, match="odd"):
        slowing_down_response(deposition, plasma, surfaces, speed_points=200)
    other = Equilibrium.solovev(6.2, 1.8, 1.7, 5.3).flux_surfaces(EDGES[::2])
    with pytest.raises(ValueError, match="different shells"):
        slowing_down_response(deposition, plasma, other)
    flat = dataclasses.replace(surfaces, axisymmetric=False)
    with pytest.raises(ValueError, match="orbit_average=False"):
        slowing_down_response(deposition, plasma, flat)


def test_3d_shells_have_no_driven_current_without_a_correction(setup):
    deposition, plasma, surfaces = setup
    flat = dataclasses.replace(surfaces, axisymmetric=False)
    time = np.linspace(0.0, 0.1, 11)
    response = slowing_down_response(deposition, plasma, flat, orbit_average=False)
    data = response.evolve(time, response.birth_power)
    assert np.all(np.isnan(data.driven_current))
    assert np.all(np.isnan(response.transfer_function(1.0).driven_current))
    corrected = slowing_down_response(
        deposition,
        plasma,
        flat,
        orbit_average=False,
        trapped_electron_correction=start_cordey_trapped_electron_correction,
    )
    assert np.all(np.isfinite(corrected.evolve(time, 1.0).driven_current))


class _TwoRanks:
    """A communicator of two ranks that holds all the births on this one."""

    def Get_size(self):
        return 2

    def allreduce(self, value, op=None):
        return value

    def Allreduce(self, send, receive, op=None):
        pass


def test_the_ages_are_agreed_over_the_ranks(setup, response):
    split = slowing_down_response(*setup, thermal_factor=3.0, comm=_TwoRanks())
    np.testing.assert_allclose(split.cumulative, response.cumulative)
