"""The fast-ion distribution f(rho, E, xi) (`freeofcharge.distribution`)."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import ParticleArrays
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from scipy.integrate import quad

import freeofcharge as foc
from freeofcharge import orbits
from freeofcharge.distribution import (
    distribution_at,
    distribution_moments,
    energy_spectrum,
    legendre_bin_integrals,
    legendre_polynomials,
    orbit_distribution,
    pitch_distribution,
    slowing_down_distribution,
)
from freeofcharge.equilibrium import FluxSurfaces
from freeofcharge.slowing_down import (
    critical_velocity,
    pitch_angle_exponent,
    pitch_at_speed,
    slowing_down_time,
)

E = ELEMENTARY_CHARGE


class _Births:
    """The parts of a `Deposition` that `slowing_down_distribution` reads."""

    def __init__(self, energy, pitch, rho, rho_edges, weight=1.0, mass_number=2):
        mass = mass_number * ATOMIC_MASS
        speed = np.sqrt(2 * np.asarray(energy, dtype=float) * E / mass)
        n = speed.size
        velocities = np.zeros((n, 3))
        velocities[:, 0] = speed
        self.births = ParticleArrays(np.zeros((n, 3)), velocities, np.full(n, weight))
        self.birth_rho = np.broadcast_to(np.asarray(rho, dtype=float), (n,))
        self.birth_pitch = np.broadcast_to(np.asarray(pitch, dtype=float), (n,))
        self.mass_number = mass_number
        self.rho_edges = np.asarray(rho_edges, dtype=float)


class _OneRank:
    """A communicator stand-in that claims two ranks but sums nothing."""

    def Get_size(self) -> int:  # noqa: N802
        return 2

    def allreduce(self, value, op=None):
        return value

    def Allreduce(self, send, recv, op=None) -> None:  # noqa: N802
        return None


def _shell(volume=1.0):
    one = np.ones(1)
    return FluxSurfaces(np.array([0.0, 1.0]), volume * one, one, 0.1 * one)


@pytest.fixture(scope="module")
def tokamak_deposition():
    """An ITER-like deuterium beam at 1 MeV, its surfaces and plasma."""
    equilibrium = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
    plasma = foc.Plasma.parabolic(
        1e20, 2e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)}
    )
    rho_edges = np.linspace(0.0, 1.0, 9)
    surfaces = equilibrium.flux_surfaces(rho_edges)
    origin, direction = foc.tangential_injection(5.3, 12.0)
    beam = foc.NeutralBeam.gaussian(
        origin, direction, 1.0e6, 16.5e6, width=0.3, markers=300, seed=1
    )
    deposition = foc.deposit(
        beam, equilibrium, plasma, step=0.05, rho_edges=rho_edges, seed=2
    )
    return deposition, plasma, surfaces


# ----------------------------------------------------------------------
# Legendre polynomials
# ----------------------------------------------------------------------


def test_legendre_polynomials_and_bin_integrals():
    x = np.linspace(-1, 1, 7)
    p = legendre_polynomials(x, 6)
    for degree in range(6):
        assert np.allclose(
            p[:, degree], np.polynomial.legendre.Legendre.basis(degree)(x)
        )
    assert legendre_polynomials(x, 1).shape == (7, 1)
    edges = np.linspace(-1, 1, 5)
    integrals = legendre_bin_integrals(edges, 5)
    for degree in range(5):
        exact = [
            quad(np.polynomial.legendre.Legendre.basis(degree), a, b)[0]
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
        assert np.allclose(integrals[:, degree], exact, atol=1e-14)
    # only P_0 has a net integral over [-1, 1]
    assert np.allclose(integrals.sum(axis=0), [2, 0, 0, 0, 0], atol=1e-14)


# ----------------------------------------------------------------------
# The local slowing-down distribution
# ----------------------------------------------------------------------


def test_classical_slowing_down_and_gaffey_pitch():
    """One Coulomb logarithm: the 1/(v^3 + v_c^3) spectrum and Gaffey's pitch decay."""
    plasma = foc.Plasma.parabolic(
        5e19, 5e3, ions={"D": (2, 1, 0.9), "C": (12, 6, 1 / 60)}
    )
    births = _Births(100e3, 0.6, 0.0, [0.0, 1.0], weight=1e20)
    ln = 17.0
    volume = 3.0
    ds = slowing_down_distribution(
        births,
        plasma,
        _shell(volume),
        energy_bins=50,
        pitch_bins=400,
        legendre_modes=24,
        ln_lambda=ln,
    )
    assert ds.f.attrs["units"] == "1/(m^3 eV)"
    assert ds.f.dims == ("rho", "energy", "pitch")
    ne, te = 5e19, 5e3
    densities = [0.9 * ne, ne / 60]
    masses = [2 * ATOMIC_MASS, 12 * ATOMIC_MASS]
    mass = 2 * ATOMIC_MASS
    tau = float(slowing_down_time(mass, 1, ne, te, ln))
    v_c = float(critical_velocity(ne, te, densities, masses, [1, 6]))
    beta = float(pitch_angle_exponent(mass, ne, densities, masses, [1, 6]))
    # Stix: ions per bin = w tau_s / 3 ln((v_hi^3 + v_c^3) / (v_lo^3 + v_c^3))
    v = np.sqrt(2 * ds.energy_edge.values * E / mass)
    exact = 1e20 * tau / 3 * np.log((v[1:] ** 3 + v_c**3) / (v[:-1] ** 3 + v_c**3))
    spectrum = energy_spectrum(ds).isel(rho=0).values
    assert np.allclose(
        spectrum * volume * np.diff(ds.energy_edge.values), exact, rtol=1e-6
    )
    # the pitch moments of each energy bin, time averages of Gaffey's decay of
    # P_1 and P_2, exponents beta / 3 and beta
    f = ds.f.isel(rho=0).values
    xi = ds.pitch.values
    density = f.sum(axis=1)
    first = (f @ xi) / density
    second = (f @ (1.5 * xi**2 - 0.5)) / density
    birth = v[-1] / v_c

    def average(power, lo, hi):
        def decay(u):
            return float(pitch_at_speed(u, birth, 1.0, beta)) ** power

        def dt(u):
            return u**2 / (u**3 + 1)

        return quad(lambda u: decay(u) * dt(u), lo, hi)[0] / quad(dt, lo, hi)[0]

    u = v / v_c
    expected = [average(1, a, b) for a, b in zip(u[:-1], u[1:], strict=True)]
    assert np.allclose(first, 0.6 * np.array(expected), atol=1e-3)
    expected = [average(3, a, b) for a, b in zip(u[:-1], u[1:], strict=True)]
    assert np.allclose(second, (1.5 * 0.36 - 0.5) * np.array(expected), atol=1e-3)
    # nearly isotropic far below the birth energy
    assert abs(first[0]) < 0.1 * first[-1]


def test_moments_reproduce_slow_down(tokamak_deposition):
    deposition, plasma, surfaces = tokamak_deposition
    local = foc.slow_down(deposition, plasma, surfaces, orbit_average=False)
    ds = slowing_down_distribution(deposition, plasma, surfaces, energy_bins=96)
    moments = distribution_moments(ds)
    density = moments.fast_ion_density.values
    assert density == pytest.approx(local.fast_ion_density, rel=1e-3)
    pressure = moments.fast_ion_pressure.values
    assert pressure == pytest.approx(local.fast_ion_pressure, rel=1e-3)
    current = moments.fast_ion_current_density.values
    assert current == pytest.approx(local.fast_ion_current_density, rel=5e-3)
    # the pressures split, p = (p_par + 2 p_perp) / 3: a tangential beam is
    # parallel at the centre, where it is born along the field
    split = moments.parallel_pressure + 2 * moments.perpendicular_pressure
    assert split.values / 3 == pytest.approx(pressure, rel=1e-12)
    assert moments.anisotropy.values[0] < 0.3
    assert moments.fast_ion_pressure.attrs["units"] == "Pa"
    # the helpers
    spectrum = energy_spectrum(ds)
    assert spectrum.dims == ("rho", "energy")
    assert (spectrum * np.diff(ds.energy_edge.values)).sum("energy").values == (
        pytest.approx(density)
    )
    pitch = pitch_distribution(ds)
    assert (pitch * np.diff(ds.pitch_edge.values)).sum("pitch").values == (
        pytest.approx(density)
    )
    at = distribution_at(ds, 0.3)
    assert at.dims == ("energy", "pitch")
    assert float(at.rho) == pytest.approx(float(ds.rho.sel(rho=0.3, method="nearest")))


def test_end_energy_and_ranks(tokamak_deposition):
    """A thermal cutoff empties the low bins; a communicator gives the same result."""
    deposition, plasma, surfaces = tokamak_deposition
    ds = slowing_down_distribution(
        deposition, plasma, surfaces, minimum_energy=100e3, energy_bins=60
    )
    spectrum = energy_spectrum(ds).values
    assert np.all(spectrum[:, ds.energy.values < 100e3 - 1e4] == 0)
    local = foc.slow_down(
        deposition, plasma, surfaces, orbit_average=False, minimum_energy=100e3
    )
    moments = distribution_moments(ds)
    assert moments.fast_ion_density.values == pytest.approx(
        local.fast_ion_density, rel=1e-3
    )
    ranks = slowing_down_distribution(
        deposition,
        plasma,
        surfaces,
        minimum_energy=100e3,
        energy_bins=60,
        comm=_OneRank(),
    )
    assert np.allclose(ranks.f.values, ds.f.values)


def test_shells_must_match(tokamak_deposition):
    deposition, plasma, _ = tokamak_deposition
    with pytest.raises(ValueError, match="shells"):
        slowing_down_distribution(deposition, plasma, _shell())


# ----------------------------------------------------------------------
# The orbit histogram
# ----------------------------------------------------------------------


def test_orbit_histogram_matches_the_local_distribution():
    """High field, large aspect ratio: narrow orbits and little mirror force."""
    eq = foc.Equilibrium.solovev(10.0, 0.6, 1.5, 10.0, 1.2, wall_psi_n=None)
    rho_edges = np.linspace(0, 1, 6)
    surfaces = eq.flux_surfaces(rho_edges)
    field = orbits.orbit_field(eq, (96, 4, 96))
    plasma = foc.Plasma.parabolic(1e20, 600.0, ions={"H": (1, 1, 1.0)})
    origin, direction = foc.tangential_injection(9.9, 13.0)
    beam = foc.NeutralBeam.gaussian(
        origin, direction, 30e3, 1e6, mass_number=1, width=0.1, markers=150, seed=1
    )
    deposition = foc.deposit(beam, eq, plasma, step=0.01, rho_edges=rho_edges, seed=2)
    energy_edges = np.linspace(0, 30e3, 16)
    pitch_edges = np.linspace(-1, 1, 11)
    local = slowing_down_distribution(
        deposition,
        plasma,
        surfaces,
        energy_edges=energy_edges,
        pitch_bins=10,
        thermal_factor=2.0,
    )
    result = orbits.follow_orbits(
        deposition,
        field,
        plasma,
        surfaces=surfaces,
        step=0.2,
        thermal_factor=2.0,
        energy_diffusion=False,
        seed=1,
        histogram=(energy_edges, pitch_edges),
    )
    # the histogram holds all the time of the moments
    assert result.histogram.sum(axis=(1, 2)) == pytest.approx(
        result.particles, rel=1e-12
    )
    traced = orbit_distribution(result, mass_number=1)
    assert traced.f.dims == local.f.dims
    a, b = distribution_moments(local), distribution_moments(traced)
    volume = surfaces.volume
    for name in ("fast_ion_density", "fast_ion_pressure", "fast_ion_current_density"):
        total_a = float(np.sum(a[name].values * volume))
        total_b = float(np.sum(b[name].values * volume))
        assert total_b == pytest.approx(total_a, rel=0.03), name
    weight = a.fast_ion_density.values * volume
    mean_a = np.nansum(a.anisotropy.values * weight) / weight.sum()
    mean_b = np.nansum(b.anisotropy.values * weight) / weight.sum()
    assert mean_b == pytest.approx(mean_a, rel=0.05)
    # the energy spectra of the shells with most ions
    for k in np.argsort(weight)[-2:]:
        ratio = energy_spectrum(traced).isel(rho=k) / energy_spectrum(local).isel(rho=k)
        assert np.nanmax(np.abs(ratio.values[1:] - 1)) < 0.1


def test_collisionless_histogram_and_errors():
    eq = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 3.0, 1.2, wall_psi_n=None)
    field = orbits.orbit_field(eq, (48, 4, 48))
    rho_edges = np.linspace(0, 1, 4)
    positions = np.array([[3.3, 0.0, 0.0], [3.2, 0.0, 0.1]])
    b = np.asarray(field.field_at(positions))
    unit = b / np.linalg.norm(b, axis=1)[:, None]
    speed = np.sqrt(2 * 25e3 * E / ATOMIC_MASS)
    births = _Births([25e3, 25e3], 1.0, 0.5, rho_edges, mass_number=1)
    births.births = ParticleArrays(positions, speed * unit, np.ones(2))
    edges = np.linspace(0, 40e3, 5)
    result = orbits.follow_orbits(
        births, field, None, max_time=2e-4, histogram=(edges, np.linspace(-1, 1, 5))
    )
    # without collisions each marker stays in its energy bin
    assert result.histogram[:, 2].sum() == pytest.approx(result.particles.sum())
    assert result.histogram.sum() == pytest.approx(4e-4)
    traced = orbit_distribution(result, mass_number=1)
    assert np.all(np.isnan(traced.f.values))  # no surfaces: no volumes
    plain = orbits.follow_orbits(births, field, None, max_time=2e-5)
    assert plain.histogram is None
    with pytest.raises(ValueError, match="histogram"):
        orbit_distribution(plain, mass_number=1)
