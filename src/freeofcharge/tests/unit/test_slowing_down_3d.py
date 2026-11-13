"""The local slowing down on the shells of a 3D equilibrium."""

from dataclasses import replace

import numpy as np
import pytest

from freeofcharge import (
    Equilibrium,
    GriddedEquilibrium,
    NeutralBeam,
    Plasma,
    deposit,
    monte_carlo_volumes,
    slow_down,
)
from freeofcharge.gridded_equilibrium import _HelicalTestField
from freeofcharge.slowing_down import (
    electron_collisionality,
    l31_coefficient,
    sauter_trapped_electron_correction,
    start_cordey_trapped_electron_correction,
)

EDGES = np.linspace(0.0, 1.0, 6)


@pytest.fixture(scope="module")
def helical():
    return _HelicalTestField()


@pytest.fixture(scope="module")
def gridded(helical):
    return GriddedEquilibrium(helical, (48, 32, 48))


@pytest.fixture(scope="module")
def surfaces(helical):
    return helical.flux_surfaces(EDGES, field_points=(32, 16))


@pytest.fixture(scope="module")
def plasma():
    return Plasma.parabolic(5e19, 3e3, ions={"H": (1, 1, 1.0)})


@pytest.fixture(scope="module")
def impure():
    """Z_eff = 1.3, so that the shielding leaves a current."""
    return Plasma.parabolic(5e19, 3e3, ions={"H": (1, 1, 0.94), "C": (12, 6, 0.01)})


@pytest.fixture(scope="module")
def deposition(gridded, plasma):
    beam = NeutralBeam.gaussian(
        [4.0, -2.5, 0.0],
        [-0.55, 1.0, 0.0],
        "55 keV",
        "1.8 MW",
        mass_number=1,
        width=0.05,
        divergence=0.01,
        markers=400,
        seed=1,
    )
    return deposit(beam, gridded, plasma, step=0.01, rho_edges=EDGES, seed=2)


def test_helical_shells_have_the_closed_form_volumes(helical, surfaces):
    exact = 2 * np.pi**2 * helical.r0 * helical.a**2 * np.diff(EDGES**2)
    np.testing.assert_allclose(surfaces.volume, exact, rtol=1e-10)
    assert not surfaces.axisymmetric
    assert surfaces.to_dataset().attrs["axisymmetric"] == 0
    # the Jacobian R0 a^2 rho is constant on a surface: equal weights
    np.testing.assert_allclose(surfaces.surface_weight, 1.0 / (32 * 16), rtol=1e-10)


def test_monte_carlo_volumes_agree_with_the_jacobian(gridded, surfaces):
    volume, error = monte_carlo_volumes(
        gridded, EDGES, samples=400_000, seed=3, chunk=150_000
    )
    assert np.all(np.abs(volume - surfaces.volume) < 4 * error + 0.01 * surfaces.volume)


def test_monte_carlo_volumes_without_one_evaluation_for_both():
    solovev = Equilibrium.solovev(3.0, 1.0, elongation=1.5, toroidal_field=2.0)
    volume, error = monte_carlo_volumes(solovev, EDGES, samples=100_000, seed=4)
    reference = solovev.flux_surfaces(EDGES).volume
    assert solovev.flux_surfaces(EDGES).axisymmetric
    assert np.all(np.abs(volume - reference) < 4 * error + 0.01 * reference)


def test_orbit_average_is_refused_in_3d(deposition, plasma, surfaces):
    with pytest.raises(ValueError, match="orbit_average=False"):
        slow_down(deposition, plasma, surfaces)


def test_local_slowing_down_in_3d(deposition, plasma, surfaces, gridded):
    assert 0.0 < deposition.shine_through_fraction < 0.9
    with pytest.warns(UserWarning, match="axisymmetry"):
        steady = slow_down(deposition, plasma, surfaces, orbit_average=False)
    # energy: the births' power goes to the electrons and the ions
    assert steady.birth_power == pytest.approx(deposition.absorbed_power, rel=1e-10)
    assert abs(steady.power_balance) < 1e-10
    heating = steady.electron_heating + steady.ion_heating
    assert heating.sum() == pytest.approx(deposition.absorbed_power, rel=1e-10)
    assert np.all(steady.electron_heating >= 0) and np.all(steady.ion_heating >= 0)
    # the current needs a trapped-electron correction, which is axisymmetric
    assert np.all(np.isnan(steady.current_density))
    assert np.isnan(steady.driven_current)
    assert np.all(np.isfinite(steady.fast_ion_current_density))
    # densities from the 3D shell volumes integrate back
    data = steady.to_dataset(surfaces)
    integral = (data.heating_density * surfaces.volume).sum("rho")
    np.testing.assert_allclose(integral.sum(), heating.sum(), rtol=1e-12)
    assert steady.stored_energy == pytest.approx(
        1.5 * float(np.sum(steady.fast_ion_pressure * surfaces.volume)), rel=1e-12
    )
    assert data.attrs["birth_power"] == steady.birth_power
    assert data.rho.attrs["radial_coordinate"] == "toroidal"
    # and in physical space: the deposited power density over Monte Carlo volumes
    volume, error = monte_carlo_volumes(gridded, EDGES, samples=400_000, seed=5)
    density = deposition.power_density(surfaces)
    assert np.sum(density * volume) == pytest.approx(
        deposition.absorbed_power, rel=0.03
    )


def test_thermalised_power_closes_the_balance(deposition, plasma, surfaces):
    with pytest.warns(UserWarning):
        steady = slow_down(
            deposition, plasma, surfaces, orbit_average=False, thermal_factor=3.0
        )
    assert steady.thermalised_power > 0.0
    assert abs(steady.power_balance) < 1e-10


def test_sauter_with_the_effective_trapped_fraction_in_3d(deposition, impure, surfaces):
    import warnings

    assert np.all((surfaces.trapped_fraction > 0) & (surfaces.trapped_fraction < 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steady = slow_down(
            deposition,
            impure,
            surfaces,
            orbit_average=False,
            trapped_electron_correction="sauter",
        )
        by_function = slow_down(
            deposition,
            impure,
            surfaces,
            orbit_average=False,
            trapped_electron_correction=sauter_trapped_electron_correction,
        )
    assert np.isfinite(steady.driven_current)
    assert steady.driven_current == by_function.driven_current
    # G = L31(f_t, nu*) of the 3D surfaces, with q = 1/|iota|
    g = sauter_trapped_electron_correction(surfaces, impure)
    assert np.all((g > 0) & (g < 1))
    zeff = impure.zeff(surfaces.rho)
    np.testing.assert_allclose(
        steady.current_density,
        steady.fast_ion_current_density * (1 - (1 - g) / zeff),
        rtol=1e-12,
    )
    # the collisionality with the mirror epsilon of |B|, not the geometric one
    ne = impure.electron_density(surfaces.rho)
    te = impure.electron_temperature(surfaces.rho)
    nu = electron_collisionality(
        ne,
        te,
        zeff,
        np.abs(surfaces.safety_factor),
        surfaces.major_radius,
        surfaces.field_epsilon,
    )
    np.testing.assert_allclose(
        g, l31_coefficient(zeff, surfaces.trapped_fraction, nu), rtol=1e-12
    )
    flipped = replace(surfaces, safety_factor=-surfaces.safety_factor)
    np.testing.assert_allclose(sauter_trapped_electron_correction(flipped, impure), g)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_no_trapped_fraction_is_no_correction(deposition, impure, surfaces):
    """|B| uniform on the surfaces: f_t = 0, so G = 0 and j = j_f (1 - 1/Z_eff).

    The mirror epsilon is zero too, so the collisionality is infinite (a warning).
    """
    uniform = replace(surfaces, field_strength=np.ones_like(surfaces.field_strength))
    np.testing.assert_allclose(uniform.trapped_fraction, 0.0, atol=1e-13)
    kwargs = {"orbit_average": False}
    sauter = slow_down(
        deposition, impure, uniform, trapped_electron_correction="sauter", **kwargs
    )
    none = slow_down(
        deposition, impure, uniform, trapped_electron_correction="none", **kwargs
    )
    np.testing.assert_allclose(sauter.current_density, none.current_density, rtol=1e-12)
    zeff = impure.zeff(surfaces.rho)
    np.testing.assert_allclose(
        none.current_density, none.fast_ion_current_density * (1 - 1 / zeff)
    )
    # trapped electrons reduce the shielding, so the corrected current is larger
    corrected = slow_down(
        deposition, impure, surfaces, trapped_electron_correction="sauter", **kwargs
    )
    np.testing.assert_allclose(zeff, 1.3)
    assert abs(corrected.driven_current) > abs(none.driven_current) > 0


def test_start_cordey_warns_in_3d(deposition, impure, surfaces):
    with pytest.warns(UserWarning, match="Start-Cordey"):
        steady = slow_down(
            deposition,
            impure,
            surfaces,
            orbit_average=False,
            trapped_electron_correction=start_cordey_trapped_electron_correction,
        )
    assert np.isfinite(steady.driven_current)


def test_unknown_correction_is_refused(deposition, impure, surfaces):
    with pytest.raises(ValueError, match="unknown trapped_electron_correction"):
        slow_down(
            deposition,
            impure,
            surfaces,
            orbit_average=False,
            trapped_electron_correction="redl",
        )


def test_sauter_needs_the_safety_factor(impure, surfaces):
    bare = replace(surfaces, safety_factor=None)
    with pytest.raises(ValueError, match="no safety factor"):
        sauter_trapped_electron_correction(bare, impure)
