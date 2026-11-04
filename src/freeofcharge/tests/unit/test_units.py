import astropy.units as u
import numpy as np
import pytest

import freeofcharge as foc
from freeofcharge import ascot5, chain
from freeofcharge.stopping import ConstantStopping, ElectronCRStopping

IONS = {"D": (2, 1, 1.0)}


@pytest.fixture(scope="module")
def equilibrium():
    return foc.Equilibrium.solovev(1.65, 0.5, 1.6, 2.5, points=65)


def test_tangential_injection_takes_quantities():
    plain = foc.tangential_injection(5.3, 12.0, height=0.1, toroidal_angle=0.5)
    quantities = foc.tangential_injection(
        5300 * u.mm, "12 m", height="10 cm", toroidal_angle=0.5 * u.rad
    )
    for a, b in zip(plain, quantities, strict=True):
        np.testing.assert_allclose(a, b)
    degrees = foc.tangential_injection(5.3, 12.0, toroidal_angle=30 * u.deg)
    radians = foc.tangential_injection(5.3, 12.0, toroidal_angle=np.pi / 6)
    np.testing.assert_allclose(degrees[0], radians[0])
    with pytest.raises(ValueError, match="tangency_radius must be in units of m"):
        foc.tangential_injection(5.3 * u.s, 12.0)


def test_gaussian_beam_takes_quantities():
    origin, direction = foc.tangential_injection(5.3, 12.0)
    kwargs = {"markers": 50, "seed": 1}
    plain = foc.NeutralBeam.gaussian(
        origin,
        direction,
        1.0e6,
        16.5e6,
        width=(0.3, 0.2),
        divergence=0.005,
        halo_fraction=0.1,
        halo_divergence=0.02,
        **kwargs,
    )
    quantities = foc.NeutralBeam.gaussian(
        origin * u.m,
        direction,
        "1 MeV",
        16.5 * u.MW,
        width=[30, 20] * u.cm,
        divergence="5 mrad",
        halo_fraction=0.1,
        halo_divergence=20 * u.mrad,
        **kwargs,
    )
    np.testing.assert_allclose(plain.markers.positions, quantities.markers.positions)
    np.testing.assert_allclose(plain.markers.velocities, quantities.markers.velocities)
    np.testing.assert_allclose(plain.markers.weights, quantities.markers.weights)
    assert quantities.power() == pytest.approx(16.5e6)
    with pytest.raises(ValueError, match="energy must be in units of eV"):
        foc.NeutralBeam.gaussian(origin, direction, "1 m", 1e6, markers=5)


def test_plasma_takes_quantities_and_kelvin():
    plain = foc.Plasma.parabolic(1e20, 2e4, 1e4, ions=IONS, points=11)
    quantities = foc.Plasma.parabolic(
        "1e20 m^-3",
        20 * u.keV,
        (1e4 * u.eV).to(u.K, u.temperature_energy()),
        ions=IONS,
        points=11,
    )
    for name in ("ne", "te", "ti"):
        np.testing.assert_allclose(getattr(plain, name), getattr(quantities, name))
    np.testing.assert_allclose(plain.ions["D"].density, quantities.ions["D"].density)
    with pytest.raises(ValueError, match="te0 must be in units of eV"):
        foc.Plasma.parabolic(1e20, "2 m", ions=IONS)


def test_plasma_profiles_take_quantity_arrays():
    rho = np.linspace(0.0, 1.0, 3)
    plasma = foc.Plasma(
        rho,
        [1e20, 5e19, 1e19] * u.m**-3,
        [5.0, 2.0, 1.0] * u.keV,
        ["5 keV", "2 keV", "1 keV"],
        ions={"D": foc.IonSpecies(2, 1, [1e14, 5e13, 1e13] * u.cm**-3)},
    )
    assert isinstance(plasma.te, np.ndarray)
    np.testing.assert_allclose(plasma.ne, [1e20, 5e19, 1e19])
    np.testing.assert_allclose(plasma.te, [5e3, 2e3, 1e3])
    np.testing.assert_allclose(plasma.ti, plasma.te)
    np.testing.assert_allclose(plasma.ions["D"].density, plasma.ne)
    plain = np.array([1.0, 2.0, 3.0])
    assert foc.IonSpecies(2, 1, plain).density is plain
    with pytest.raises(ValueError, match="ne must be in units of m"):
        foc.Plasma(rho, [1.0, 2.0, 3.0] * u.eV, plain, plain)


def test_solovev_takes_quantities(equilibrium):
    other = foc.Equilibrium.solovev("1650 mm", 50 * u.cm, 1.6, 25 * u.kG, points=65)
    np.testing.assert_allclose(other.psi_grid, equilibrium.psi_grid)
    np.testing.assert_allclose(other.f, equilibrium.f)
    with pytest.raises(ValueError, match="toroidal_field must be in units of T"):
        foc.Equilibrium.solovev(1.65, 0.5, 1.6, 2.5 * u.m)


def test_equilibrium_constructor_takes_quantities(equilibrium):
    e = equilibrium
    rebuilt = foc.Equilibrium(
        e.r * u.m,
        (e.z * u.m).to(u.cm),
        e.psi_grid * u.Wb / u.rad,
        f"{e.psi_axis} Wb / rad",
        e.psi_boundary * u.Wb / u.rad,
        e.f * u.T * u.m,
        axis=(e.axis[0] * u.m, "0 cm"),
        boundary=e.boundary.vertices * u.m,
        wall=e.wall.vertices * 1000 * u.mm,
    )
    np.testing.assert_allclose(rebuilt.z, e.z)
    assert rebuilt.axis == pytest.approx(e.axis)
    np.testing.assert_allclose(rebuilt.wall.vertices, e.wall.vertices)
    point = np.array([[1.9, 0.0, 0.1]])
    np.testing.assert_allclose(rebuilt.field_at(point), e.field_at(point))
    with pytest.raises(ValueError, match="psi must be in units of Wb/rad"):
        foc.Equilibrium(
            e.r, e.z, e.psi_grid * u.Wb, 0.0, 1.0, e.f,
            axis=e.axis, boundary=e.boundary.vertices,
        )  # fmt: skip


def test_deposit_and_slow_down_take_quantities(equilibrium):
    plasma = foc.Plasma.parabolic(5e19, 2.5e3, ions=IONS, points=21)
    origin, direction = foc.tangential_injection(1.0, 3.5)
    beam = foc.NeutralBeam.gaussian(
        origin, direction, "60 keV", "2.5 MW", markers=20, seed=1
    )
    stopping = ConstantStopping("1e-20 m^2")
    assert stopping.sigma == pytest.approx(1e-20)
    common = {"rho_edges": np.linspace(0.0, 1.0, 6), "births_per_marker": 1}
    plain = foc.deposit(
        beam, equilibrium, plasma, stopping, step=0.02, max_length=3.0, seed=2, **common
    )
    quantities = foc.deposit(
        beam, equilibrium, plasma, stopping, step="2 cm", max_length=3 * u.m, seed=2,
        **common,
    )  # fmt: skip
    np.testing.assert_allclose(plain.power, quantities.power)
    surfaces = equilibrium.flux_surfaces(
        plain.rho_edges, resolution=50, field_points=32
    )
    a = foc.slow_down(plain, plasma, surfaces, minimum_energy=1.0e3, speed_points=21)
    b = foc.slow_down(plain, plasma, surfaces, minimum_energy="1 keV", speed_points=21)
    assert a.driven_current == pytest.approx(b.driven_current)
    with pytest.raises(ValueError, match="step must be in units of m"):
        foc.deposit(beam, equilibrium, plasma, stopping, step="1 s")
    with pytest.raises(ValueError, match="minimum_energy must be in units of eV"):
        foc.slow_down(plain, plasma, surfaces, minimum_energy=1 * u.m)
    with pytest.raises(ValueError, match="sigma must be in units of m"):
        ConstantStopping(1 * u.m)


def test_chain_stages_take_quantities():
    assert chain.child_langmuir("9 kV", 6 * u.mm) == pytest.approx(
        chain.child_langmuir(9.0e3, 6.0e-3)
    )
    np.testing.assert_allclose(
        chain.child_langmuir([9.0, 18.0] * u.kV, 6.0e-3, mass="3.344e-27 kg"),
        chain.child_langmuir(np.array([9.0e3, 18.0e3]), 6.0e-3, mass=3.344e-27),
    )
    np.testing.assert_allclose(
        chain.neutral_fraction([1e16, 3e16] * u.cm**-2),
        chain.neutral_fraction([1e20, 3e20]),
    )
    sigma_cm = {
        key: value * 1e4 * u.cm**2 for key, value in chain.GAS_NEUTRALIZER.items()
    }
    np.testing.assert_allclose(
        chain.neutral_fraction("1.4e16 cm^-2", sigma_cm), chain.neutral_fraction(1.4e20)
    )
    np.testing.assert_allclose(
        chain.neutral_fraction_closed_form("1.4e20 m^-2", sigma_cm),
        chain.neutral_fraction_closed_form(1.4e20),
    )
    assert chain.duct_transmission("10 cm", 5 * u.mrad, 23 * u.m, "25 cm") == (
        pytest.approx(chain.duct_transmission(0.1, 5e-3, 23.0, 0.25))
    )
    with pytest.raises(ValueError, match="voltage must be in units of V"):
        chain.child_langmuir("9 A", 6e-3)
    with pytest.raises(ValueError, match="divergence must be in units of rad"):
        chain.duct_transmission(0.1, "5 m", 23.0, 0.25)


def test_deuterium_source_takes_quantities():
    pytest.importorskip("wellbalanced")
    plain = chain.deuterium_source(2e5, gas_temperature=0.05)
    gas = (0.05 * u.eV).to(u.K, u.temperature_energy())
    quantities = chain.deuterium_source(
        "200 kW",
        pressure=300 * u.mPa,
        volume="0.4 m^3",
        wall_area=6 * u.m**2,
        gas_temperature=gas,
    )
    assert quantities.rf_power == 2e5
    assert quantities.current_density == pytest.approx(plain.current_density)
    with pytest.raises(ValueError, match="pressure must be in units of Pa"):
        chain.deuterium_source(2e5, pressure="0.3 m")


def test_ascot5_setup_takes_quantities():
    case = ascot5.benchmark_case(markers=10, grid_points=33, toroidal_field="25 kG")
    assert case.parameters["toroidal_field"] == pytest.approx(2.5)
    a = ascot5.injector(case.beam, divergence="5 mrad")
    assert a["divh"] == pytest.approx(5e-3)


def test_electron_cr_stopping_converts_before_opencr():
    with pytest.raises(ValueError, match="temperatures must be in units of eV"):
        ElectronCRStopping(temperatures=[1.0, 2.0] * u.m)
