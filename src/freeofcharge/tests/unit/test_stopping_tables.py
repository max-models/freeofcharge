import time

import numpy as np
import pytest

from freeofcharge import (
    ConstantStopping,
    Equilibrium,
    NeutralBeam,
    Plasma,
    SuzukiStopping,
    deposit,
    tangential_injection,
)
from freeofcharge.stopping_tables import (
    Adf21,
    TabulatedStopping,
    read_adf21,
    write_adf21,
)

AMU = 1.66053906660e-27
E_CHARGE = 1.602176634e-19


def speed(energy_per_amu):
    return np.sqrt(2.0 * np.asarray(energy_per_amu) * E_CHARGE / AMU)


def grid(start, stop, per_decade, extra=()):
    count = int(np.ceil(per_decade * np.log10(stop / start))) + 1
    return np.union1d(np.geomspace(start, stop, count), extra)


# Suzuki's fit has kinks at the edges of its blend window: nodes there
ENERGIES = grid(10e3, 5e6, 15, (1e5 / 1.1, 1.1e5))
DENSITIES = grid(1e18, 3e20, 15)
TEMPERATURES = grid(100.0, 3e4, 15)


@pytest.fixture(scope="module")
def suzuki_table():
    return TabulatedStopping.from_model(
        SuzukiStopping(),
        ENERGIES,
        DENSITIES,
        TEMPERATURES,
        [(2, 1), (3, 1), (4, 2), (12, 6)],
    )


def random_points(count, seed=1):
    rng = np.random.default_rng(seed)

    def log_uniform(nodes):
        return np.exp(rng.uniform(np.log(nodes[0]), np.log(nodes[-1]), count))

    return log_uniform(ENERGIES), log_uniform(DENSITIES), log_uniform(TEMPERATURES)


PLASMAS = {
    "D": [(1.0, 2, 1)],
    "DT": [(0.5, 2, 1), (0.5, 3, 1)],
    "D+He": [(0.9, 2, 1), (0.05, 4, 2)],
    "D+C": [(0.88, 2, 1), (0.02, 12, 6)],
    "D+He+C": [(0.86, 2, 1), (0.04, 4, 2), (0.01, 12, 6)],
}


def ion_args(ne, ions):
    return (
        [share * ne for share, _, _ in ions],
        [a for _, a, _ in ions],
        [z for _, _, z in ions],
    )


@pytest.mark.parametrize("plasma", list(PLASMAS))
def test_table_of_suzuki_reproduces_suzuki_inside_the_grid(suzuki_table, plasma):
    e, n, t = random_points(50_000)
    args = (e, n, t, *ion_args(n, PLASMAS[plasma]))
    error = np.asarray(suzuki_table(*args)) / np.asarray(SuzukiStopping()(*args)) - 1
    assert np.max(np.abs(error)) < 5e-3
    assert np.mean(np.abs(error)) < 5e-4


def test_table_of_suzuki_is_exact_at_the_nodes(suzuki_table):
    e, n, t = np.meshgrid(ENERGIES[::7], DENSITIES[::5], TEMPERATURES[::5])
    # Suzuki's fit is of the composite form: exact for every mixture
    for ions in PLASMAS.values():
        args = (e, n, t, *ion_args(n, ions))
        np.testing.assert_allclose(
            suzuki_table(*args), SuzukiStopping()(*args), rtol=1e-12
        )


def test_species_tables_are_pure_plasmas(suzuki_table):
    sigma = suzuki_table.species_sigma(2e5, 5e19, 3e3)
    assert sigma.shape == (4,)
    # the pure-carbon plasma of Suzuki's composite form: sigma_H (1 + 5 S_C)
    assert sigma[3] > sigma[2] > sigma[0]
    pure_d = SuzukiStopping()(2e5, 5e19, 3e3, [5e19], [2], [1])
    assert sigma[0] == pytest.approx(float(pure_d), rel=1e-6)


def test_table_is_faster_than_suzuki(suzuki_table):
    e, n, t = random_points(200_000, seed=2)
    args = (e, n, t, *ion_args(n, PLASMAS["D+He+C"]))

    def best(model):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            model(*args)
            times.append(time.perf_counter() - start)
        return min(times)

    assert best(SuzukiStopping()) > 1.5 * best(suzuki_table)


def test_save_and_load(tmp_path, suzuki_table):
    path = tmp_path / "suzuki.nc"
    suzuki_table.save(path)
    loaded = TabulatedStopping.load(path)
    assert loaded.density_rule == "electron"
    assert loaded.table.attrs["source"] == "sampled from SuzukiStopping"
    e, n, t = random_points(100)
    args = (e, n, t, *ion_args(n, PLASMAS["D+C"]))
    np.testing.assert_array_equal(loaded(*args), suzuki_table(*args))


def test_outside_the_grid(suzuki_table):
    args = (5e19, 3e3, [5e19], [2], [1])
    with pytest.raises(ValueError, match="energy 1e\\+04 to 5e\\+06 eV/amu"):
        suzuki_table(5e3, *args)
    with pytest.raises(ValueError, match="density"):
        suzuki_table(1e5, 1e17, 3e3, [1e17], [2], [1])
    table = suzuki_table.table
    clamped = TabulatedStopping(table, outside="clamp")
    assert float(clamped(5e3, *args)) == float(clamped(10e3, *args))
    # per axis: the density clamps, the energy still raises
    mixed = TabulatedStopping(table, outside={"density": "clamp"})
    assert float(mixed(1e5, 1e17, 3e3, [1e17], [2], [1])) == float(
        mixed(1e5, 1e18, 3e3, [1e18], [2], [1])
    )
    with pytest.raises(ValueError, match="energy"):
        mixed(5e3, *args)
    # extrapolation continues the last cell, linear in the logarithms
    extrapolated = TabulatedStopping(table, outside="extrapolate")
    e = np.array([ENERGIES[0], ENERGIES[1], ENERGIES[0] ** 2 / ENERGIES[1]])
    log_sigma = np.log(np.asarray(extrapolated(e, *args)))
    assert log_sigma[2] == pytest.approx(2 * log_sigma[0] - log_sigma[1], rel=1e-12)


def test_invalid_options(suzuki_table):
    table = suzuki_table.table
    with pytest.raises(ValueError, match="outside"):
        TabulatedStopping(table, outside="wrap")
    with pytest.raises(ValueError, match="unknown axes"):
        TabulatedStopping(table, outside={"time": "clamp"})
    with pytest.raises(ValueError, match="density_rule"):
        TabulatedStopping(table, density_rule="ion")
    with pytest.raises(ValueError, match="positive"):
        TabulatedStopping(table.assign(sigma=-table.sigma))
    bad_grid = table.assign_coords(energy=table.energy.values[::-1])
    with pytest.raises(ValueError, match="increasing"):
        TabulatedStopping(bad_grid)


def test_species_matching(suzuki_table):
    # boron has no table
    with pytest.raises(ValueError, match="Z = 5"):
        suzuki_table(1e5, 5e19, 3e3, [4e19, 2e18], [2, 11], [1, 5])
    # an ion without a table but with zero density is ignored
    with_zero = suzuki_table(1e5, 5e19, 3e3, [5e19, 0.0], [2, 11], [1, 5])
    pure = suzuki_table(1e5, 5e19, 3e3, [5e19], [2], [1])
    assert float(with_zero) == float(pure)
    # D and T have tables of their own; an unknown isotope of charge 1 has not
    with pytest.raises(ValueError, match="A = 1"):
        suzuki_table(1e5, 5e19, 3e3, [5e19], [1], [1])
    with pytest.raises(ValueError, match="ion densities"):
        suzuki_table(1e5, 5e19, 3e3, [], [], [])


def test_single_species_without_ions_and_constant_axes():
    table = TabulatedStopping.from_model(
        ConstantStopping(2e-20), [1e4, 1e6], [1e19], [1e3], [(2, 1)]
    )
    # one node in density and temperature: independent of them, no range check
    sigma = table(np.array([1e4, 1e5]), 1e21, 5.0, [], [], [])
    np.testing.assert_allclose(sigma, 2e-20, rtol=1e-14)
    # one node in energy
    table = TabulatedStopping.from_model(
        ConstantStopping(3e-20), [1e5], [1e19, 1e20], [1e3], [(2, 1)]
    )
    assert float(table(1e6, 3e19, 1e3, [3e19], [2], [1])) == pytest.approx(3e-20)


def test_from_model_checks():
    with pytest.raises(ValueError, match="hydrogen"):
        TabulatedStopping.from_model(
            ConstantStopping(1e-20), [1e5], [1e19], [1e3], [(4, 2)]
        )
    with pytest.raises(ValueError, match="impurity_fraction"):
        TabulatedStopping.from_model(
            ConstantStopping(1e-20),
            [1e5],
            [1e19],
            [1e3],
            [(2, 1), (4, 2)],
            impurity_fraction=1.0,
        )

    def falling(e, ne, te, densities, masses, charges):
        # stopping that falls with impurities: not of the composite form
        share = sum(n * z for n, z in zip(densities, charges, strict=True) if z > 1)
        return 1e-20 * (1.0 - 5.0 * share / ne)

    with pytest.raises(ValueError, match="composite"):
        TabulatedStopping.from_model(falling, [1e5], [1e19], [1e3], [(2, 1), (4, 2)])


# ----------------------------------------------------------------------
# ADAS adf21
# ----------------------------------------------------------------------

# A synthetic file in the adf21 layout (made-up numbers, not ADAS data):
# 3 energies x 2 densities at TREF = 2 keV, and 4 temperatures at EREF =
# 50 keV/amu, NREF = 1e13 cm^-3; SVREF = S(EREF, NREF, TREF).
SYNTHETIC = """\
    1 /SVREF= 2.000D-08 /SPEC=H  /DATE=13/11/26 /CODE=synthetic
-------------------------------------------------------------------------------
    3    2 /TREF= 2.000D+03
-------------------------------------------------------------------------------
  2.000D+04 5.000D+04 2.000D+05
  1.000D+13 1.000D+14
-------------------------------------------------------------------------------
  2.500D-08 2.000D-08 1.000D-08
  3.000D-08 2.600D-08 1.400D-08
-------------------------------------------------------------------------------
    4 /EREF= 5.000D+04 /NREF= 1.000D+13
-------------------------------------------------------------------------------
  1.000D+02 1.000D+03 2.000D+03 1.000D+04
-------------------------------------------------------------------------------
  1.800D-08 1.950D-08 2.000D-08 2.100D-08
-------------------------------------------------------------------------------
C  synthetic stopping coefficients for the freeofcharge tests
"""


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "synthetic_h1.dat"
    path.write_text(SYNTHETIC)
    return path


def test_read_adf21(synthetic):
    data = read_adf21(synthetic)
    assert data.charge == 1 and data.element == "H"
    np.testing.assert_array_equal(data.energies, [2e4, 5e4, 2e5])
    np.testing.assert_array_equal(data.densities, [1e13, 1e14])
    np.testing.assert_array_equal(
        data.coefficients, [[2.5e-8, 3.0e-8], [2.0e-8, 2.6e-8], [1.0e-8, 1.4e-8]]
    )
    np.testing.assert_array_equal(data.temperatures, [1e2, 1e3, 2e3, 1e4])
    assert data.reference == 2e-8
    assert (data.reference_energy, data.reference_density) == (5e4, 1e13)
    assert data.reference_temperature == 2e3


def test_adf21_table_is_exact_at_the_nodes(synthetic):
    data = read_adf21(synthetic)
    table = TabulatedStopping.from_adf21([synthetic], mass_numbers=[1])
    assert table.density_rule == "equivalent"
    assert list(table.table.species.values) == ["H1+"]
    e, n, t = np.meshgrid(
        data.energies, data.densities, data.temperatures, indexing="ij"
    )
    expected = (
        data.coefficients[:, :, None]
        * data.temperature_coefficients[None, None, :]
        / data.reference
    )
    sigma = table(e, n * 1e6, t, [n * 1e6], [1], [1])
    np.testing.assert_allclose(sigma, expected * 1e-6 / speed(e), rtol=1e-14)
    # the reference point: S = SVREF
    s_ref = table(5e4, 1e19, 2e3, [1e19], [2], [1]) * speed(5e4) * 1e6
    assert float(s_ref) == pytest.approx(2e-8, rel=1e-14)


def test_adf21_table_follows_the_separable_rule_between_nodes(synthetic):
    data = read_adf21(synthetic)
    table = TabulatedStopping.from_adf21([synthetic])
    rng = np.random.default_rng(3)
    e = np.exp(rng.uniform(np.log(2e4), np.log(2e5), 200))
    n = np.exp(rng.uniform(np.log(1e13), np.log(1e14), 200))
    t = np.exp(rng.uniform(np.log(1e2), np.log(1e4), 200))

    def log_lerp(nodes, values, x):
        return np.exp(np.interp(np.log(x), np.log(nodes), np.log(values)))

    # S_En bilinear in (ln E, ln n) of ln S: along E at both densities, then n
    s_e = np.stack(
        [log_lerp(data.energies, data.coefficients[:, j], e) for j in range(2)]
    )
    w = np.log(n / 1e13) / np.log(10.0)
    s_en = np.exp((1 - w) * np.log(s_e[0]) + w * np.log(s_e[1]))
    s_t = log_lerp(data.temperatures, data.temperature_coefficients, t)
    expected = s_en * s_t / data.reference * 1e-6 / speed(e)
    sigma = table(e, n * 1e6, t, [n * 1e6], [1], [1])
    np.testing.assert_allclose(sigma, expected, rtol=1e-12)


def test_write_adf21_round_trip_in_fixed_columns(tmp_path, synthetic):
    data = read_adf21(synthetic)
    path = tmp_path / "written.dat"
    write_adf21(path, data, comment="round trip")
    again = read_adf21(path)
    for name in (
        "energies",
        "densities",
        "coefficients",
        "temperatures",
        "temperature_coefficients",
    ):
        np.testing.assert_array_equal(getattr(again, name), getattr(data, name))
    assert again.reference == data.reference
    assert again.reference_temperature == data.reference_temperature
    # the fixed columns of CHERAB's reader (cherab.openadas.parse.utility)
    lines = path.read_text().splitlines()
    assert int(lines[0][3:5]) == 1
    assert float(lines[0][13:22].replace("D", "E")) == data.reference
    assert lines[0][29:31] == "H "
    assert (int(lines[2][1:5]), int(lines[2][6:10])) == (3, 2)
    assert float(lines[2][17:26].replace("D", "E")) == 2e3
    assert float(lines[4][11:20].replace("D", "E")) == 5e4  # second energy
    header = next(line for line in lines if "/EREF=" in line)
    assert int(header[1:5]) == 4
    assert float(header[12:21].replace("D", "E")) == 5e4
    assert float(header[28:37].replace("D", "E")) == 1e13


def carbon_file(tmp_path):
    """A second receiver (Z = 6) on a different, finer grid."""
    energies = np.array([1e4, 2e4, 3e4, 5e4, 1e5, 3e5])
    densities = np.array([5e12, 1e13, 3e13, 1e14, 2e14])
    coefficients = (
        6e-8 * (energies[:, None] / 5e4) ** -0.6 * (densities[None, :] / 1e13) ** 0.1
    )
    data = Adf21(
        charge=6,
        element="C",
        energies=energies,
        densities=densities,
        coefficients=np.round(coefficients, 11),
        temperatures=[1e2, 1e4],
        temperature_coefficients=[6e-8, 6e-8],
        reference=6e-8,
        reference_energy=5e4,
        reference_density=1e13,
        reference_temperature=2e3,
    )
    path = tmp_path / "synthetic_c6.dat"
    write_adf21(path, data)
    return path


def test_adf21_composite_rule(tmp_path, synthetic):
    carbon = carbon_file(tmp_path)
    table = TabulatedStopping.from_adf21(
        {"H": synthetic, "C": carbon}, mass_numbers={"C": 12}
    )
    # the union of the nodes inside the shared range
    np.testing.assert_array_equal(table.table.energy, [2e4, 3e4, 5e4, 1e5, 2e5])
    np.testing.assert_array_equal(table.table.density, [1e19, 3e19, 1e20])
    # each species keeps its own file's interpolation
    alone = {
        "H": TabulatedStopping.from_adf21([synthetic]),
        "C": TabulatedStopping.from_adf21([read_adf21(carbon)]),
    }
    e, t = 7e4, 3e3
    n_d, n_c = 5e19, 1e18
    ne = n_d + 6 * n_c
    z2 = n_d + 36 * n_c
    sigma_h = alone["H"](e, z2, t, [], [], [])
    sigma_c = alone["C"](e, z2 / 6, t, [], [], [])
    expected = (n_d * sigma_h + 6 * n_c * sigma_c) / ne
    sigma = table(e, ne, t, [n_d, n_c], [2, 12], [1, 6])
    assert float(sigma) == pytest.approx(float(expected), rel=1e-12)
    # the electron rule reads both at n_e
    electron = TabulatedStopping(table.table, density_rule="electron")
    expected = (
        n_d * alone["H"](e, ne, t, [], [], [])
        + 6 * n_c * alone["C"](e, ne, t, [], [], [])
    ) / ne
    assert float(electron(e, ne, t, [n_d, n_c], [2, 12], [1, 6])) == pytest.approx(
        float(expected), rel=1e-12
    )


def test_adf21_errors(tmp_path, synthetic):
    path = tmp_path / "short.dat"
    path.write_text("\n".join(SYNTHETIC.splitlines()[:6]))
    with pytest.raises(ValueError, match="not an adf21 file"):
        read_adf21(path)
    path.write_text(SYNTHETIC.replace("  1.000D+13 1.000D+14\n", "  1.000D+13\n"))
    with pytest.raises(ValueError, match="grid values"):
        read_adf21(path)
    path.write_text(SYNTHETIC.replace("  3.000D-08 2.600D-08 1.400D-08\n", ""))
    with pytest.raises(ValueError, match="coefficients, expected"):
        read_adf21(path)
    path.write_text(SYNTHETIC.replace(" 2.100D-08", ""))
    with pytest.raises(ValueError, match="temperatures"):
        read_adf21(path)
    path.write_text(SYNTHETIC.replace("/TREF= 2.000D+03", ""))
    with pytest.raises(ValueError, match="TREF"):
        read_adf21(path)
    data = read_adf21(synthetic)
    kwargs = {
        name: getattr(data, name)
        for name in (
            "charge",
            "element",
            "energies",
            "densities",
            "coefficients",
            "temperatures",
            "temperature_coefficients",
            "reference",
            "reference_energy",
            "reference_density",
            "reference_temperature",
        )
    }
    with pytest.raises(ValueError, match="shape"):
        Adf21(**{**kwargs, "coefficients": data.coefficients.T})
    with pytest.raises(ValueError, match="differ in length"):
        Adf21(**{**kwargs, "temperatures": data.temperatures[:3]})
    with pytest.raises(ValueError, match="increasing"):
        Adf21(**{**kwargs, "energies": data.energies[::-1]})
    with pytest.raises(ValueError, match="positive"):
        Adf21(**{**kwargs, "reference": 0.0})
    other = Adf21(**{**kwargs, "energies": data.energies * 100})
    with pytest.raises(ValueError, match="no energy range"):
        TabulatedStopping.from_adf21([data, other])


# ----------------------------------------------------------------------
# In the deposition
# ----------------------------------------------------------------------


def test_deposition_with_the_table_equals_suzuki(suzuki_table):
    r0, a = 3.0, 0.9
    equilibrium = Equilibrium.solovev(r0, a, 1.6, 2.5, 1.2, wall_psi_n=1.3)
    plasma = Plasma.parabolic(
        3e19, 5e3, ions={"D": (2, 1, 0.9), "C": (12, 6, 1.0 / 60.0)}
    )
    origin, direction = tangential_injection(r0, r0 + 3 * a)
    beam = NeutralBeam.gaussian(
        origin, direction, 500e3, 5e6, width=0.1, divergence=0.01, markers=100, seed=5
    )
    kwargs = {"step": 5e-3, "births_per_marker": 0, "rho_edges": np.linspace(0, 1, 11)}
    model = deposit(beam, equilibrium, plasma, SuzukiStopping(), **kwargs)
    table = deposit(beam, equilibrium, plasma, suzuki_table, **kwargs)
    assert table.shine_through_fraction == pytest.approx(
        model.shine_through_fraction, rel=1e-3
    )
    np.testing.assert_allclose(table.power, model.power, rtol=1e-3)


def test_tabulate_the_electron_cr_model(tmp_path):
    pytest.importorskip("opencr")
    from freeofcharge.stopping import ElectronCRStopping

    model = ElectronCRStopping(
        temperatures=np.geomspace(100.0, 2.0e4, 8), densities=[1e19, 1e20], n_max=6
    )
    table = TabulatedStopping.from_model(
        model,
        np.geomspace(2e4, 1e6, 12),
        [1e19, 1e20],
        np.geomspace(100, 2e4, 8),
        [(2, 1)],
    )
    table.save(tmp_path / "cr.nc")
    loaded = TabulatedStopping.load(tmp_path / "cr.nc")
    # at the nodes the model's values
    args = (np.array([2e4, 1e6]), 1e20, 2e4, [1e20], [2], [1])
    np.testing.assert_allclose(loaded(*args), model(*args), rtol=1e-12)
