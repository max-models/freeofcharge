import numpy as np
import pytest

from freeofcharge import MagneticEquilibrium

pytest.importorskip("desc")

from desc.grid import Grid  # noqa: E402

from freeofcharge.desc_equilibrium import DESCEquilibrium  # noqa: E402


@pytest.fixture(scope="module")
def example():
    return DESCEquilibrium.from_example("W7-X")


def _points(equilibrium, n=12, seed=0):
    rng = np.random.default_rng(seed)
    nodes = np.stack(
        [0.05 + 0.9 * rng.random(n), rng.random(n) * 2 * np.pi, rng.random(n) * 6],
        axis=-1,
    )
    data = equilibrium.equilibrium.compute(
        ["R", "phi", "Z", "B"], grid=Grid(nodes, sort=False)
    )
    r, phi = np.asarray(data["R"]), np.asarray(data["phi"])
    points = np.stack([r * np.cos(phi), r * np.sin(phi), np.asarray(data["Z"])], -1)
    b = np.asarray(data["B"])
    field = np.stack(
        [
            b[:, 0] * np.cos(phi) - b[:, 1] * np.sin(phi),
            b[:, 0] * np.sin(phi) + b[:, 1] * np.cos(phi),
            b[:, 2],
        ],
        -1,
    )
    return nodes, points, field


def test_is_a_magnetic_equilibrium(example):
    assert isinstance(example, MagneticEquilibrium)
    lower, upper = example.bounds
    assert np.all(lower < upper)
    assert not example.has_wall


def test_inverse_map_and_field(example):
    nodes, points, field = _points(example)
    rho, theta, _ = example.flux_coordinates(points.reshape(3, 4, 3))
    np.testing.assert_allclose(rho.ravel(), nodes[:, 0], atol=1e-8)
    angle = np.angle(np.exp(1j * (theta.ravel() - nodes[:, 1])))
    np.testing.assert_allclose(angle, 0.0, atol=1e-6)
    np.testing.assert_allclose(example.field_at(points), field, rtol=1e-6, atol=1e-9)
    assert np.all(example.inside_at(points))


def test_points_outside(example):
    _, upper = example.bounds
    points = np.array([[0.1, 0.0, 0.0], [upper[0] + 0.5, 0.0, 0.0]])
    assert np.all(example.rho_at(points) > 1.0)
    assert np.all(example.inside_wall_at(points))
    assert example.field_at(points).shape == (2, 3)


def test_flux_surfaces_match_desc_volume(example):
    surfaces = example.flux_surfaces(np.linspace(0.0, 1.0, 5), field_points=(24, 16))
    total = float(example.equilibrium.compute("V")["V"])
    np.testing.assert_allclose(surfaces.volume.sum(), total, rtol=1e-3)
    np.testing.assert_allclose(surfaces.surface_weight.sum(axis=1), 1.0)
    assert np.all((surfaces.trapped_fraction > 0) & (surfaces.trapped_fraction < 1))
    assert np.all(surfaces.inverse_aspect_ratio > 0)


def test_from_file(example, tmp_path):
    path = tmp_path / "w7x.h5"
    example.equilibrium.save(path)
    loaded = DESCEquilibrium.from_file(path)
    assert loaded.nfp == example.nfp
    np.testing.assert_allclose(loaded.bounds[1], example.bounds[1])


def test_rho_and_field_at_once_and_symmetry(example):
    _, points, field = _points(example, n=6, seed=1)
    rho, both = example.rho_and_field_at(points)
    np.testing.assert_allclose(rho, example.rho_at(points))
    np.testing.assert_allclose(both, field, rtol=1e-6, atol=1e-9)
    # stellarator symmetry: (x, y, z) -> (x, -y, -z) keeps rho, B_R flips, B_phi, B_Z stay
    assert example.stellarator_symmetric
    mirrored = points * np.array([1.0, -1.0, -1.0])
    rho_m, field_m = example.rho_and_field_at(mirrored)
    np.testing.assert_allclose(rho_m, rho, atol=1e-8)
    np.testing.assert_allclose(field_m, field * np.array([-1.0, 1.0, 1.0]), atol=1e-6)


def test_gridded(example):
    from freeofcharge import GriddedEquilibrium

    # coarse, a test of the sampling and the symmetry: errors of a few 1e-2
    gridded = GriddedEquilibrium(example, (16, 20, 16), r_range=(4.64, 6.21))
    assert gridded.nfp == 5 and gridded.stellarator_symmetric
    nodes, points, field = _points(example, n=40, seed=2)
    plasma = nodes[:, 0] < 0.8
    rho = np.asarray(gridded.rho_at(points))
    # rho**2 is what is interpolated; rho itself has a cone on the axis, where a small
    # error in rho**2 is a large one in rho
    # on this coarse grid: rms about 7e-3, max about 3e-2 (it varies with the platform)
    error2 = rho[plasma] ** 2 - nodes[plasma, 0] ** 2
    assert np.sqrt(np.mean(error2**2)) < 0.015
    assert np.max(np.abs(error2)) < 0.06
    error = np.linalg.norm(np.asarray(gridded.field_at(points)) - field, axis=-1)
    assert np.all(error[plasma] < 3e-2 * np.linalg.norm(field[plasma], axis=-1))


def test_rho_and_field_in_planes(example):
    nodes, points, field = _points(example, n=30, seed=3)
    nodes[0, 0] = 0.0  # on the axis
    phi = np.mod(nodes[:, 2], 2 * np.pi)
    data = example.equilibrium.compute(["R", "Z", "B"], grid=Grid(nodes, sort=False))
    r, z = np.asarray(data["R"]), np.asarray(data["Z"])
    rho, b = example.rho_and_field_in_planes(phi, r[:, None], z[:, None])
    # rho**2: rho has a cone on the axis
    np.testing.assert_allclose(rho[:, 0] ** 2, nodes[:, 0] ** 2, atol=1e-5)
    np.testing.assert_allclose(b[:, 0], np.asarray(data["B"]), rtol=1e-4, atol=1e-6)
    # outside: 1 + residual / a, as the inverse map gives it. Near the boundary the
    # two inversions agree; far out the residual of the unconverged Newton iteration
    # depends on its path (and on the machine), so there only rho > 1 is checked.
    _, upper = example.bounds
    outside = np.array([[upper[0] + 0.2, 1.0]])
    rho, b = example.rho_and_field_in_planes([0.3], outside, [[0, 0]])
    cartesian = np.stack([outside * np.cos(0.3), outside * np.sin(0.3), [[0, 0]]], -1)
    np.testing.assert_allclose(rho[:, 0], example.rho_at(cartesian)[:, 0], rtol=0.05)
    assert np.all(rho > 1.0) and np.all(np.isfinite(b))
    assert np.all(np.asarray(example.rho_at(cartesian)) > 1.0)
    assert example.radial_coordinate == "toroidal"


def test_gridded_deposition_matches_the_direct_one(example):
    import freeofcharge as foc
    from freeofcharge import GriddedEquilibrium
    from freeofcharge.injectors import w7x_injectors

    gridded = GriddedEquilibrium(example, (64, 32, 64))
    assert gridded.radial_coordinate == "toroidal"
    plasma = foc.Plasma.parabolic(
        2e19, 2e3, ions={"H": (1, 1, 1.0)}, radial_coordinate="toroidal"
    )
    beam = w7x_injectors()[1].sources[0].beams(2, seed=1)["full"]
    kwargs = {"step": 0.03, "rho_edges": np.linspace(0, 1, 6), "seed": 2}
    on_grid = foc.deposit(beam, gridded, plasma, **kwargs)
    direct = foc.deposit(beam, example, plasma, **kwargs)
    assert 0.05 < direct.shine_through_fraction < 0.5
    assert on_grid.shine_through_fraction == pytest.approx(
        direct.shine_through_fraction, abs=5e-3
    )
    np.testing.assert_allclose(
        on_grid.birth_rate, direct.birth_rate, atol=0.05 * direct.birth_rate.max()
    )
    np.testing.assert_allclose(on_grid.birth_rho, direct.birth_rho, atol=0.02)
    assert on_grid.to_dataset().rho.attrs["long_name"] == r"$\rho_\mathrm{tor}$"


def test_surface_averages_match_desc(example):
    from desc.grid import LinearGrid

    surfaces = example.flux_surfaces(np.linspace(0.1, 0.9, 5), field_points=(32, 32))
    assert not surfaces.axisymmetric
    eq = example.equilibrium
    grid = LinearGrid(rho=surfaces.rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute(["<|B|^2>", "trapped fraction"], grid=grid)
    b2 = grid.compress(np.asarray(data["<|B|^2>"]))
    trapped = grid.compress(np.asarray(data["trapped fraction"]))
    # the Jacobian weights of the 3D flux-surface average
    average = surfaces.surface_average(surfaces.field_strength**2)
    np.testing.assert_allclose(average, b2, rtol=1e-4)
    np.testing.assert_allclose(surfaces.trapped_fraction, trapped, rtol=0.02)


def test_vmec_reads_desc_wout(example, tmp_path):
    """DESC's own wout export through the VMEC adapter: the same equilibrium."""
    pytest.importorskip("vmecpp")
    from desc.vmec import VMECIO

    from freeofcharge.vmec_equilibrium import VMECEquilibrium

    path = tmp_path / "wout_w7x.nc"
    VMECIO.save(example.equilibrium, path, surfs=17, verbose=0)
    vmec = VMECEquilibrium.from_wout(path)
    nodes, points, field = _points(example, n=40, seed=1)
    # 3e-6 from VMEC's 17 radial surfaces; 1e-7 at 33
    np.testing.assert_allclose(vmec.rho_at(points), nodes[:, 0], atol=1e-5)
    # 2.6e-4 at most, at any number of surfaces: the Fourier truncation of DESC's export
    difference = np.linalg.norm(vmec.field_at(points) - field, axis=-1)
    assert np.max(difference / np.linalg.norm(field, axis=-1)) < 1e-3


def test_vmec_run_agrees_with_desc(example, tmp_path):
    """VMEC++ on DESC's W7-X input (boundary, pressure, iota) against DESC's equilibrium.

    The difference is that of the two equilibria, not of the adapters: it does not
    shrink with VMEC's radial resolution (17 to 65 surfaces), and VMEC's axis lies
    0.9 mm from DESC's at phi = 0 (rho about 2e-3 there).
    """
    vmecpp = pytest.importorskip("vmecpp")
    from desc.vmec import VMECIO

    from freeofcharge.vmec_equilibrium import VMECEquilibrium

    path = tmp_path / "input.w7x"
    VMECIO.write_vmec_input(example.equilibrium, str(path))
    vmec_input = vmecpp.VmecInput.from_file(path)
    vmec_input.ns_array = np.array([9, 17])
    vmec_input.ftol_array = np.array([1e-10, 1e-12])
    vmec_input.niter_array = np.array([10000, 20000])
    vmec = VMECEquilibrium.from_input(vmec_input)
    nodes, points, field = _points(example, n=40, seed=2)
    rho = np.asarray(vmec.rho_at(points))
    assert np.mean(np.abs(rho - nodes[:, 0])) < 5e-3
    assert np.max(np.abs(rho - nodes[:, 0])) < 1.5e-2
    difference = np.linalg.norm(vmec.field_at(points) - field, axis=-1)
    difference /= np.linalg.norm(field, axis=-1)
    assert np.mean(difference) < 1e-3
    assert np.max(difference) < 4e-3
