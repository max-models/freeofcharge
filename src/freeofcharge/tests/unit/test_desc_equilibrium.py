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
