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

    # coarse, to keep the sampling (about 2800 inversions) short: errors of a few 1e-2
    gridded = GriddedEquilibrium(example, (16, 20, 16), r_range=(4.64, 6.21))
    assert gridded.nfp == 5 and gridded.stellarator_symmetric
    nodes, points, field = _points(example, n=40, seed=2)
    plasma = nodes[:, 0] < 0.8
    rho = np.asarray(gridded.rho_at(points))
    np.testing.assert_allclose(rho[plasma], nodes[plasma, 0], atol=0.08)
    error = np.linalg.norm(np.asarray(gridded.field_at(points)) - field, axis=-1)
    assert np.all(error[plasma] < 3e-2 * np.linalg.norm(field[plasma], axis=-1))
