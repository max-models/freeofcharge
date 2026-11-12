import numpy as np
import pytest

from freeofcharge import Equilibrium, GriddedEquilibrium, MagneticEquilibrium
from freeofcharge.gridded_equilibrium import _HelicalTestField


@pytest.fixture(scope="module")
def helical():
    return _HelicalTestField()


@pytest.fixture(scope="module")
def gridded(helical):
    return GriddedEquilibrium(helical, (48, 32, 48))


@pytest.fixture(scope="module")
def solovev():
    return Equilibrium.solovev(3.0, 1.0, elongation=1.5, toroidal_field=2.0)


def _cartesian(r, phi, z):
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _plasma_points(field, n=500, seed=0, rho=(0.1, 0.9)):
    """Points of the helical test field at radii in ``rho``."""
    rng = np.random.default_rng(seed)
    radius = rng.uniform(*rho, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    phi = rng.uniform(-np.pi, 3.0 * np.pi, n)
    u = field.a * radius * np.cos(theta) + field.delta * np.cos(field.nfp * phi)
    v = field.a * radius * np.sin(theta) + field.delta * np.sin(field.nfp * phi)
    r = np.sqrt(field.r0**2 + 2.0 * field.r0 * u)
    return radius, _cartesian(r, phi, v)


def test_is_a_magnetic_equilibrium(gridded, helical):
    assert isinstance(gridded, MagneticEquilibrium)
    assert gridded.nfp == helical.nfp
    assert gridded.stellarator_symmetric
    np.testing.assert_array_equal(gridded.bounds[0], helical.bounds[0])
    assert not gridded.has_wall


def test_matches_the_exact_field(gridded, helical):
    rho, points = _plasma_points(helical)
    np.testing.assert_allclose(gridded.rho_at(points), rho, atol=1e-5)
    field = helical.field_at(points)
    np.testing.assert_allclose(gridded.field_at(points), field, atol=1e-6)
    assert np.all(gridded.inside_at(points))


def test_leading_shapes_are_kept(gridded, helical):
    _, points = _plasma_points(helical, n=12)
    points = points.reshape(3, 4, 3)
    assert gridded.rho_at(points).shape == (3, 4)
    assert gridded.field_at(points).shape == (3, 4, 3)
    assert gridded.inside_at(points).shape == (3, 4)


def test_symmetry_and_full_period_agree(gridded, helical):
    full = GriddedEquilibrium(helical, (48, 32, 48), symmetric=False)
    assert not full.stellarator_symmetric
    _, points = _plasma_points(helical, seed=1)
    # the full period samples the same grid: only round-off differs
    np.testing.assert_allclose(full.rho_at(points), gridded.rho_at(points), atol=1e-12)
    np.testing.assert_allclose(
        full.field_at(points), gridded.field_at(points), atol=1e-12
    )


def test_field_periods_rotate_the_field(gridded, helical):
    _, points = _plasma_points(helical, n=50, seed=2)
    angle = 2.0 * np.pi / helical.nfp
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = points @ rotation.T
    np.testing.assert_allclose(gridded.rho_at(turned), gridded.rho_at(points))
    np.testing.assert_allclose(
        gridded.field_at(turned), gridded.field_at(points) @ rotation.T, atol=1e-12
    )


def test_trilinear_is_less_accurate(helical, gridded):
    linear = GriddedEquilibrium(helical, (48, 32, 48), order=1)
    rho, points = _plasma_points(helical)
    error_linear = np.abs(linear.rho_at(points) - rho).max()
    error_cubic = np.abs(gridded.rho_at(points) - rho).max()
    assert error_cubic < 1e-2 * error_linear < 1e-4


def test_outside_the_grid_is_nan(gridded):
    points = np.array([[0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [3.0, 0.0, 5.0]])
    assert np.all(np.isnan(gridded.rho_at(points)))
    assert np.all(np.isnan(gridded.field_at(points)))
    assert not np.any(gridded.inside_at(points))
    assert np.all(gridded.inside_wall_at(points))


def test_boundary_layer(gridded, helical):
    rho, points = _plasma_points(helical, n=2000, seed=3, rho=(0.9, 1.1))
    interpolated = gridded.rho_at(points)
    # smooth across rho = 1 here, so no kink: still fourth order
    np.testing.assert_allclose(interpolated, rho, atol=1e-5)
    assert np.mean(gridded.inside_at(points) != (rho < 1.0)) < 1e-2


def test_save_and_load(gridded, helical, tmp_path):
    path = tmp_path / "grid.nc"
    gridded.save(path)
    alone = GriddedEquilibrium.load(path)
    _, points = _plasma_points(helical, n=50, seed=4)
    np.testing.assert_allclose(alone.rho_at(points), gridded.rho_at(points))
    np.testing.assert_allclose(alone.field_at(points), gridded.field_at(points))
    assert alone.nfp == gridded.nfp and alone.stellarator_symmetric
    np.testing.assert_array_equal(alone.bounds[1], gridded.bounds[1])
    assert not alone.has_wall
    assert np.all(alone.inside_wall_at(points))
    with pytest.raises(ValueError, match="needs the equilibrium"):
        alone.flux_surfaces([0.0, 1.0])


def test_wraps_an_axisymmetric_equilibrium(solovev, tmp_path):
    gridded = GriddedEquilibrium(solovev, (64, 4, 64), symmetric=True)
    assert gridded.nfp == 1 and gridded.has_wall
    rng = np.random.default_rng(5)
    points = _cartesian(
        rng.uniform(2.2, 3.8, 1000),
        rng.uniform(0, 2 * np.pi, 1000),
        rng.uniform(-1, 1, 1000),
    )
    rho = solovev.rho_at(points)
    plasma = rho < 0.9
    np.testing.assert_allclose(gridded.rho_at(points)[plasma], rho[plasma], atol=1e-5)
    np.testing.assert_allclose(
        gridded.field_at(points)[plasma], solovev.field_at(points)[plasma], atol=1e-5
    )
    np.testing.assert_array_equal(
        gridded.inside_wall_at(points), solovev.inside_wall_at(points)
    )
    shells = gridded.flux_surfaces([0.0, 0.5, 1.0], resolution=50, field_points=16)
    np.testing.assert_allclose(
        shells.volume,
        solovev.flux_surfaces([0.0, 0.5, 1.0], resolution=50, field_points=16).volume,
    )
    gridded.save(tmp_path / "solovev.nc")
    assert GriddedEquilibrium.load(tmp_path / "solovev.nc", solovev).has_wall


def test_explicit_ranges(helical):
    gridded = GriddedEquilibrium(
        helical, 24, symmetric=False, r_range=(2.3, 3.7), z_range=(-0.7, 0.7)
    )
    r = gridded.samples["R"].values
    z = gridded.samples["Z"].values
    np.testing.assert_allclose([r[0], r[-1]], [2.3 - 0.14, 3.7 + 0.14])
    np.testing.assert_allclose([z[0], z[-1]], [-0.7 - 0.14, 0.7 + 0.14])


class _PartialMap(_HelicalTestField):
    """rho inf and the field NaN outside rho = 1, as PolarFEEC."""

    def rho_at(self, points):
        rho = super().rho_at(points)
        return np.where(rho < 1.0, rho, np.inf)

    def field_at(self, points):
        field = super().field_at(points)
        return np.where((super().rho_at(points) < 1.0)[..., None], field, np.nan)


def test_fills_points_outside_the_map():
    partial = _PartialMap()
    gridded = GriddedEquilibrium(partial, (48, 32, 48))
    assert np.all(np.isfinite(gridded.samples["rho_squared"]))
    assert np.all(np.isfinite(gridded.samples["b_r"]))
    rho, points = _plasma_points(partial, n=500, seed=6, rho=(0.1, 0.8))
    np.testing.assert_allclose(gridded.rho_at(points), rho, atol=1e-2)
    outside_rho, outside = _plasma_points(partial, n=500, seed=7, rho=(1.05, 1.15))
    assert np.all(gridded.rho_at(outside) > 1.0)
    assert np.all(np.isfinite(gridded.field_at(outside)))


class _Nowhere(_HelicalTestField):
    def rho_at(self, points):
        return np.full(np.asarray(points).shape[:-1], np.inf)

    def inside_at(self, points):
        return np.zeros(np.asarray(points).shape[:-1], dtype=bool)


def test_errors(helical):
    with pytest.raises(ValueError, match="order"):
        GriddedEquilibrium(helical, 8, order=7)
    with pytest.raises(ValueError, match="even"):
        GriddedEquilibrium(helical, (8, 7, 8))
    with pytest.raises(ValueError, match="an equilibrium is needed"):
        GriddedEquilibrium(None)
    with pytest.raises(ValueError, match="probe found no point"):
        GriddedEquilibrium(_Nowhere(), 8)
    with pytest.raises(ValueError, match="no finite sample"):
        GriddedEquilibrium(_Nowhere(), 8, r_range=(2.5, 3.5))


class _BothAtOnce(_HelicalTestField):
    """An adapter with one inversion for rho and B, as DESC and GVEC."""

    calls = 0

    def rho_and_field_at(self, points):
        self.calls += 1
        return self.rho_at(points), self.field_at(points)


def test_uses_one_call_for_rho_and_field(gridded, helical):
    both = _BothAtOnce()
    gridded_both = GriddedEquilibrium(both, (48, 32, 48), chunk=20000)
    assert both.calls == -(-48 * 17 * 48 // 20000)
    _, points = _plasma_points(helical, n=50, seed=8)
    np.testing.assert_allclose(gridded_both.rho_at(points), gridded.rho_at(points))


def test_nan_points_are_outside(gridded):
    points = np.array([[np.nan, 0.0, 0.0], [3.0, np.nan, 0.0], [3.0, 0.0, np.nan]])
    assert np.all(np.isnan(gridded.rho_at(points)))
    assert not np.any(gridded.inside_at(points))


def test_rho_and_inside_at_once(gridded, helical):
    _, points = _plasma_points(helical, n=40, seed=9)
    off = np.array([[100.0, 0.0, 0.0]])
    rho, inside = gridded.rho_and_inside_at(np.concatenate([points, off]))
    np.testing.assert_array_equal(rho[:-1], gridded.rho_at(points))
    assert np.all(inside[:-1]) and not inside[-1] and np.isnan(rho[-1])
    # no point on the grid at all
    assert np.all(np.isnan(gridded.field_at(np.repeat(off, 3, axis=0))))


def test_cached_samples_once(solovev, tmp_path):
    path = tmp_path / "cache" / "grid.nc"
    first = GriddedEquilibrium.cached(path, solovev, (16, 4, 16), symmetric=False)
    assert path.exists() and first.radial_coordinate == "poloidal"
    again = GriddedEquilibrium.cached(path, solovev, (16, 4, 16), symmetric=False)
    np.testing.assert_array_equal(again.samples.rho_squared, first.samples.rho_squared)
    assert again.equilibrium is solovev
    other = GriddedEquilibrium.cached(path, solovev, 12, symmetric=False)
    assert other.samples.rho_squared.shape == (12, 12, 12)
    # the radial coordinate is kept in the file
    assert GriddedEquilibrium.load(path).radial_coordinate == "poloidal"


def test_radial_coordinate_of_older_files(solovev):
    gridded = GriddedEquilibrium(solovev, (12, 4, 12), symmetric=False)
    samples = gridded.samples.copy()
    del samples.attrs["radial_coordinate"]
    older = GriddedEquilibrium(solovev, _samples=samples)
    assert older.radial_coordinate == "poloidal"
