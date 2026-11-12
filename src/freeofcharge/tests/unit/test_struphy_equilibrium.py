from pathlib import Path

import numpy as np
import pytest

from freeofcharge import MagneticEquilibrium

pytest.importorskip("struphy")

from struphy.fields_background.equils import AdhocTorus  # noqa: E402
from struphy.geometry.domains import HollowTorus  # noqa: E402

from freeofcharge.struphy_equilibrium import StruphyEquilibrium  # noqa: E402

A, R0 = 1.0, 3.0
DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def torus():
    """Struphy's ad hoc tokamak on a third of a hollow torus (hole r < 0.1)."""
    equilibrium = AdhocTorus(a=A, R0=R0, B0=2.0)
    equilibrium.domain = HollowTorus(a1=0.1, a2=A, R0=R0, tor_period=3)
    return StruphyEquilibrium(equilibrium, rho_inner=0.1)


def _torus_points(r, theta, phi):
    big_r = R0 + r * np.cos(theta)
    return np.stack(
        [big_r * np.cos(phi), big_r * np.sin(phi), r * np.sin(theta)], axis=-1
    )


def test_is_a_magnetic_equilibrium(torus):
    assert isinstance(torus, MagneticEquilibrium)
    assert not torus.has_wall
    assert torus.nfp == 3
    # HollowTorus winds clockwise: y = -R sin(2 pi eta3 / 3)
    np.testing.assert_allclose(torus.period, -2 * np.pi / 3)
    np.testing.assert_allclose(torus.minor_radius, A)
    lower, upper = torus.bounds
    np.testing.assert_allclose(upper, [R0 + A, R0 + A, A], rtol=1e-3)
    assert np.all(lower < upper)


def test_inverse_map_and_field_on_the_whole_torus(torus):
    rng = np.random.default_rng(1)
    r = 0.15 + 0.8 * rng.random(40)
    theta, phi = 2 * np.pi * rng.random((2, 40))
    points = _torus_points(r, theta, phi).reshape(4, 10, 3)
    rho = torus.rho_at(points)
    assert rho.shape == (4, 10)
    np.testing.assert_allclose(rho.ravel(), r / A, atol=1e-10)
    assert np.all(torus.inside_at(points))
    assert np.all(torus.inside_wall_at(points))
    flat = points.reshape(-1, 3)
    expected = np.stack(torus.equilibrium.b_xyz(*flat.T), -1)
    np.testing.assert_allclose(
        torus.field_at(points).reshape(-1, 3), expected, rtol=1e-9, atol=1e-12
    )


def test_hole_and_outside(torus):
    r = np.array([0.0, 0.03, 0.08, 1.2, 1.5])
    theta = np.array([0.0, 1.0, 4.0, 2.5, 0.3])
    phi = np.array([0.3, 2.5, 5.0, 1.0, 4.0])
    points = _torus_points(r, theta, phi)
    rho = torus.rho_at(points)
    # on a circle the hole's linear radius is exact (theta is undefined on the axis)
    np.testing.assert_allclose(rho[:3], r[:3] / A, atol=1e-5)
    # outside: 1 plus the residual of the clamped Newton iteration
    np.testing.assert_allclose(rho[3:], r[3:] / A, rtol=0.02)
    assert list(torus.inside_at(points)) == [True, True, True, False, False]
    field = torus.field_at(points)
    assert np.all(np.isfinite(field))
    # outside: the field on the last closed surface
    edge = _torus_points(np.full(2, A), theta[3:], phi[3:])
    expected = np.stack(torus.equilibrium.b_xyz(*edge.T), -1)
    np.testing.assert_allclose(field[3:], expected, rtol=1e-8)


def test_flux_surfaces(torus):
    surfaces = torus.flux_surfaces(np.linspace(0.1, 1.0, 4), field_points=(32, 8))
    shells = 2 * np.pi**2 * R0 * np.diff(np.linspace(0.1, 1.0, 4) ** 2)
    np.testing.assert_allclose(surfaces.volume, shells, rtol=1e-6)
    np.testing.assert_allclose(surfaces.area, np.pi * np.diff([0.01, 0.16, 0.49, 1.0]))
    np.testing.assert_allclose(
        np.abs(surfaces.safety_factor), torus.equilibrium.q_r(surfaces.rho), rtol=1e-6
    )
    np.testing.assert_allclose(surfaces.major_radius, R0, rtol=1e-12)
    np.testing.assert_allclose(
        surfaces.inverse_aspect_ratio, surfaces.rho / R0, rtol=1e-6
    )
    assert np.all((surfaces.trapped_fraction > 0) & (surfaces.trapped_fraction < 1))


def test_a_cartesian_domain_is_refused():
    from struphy.fields_background.equils import HomogenSlab

    equilibrium = HomogenSlab()
    from struphy.geometry.domains import Cuboid

    equilibrium.domain = Cuboid()
    with pytest.raises(ValueError, match="not toroidal"):
        StruphyEquilibrium(equilibrium)


@pytest.fixture(scope="module")
def w7x():
    """Struphy's DESC equilibrium (DESC's W7-X example) on its default spline map."""
    pytest.importorskip("desc")
    from struphy.fields_background.equils import DESCequilibrium

    return StruphyEquilibrium(DESCequilibrium())


def _desc_points(equilibrium, n=40, seed=0):
    """Flux coordinates, Cartesian points and DESC's B there."""
    from desc.grid import Grid

    rng = np.random.default_rng(seed)
    nodes = np.stack(
        [0.05 + 0.9 * rng.random(n), rng.random(n) * 2 * np.pi, rng.random(n) * 6],
        axis=-1,
    )
    data = equilibrium.compute(["R", "phi", "Z", "B"], grid=Grid(nodes, sort=False))
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


def test_desc_w7x_against_desc(w7x):
    """Struphy's spline map (16, 16, 50 cells) against DESC itself.

    On 200 points the errors are 3.0e-3 in rho and 0.82 % in B at most (4.3e-4
    and 0.11 % rms), 1.5e-4 and 0.033 % with (32, 32, 100) cells: the
    interpolation error of the map.
    """
    assert (w7x.nfp, w7x.rho_inner) == (5, 0.01)
    np.testing.assert_allclose(w7x.period, 2 * np.pi / 5)
    nodes, points, field = _desc_points(w7x.equilibrium.eq)
    rho = w7x.rho_at(points)
    np.testing.assert_allclose(rho, nodes[:, 0], atol=3e-3)
    assert np.sqrt(np.mean((rho - nodes[:, 0]) ** 2)) < 1e-3
    error = np.linalg.norm(w7x.field_at(points) - field, axis=-1)
    assert np.all(error < 1e-2 * np.linalg.norm(field, axis=-1))
    # the cache of Struphy's equilibrium stays empty
    assert not w7x.equilibrium._cache["bv"]["grids"]


def test_desc_w7x_near_the_axis(w7x):
    nodes = np.array([[0.0, 0.0, 0.1], [0.003, 2.0, 1.0], [0.006, 4.0, 3.0]])
    from desc.grid import Grid

    data = w7x.equilibrium.eq.compute(["R", "phi", "Z"], grid=Grid(nodes, sort=False))
    r, phi = np.asarray(data["R"]), np.asarray(data["phi"])
    points = np.stack([r * np.cos(phi), r * np.sin(phi), np.asarray(data["Z"])], -1)
    # inside the hole rho < 0.01 from the map taken as affine there
    np.testing.assert_allclose(w7x.rho_at(points), nodes[:, 0], atol=5e-4)


def test_desc_w7x_flux_surfaces(w7x):
    surfaces = w7x.flux_surfaces(
        np.linspace(0.0, 1.0, 3), field_points=(16, 8), radial_points=3
    )
    eq = w7x.equilibrium.eq
    np.testing.assert_allclose(
        surfaces.volume.sum(), float(eq.compute("V")["V"]), rtol=1e-3
    )
    from desc.grid import LinearGrid

    grid = LinearGrid(rho=surfaces.rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    iota = grid.compress(np.asarray(eq.compute("iota", grid=grid)["iota"]))
    np.testing.assert_allclose(surfaces.rotational_transform, iota, rtol=1e-12)


def test_desc_w7x_iota_from_the_field(w7x, monkeypatch):
    """The generic iota, N_fp <sqrt(g) B^eta2> / <sqrt(g) B^eta3>, against DESC's."""
    monkeypatch.setattr(w7x, "_iota_profile", lambda rho: None)
    theta, zeta = np.meshgrid(
        np.linspace(0, 2 * np.pi, 32, endpoint=False),
        np.linspace(0, 2 * np.pi / 5, 24, endpoint=False),
        indexing="ij",
    )
    iota = w7x._shape(np.full(theta.size, 0.5), theta.ravel(), zeta.ravel())[-1]
    from desc.grid import LinearGrid

    eq = w7x.equilibrium.eq
    grid = LinearGrid(rho=np.array([0.5]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    expected = grid.compress(np.asarray(eq.compute("iota", grid=grid)["iota"]))
    np.testing.assert_allclose(iota, expected[0], rtol=1e-3)


@pytest.fixture(scope="module")
def ellipstell():
    """Struphy's GVEC equilibrium and freeofcharge's GVEC adapter on one state."""
    pytest.importorskip("gvec")
    from struphy.fields_background.equils import GVECequilibrium

    from freeofcharge.gvec_equilibrium import GVECEquilibrium

    equilibrium = GVECequilibrium(
        rel_path=False,
        dat_file=str(DATA / "ellipse_State_final.dat"),
        param_file=str(DATA / "parameter_ellipse_final.ini"),
    )
    return StruphyEquilibrium(equilibrium), GVECEquilibrium(equilibrium.state)


def _gvec_points(gvec_adapter, n=100, seed=0):
    rng = np.random.default_rng(seed)
    rho = 0.05 + 0.9 * rng.random(n)
    theta, zeta = 2 * np.pi * rng.random((2, n))
    r = gvec_adapter._evaluate("X1", rho, theta, zeta)[0]
    z = gvec_adapter._evaluate("X2", rho, theta, zeta)[0]
    points = np.stack([r * np.cos(zeta), -r * np.sin(zeta), z], -1)
    return rho, points, gvec_adapter._field(rho, theta, zeta)


def test_gvec_against_the_gvec_adapter(ellipstell):
    """Struphy's spline map (16 cells per direction): 8.1e-5 in rho at most."""
    struphy, gvec = ellipstell
    # GVEC's zeta runs against phi: x = (R cos zeta, -R sin zeta, Z)
    assert struphy.nfp == 2
    np.testing.assert_allclose(struphy.period, -np.pi)
    rho, points, _ = _gvec_points(gvec)
    np.testing.assert_allclose(struphy.rho_at(points), rho, atol=2e-4)
    # around the axis, inside the hole rho < 0.01 too
    rho = np.array([1e-9, 0.003, 0.006, 0.02])
    theta, zeta = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.1, 0.5, 1.0, 2.0])
    r = gvec._evaluate("X1", rho, theta, zeta)[0]
    z = gvec._evaluate("X2", rho, theta, zeta)[0]
    points = np.stack([r * np.cos(zeta), -r * np.sin(zeta), z], -1)
    np.testing.assert_allclose(struphy.rho_at(points), rho, atol=5e-5)


def test_gvec_flux_surfaces(ellipstell):
    struphy, gvec = ellipstell
    edges = np.linspace(0.0, 1.0, 3)
    ours, theirs = struphy.flux_surfaces(edges), gvec.flux_surfaces(edges)
    np.testing.assert_allclose(ours.volume, theirs.volume, rtol=3e-3)
    np.testing.assert_allclose(ours.safety_factor, theirs.safety_factor, rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="struphy 3.6.1 evaluates GVEC's field at zeta = 2 pi eta3, its map at "
    "zeta = 2 pi eta3 / nfp (GVECequilibrium._gvec_evaluations)",
)
def test_gvec_field_against_the_gvec_adapter(ellipstell):
    struphy, gvec = ellipstell
    _, points, field = _gvec_points(gvec, n=20)
    np.testing.assert_allclose(struphy.field_at(points), field, rtol=1e-3, atol=1e-3)
