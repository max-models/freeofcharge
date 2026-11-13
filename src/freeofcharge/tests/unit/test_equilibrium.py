import numpy as np
import pytest

from freeofcharge import Equilibrium, FluxSurfaces, MagneticEquilibrium
from freeofcharge.geqdsk import read

R0, A, KAPPA, B0, Q0 = 3.0, 0.9, 1.6, 2.5, 1.2


@pytest.fixture(scope="module")
def solovev():
    return Equilibrium.solovev(R0, A, KAPPA, B0, Q0, wall_psi_n=1.3)


def exact_flux(r, z):
    c = B0 / (2 * R0**2 * KAPPA * Q0)
    return c * (r**2 * z**2 + 0.25 * KAPPA**2 * (r**2 - R0**2) ** 2)


def test_boundary_passes_through_the_outboard_midplane(solovev):
    assert solovev.psi_normalized(R0 + A, 0.0) == pytest.approx(1.0, rel=1e-4)
    assert solovev.boundary.vertices[:, 0].max() == pytest.approx(R0 + A, rel=1e-4)
    r_inner = np.sqrt(2 * R0**2 - (R0 + A) ** 2)
    assert solovev.boundary.vertices[:, 0].min() == pytest.approx(r_inner, rel=1e-4)


def test_flux_surface_lies_on_its_flux(solovev):
    surface = solovev.flux_surface(0.49, points=64)
    np.testing.assert_allclose(
        solovev.psi_normalized(surface[:, 0], surface[:, 1]), 0.49, rtol=1e-6
    )


def test_field_matches_the_analytic_derivatives(solovev):
    rng = np.random.default_rng(0)
    r = R0 + rng.uniform(-0.5, 0.5, 50)
    z = rng.uniform(-0.6, 0.6, 50)
    c = B0 / (2 * R0**2 * KAPPA * Q0)
    dpsi_dr = c * (2 * r * z**2 + KAPPA**2 * r * (r**2 - R0**2))
    dpsi_dz = c * 2 * r**2 * z
    b_r, b_z, b_phi = solovev.magnetic_field(r, z)
    # cubic interpolation on a 129-point grid: errors of about 1e-5 of the field
    b_pol = np.max(np.hypot(dpsi_dr, dpsi_dz) / r)
    np.testing.assert_allclose(b_r, -dpsi_dz / r, atol=1e-4 * b_pol)
    np.testing.assert_allclose(b_z, dpsi_dr / r, atol=1e-4 * b_pol)
    np.testing.assert_allclose(b_phi, B0 * R0 / r, rtol=1e-12)
    np.testing.assert_allclose(
        solovev.psi(r, z), exact_flux(r, z), atol=1e-4 * exact_flux(R0 + A, 0.0)
    )


def test_cartesian_field_is_the_rotated_cylindrical_one(solovev):
    phi = 0.7
    r, z = R0 + 0.3, 0.2
    b_r, b_z, b_phi = (float(b) for b in solovev.magnetic_field(r, z))
    field = solovev.field_at(np.array([[r * np.cos(phi), r * np.sin(phi), z]]))[0]
    expected = [
        b_r * np.cos(phi) - b_phi * np.sin(phi),
        b_r * np.sin(phi) + b_phi * np.cos(phi),
        b_z,
    ]
    np.testing.assert_allclose(field, expected, rtol=1e-12)


def test_inside(solovev):
    assert solovev.inside(
        np.array([R0, R0 + 1.2 * A]), np.array([0.0, 0.0])
    ).tolist() == [True, False]
    assert solovev.inside_wall(np.array([R0 + 1.05 * A]), np.array([0.0])).tolist() == [
        True
    ]


def test_volume_matches_the_boundary_integral(solovev):
    surfaces = solovev.flux_surfaces(np.linspace(0.0, 1.0, 11), resolution=600)
    # V = 2 pi int R dR dZ = 2 pi oint R^2 / 2 dZ (counter-clockwise boundary)
    r, z = solovev.flux_surface(1.0, points=4096).T
    dz = np.roll(z, -1) - z
    r_mid = 0.5 * (r + np.roll(r, -1))
    volume = 2 * np.pi * np.sum(0.5 * r_mid**2 * dz)
    assert surfaces.volume.sum() == pytest.approx(volume, rel=2e-3)
    assert np.all(np.diff(surfaces.inverse_aspect_ratio) > 0)
    r_inner = np.sqrt(2 * R0**2 - (R0 + A) ** 2)
    # the last shell's centre (rho = 0.95) lies just inside the boundary
    assert surfaces.inverse_aspect_ratio[-1] < (R0 + A - r_inner) / (R0 + A + r_inner)
    assert surfaces.inverse_aspect_ratio[-1] > 0.9 * (R0 + A - r_inner) / (
        R0 + A + r_inner
    )


def test_geqdsk_round_trip(solovev, tmp_path):
    path = tmp_path / "solovev.geqdsk"
    solovev.to_geqdsk(path)
    again = Equilibrium.from_geqdsk(path)
    np.testing.assert_allclose(again.psi_grid, solovev.psi_grid, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(again.r, solovev.r, rtol=1e-9)
    np.testing.assert_allclose(again.z, solovev.z, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        again.boundary.vertices, solovev.boundary.vertices, rtol=1e-8
    )
    np.testing.assert_allclose(again.wall.vertices, solovev.wall.vertices, rtol=1e-8)
    assert again.psi_boundary == pytest.approx(solovev.psi_boundary, rel=1e-9)


def test_geqdsk_reads_numbers_that_run_together(tmp_path):
    lines = ["  test" + " " * 42 + "   0   2   2"]
    scalars = [1.0, 2.0, 0.0, 1.0, 0.0, 1.5, 0.0, -1.0, -2.0, 1.0]
    scalars += [0.0, -1.0, 0.0, 1.5, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0]
    # every block starts on a new line; negative numbers fill all 16 columns
    for block in (scalars, [3.0, 3.0] * 4, [-1.0, -1.5, -1.25, -2.0], [1.0, 1.0]):
        for i in range(0, len(block), 5):
            lines.append("".join(f"{v:16.9e}" for v in block[i : i + 5]))
    assert "e+00-" in lines[-2]
    lines.append("    0    0")
    path = tmp_path / "packed.geqdsk"
    path.write_text("\n".join(lines) + "\n")
    data = read(path)
    assert data.psi.tolist() == [[-1.0, -1.25], [-1.5, -2.0]]
    assert data.psi_boundary == -2.0


def test_star_shaped_inside_matches_the_polygon_test():
    """The fast boundary and wall tests agree with ray casting on every polygon."""
    from freeofcharge.equilibrium import _StarShapedPolygon

    eq = Equilibrium.solovev(6.2, 1.8, 1.7, 5.3)
    assert eq._fast_boundary is not None and eq._fast_wall is not None
    rng = np.random.default_rng(0)
    r, z = rng.uniform(3.0, 10.0, 200_000), rng.uniform(-7.0, 7.0, 200_000)
    for polygon in (eq.boundary, eq.wall):
        vertices = np.asarray(polygon.vertices)
        slow = np.asarray(polygon.contains_points(np.stack([r, z])))
        for order in (vertices, vertices[::-1], np.roll(vertices, 37, axis=0)):
            fast = _StarShapedPolygon(order, eq.axis).contains(r, z)
            np.testing.assert_array_equal(fast, slow)


def test_a_polygon_that_is_not_star_shaped_falls_back():
    from freeofcharge.equilibrium import _StarShapedPolygon

    c_shape = np.array(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 1.5], [1.5, 1.5], [1.5, 0.5], [0, 0.5]],
        dtype=float,
    )
    assert _StarShapedPolygon.try_build(c_shape, (1.75, 1.0)) is None


def test_equilibrium_is_a_magnetic_equilibrium(solovev):
    assert isinstance(solovev, MagneticEquilibrium)
    lower, upper = solovev.bounds
    assert lower.shape == upper.shape == (3,)
    assert np.all(lower < upper)


def test_cartesian_methods_match_the_cylindrical_ones(solovev):
    phi = np.linspace(0.0, 2.0 * np.pi, 7)
    r = np.linspace(R0 - 0.8 * A, R0 + 1.3 * A, 7)
    z = np.linspace(-0.4, 0.3, 7)
    points = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1).reshape(7, 1, 3)
    np.testing.assert_allclose(
        np.ravel(solovev.rho_at(points)), solovev.rho(r, z), rtol=1e-12
    )
    assert np.ravel(solovev.inside_at(points)).tolist() == solovev.inside(r, z).tolist()
    assert (
        np.ravel(solovev.inside_wall_at(points)).tolist()
        == solovev.inside_wall(r, z).tolist()
    )
    assert solovev.has_wall
    assert solovev.field_at(points).shape == (7, 1, 3)


def test_rotational_transform_is_one_over_q(solovev):
    surfaces = solovev.flux_surfaces(np.linspace(0.0, 1.0, 5), resolution=50)
    np.testing.assert_allclose(
        surfaces.rotational_transform * surfaces.safety_factor, 1.0
    )
    bare = FluxSurfaces(np.linspace(0.0, 1.0, 3), np.zeros(2), np.zeros(2), np.zeros(2))
    assert bare.rotational_transform is None


def test_flux_surfaces_convert_to_a_dataset(solovev):
    surfaces = solovev.flux_surfaces(np.linspace(0.0, 1.0, 5), resolution=50)
    shells = surfaces.to_dataset()
    np.testing.assert_allclose(shells.rho, surfaces.rho)
    np.testing.assert_allclose(shells.attrs["rho_edges"], surfaces.rho_edges)
    np.testing.assert_allclose(shells.volume, surfaces.volume)
    np.testing.assert_allclose(
        shells.rotational_transform, surfaces.rotational_transform
    )
    np.testing.assert_allclose(shells.trapped_fraction, surfaces.trapped_fraction)
    assert shells.volume.attrs["units"] == "m^3"
    bare = FluxSurfaces(np.linspace(0.0, 1.0, 3), np.zeros(2), np.zeros(2), np.zeros(2))
    assert set(bare.to_dataset().data_vars) == {
        "volume",
        "area",
        "inverse_aspect_ratio",
    }


def _polar_flux_coordinates(equilibrium, step=1e-4):
    r"""The flux coordinates $(\rho, \theta)$ of an axisymmetric equilibrium.

    $\theta$ the geometric angle about the axis; the radius along each ray by
    bisection of $\psi_n = \rho^2$, the derivatives by central differences.
    """
    r0, z0 = equilibrium.axis
    lower = np.asarray(equilibrium.grid.lower_bound)
    upper = np.asarray(equilibrium.grid.upper_bound)

    def surface(rho, theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        with np.errstate(divide="ignore"):
            reach = np.min(
                np.abs(
                    [
                        np.where(cos_t > 0, upper[0] - r0, lower[0] - r0) / cos_t,
                        np.where(sin_t > 0, upper[1] - z0, lower[1] - z0) / sin_t,
                    ]
                ),
                axis=0,
            )
        low, high = np.zeros_like(rho), 0.999 * reach
        for _ in range(60):
            mid = 0.5 * (low + high)
            above = (
                np.asarray(
                    equilibrium.psi_normalized(r0 + mid * cos_t, z0 + mid * sin_t)
                )
                >= rho**2
            )
            high, low = np.where(above, mid, high), np.where(above, low, mid)
        distance = 0.5 * (low + high)
        return r0 + distance * cos_t, z0 + distance * sin_t

    def shape(rho, theta, zeta):
        r, z = surface(rho, theta)
        r_up, z_up = surface(rho + step, theta)
        r_down, z_down = surface(rho - step, theta)
        r_right, z_right = surface(rho, theta + step)
        r_left, z_left = surface(rho, theta - step)
        b_r, b_z, b_phi = (np.asarray(b) for b in equilibrium.magnetic_field(r, z))
        return (
            r,
            (r_up - r_down) / (2 * step),
            (r_right - r_left) / (2 * step),
            (z_up - z_down) / (2 * step),
            (z_right - z_left) / (2 * step),
            np.sqrt(b_r**2 + b_z**2 + b_phi**2),
            np.ones_like(r),
        )

    return shape


def test_trapped_fraction_of_the_3d_path_is_the_axisymmetric_one():
    """Solov'ev through `FluxSurfaces.from_flux_coordinates`: the Jacobian weights
    of the 3D average are the 2D ones, dl / B_p."""
    solovev = Equilibrium.solovev(3.0, 1.0, elongation=1.5, toroidal_field=2.0)
    edges = np.linspace(0.0, 1.0, 6)
    flat = FluxSurfaces.from_flux_coordinates(
        _polar_flux_coordinates(solovev),
        1,
        edges,
        field_points=(256, 1),
        radial_points=2,
        radial_coordinate="poloidal",
    )
    surfaces = solovev.flux_surfaces(edges)
    assert not flat.axisymmetric
    assert np.all(surfaces.trapped_fraction > 0.25)
    # 5e-6 at most
    np.testing.assert_allclose(
        flat.trapped_fraction, surfaces.trapped_fraction, atol=1e-4
    )
    np.testing.assert_allclose(flat.volume, surfaces.volume, rtol=1e-3)


def test_field_epsilon_is_the_inverse_aspect_ratio_where_b_goes_as_one_over_r():
    solovev = Equilibrium.solovev(10.0, 1.0, toroidal_field=2.0, q_axis=3.0)
    surfaces = solovev.flux_surfaces(np.linspace(0.0, 1.0, 6), resolution=50)
    # B_phi = F / R dominates at large aspect ratio and q
    np.testing.assert_allclose(
        surfaces.field_epsilon, surfaces.inverse_aspect_ratio, rtol=0.02
    )
    bare = FluxSurfaces(np.linspace(0.0, 1.0, 3), np.zeros(2), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="no field points"):
        _ = bare.field_epsilon
