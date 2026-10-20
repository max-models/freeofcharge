import numpy as np
import pytest

from freeofcharge import Equilibrium
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
    field = solovev.magnetic_field_cartesian(
        np.array([[r * np.cos(phi), r * np.sin(phi), z]])
    )[0]
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
