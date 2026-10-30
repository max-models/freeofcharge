from pathlib import Path

import numpy as np
import pytest

from freeofcharge import MagneticEquilibrium

gvec = pytest.importorskip("gvec")

from freeofcharge.gvec_equilibrium import GVECEquilibrium  # noqa: E402

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def ellipse():
    """A small rotating ellipse with two field periods."""
    return GVECEquilibrium.from_files(
        DATA / "parameter_ellipse_final.ini", DATA / "ellipse_State_final.dat"
    )


def _gvec_points(equilibrium, rho, theta, zeta):
    ev = gvec.evaluate(equilibrium.state, "pos", "B", rho=rho, theta=theta, zeta=zeta)
    pos = ev["pos"].transpose("rad", "pol", "tor", "xyz").values.reshape(-1, 3)
    field = ev["B"].transpose("rad", "pol", "tor", "xyz").values.reshape(-1, 3)
    rr, tt, zz = np.meshgrid(rho, theta, zeta, indexing="ij")
    return pos, field, rr.ravel(), tt.ravel(), zz.ravel()


def test_is_a_magnetic_equilibrium(ellipse):
    assert isinstance(ellipse, MagneticEquilibrium)
    lower, upper = ellipse.bounds
    assert np.all(lower < upper)
    assert not ellipse.has_wall


def test_inverse_map_recovers_the_flux_coordinates(ellipse):
    rho = np.array([0.0, 0.05, 0.3, 0.7, 0.99])
    theta = np.linspace(0.1, 6.0, 5)
    zeta = np.linspace(0.0, 2.0 * np.pi, 7)
    pos, _, rr, tt, zz = _gvec_points(ellipse, rho, theta, zeta)
    found_rho, found_theta, found_zeta = ellipse.flux_coordinates(pos)
    np.testing.assert_allclose(found_rho, rr, atol=1e-8)
    away = rr > 0.01
    angle = np.angle(np.exp(1j * (found_theta - tt)))
    np.testing.assert_allclose(angle[away], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.exp(1j * found_zeta), np.exp(1j * zz), atol=1e-12)
    assert np.all(ellipse.inside_at(pos.reshape(5, -1, 3)) == (rr < 1.0).reshape(5, -1))


def test_field_matches_gvec(ellipse):
    pos, field, *_ = _gvec_points(
        ellipse, np.array([0.2, 0.6, 0.9]), np.linspace(0, 6, 4), np.linspace(0, 3, 4)
    )
    np.testing.assert_allclose(ellipse.field_at(pos), field, rtol=1e-7, atol=1e-10)


def test_points_outside_have_rho_above_one(ellipse):
    lower, upper = ellipse.bounds
    points = np.array([[0.0, 0.0, 0.0], [upper[0] + 0.5, 0.0, 0.0]])
    rho = ellipse.rho_at(points)
    assert np.all(rho > 1.0)
    assert not np.any(ellipse.inside_at(points))
    assert np.all(ellipse.inside_wall_at(points))
    assert ellipse.field_at(points).shape == (2, 3)


def test_divergence_free(ellipse):
    centre = ellipse.flux_coordinates(np.array([[3.2, 0.3, 0.1]]))
    assert centre[0][0] < 1.0
    x0, h = np.array([3.2, 0.3, 0.1]), 1.0e-4
    divergence = 0.0
    for i in range(3):
        step = np.zeros(3)
        step[i] = h
        b = ellipse.field_at(np.stack([x0 + step, x0 - step]))
        divergence += (b[0, i] - b[1, i]) / (2.0 * h)
    assert abs(divergence) < 1e-4 * np.linalg.norm(ellipse.field_at(x0[None])) / 0.6


def test_flux_surfaces_match_gvec_volume(ellipse):
    surfaces = ellipse.flux_surfaces(np.linspace(0.0, 1.0, 6))
    total = float(
        gvec.evaluate(ellipse.state, "V", rho=None, theta=None, zeta=None)["V"]
    )
    np.testing.assert_allclose(surfaces.volume.sum(), total, rtol=1e-4)
    # an ellipse of half axes about 0.4 and 0.6
    np.testing.assert_allclose(surfaces.area.sum(), np.pi * 0.4 * 0.6, rtol=0.05)
    np.testing.assert_allclose(surfaces.surface_weight.sum(axis=1), 1.0)
    assert np.all((surfaces.trapped_fraction > 0) & (surfaces.trapped_fraction < 1))
    np.testing.assert_allclose(
        surfaces.rotational_transform,
        ellipse.state.evaluate_profile("iota", surfaces.rho),
    )
    assert np.all(np.abs(surfaces.major_radius - 3.0) < 0.1)
