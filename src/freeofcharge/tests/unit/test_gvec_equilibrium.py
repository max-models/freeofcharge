import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from freeofcharge import MagneticEquilibrium

gvec = pytest.importorskip("gvec")

from freeofcharge.gvec_equilibrium import GVECEquilibrium  # noqa: E402

DATA = Path(__file__).parents[1] / "data"
PARAMETERS = DATA / "parameter_ellipse_final.ini"
STATE = DATA / "ellipse_State_final.dat"


def _gvec_runs() -> str | None:
    """Load the test state in a subprocess; why not, if GVEC's library crashes there.

    A GVEC built for a newer CPU (a wheel from the build cache of another runner) dies with
    SIGILL, which no test can catch in its own process.
    """
    code = (
        "from freeofcharge.gvec_equilibrium import GVECEquilibrium; "
        f"GVECEquilibrium.from_files({str(PARAMETERS)!r}, {str(STATE)!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if result.returncode < 0:
        return (
            f"GVEC's compiled library crashed (signal {-result.returncode}) on this CPU"
        )
    return None


_crash = _gvec_runs()
if _crash:
    pytest.skip(_crash, allow_module_level=True)


@pytest.fixture(scope="module")
def ellipse():
    """A small rotating ellipse with two field periods."""
    return GVECEquilibrium.from_files(PARAMETERS, STATE)


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


def test_rho_and_field_at_once(ellipse):
    pos, field, rr, *_ = _gvec_points(
        ellipse, np.array([0.2, 0.6]), np.linspace(0, 6, 3), np.linspace(0, 3, 3)
    )
    rho, both = ellipse.rho_and_field_at(pos)
    np.testing.assert_allclose(rho, rr, atol=1e-8)
    np.testing.assert_allclose(both, field, rtol=1e-7, atol=1e-10)


def test_gridded(ellipse):
    from freeofcharge import GriddedEquilibrium

    gridded = GriddedEquilibrium(ellipse, (32, 16, 32))
    assert gridded.nfp == ellipse.nfp
    pos, field, rr, *_ = _gvec_points(
        ellipse, np.array([0.3, 0.6, 0.8]), np.linspace(0, 6, 5), np.linspace(0, 3, 5)
    )
    np.testing.assert_allclose(gridded.rho_at(pos), rr, atol=1e-2)
    np.testing.assert_allclose(gridded.field_at(pos), field, rtol=1e-2, atol=1e-2)


def test_transport_geometry_matches_gvec_and_the_shells(ellipse):
    rho = np.linspace(0.05, 1.0, 12)
    geo = ellipse.transport_geometry(rho)
    ev = gvec.evaluate(ellipse.state, "dV_dPhi_n", "iota", rho=rho, theta=8, zeta=8)
    # V' = dV/drho = 2 rho dV/dPhi_n, GVEC's own
    np.testing.assert_allclose(
        geo.volume_derivative, 2.0 * rho * ev["dV_dPhi_n"].values, rtol=1e-10
    )
    np.testing.assert_allclose(geo.rotational_transform, ev["iota"].values)
    # the integral of V' over a shell is the shell volume of FluxSurfaces
    edges = np.array([0.2, 0.5, 0.8])
    shells = ellipse.flux_surfaces(edges)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    for k in range(2):
        lo, hi = edges[k], edges[k + 1]
        points = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
        vp = ellipse.transport_geometry(points).volume_derivative.values
        np.testing.assert_allclose(
            0.5 * (hi - lo) * np.sum(weights * vp), shells.volume[k], rtol=1e-6
        )
    centres = ellipse.transport_geometry(shells.rho)
    np.testing.assert_allclose(
        centres.trapped_fraction, shells.trapped_fraction, rtol=1e-2
    )
    assert np.all(geo.grad_rho2 > geo.grad_rho**2 * (1 - 1e-12))


def test_transport_geometry_gradient_from_the_shape(ellipse):
    """|grad rho|^2 from GVEC's metric against cross products of the (R, Z) map."""
    rho = np.array([0.3, 0.7])
    geo = ellipse.transport_geometry(rho, angles=(32, 32))
    ev = gvec.evaluate(ellipse.state, "X1", "X2", rho=rho, theta=32, zeta=32)
    r = ev["X1"].values
    e_t = np.stack([ev["dX1_dt"].values, np.zeros_like(r), ev["dX2_dt"].values], -1)
    e_z = np.stack([ev["dX1_dz"].values, -r, ev["dX2_dz"].values], -1)
    e_r = np.stack([ev["dX1_dr"].values, np.zeros_like(r), ev["dX2_dr"].values], -1)
    normal = np.cross(e_t, e_z)
    jac = np.einsum("...i,...i", e_r, normal)
    grad2 = np.einsum("...i,...i", normal, normal) / jac**2
    w = np.abs(jac) / np.abs(jac).sum(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(geo.grad_rho2, (w * grad2).sum(axis=(1, 2)), rtol=1e-8)
