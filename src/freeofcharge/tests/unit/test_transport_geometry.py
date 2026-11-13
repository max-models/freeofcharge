import numpy as np
import pytest

from freeofcharge import surface_geometry
from freeofcharge.equilibrium import effective_trapped_fraction


def _circular_torus(rho, major=3.0, minor=1.0, field=2.0, n_theta=64, n_zeta=8):
    """Concentric circles: R = R0 + a rho cos(theta), Z = a rho sin(theta)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
    rr, tt, _ = np.meshgrid(rho, theta, zeta, indexing="ij")
    r = major + minor * rr * np.cos(tt)
    z = minor * rr * np.sin(tt)
    jacobian = minor**2 * rr * r
    grad_rho2 = np.full_like(r, 1.0 / minor**2)
    strength = field * major / r
    return jacobian, grad_rho2, r, z, strength


def test_circular_torus_matches_the_analytic_averages():
    major, minor = 3.0, 1.0
    rho = np.array([0.1, 0.5, 0.9])
    geo = surface_geometry(rho, *_circular_torus(rho, major, minor))
    r = minor * rho
    np.testing.assert_allclose(
        geo.volume_derivative, 4.0 * np.pi**2 * major * minor**2 * rho, rtol=1e-12
    )
    np.testing.assert_allclose(geo.grad_rho2, 1.0 / minor**2, rtol=1e-12)
    np.testing.assert_allclose(geo.grad_rho, 1.0 / minor, rtol=1e-12)
    # <1/R^2> with the weight R: (1/2 pi R0) int dtheta / R = 1 / (R0 sqrt(R0^2 - r^2))
    np.testing.assert_allclose(
        geo.inv_R2, 1.0 / (major * np.sqrt(major**2 - r**2)), rtol=1e-10
    )
    np.testing.assert_allclose(geo.inv_R, 1.0 / major, rtol=1e-12)
    np.testing.assert_allclose(geo.grad_rho2_over_R2, geo.inv_R2 / minor**2)
    np.testing.assert_allclose(geo.R_in, major - r, rtol=1e-12)
    np.testing.assert_allclose(geo.R_out, major + r, rtol=1e-12)
    np.testing.assert_allclose(geo.elongation, 1.0, rtol=1e-2)
    np.testing.assert_allclose(geo.triangularity_upper, 0.0, atol=1e-12)
    # B = B0 R0 / R: <B^2> = B0^2 R0^2 <1/R^2>
    np.testing.assert_allclose(geo.B2, 4.0 * major**2 * geo.inv_R2, rtol=1e-12)
    assert np.all(np.diff(geo.trapped_fraction) > 0)
    assert geo.rho.attrs and geo.attrs["nfp"] == 1


def test_trapped_fraction_of_a_uniform_field_is_zero():
    b = np.ones((2, 16))
    w = np.full((2, 16), 1.0 / 16)
    np.testing.assert_allclose(effective_trapped_fraction(b, w), 0.0, atol=1e-12)


def test_optional_profiles_and_the_axis():
    rho = np.array([0.2, 0.4])
    geo = surface_geometry(
        rho,
        *_circular_torus(rho),
        rotational_transform=0.5,
        toroidal_current=[1.0, 2.0],
        toroidal_flux_edge=0.3,
    )
    np.testing.assert_allclose(geo.rotational_transform, 0.5)
    np.testing.assert_allclose(geo.toroidal_current, [1.0, 2.0])
    assert geo.attrs["toroidal_flux_edge"] == 0.3
    with pytest.raises(ValueError, match="rho > 0"):
        surface_geometry(np.array([0.0, 0.5]), *_circular_torus(np.array([0.0, 0.5])))
