from pathlib import Path

import numpy as np
import pytest

from freeofcharge import MagneticEquilibrium

pytest.importorskip("polarfeec")
pytest.importorskip("gvec")

from freeofcharge.gvec_equilibrium import GVECEquilibrium  # noqa: E402
from freeofcharge.polarfeec_equilibrium import PolarFEECEquilibrium  # noqa: E402

DATA = Path(__file__).parents[1] / "data"
FILES = (DATA / "parameter_ellipse_final.ini", DATA / "ellipse_State_final.dat")


@pytest.fixture(scope="module")
def ellipse():
    """The rotating ellipse of the GVEC tests, read by PolarFEEC.jl."""
    return PolarFEECEquilibrium.from_gvec_files(*FILES)


@pytest.fixture(scope="module")
def reference():
    """The same equilibrium through the GVEC adapter."""
    return GVECEquilibrium.from_files(*FILES)


@pytest.fixture(scope="module")
def points(reference):
    """Random points in the box, clear of the last closed surface."""
    lower, upper = reference.bounds
    points = lower + (upper - lower) * np.random.default_rng(0).random((3000, 3))
    rho = reference.rho_at(points)
    return points[np.abs(rho - 1.0) > 1e-6]


def test_is_a_magnetic_equilibrium(ellipse, reference):
    assert isinstance(ellipse, MagneticEquilibrium)
    assert not ellipse.has_wall
    for mine, theirs in zip(ellipse.bounds, reference.bounds, strict=True):
        np.testing.assert_allclose(mine, theirs, rtol=1e-3)


def test_rho_and_inside_match_gvec(ellipse, reference, points):
    inside = reference.inside_at(points)
    assert 0 < inside.sum() < len(points)
    np.testing.assert_array_equal(ellipse.inside_at(points), inside)
    np.testing.assert_allclose(
        ellipse.rho_at(points[inside]), reference.rho_at(points[inside]), atol=1e-8
    )
    assert np.all(np.isinf(ellipse.rho_at(points[~inside])))


def test_field_matches_gvec(ellipse, reference, points):
    inside = points[reference.inside_at(points)]
    np.testing.assert_allclose(
        ellipse.field_at(inside), reference.field_at(inside), rtol=1e-7, atol=1e-10
    )
    outside = points[~reference.inside_at(points)]
    assert np.all(np.isnan(ellipse.field_at(outside)))


def test_leading_shapes_are_kept(ellipse, points):
    grid = points[:12].reshape(3, 4, 3)
    assert ellipse.rho_at(grid).shape == (3, 4)
    assert ellipse.field_at(grid).shape == (3, 4, 3)
    assert ellipse.rho_at(np.zeros((0, 3))).shape == (0,)


def test_flux_surfaces_match_gvec(ellipse, reference):
    edges = np.linspace(0.0, 1.0, 6)
    mine, theirs = ellipse.flux_surfaces(edges), reference.flux_surfaces(edges)
    np.testing.assert_allclose(mine.volume, theirs.volume, rtol=1e-6)
    np.testing.assert_allclose(mine.area, theirs.area, rtol=1e-6)
    np.testing.assert_allclose(mine.major_radius, theirs.major_radius, rtol=1e-8)
    np.testing.assert_allclose(
        mine.inverse_aspect_ratio, theirs.inverse_aspect_ratio, rtol=1e-6
    )
    np.testing.assert_allclose(mine.safety_factor, theirs.safety_factor, rtol=1e-8)
    np.testing.assert_allclose(mine.field_strength, theirs.field_strength, rtol=1e-7)
    np.testing.assert_allclose(
        mine.trapped_fraction, theirs.trapped_fraction, rtol=1e-6
    )
