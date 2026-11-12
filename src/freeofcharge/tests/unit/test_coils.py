import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from freeofcharge import MagneticEquilibrium, NeutralBeam
from freeofcharge.coils import (
    MU_0,
    CoilField,
    CoilSet,
    PlasmaBoundary,
    blocked_fraction,
    injection_access,
    normal_field_error,
)
from freeofcharge.equilibrium import Equilibrium

DATA = Path(__file__).parents[1] / "data" / "coilstellaration"


def loop(radius=1.0, current=1.0e6, n_points=512, z=0.0):
    """One horizontal circular loop around the z axis, counter-clockwise seen from above."""
    s = 2 * np.pi * np.arange(n_points) / n_points
    points = np.stack([radius * np.cos(s), radius * np.sin(s), np.full_like(s, z)], -1)
    return CoilSet(points[None], [current])


def loop_field(radius, current, r, z):
    """The closed-form field (B_R, B_Z) of a circular loop (elliptic integrals)."""
    m = 4 * radius * r / ((radius + r) ** 2 + z**2)
    k, e = ellipk(m), ellipe(m)
    common = MU_0 * current / (2 * np.pi * np.sqrt((radius + r) ** 2 + z**2))
    denom = (radius - r) ** 2 + z**2
    b_z = common * (k + (radius**2 - r**2 - z**2) / denom * e)
    b_r = common * z / r * (-k + (radius**2 + r**2 + z**2) / denom * e)
    return b_r, b_z


def test_loop_on_axis_and_off_axis():
    coils = loop()
    z = np.array([0.0, 0.5, 2.0])
    on_axis = coils.field_at(np.stack([0 * z, 0 * z, z], -1))
    exact = MU_0 * 1e6 / (2 * (1 + z**2) ** 1.5)
    np.testing.assert_allclose(on_axis[:, 2], exact, rtol=1e-4)
    np.testing.assert_allclose(on_axis[:, :2], 0.0, atol=1e-12)
    r, z = np.array([0.3, 0.7, 1.6]), np.array([0.2, -0.4, 0.5])
    field = coils.field_at(np.stack([r, 0 * r, z], -1))
    b_r, b_z = loop_field(1.0, 1e6, r, z)
    np.testing.assert_allclose(field[:, 0], b_r, rtol=1e-4)
    np.testing.assert_allclose(field[:, 2], b_z, rtol=1e-4)
    # finite on the filament: a segment adds nothing at its own end points
    assert np.all(np.isfinite(coils.field_at(coils.filaments[0, :3])))


def test_polyline_error_is_second_order():
    point = np.array([[0.4, 0.1, 0.3]])
    exact = loop(n_points=4096).field_at(point)
    errors = [
        np.linalg.norm(loop(n_points=n).field_at(point) - exact) for n in (32, 64)
    ]
    assert 3.5 < errors[0] / errors[1] < 4.5


def test_toroidal_field_of_planar_coils():
    coils = CoilSet.planar_circular(36, 5.0, 1.5, 1.0e5, n_points=128)
    assert coils.n_coils == 36 and coils.nfp == 36
    np.testing.assert_allclose(coils.lengths, 2 * np.pi * 1.5, rtol=1e-3)
    # midway between two coils, inside: B = mu0 N I / (2 pi R) along +phi, ripple small
    phi = np.pi / 36
    r = np.array([4.5, 5.0, 5.5])
    points = np.stack([r * np.cos(phi), r * np.sin(phi), 0 * r + 0.2], -1)
    field = coils.field_at(points)
    b_phi = -np.sin(phi) * field[:, 0] + np.cos(phi) * field[:, 1]
    np.testing.assert_allclose(b_phi, MU_0 * 36 * 1e5 / (2 * np.pi * r), rtol=1e-2)
    # outside the coils the field vanishes (Ampère)
    outside = coils.field_at([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.all(np.abs(outside) < 1e-3 * b_phi.min())


def test_distance_and_validation():
    coils = loop()
    np.testing.assert_allclose(
        coils.distance_at([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]]), [1.0, 0.5], rtol=1e-4
    )
    with pytest.raises(ValueError, match="filaments"):
        CoilSet(np.zeros((2, 2, 3)), [1.0, 1.0])
    with pytest.raises(ValueError, match="one current per coil"):
        CoilSet(np.zeros((2, 4, 3)), [1.0])


def test_first_hit_and_clearance_of_a_loop():
    coils = loop()
    starts = np.array(
        [
            [-3.0, 0.0, 0.0],  # along x through the loop: hits at x = -1 - r
            [-3.0, 0.0, 0.5],  # passes 0.5 above: misses a tube of 0.1
            [0.0, 0.0, 3.0],  # down the axis: misses
            [1.0, 0.0, 0.05],  # starts inside the tube
            [1.0, 0.0, 0.0],  # starts on the filament, along it
        ]
    )
    directions = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0]], dtype=float
    )
    distance, coil = coils.first_hit(starts, directions, "10 cm")
    np.testing.assert_allclose(distance[0], 3.0 - 1.0 - 0.1, atol=1e-3)
    assert np.isinf(distance[1]) and np.isinf(distance[2])
    np.testing.assert_allclose(distance[3:], 0.0)
    np.testing.assert_array_equal(coil, [0, -1, -1, 0, 0])
    gaps = coils.clearance(starts[:3], directions[:3], [6.0, 6.0, 6.0])
    np.testing.assert_allclose(gaps, [0.0, 0.5, 1.0], atol=1e-3)
    # a short segment ends before the loop
    np.testing.assert_allclose(
        coils.clearance(starts[:1], directions[:1], 1.0), 1.0, atol=1e-3
    )


def test_first_hit_matches_sampling():
    rng = np.random.default_rng(3)
    coils = CoilSet.planar_circular(6, 3.0, 1.0, 1.0, n_points=48)
    starts = rng.uniform(-5, 5, size=(150, 3))
    directions = rng.uniform(-1, 1, size=(150, 3)) * [1, 1, 0.3] - 0.2 * starts
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distance, _ = coils.first_hit(starts, directions, 0.2)
    s = np.linspace(0, 12, 2401)
    near = (
        coils.distance_at(starts[:, None] + s[None, :, None] * directions[:, None])
        <= 0.2
    )
    sampled = np.where(near.any(1), s[np.argmax(near, axis=1)], np.inf)
    hit = np.isfinite(sampled) & (sampled < 12)
    assert hit.sum() > 10
    np.testing.assert_allclose(distance[hit], sampled[hit], atol=6e-3)
    assert np.all(distance[~hit] > 12 - 6e-3)


@pytest.fixture(scope="module")
def sample():
    coilset = json.loads((DATA / "coilstellaration_coilset.json").read_text())
    boundary = json.loads((DATA / "coilstellaration_boundary.json").read_text())
    meta = json.loads((DATA / "coilstellaration_sample.json").read_text())
    return coilset, boundary, meta


def test_field_error_of_the_coilstellaration_sample(sample):
    plasma_plots = pytest.importorskip("plasma_plots")
    coilset, boundary, meta = sample
    coils = CoilSet.from_coilset(coilset, n_points=128)
    assert coils.n_coils == 2 * 5 * 3 and coils.nfp == 3
    surface = plasma_plots.from_boundary(
        boundary, theta=64, zeta=np.linspace(0, 2 * np.pi, 144, endpoint=False)
    )
    error = normal_field_error(coils, surface)
    assert error.dims == ("theta", "zeta")
    # the dataset's own metric, from DESC's field on its own grid
    expected = meta["desc_metrics/normalized_field_error/mean"]
    assert abs(float(error.mean()) / expected - 1) < 0.15
    # plain arrays: a coarser surface, the same values as through a Dataset
    coarse = surface.isel(theta=slice(None, None, 4), zeta=slice(None, None, 4))
    points = np.stack([coarse[a].isel(rho=0).values for a in "XYZ"], -1)
    np.testing.assert_allclose(
        normal_field_error(coils, points), normal_field_error(coils, coarse).values
    )


@pytest.fixture(scope="module")
def tokamak():
    equilibrium = Equilibrium.solovev(3.0, 1.0, toroidal_field=2.0, q_axis=1.5)
    coils = CoilSet.planar_circular(16, 3.0, 1.8, 1.0, n_points=96)
    return equilibrium, coils


def test_coil_field_joins_the_two_regions(tokamak):
    equilibrium, coils = tokamak
    # scale the coils to the vacuum field 2 T at R = 3 m
    scale = 2.0 / float(coils.field_at([[3.0, 0.0, 0.0]])[0, 1])
    field = CoilField(equilibrium, coils, current_scale=scale)
    assert isinstance(field, MagneticEquilibrium)
    assert field.has_wall == equilibrium.has_wall
    np.testing.assert_array_equal(field.bounds[0], equilibrium.bounds[0])
    points = np.array([[3.2, 0.0, 0.1], [3.0, 0.0, 1.5], [0.0, 3.0, 1.4]])
    inside = equilibrium.inside_at(points)
    np.testing.assert_array_equal(inside, [True, False, False])
    np.testing.assert_array_equal(field.rho_at(points), equilibrium.rho_at(points))
    np.testing.assert_array_equal(field.inside_at(points), inside)
    np.testing.assert_array_equal(
        field.inside_wall_at(points), equilibrium.inside_wall_at(points)
    )
    combined = field.field_at(points.reshape(1, 3, 3)).reshape(3, 3)
    np.testing.assert_allclose(combined[0], equilibrium.field_at(points[:1])[0])
    np.testing.assert_allclose(combined[1:], scale * coils.field_at(points[1:]))
    everywhere = CoilField(equilibrium, coils, inside="coils", current_scale=scale)
    np.testing.assert_allclose(
        everywhere.field_at(points), scale * coils.field_at(points)
    )
    # the toroidal fields agree to the ripple and the poloidal field (both missing here)
    edge = np.array([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.all(field.jump(edge) < 0.5)
    assert field.flux_surfaces(np.linspace(0, 1, 5)).rho.shape == (4,)
    with pytest.raises(ValueError, match="inside"):
        CoilField(equilibrium, coils, inside="nowhere")


def test_blocked_fraction(tokamak):
    equilibrium, coils = tokamak
    # markers along -x at y = 0 (inside a coil plane) and between two coils
    phi = np.pi / 16
    starts = np.array(
        [[6.0, 0.0, 0.0], [6.0 * np.cos(phi), 6.0 * np.sin(phi), 0.0], [6.0, 0.0, 5.0]]
    )
    directions = np.array(
        [[-1.0, 0.0, 0.0], [-np.cos(phi), -np.sin(phi), 0.0], [-1, 0, 0]]
    )
    from plasmacoll import ParticleArrays

    beam = NeutralBeam(
        ParticleArrays(starts, 1e6 * directions, np.array([1.0, 1.0, 2.0])), 2
    )
    result = blocked_fraction(beam, coils, equilibrium, 0.1, max_length=8.0)
    np.testing.assert_array_equal(result["blocked"], [True, False, False])
    np.testing.assert_array_equal(result["coil"][1:], [-1, -1])
    np.testing.assert_allclose(result["coil_distance"][0], 6.0 - 4.8 - 0.1, atol=1e-3)
    assert np.isinf(result["plasma_distance"][2])
    entry = starts[1] + result["plasma_distance"][1] * directions[1]
    assert equilibrium.inside_at(entry[None])[0]
    assert not equilibrium.inside_at((entry - 0.01 * directions[1])[None])[0]
    assert result["fraction"] == pytest.approx(0.25)


def test_injection_access(tokamak):
    equilibrium, coils = tokamak
    phis = np.array([0.0, np.pi / 16])
    access = injection_access(
        coils,
        equilibrium,
        [3.0],
        phis,
        [0.0, 0.5],
        start_distance="6 m",
        coil_radius=0.1,
        beam_radius=0.1,
    )
    assert access.admissible.dims == (
        "tangency_radius",
        "toroidal_angle",
        "vertical_angle",
    )
    # whether the axis passes the outer coil legs depends on the toroidal angle
    assert bool(access.admissible.any()) and not bool(access.admissible.all())
    assert np.all(np.isfinite(access.plasma_distance))
    assert np.all(access.plasma_distance < 6.0)
    # each clearance against sampled points of the axis up to the plasma
    for index in np.ndindex(access.clearance.shape):
        r, phi, alpha = (
            access[c].values[i]
            for c, i in zip(access.clearance.dims, index, strict=True)
        )
        d = np.array(
            [-np.cos(alpha) * np.sin(phi), np.cos(alpha) * np.cos(phi), np.sin(alpha)]
        )
        start = np.array([r * np.cos(phi), r * np.sin(phi), 0.0]) - 6.0 * d
        s = np.linspace(0, float(access.plasma_distance.values[index]), 2001)
        sampled = coils.distance_at(start + s[:, None] * d).min() - 0.1
        assert abs(sampled - access.clearance.values[index]) < 2e-3
    # above the plasma: no entry, not admissible
    above = injection_access(
        coils,
        equilibrium,
        [3.0],
        [0.1],
        start_distance=6.0,
        coil_radius=0.1,
        height=2.5,
    )
    assert np.isinf(above.plasma_distance).all() and not above.admissible.any()
    # wider coils than the gap close every line
    closed = injection_access(
        coils, equilibrium, [3.0], phis, start_distance=6.0, coil_radius=1.5
    )
    assert not bool(closed.admissible.any())
    with pytest.raises(ValueError, match="sense"):
        injection_access(
            coils,
            equilibrium,
            [3.0],
            [0.0],
            start_distance=6.0,
            coil_radius=0.1,
            sense=0,
        )


def test_plasma_boundary_of_a_torus_and_the_sample(sample):
    # a circular torus R0 = 3, a = 1: r_cos[0, N] = 3, r_cos[1, N] = z_sin[1, N] = 1
    r_cos, z_sin = np.zeros((2, 3)), np.zeros((2, 3))
    r_cos[0, 1], r_cos[1, 1], z_sin[1, 1] = 3.0, 1.0, 1.0
    torus = PlasmaBoundary(r_cos, z_sin, 2)
    rng = np.random.default_rng(0)
    points = rng.uniform(-4.5, 4.5, size=(500, 3)) * [1, 1, 0.3]
    distance = np.hypot(np.hypot(points[:, 0], points[:, 1]) - 3.0, points[:, 2])
    keep = np.abs(distance - 1.0) > 1e-3
    np.testing.assert_array_equal(torus.inside_at(points)[keep], (distance < 1.0)[keep])
    np.testing.assert_allclose(torus.bounds[1], [4.0, 4.0, 1.0], rtol=1e-3)
    with pytest.raises(ValueError, match="r_cos"):
        PlasmaBoundary(np.zeros((2, 2)), np.zeros((2, 2)), 1)
    # the sample: the axis inside, far points outside, and points on the surface
    _, boundary, _ = sample
    plasma = PlasmaBoundary.from_boundary(json.dumps(boundary))
    r, z = plasma.rz(np.array([0.0, 2.0]), 0.4)
    centre = plasma.rz(np.linspace(0, 2 * np.pi, 64), 0.4)
    inner = np.array([np.mean(centre[0]), np.mean(centre[1])])
    on = np.stack([r * np.cos(0.4), r * np.sin(0.4), z], -1)
    axis = np.array([inner[0] * np.cos(0.4), inner[0] * np.sin(0.4), inner[1]])
    just_in = axis + 0.99 * (on - axis)
    just_out = axis + 1.01 * (on - axis)
    flags = plasma.inside_at(np.concatenate([just_in, just_out, [[0.0, 0.0, 0.0]]]))
    np.testing.assert_array_equal(flags, [True, True, False, False, False])
