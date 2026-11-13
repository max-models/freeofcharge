"""Shine-through heat loads on the wall (`freeofcharge.shine_through`)."""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll import ParticleArrays
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from simgeometry.objects.fourier_surface import FourierSurface
from simgeometry.objects.triangle_mesh import TriangleMesh

import freeofcharge as foc
from freeofcharge.deposition import Deposition
from freeofcharge.shine_through import _arc_length, shine_through_loads, wall_surface

E = ELEMENTARY_CHARGE


@pytest.fixture(scope="module")
def iter_like():
    """The tokamak of the stopping tutorial, a low density and a 1 MeV D beam."""
    eq = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
    plasma = foc.Plasma.parabolic(1e19, 1e4, ions={"D": (2, 1, 1.0)})
    origin, direction = foc.tangential_injection(5.3, 12.0)
    beam = foc.NeutralBeam.gaussian(
        origin,
        direction,
        1.0e6,
        16.5e6,
        width=0.3,
        divergence=0.005,
        markers=300,
        seed=1,
    )
    deposition = foc.deposit(beam, eq, plasma, step=0.03, births_per_marker=0)
    return eq, plasma, beam, deposition


def _plate(n: int, size: float, alpha: float, centre) -> TriangleMesh:
    """A square plate of ``n`` x ``n`` squares (two triangles each), tilted by ``alpha`` about y."""
    u = np.linspace(-size / 2, size / 2, n + 1)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    e_u = np.array([np.cos(alpha), 0.0, -np.sin(alpha)])
    e_v = np.array([0.0, 1.0, 0.0])
    vertices = (
        np.asarray(centre) + uu.ravel()[:, None] * e_u + vv.ravel()[:, None] * e_v
    )
    i, j = (k.ravel() for k in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
    c, d = b + 1, a + 1
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return TriangleMesh(vertices, faces, closed=False)


def _downward(xy, weights, energy=1e5, height=5.0) -> foc.NeutralBeam:
    """Deuterium markers at ``xy`` [m], height ``height``, flying along -z."""
    n = len(xy)
    speed = np.sqrt(2 * energy * E / (2 * ATOMIC_MASS))
    positions = np.column_stack([xy, np.full(n, height)])
    velocities = np.tile([0.0, 0.0, -speed], (n, 1))
    return foc.NeutralBeam(
        ParticleArrays(positions, velocities, np.asarray(weights, float)), 2
    )


def _unattenuated(beam: foc.NeutralBeam) -> Deposition:
    """The deposition of a beam in a plasma that does not stop it: all shines through."""
    eq = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 3.0, 1.2, wall_psi_n=None)
    plasma = foc.Plasma.parabolic(1e19, 1e3, ions={"D": (2, 1, 1.0)})
    deposition = foc.deposit(
        beam, eq, plasma, foc.ConstantStopping(0.0), step=0.05, births_per_marker=0
    )
    assert np.all(np.asarray(deposition.transmission) == 1.0)
    return deposition


# ----------------------------------------------------------------------
# Power conservation and the deposition's wall hits
# ----------------------------------------------------------------------


def test_the_shine_through_power_reaches_the_wall(iter_like):
    eq, _, beam, deposition = iter_like
    assert 0.3 < deposition.shine_through_fraction < 0.8
    loads = shine_through_loads(beam, deposition, eq)
    assert loads.attrs["missed_power"] == 0.0
    assert loads.attrs["wall_power"] == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )
    assert float(loads.power.sum()) == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )
    assert float(loads.hit_power.sum()) == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )
    # the map holds the same power: heat flux times the area of its faces
    area = loads.area.values
    assert np.nansum(loads.wall_load.values * area) == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )
    assert loads.attrs["peak_wall_load"] == pytest.approx(
        float(loads.wall_load.max()), rel=1e-15
    )
    assert loads.wall_load_map.dims == ("s", "phi")
    # on the far outboard wall, as the deposition's (step-resolved) wall hits
    hits = loads.hit_position.values
    assert np.all(np.hypot(hits[:, 0], hits[:, 1]) > 8.0)
    reference = np.asarray(deposition.wall_hits.positions)
    assert np.max(np.linalg.norm(hits - reference, axis=1)) < 1.2 * 0.03


def test_resampling_in_a_cone_keeps_the_power(iter_like):
    eq, _, beam, deposition = iter_like
    loads = shine_through_loads(
        beam, deposition, eq, divergence=(0.004, 0.002), resample=8, seed=3
    )
    assert loads.sizes["hit"] == 8 * beam.size
    total = loads.attrs["wall_power"] + loads.attrs["missed_power"]
    assert total == pytest.approx(deposition.shine_through_power, rel=1e-12)
    assert loads.attrs["missed_power"] == 0.0
    assert np.isclose(
        loads.attrs["shine_through_power"], deposition.shine_through_power
    )
    assert loads.attrs["injected_power"] == pytest.approx(16.5e6, rel=1e-12)


def test_an_open_panel_misses_part_of_the_beam():
    """Lines that pass beside a panel carry the missed power."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.5, 0.5, (2000, 2)) + [20.0, 0.0]
    beam = _downward(xy, np.full(2000, 1e19))
    deposition = _unattenuated(beam)
    loads = shine_through_loads(beam, deposition, _plate(4, 0.5, 0.0, (20.0, 0, 0)))
    total = loads.attrs["wall_power"] + loads.attrs["missed_power"]
    assert total == pytest.approx(deposition.shine_through_power, rel=1e-12)
    # a quarter of the area: about a quarter of the power
    assert loads.attrs["wall_power"] / total == pytest.approx(0.25, abs=0.03)
    assert "wall_load_map" not in loads


# ----------------------------------------------------------------------
# A pencil beam on a plate: P cos(alpha) / A, and the peak under refinement
# ----------------------------------------------------------------------


def test_lines_through_the_edges_of_an_open_plate_hit_it():
    """Lines aimed exactly at the plate's edges and vertices are traced again, not lost."""
    side, m, power, energy = 0.6, 600, 1e6, 1e5
    pitch = side / m
    grid = (np.arange(m) + 1.0) * pitch - side / 2
    x, y = np.meshgrid(grid, grid, indexing="ij")
    xy = np.column_stack([x.ravel(), y.ravel()]) + [20.0, 0.0]
    weights = np.full(len(xy), power / (energy * E) / len(xy))
    beam = _downward(xy, weights, energy)
    loads = shine_through_loads(
        beam, _unattenuated(beam), _plate(10, 2.0, 0.0, (20, 0, 0))
    )
    assert loads.attrs["wall_power"] == pytest.approx(power, rel=1e-12)
    assert loads.attrs["missed_power"] == 0.0


@pytest.mark.parametrize("alpha_deg", [0.0, 30.0, 60.0])
def test_a_uniform_beam_on_a_tilted_plate(alpha_deg):
    alpha = np.radians(alpha_deg)
    side, m, power, energy = 0.6, 600, 1e6, 1e5
    pitch = side / m
    grid = (np.arange(m) + 0.5) * pitch - side / 2 + 0.1234 * pitch
    x, y = np.meshgrid(grid, grid, indexing="ij")
    xy = np.column_stack([x.ravel(), y.ravel()]) + [20.0, 0.0]
    weights = np.full(len(xy), power / (energy * E) / len(xy))
    beam = _downward(xy, weights, energy)
    deposition = _unattenuated(beam)
    loads = shine_through_loads(beam, deposition, _plate(10, 2.0, alpha, (20, 0, 0)))
    assert loads.attrs["wall_power"] == pytest.approx(power, rel=1e-12)
    # faces well inside the footprint (x within 0.3, y within 0.3)
    inner = (np.abs(loads.X - 20.0) < 0.2) & (np.abs(loads.Y) < 0.2)
    assert int(inner.sum()) >= 8
    analytic = power * np.cos(alpha) / side**2
    np.testing.assert_allclose(loads.wall_load.values[inner], analytic, rtol=5e-3)
    np.testing.assert_allclose(loads.incidence.values[inner], alpha, atol=1e-9)
    np.testing.assert_allclose(loads.hit_incidence.values, alpha, atol=1e-9)
    assert loads.attrs["peak_wall_load"] == pytest.approx(analytic, rel=5e-3)


def test_the_peak_of_a_gaussian_beam_converges_with_the_mesh():
    r"""Face averages approach the peak $P\cos\alpha / (2\pi\sigma^2)$ at second order."""
    alpha, sigma, power, energy, size = np.radians(60.0), 0.1, 1e6, 1e5, 1.0
    # a lattice aligned with the plate's squares; each point split across the diagonal
    px, py = size * np.cos(alpha) / 256, size / 256
    gx = (np.arange(int(0.9 / px)) - int(0.45 / px) + 0.5) * px
    gy = (np.arange(int(0.9 / py)) - int(0.45 / py) + 0.5) * py
    x, y = np.meshgrid(gx, gy, indexing="ij")
    xy = np.column_stack([x.ravel(), y.ravel()])
    density = np.exp(-np.sum(xy**2, axis=1) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    shift = 1e-3 * np.array([px, -py])
    xy = np.concatenate([xy + shift, xy - shift]) + [20.0, 0.0]
    weights = np.tile(0.5 * density * px * py, 2) * power / (energy * E)
    beam = _downward(xy, weights, energy)
    deposition = _unattenuated(beam)
    peak = power * np.cos(alpha) / (2 * np.pi * sigma**2)
    errors = []
    for n in (4, 8, 16, 32):
        loads = shine_through_loads(
            beam, deposition, _plate(n, size, alpha, (20, 0, 0))
        )
        errors.append(loads.attrs["peak_wall_load"] / peak - 1.0)
    errors = np.array(errors)
    assert np.all(errors < 0.0)  # a face average stays below the peak
    ratios = errors[:-1] / errors[1:]
    assert np.all(ratios > 2.5)
    assert ratios[-1] == pytest.approx(4.0, abs=0.3)  # second order
    assert abs(errors[-1]) < 0.015


def test_a_cone_spreads_a_pencil_by_its_divergence():
    """One line spread over a cone of 2 mrad lands with a spread of 2 mrad times the distance."""
    beam = _downward(np.array([[20.0, 0.0]]), np.array([1e19]), height=10.0)
    deposition = _unattenuated(beam)
    loads = shine_through_loads(
        beam,
        deposition,
        _plate(2, 1.0, 0.0, (20.0, 0, 0)),
        divergence=0.002,
        resample=4000,
        seed=1,
    )
    hits = loads.hit_position.values
    assert loads.sizes["hit"] == 4000
    np.testing.assert_allclose(hits[:, :2].std(axis=0), 0.02, rtol=0.05)
    np.testing.assert_allclose(hits[:, :2].mean(axis=0), [20.0, 0.0], atol=2e-3)
    assert loads.attrs["wall_power"] == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )


# ----------------------------------------------------------------------
# Walls: revolved outlines, Fourier surfaces
# ----------------------------------------------------------------------


def test_arc_length_on_a_circle():
    theta = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    outline = np.column_stack([3 + np.cos(theta), np.sin(theta)])
    probe = np.array([0.0, 0.5, 1.5, 3.0, 4.5])
    s, perimeter = _arc_length(outline[::-1], 3 + np.cos(probe), np.sin(probe))
    assert perimeter == pytest.approx(2 * np.pi, rel=1e-5)
    # from the lowest point (theta = -pi/2), counter-clockwise
    np.testing.assert_allclose(s, np.mod(probe + np.pi / 2, 2 * np.pi), atol=1e-5)


def test_wall_surface_accepts_outlines_and_surfaces():
    theta = np.linspace(0, 2 * np.pi, 33)
    outline = np.column_stack([3 + np.cos(theta), np.sin(theta)])  # closed
    mesh, kept = wall_surface(outline, n_phi=16)
    assert kept.shape == (32, 2)
    assert mesh.n_faces == 2 * 32 * 16
    assert wall_surface(mesh) == (mesh, None)
    with pytest.raises(ValueError, match="no axisymmetric wall"):
        wall_surface(foc.Equilibrium.solovev(3.0, 0.6, 1.5, 3.0, wall_psi_n=None))
    with pytest.raises(ValueError, match="outline"):
        wall_surface(np.zeros((4, 3)))


def test_a_fourier_wall_gives_a_theta_phi_map():
    """A radial beam through a circular torus exits on the inboard midplane."""
    r_cos, z_sin = np.zeros((2, 1)), np.zeros((2, 1))
    r_cos[0, 0], r_cos[1, 0], z_sin[1, 0] = 3.0, 1.0, 1.0
    torus = FourierSurface(r_cos, z_sin, 1, resolution=(32, 32))
    speed = np.sqrt(2 * 1e5 * E / (2 * ATOMIC_MASS))
    n = 50
    rng = np.random.default_rng(2)
    positions = np.column_stack(
        [np.full(n, 6.0), rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)]
    )
    beam = foc.NeutralBeam(
        ParticleArrays(positions, np.tile([-speed, 0, 0], (n, 1)), np.full(n, 1e18)),
        2,
    )
    deposition = _unattenuated(beam)
    loads = shine_through_loads(beam, deposition, torus, map_bins=(16, 32))
    assert loads.attrs["wall_power"] == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )
    hits = loads.hit_position.values
    np.testing.assert_allclose(np.hypot(hits[:, 0], hits[:, 1]), 2.0, atol=0.01)
    assert hits[:, 0].max() < 2.01  # the exit, not the entry at R = 4
    assert loads.wall_load_map.dims == ("theta", "phi")
    hot = loads.wall_load_map.fillna(0.0)
    where = hot.argmax(...)
    assert float(hot.theta[where["theta"]]) == pytest.approx(np.pi, abs=0.25)
    assert (
        min(float(hot.phi[where["phi"]]), 2 * np.pi - float(hot.phi[where["phi"]]))
        < 0.2
    )


# ----------------------------------------------------------------------
# The transmission of the deposition, and errors
# ----------------------------------------------------------------------


def test_combined_depositions_keep_the_transmission(iter_like):
    eq, plasma, beam, deposition = iter_like
    other = foc.deposit(beam, eq, plasma, step=0.03, births_per_marker=0)
    both = Deposition.combine([deposition, other])
    assert np.asarray(both.transmission).shape == (2 * beam.size,)
    joined = foc.NeutralBeam.concatenate([beam, beam])
    loads = shine_through_loads(joined, both, eq)
    assert loads.attrs["wall_power"] == pytest.approx(
        both.shine_through_power, rel=1e-12
    )
    from dataclasses import replace

    assert (
        Deposition.combine([replace(deposition, transmission=None), other]).transmission
        is None
    )


def test_errors(iter_like):
    from dataclasses import replace

    eq, _, beam, deposition = iter_like
    with pytest.raises(ValueError, match="no transmission"):
        shine_through_loads(beam, replace(deposition, transmission=None), eq)
    with pytest.raises(ValueError, match="markers"):
        shine_through_loads(foc.NeutralBeam.concatenate([beam, beam]), deposition, eq)
    with pytest.raises(ValueError, match="at least 1"):
        shine_through_loads(beam, deposition, eq, resample=0)
    with pytest.raises(ValueError, match="divergence"):
        shine_through_loads(beam, deposition, eq, resample=4)


def test_max_length_short_of_the_wall_misses_it(iter_like):
    eq, _, beam, deposition = iter_like
    loads = shine_through_loads(beam, deposition, eq, max_length=1.0)
    assert loads.attrs["wall_power"] == 0.0
    assert loads.attrs["missed_power"] == pytest.approx(
        deposition.shine_through_power, rel=1e-12
    )
