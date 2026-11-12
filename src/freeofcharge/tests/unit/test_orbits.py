"""Guiding-centre orbits with collisions (`freeofcharge.orbits`) and their kernel."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

import freeofcharge as foc
from freeofcharge import orbits
from freeofcharge.kernels.guiding_centre_push import guiding_centre_push_numpy
from freeofcharge.kernels.guiding_centre_push import kernel as push_kernel

E = ELEMENTARY_CHARGE
PYCCEL = pytest.mark.skipif(
    importlib.util.find_spec("pyccel") is None, reason="Pyccel is not installed"
)


@pytest.fixture(scope="module")
def tokamak():
    """A Solov'ev tokamak, its gridded field (covering the wall) and the wall."""
    eq = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 3.0, 1.2, wall_psi_n=1.3)
    field = orbits.orbit_field(eq, (96, 8, 96), box=((2.0, 3.9), (-1.25, 1.25)))
    wall = orbits.revolved_wall(eq.wall.vertices, n_phi=96)
    return eq, field, wall


def _markers(field, positions, pitch, energy=55e3, mass=ATOMIC_MASS):
    """Guiding centres from particles at ``positions`` with ``pitch`` and ``energy`` [eV]."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    pitch = np.broadcast_to(np.asarray(pitch, dtype=float), positions.shape[:1])
    b = np.asarray(field.field_at(positions))
    unit = b / np.linalg.norm(b, axis=1)[:, None]
    e1 = np.cross(unit, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    speed = np.sqrt(2 * energy * E / mass)
    velocity = speed * (pitch[:, None] * unit + np.sqrt(1 - pitch**2)[:, None] * e1)
    weights = np.ones(len(positions))
    return orbits.guiding_centres(positions, velocity, weights, field, mass), velocity


def _beam_deposition(eq, plasma, markers, rho_edges, energy=55e3):
    origin, direction = foc.tangential_injection(2.9, 6.0)
    beam = foc.NeutralBeam.gaussian(
        origin,
        direction,
        energy,
        1e6,
        mass_number=1,
        width=0.1,
        divergence=0.005,
        markers=markers,
        seed=1,
    )
    return foc.deposit(
        beam, eq, plasma, step=0.01, births_per_marker=1, rho_edges=rho_edges, seed=2
    )


# ----------------------------------------------------------------------
# The gridded field and the kernel
# ----------------------------------------------------------------------


def test_field_jacobian_matches_finite_differences(tokamak):
    _, field, _ = tokamak
    rng = np.random.default_rng(0)
    points = np.stack(
        [
            rng.uniform(2.7, 3.4, 20),
            rng.uniform(-0.5, 0.5, 20),
            rng.uniform(-0.4, 0.4, 20),
        ],
        -1,
    )
    b, jacobian = field.field_jacobian_at(points)
    assert np.allclose(b, np.asarray(field.field_at(points)), atol=1e-13)
    h = 1e-5
    numeric = np.stack(
        [
            (
                np.asarray(field.field_at(points + h * e))
                - np.asarray(field.field_at(points - h * e))
            )
            / (2 * h)
            for e in np.eye(3)
        ],
        -1,
    )
    assert np.allclose(jacobian, numeric, atol=1e-8)
    # divergence free to the interpolation error
    assert np.max(np.abs(np.trace(jacobian, axis1=1, axis2=2))) < 1e-6
    b_off, j_off = field.field_jacobian_at(np.array([[100.0, 0.0, 0.0]]))
    assert np.isnan(b_off).all() and np.isnan(j_off).all()


def test_spline_arrays_need_cubic_splines(tokamak):
    eq, field, _ = tokamak
    linear = foc.GriddedEquilibrium(eq, (16, 4, 16), order=1)
    with pytest.raises(ValueError, match="cubic"):
        linear.spline_arrays  # noqa: B018
    with pytest.raises(ValueError, match="shape"):
        field.spline_coefficients(np.zeros((2, 2, 2)))
    assert orbits.orbit_field(field) is field
    assert orbits.orbit_field(linear).order == 3


def _kernel_arguments(field, n, wall=None, seed=0):
    rng = np.random.default_rng(seed)
    coefficients, grid, shape = field.spline_arrays
    radius = rng.uniform(3.1, 3.5, n)
    angle = rng.uniform(0, 2 * np.pi, n)
    state = np.stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            rng.uniform(-0.1, 0.1, n),
            rng.uniform(-2e6, 2e6, n),
        ]
    )
    mu = rng.uniform(0.0, 1e12, n) / 3.0
    edges = np.linspace(0.0, 1.0, 6)
    return (
        state,
        mu,
        np.full(n, 3e-6),
        np.full(n, 3e-8),
        coefficients.reshape(-1).copy(),
        grid,
        shape,
        E / ATOMIC_MASS,
        0.5,
        np.zeros(1) if wall is None else wall.reshape(-1).copy(),
        0.02,
        np.inf,
        edges,
        np.zeros((3, n)),
        np.zeros(n),
        np.zeros(n, dtype=np.int64),
        np.zeros((5, n)),
        np.zeros((5, n)),
        np.zeros((8, 3, n)),
        np.zeros(n, dtype=np.int64),
    )


@PYCCEL
def test_compiled_kernel_matches_numpy(tokamak):
    _, field, wall = tokamak
    clearance = orbits.wall_clearance(wall, field)
    compiled = _kernel_arguments(field, 40, clearance)
    reference = _kernel_arguments(field, 40, clearance)
    push_kernel(*compiled)
    guiding_centre_push_numpy.guiding_centre_push(*reference)
    assert push_kernel.selected() == "pyccel"
    for a, b in zip(compiled, reference, strict=True):
        if isinstance(a, np.ndarray):
            assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
    status = compiled[15]
    # some markers stop at rho^2 = 0.5, the others run their time
    assert set(np.unique(status)) <= {0, 1}
    assert np.any(status == 1) and np.any(status == 0)
    assert np.allclose(compiled[14][status == 0], 3e-6)
    assert np.all(compiled[19][status == 0] >= 100)
    # the shells hold the time pushed
    assert np.allclose(compiled[16].sum(axis=0), compiled[14])


def test_numpy_kernel_stops_off_the_grid(tokamak):
    _, field, _ = tokamak
    args = list(_kernel_arguments(field, 3))
    args[0][0] = [3.795, 3.795, 50.0]  # at the grid's edge, moving out, far off
    args[0][1] = 0.0
    args[0][2] = [0.0, 0.0, 0.0]
    args[8] = np.inf
    guiding_centre_push_numpy.guiding_centre_push(*args)
    assert args[15][2] == 2


# ----------------------------------------------------------------------
# Guiding centres and conservation
# ----------------------------------------------------------------------


def test_guiding_centre_is_a_larmor_radius_from_the_particle(tokamak):
    _, field, _ = tokamak
    markers, velocity = _markers(field, [[3.3, 0.0, 0.1]], 0.3)
    b = np.linalg.norm(np.asarray(field.field_at(np.array([[3.3, 0.0, 0.1]]))))
    v_perp = np.sqrt(np.sum(velocity**2) - markers.v_parallel[0] ** 2)
    larmor = ATOMIC_MASS * v_perp / (E * b)
    shift = np.linalg.norm(markers.positions[0] - [3.3, 0.0, 0.1])
    assert shift == pytest.approx(larmor, rel=1e-12)
    assert markers.energy(field)[0] == pytest.approx(55e3 * E, rel=1e-12)
    assert markers.charge_over_mass == pytest.approx(E / ATOMIC_MASS)


def test_energy_and_toroidal_momentum_are_conserved(tokamak):
    eq, field, _ = tokamak
    pitch = np.array([-0.9, -0.3, 0.1, 0.5, 0.95])
    positions = np.tile([3.3, 0.0, 0.0], (5, 1))
    markers, _ = _markers(field, positions, pitch)
    traces = orbits.trace_orbits(markers, field, 5e-4, samples=101, step=0.1)
    energy = traces.energy.values
    assert np.max(np.abs(energy / energy[0] - 1)) < 1e-6
    r, z = traces.R.values, traces.z.values
    points = np.stack([traces.x.values, traces.y.values, z], -1)
    b = np.asarray(field.field_at(points))
    b_phi = (-traces.y.values * b[..., 0] + traces.x.values * b[..., 1]) / r
    p_phi = ATOMIC_MASS * r * traces.v_parallel.values * b_phi / traces.B.values
    p_phi += E * eq.psi(r, z)
    span = E * (eq.psi_boundary - eq.psi_axis)
    assert np.max(np.abs(p_phi - p_phi[0])) / span < 1e-5
    # trapped (pitch 0.1) and passing (0.95) orbits both stay inside
    assert np.all(traces.rho.values < 1.0)
    assert np.any(traces.v_parallel.values[:, 2] < 0)
    assert np.all(traces.v_parallel.values[:, 4] > 0)


def test_passing_orbit_width():
    r0, a, b0, q0 = 10.0, 1.0, 2.0, 1.5
    eq = foc.Equilibrium.solovev(r0, a, 1.0, b0, q0, wall_psi_n=None)
    field = orbits.orbit_field(eq, (160, 4, 160))
    surfaces = eq.flux_surfaces(np.linspace(0, 1, 21))
    markers, _ = _markers(field, [[r0 + 0.5, 0.0, 0.0]], -0.9)
    traces = orbits.trace_orbits(markers, field, 1.5e-4, samples=6001, step=0.05)
    r, z, v_par = (traces[k].values[:, 0] for k in ("R", "z", "v_parallel"))
    u = (r**2 - r0**2) / (2 * r0)  # a flux label on the midplane
    k = np.nonzero(np.sign(z[1:]) != np.sign(z[:-1]))[0]
    t = z[k] / (z[k] - z[k + 1])
    u_cross = u[k] + t * (u[k + 1] - u[k])
    v_cross = v_par[k] + t * (v_par[k + 1] - v_par[k])
    r_cross = r[k] + t * (r[k + 1] - r[k])
    b_cross = np.linalg.norm(
        np.asarray(field.field_at(np.stack([r_cross, 0 * r_cross, 0 * r_cross], -1))),
        axis=-1,
    )
    out, inner = u_cross > 0, u_cross < 0
    width = u_cross[out].mean() + u_cross[inner].mean()
    # exact: e psi + m F v_par / B is conserved, psi = c R0^2 u^2 on the midplane
    c = b0 / (2 * r0**2 * q0)
    r_mid = 0.5 * (u_cross[out].mean() - u_cross[inner].mean())
    d_psi = (ATOMIC_MASS * b0 * r0 / E) * (
        v_cross[inner].mean() / b_cross[inner].mean()
        - v_cross[out].mean() / b_cross[out].mean()
    )
    assert width == pytest.approx(d_psi / (2 * c * r0**2 * r_mid), rel=1e-3)
    # the estimate 2 q rho_par, to the aspect-ratio corrections
    q = np.interp(float(eq.rho(r0 + r_mid, 0.0)), surfaces.rho, surfaces.safety_factor)
    estimate = 2 * q * ATOMIC_MASS * np.abs(v_cross).mean() / (E * b0)
    assert abs(width) == pytest.approx(estimate, rel=0.15)


# ----------------------------------------------------------------------
# Collisions
# ----------------------------------------------------------------------


def test_collisions_follow_the_drag_of_slow_down():
    rng = np.random.default_rng(1)
    a_e, a_i, v0 = 50.0, 2.0e20, 3.0e6
    speed = np.full(4, v0)
    total = np.zeros(4)
    for _ in range(100):
        new, pitch, to_e, to_i = orbits.collide(
            speed,
            np.zeros(4),
            1e-4,
            a_e,
            a_i,
            0.0,
            1e3,
            1e3,
            ATOMIC_MASS,
            rng,
            energy_diffusion=False,
        )
        assert np.allclose(to_e + to_i, 0.5 * ATOMIC_MASS * (speed**2 - new**2))
        total += to_i
        speed = new
    # v^3 + a_i / a_e decays as exp(-3 a_e t)
    t = 100 * 1e-4
    expected = np.cbrt((v0**3 + a_i / a_e) * np.exp(-3 * a_e * t) - a_i / a_e)
    assert speed == pytest.approx(np.full(4, expected), rel=1e-12)
    # the ions' share of the energy: m int a_i / v dt
    from scipy.integrate import solve_ivp

    solution = solve_ivp(
        lambda _, y: [-a_e * y[0] - a_i / y[0] ** 2, ATOMIC_MASS * a_i / y[0]],
        (0, t),
        [v0, 0.0],
        rtol=1e-10,
        atol=1e-12,
    )
    assert total[0] == pytest.approx(solution.y[1, -1], rel=1e-3)


def test_pitch_scattering_moments():
    rng = np.random.default_rng(2)
    n, nu, dt = 20000, 100.0, 1e-4
    pitch = np.full(n, 0.6)
    speed = np.full(n, 1e6)
    for _ in range(50):
        speed, pitch, _, _ = orbits.collide(
            speed,
            pitch,
            dt,
            1e-30,
            0.0,
            nu * 1e18,
            0.0,
            0.0,
            ATOMIC_MASS,
            rng,
            energy_diffusion=False,
        )
    # d<xi>/dt = -nu xi, d<xi^2>/dt = nu (1 - 3 xi^2) (Lorentz)
    t = 50 * dt
    assert pitch.mean() == pytest.approx(0.6 * np.exp(-nu * t), abs=0.01)
    second = 1 / 3 + (0.36 - 1 / 3) * np.exp(-3 * nu * t)
    assert np.mean(pitch**2) == pytest.approx(second, abs=0.01)


def test_energy_diffusion_relaxes_to_the_electron_temperature():
    rng = np.random.default_rng(3)
    te = 2e3
    speed = np.full(20000, np.sqrt(2 * 10 * te * E / ATOMIC_MASS))
    for _ in range(400):
        speed, _, _, _ = orbits.collide(
            speed, np.zeros_like(speed), 0.01, 1.0, 0.0, 0.0, te, te, ATOMIC_MASS, rng
        )
    energy = 0.5 * ATOMIC_MASS * speed**2 / E
    assert energy.mean() == pytest.approx(1.5 * te, rel=0.03)


# ----------------------------------------------------------------------
# Following births
# ----------------------------------------------------------------------


def test_orbits_reproduce_the_local_slowing_down():
    """High field, small orbits: the orbit-following moments are slow_down's."""
    eq = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 10.0, 1.2, wall_psi_n=None)
    rho_edges = np.linspace(0, 1, 6)
    surfaces = eq.flux_surfaces(rho_edges)
    field = orbits.orbit_field(eq, (96, 4, 96))
    plasma = foc.Plasma.parabolic(1e20, 600.0, ions={"H": (1, 1, 1.0)})
    deposition = _beam_deposition(eq, plasma, 120, rho_edges, energy=30e3)
    local = foc.slow_down(
        deposition, plasma, surfaces, orbit_average=False, thermal_factor=2.0
    )
    result = orbits.follow_orbits(
        deposition,
        field,
        plasma,
        surfaces=surfaces,
        step=0.2,
        thermal_factor=2.0,
        energy_diffusion=False,
        seed=1,
    )
    assert result.fate_fractions()["thermalised"] == 1.0
    assert abs(result.power_balance) < 1e-12
    fractions = result.power_fractions()
    assert fractions["electrons"] == pytest.approx(
        local.electron_heating.sum() / local.birth_power, abs=0.01
    )
    assert fractions["ions"] == pytest.approx(
        local.ion_heating.sum() / local.birth_power, abs=0.01
    )
    assert abs(fractions["numerical"]) < 1e-6
    assert result.stored_energy == pytest.approx(local.stored_energy, rel=0.03)
    assert result.particles.sum() == pytest.approx(
        np.sum(local.fast_ion_density * surfaces.volume), rel=0.03
    )
    profiles = result.to_dataset()
    assert profiles.fast_ion_density.attrs["units"] == "1/m^3"
    assert profiles.attrs["stored_energy"] == pytest.approx(result.stored_energy)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _FakeDeposition:
    """The parts of a `Deposition` that `follow_orbits` reads."""

    def __init__(self, positions, velocities, rho_edges, mass_number=1):
        from plasmacoll import ParticleArrays

        positions = np.asarray(positions, dtype=float)
        self.births = ParticleArrays(
            positions, np.asarray(velocities, dtype=float), np.ones(len(positions))
        )
        self.mass_number = mass_number
        self.rho_edges = np.asarray(rho_edges, dtype=float)


def test_wall_losses_and_loads(tokamak):
    eq, field, wall = tokamak
    # trapped ions of 1.5 MeV born near the edge: their banana orbits reach the wall
    rng = np.random.default_rng(4)
    n = 40
    radius = rng.uniform(3.45, 3.6, n)
    angle = rng.uniform(0, 2 * np.pi, n)
    positions = np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)], -1
    )
    markers, velocity = _markers(field, positions, rng.uniform(-0.4, 0.4, n), 1.5e6)
    deposition = _FakeDeposition(positions, velocity, np.linspace(0, 1, 6))
    result = orbits.follow_orbits(
        deposition, field, wall=wall, max_time=1e-4, interval=5e-5, step=0.05
    )
    fates = result.fate_fractions()
    assert fates["wall"] > 0.02
    assert fates["wall"] + fates["max_time"] == pytest.approx(1.0)
    assert abs(result.power_balance) < 1e-12
    lost = result.losses()
    hits = np.stack([lost.end_x, lost.end_y, lost.end_z], -1)
    assert np.max(np.abs(orbits.wall_distance(wall, hits))) < 1e-6
    assert np.all(lost.face.values >= 0)
    loads = result.wall_loads()
    assert loads.power.sum() == pytest.approx(result.lost_power)
    assert loads.wall_load.attrs["units"] == "W/m^2"
    assert "wall_load_map" not in loads
    # with the Larmor circle, at least as many losses
    larmor = orbits.follow_orbits(
        deposition,
        field,
        wall=wall,
        max_time=1e-4,
        interval=5e-5,
        step=0.05,
        larmor_points=8,
    )
    assert larmor.fate_fractions()["wall"] >= fates["wall"]
    # the same with safe_rho only (no distance field), for a few markers
    few = _FakeDeposition(positions[:10], velocity[:10], np.linspace(0, 1, 6))
    plain = orbits.follow_orbits(
        few,
        field,
        wall=wall,
        max_time=1e-4,
        interval=5e-5,
        step=0.05,
        clearance=None,
    )
    reference = orbits.follow_orbits(
        few, field, wall=wall, max_time=1e-4, interval=5e-5, step=0.05
    )
    assert np.array_equal(plain.markers.fate, reference.markers.fate)


def test_losses_at_rho_limit_without_a_wall(tokamak):
    _, field, _ = tokamak
    rng = np.random.default_rng(5)
    n = 30
    positions = np.stack([rng.uniform(3.4, 3.55, n), np.zeros(n), np.zeros(n)], -1)
    markers, velocity = _markers(field, positions, rng.uniform(-0.4, 0.4, n), 1.5e6)
    deposition = _FakeDeposition(positions, velocity, np.linspace(0, 1, 6))
    result = orbits.follow_orbits(deposition, field, max_time=2e-4, rho_limit=0.95)
    lost = result.losses()
    assert len(lost.marker) > 0
    assert np.all(lost.fate.values == orbits.FATES.index("rho_limit"))
    assert np.allclose(lost.end_rho.values, 0.95, atol=2e-3)
    with pytest.raises(ValueError, match="no wall"):
        result.wall_loads()
    assert np.isnan(result.to_dataset().fast_ion_density).all()


def test_fourier_wall_gives_a_map(tokamak):
    from simgeometry.objects.fourier_surface import FourierSurface

    eq, field, _ = tokamak
    torus = FourierSurface(
        np.array([[3.0], [0.75]]), np.array([[0.0], [1.0]]), 1, resolution=(48, 64)
    )
    rng = np.random.default_rng(6)
    n = 40
    positions = np.stack([rng.uniform(3.4, 3.55, n), np.zeros(n), np.zeros(n)], -1)
    markers, velocity = _markers(field, positions, rng.uniform(-0.4, 0.4, n), 1.5e6)
    deposition = _FakeDeposition(positions, velocity, np.linspace(0, 1, 6))
    result = orbits.follow_orbits(
        deposition, field, wall=torus, max_time=1e-4, interval=5e-5
    )
    loads = result.wall_loads(theta_bins=12, phi_bins=8)
    assert loads.wall_load_map.dims == ("theta", "phi")
    assert result.lost_power > 0
    assert np.nanmax(loads.wall_load_map.values) > 0
    assert loads.attrs["lost_power"] == pytest.approx(result.lost_power)


def _shifted_outline():
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    return np.stack([3.0 + 0.7 * np.cos(angle), 0.9 * np.sin(angle)], -1)


class _Shifted:
    """A wall moved off the axis of symmetry: it does not repeat with any period."""

    def __init__(self, mesh, shift):
        from simgeometry.objects.triangle_mesh import TriangleMesh

        vertices = np.asarray(mesh.vertices) + [shift, 0.0, 0.0]
        self.mesh = TriangleMesh(vertices, np.asarray(mesh.faces), closed=True)

    def inside_at(self, points):
        return self.mesh.inside_at(points)


def test_wall_must_repeat_with_the_period(tokamak):
    eq, field, wall = tokamak
    # the field sampled as if it had four periods, a wall off the axis
    periodic = orbits.orbit_field(
        eq, (48, 8, 48), nfp=4, box=((2.0, 3.9), (-1.25, 1.25))
    )
    shifted = orbits.revolved_wall(_shifted_outline(), 64)
    with pytest.raises(ValueError, match="repeat"):
        orbits.wall_clearance(_Shifted(shifted, 0.2), periodic)
    with pytest.raises(ValueError, match="safe_rho"):
        orbits.follow_orbits(
            _FakeDeposition(np.array([[3.3, 0, 0]]), np.array([[0, 1e6, 0]]), [0, 1]),
            field,
            wall=wall,
            safe_rho=1.5,
        )


def test_births_must_be_on_the_grid(tokamak):
    _, field, _ = tokamak
    deposition = _FakeDeposition(
        np.array([[30.0, 0, 0]]), np.array([[0, 1e6, 0]]), [0.0, 1.0]
    )
    with pytest.raises(ValueError, match="grid"):
        orbits.follow_orbits(deposition, field)


def test_coil_field_outside_the_plasma():
    eq = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 3.0, 1.2, wall_psi_n=1.3)
    coils = foc.CoilSet.planar_circular(
        12, 3.0, 1.4, 3.0 * 2 * np.pi * 3.0 / (12 * 4e-7 * np.pi), n_points=48
    )
    combined = foc.CoilField(eq, coils)
    field = orbits.orbit_field(combined, (40, 8, 40), box=((2.2, 3.8), (-1.2, 1.2)))
    inside = np.array([[3.2, 0.0, 0.1]])
    outside = np.array([[3.75, 0.0, 0.0]])
    assert np.allclose(
        field.field_at(inside), eq.field_at(inside), rtol=1e-3, atol=1e-3
    )
    assert np.allclose(
        field.field_at(outside), coils.field_at(outside), rtol=0.05, atol=0.05
    )


def test_traces_have_plasma_plots_layout(tokamak):
    _, field, _ = tokamak
    markers, _ = _markers(field, [[3.3, 0, 0], [3.2, 0, 0]], [0.5, -0.5])
    traces = orbits.trace_orbits(markers, field, 1e-5, samples=5)
    assert traces.x.dims == ("t", "marker")
    plasma_plots = pytest.importorskip("plasma_plots.plotting")
    subset = plasma_plots.prepare_orbits(traces, required=("x", "y", "z"))
    assert subset.sizes["marker"] == 2


def test_pyccel_source_runs_as_python(tokamak):
    """The Pyccel source, uncompiled, gives the NumPy version's result."""
    from freeofcharge.kernels.guiding_centre_push import guiding_centre_push_pyccel

    _, field, wall = tokamak
    clearance = orbits.wall_clearance(wall, field)
    python = list(_kernel_arguments(field, 3, clearance))
    reference = list(_kernel_arguments(field, 3, clearance))
    for args in (python, reference):
        args[2] = np.full(3, 9e-8)
    guiding_centre_push_pyccel.guiding_centre_push(*python)
    guiding_centre_push_numpy.guiding_centre_push(*reference)
    for a, b in zip(python, reference, strict=True):
        if isinstance(a, np.ndarray):
            assert np.allclose(a, b, rtol=1e-9, atol=1e-9)
    assert np.all(python[19] == 3)
