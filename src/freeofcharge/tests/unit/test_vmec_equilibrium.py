import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from freeofcharge import CoilField, CoilSet, MagneticEquilibrium
from freeofcharge.equilibrium import Equilibrium

vmecpp = pytest.importorskip("vmecpp")

from freeofcharge.vmec_equilibrium import (  # noqa: E402
    _WOUT_NAMES,
    VMECEquilibrium,
    _defaults,
)

DATA = Path(__file__).parents[1] / "data" / "coilstellaration"
BOUNDARY = json.loads((DATA / "coilstellaration_boundary.json").read_text())


@pytest.fixture(scope="module")
def vacuum():
    """VMEC++ at beta = 0 inside the CoilStellaration sample's boundary (about 1.5 s)."""
    return VMECEquilibrium.from_boundary(BOUNDARY, mpol=8, ntor=7, ns=(9, 17))


def _points(equilibrium, n=60, seed=0, rho=(0.05, 0.95)):
    """Cartesian points of random flux coordinates, and those coordinates."""
    rng = np.random.default_rng(seed)
    nodes = np.stack(
        [
            rho[0] + (rho[1] - rho[0]) * rng.random(n),
            2 * np.pi * rng.random(n),
            2 * np.pi * rng.random(n),
        ]
    )
    r = equilibrium._r(*nodes)[0]
    z = equilibrium._z(*nodes)[0]
    phi = nodes[2]
    return nodes, np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def test_is_a_magnetic_equilibrium(vacuum):
    assert isinstance(vacuum, MagneticEquilibrium)
    lower, upper = vacuum.bounds
    assert np.all(lower < upper)
    assert not vacuum.has_wall
    assert vacuum.radial_coordinate == "toroidal"
    assert vacuum.nfp == 3
    # the sample's device: major radius 1 m, minor radius about 0.16 m
    assert 0.1 < vacuum.minor_radius < 0.2
    # beta = 0 and no current
    np.testing.assert_allclose(vacuum.wout.betatotal, 0.0, atol=1e-12)


def test_inverse_map(vacuum):
    nodes, points = _points(vacuum)
    rho, theta, phi = vacuum.flux_coordinates(points.reshape(3, 20, 3))
    assert rho.shape == (3, 20)
    np.testing.assert_allclose(rho.ravel(), nodes[0], atol=1e-9)
    angle = np.angle(np.exp(1j * (theta.ravel() - nodes[1])))
    np.testing.assert_allclose(angle, 0.0, atol=1e-8)
    np.testing.assert_allclose(np.cos(phi.ravel() - nodes[2]), 1.0)
    assert np.all(vacuum.inside_at(points))
    assert np.all(vacuum.inside_wall_at(points))
    # near the axis, where the angle is undefined
    near, axis_points = _points(vacuum, n=10, seed=1, rho=(0.0, 0.02))
    np.testing.assert_allclose(vacuum.rho_at(axis_points), near[0], atol=1e-8)


def test_field_strength_matches_vmec(vacuum):
    # |B^u e_u + B^v e_v| against VMEC's own |B| series: up to Fourier truncation
    nodes, points = _points(vacuum, seed=2, rho=(0.0, 1.0))
    strength = np.linalg.norm(vacuum.field_at(points), axis=-1)
    np.testing.assert_allclose(strength, vacuum._b(*nodes)[0], rtol=2e-3)


def test_points_outside(vacuum):
    _, upper = vacuum.bounds
    points = np.array([[0.1, 0.0, 0.0], [upper[0] + 0.2, 0.0, 0.0]])
    rho = vacuum.rho_at(points)
    assert np.all(rho > 1.0)
    assert not np.any(vacuum.inside_at(points))
    field = vacuum.field_at(points)
    assert field.shape == (2, 3) and np.all(np.isfinite(field))


def test_flux_surfaces(vacuum):
    surfaces = vacuum.flux_surfaces(np.linspace(0.0, 1.0, 5), field_points=(32, 16))
    np.testing.assert_allclose(surfaces.volume.sum(), vacuum.wout.volume_p, rtol=1e-4)
    np.testing.assert_allclose(surfaces.surface_weight.sum(axis=1), 1.0)
    assert np.all((surfaces.trapped_fraction > 0) & (surfaces.trapped_fraction < 1))
    np.testing.assert_allclose(vacuum.iota([0.0, 1.0]), vacuum.wout.iotaf[[0, -1]])


def test_from_wout_and_from_input_files(vacuum, tmp_path):
    path = tmp_path / "wout_sample.nc"
    vacuum.wout.save(path)
    loaded = VMECEquilibrium.from_wout(path)
    _, points = _points(vacuum, n=10, seed=3)
    np.testing.assert_allclose(loaded.field_at(points), vacuum.field_at(points))
    np.testing.assert_allclose(loaded.rho_at(points), vacuum.rho_at(points))
    # a VMEC++ input file, here at low resolution
    vmec_input = VMECEquilibrium.vacuum_input(BOUNDARY, ns=(9,))
    vmec_input.save(tmp_path / "input.json")
    rerun = VMECEquilibrium.from_input(str(tmp_path / "input.json"))
    assert rerun.wout.ns == 9


def test_non_symmetric_series(vacuum):
    # the sine series of lasym equilibria, zero here: the same field
    wout = {name: getattr(vacuum.wout, name, None) for name in _WOUT_NAMES}
    wout["lasym__logical__"] = 1
    for name in ("rmns", "zmnc", "bsupumns", "bsupvmns", "bmns"):
        partner = name[:-1] + ("c" if name.endswith("s") else "s")
        wout[name] = np.zeros_like(wout[partner])
    asymmetric = VMECEquilibrium(wout)
    _, points = _points(vacuum, n=10, seed=4)
    np.testing.assert_allclose(asymmetric.field_at(points), vacuum.field_at(points))
    # and a non-symmetric boundary gives a lasym input
    boundary = dict(BOUNDARY)
    boundary["r_sin"] = 1e-3 * np.ones_like(np.asarray(BOUNDARY["r_cos"]))
    vmec_input = VMECEquilibrium.vacuum_input(boundary, mpol=3, ntor=2)
    assert vmec_input.lasym
    assert vmec_input.rbs.shape == (3, 5)
    np.testing.assert_allclose(vmec_input.rbs, 1e-3)
    np.testing.assert_allclose(vmec_input.zbc, 0.0)


def test_coils_reproduce_the_vacuum_field(vacuum):
    """The coils' Biot-Savart field against VMEC's inside the plasma, at beta = 0.

    With the currents scaled to the equilibrium's field (least squares), they differ
    by the coils' field error (the dataset's mean |B_n|/|B| is 0.27 %) plus VMEC's
    resolution: 0.21 % on average, 0.68 % at most here; 1.1 % with VMEC at the
    boundary's own modes (mpol 5, ntor 4), 0.16 % at mpol 12.
    """
    pytest.importorskip("plasma_plots")
    coils = CoilSet.from_coilset(
        json.loads((DATA / "coilstellaration_coilset.json").read_text()),
        n_points=128,
    )
    _, points = _points(vacuum, n=200, seed=5)
    inner = np.asarray(vacuum.field_at(points))
    outer = np.asarray(coils.field_at(points))
    scale = np.sum(inner * outer) / np.sum(outer * outer)
    combined = CoilField(vacuum, coils, current_scale=scale)
    difference = np.asarray(combined.jump(points))
    assert scale < 0  # the coils' currents run the other way round
    assert np.mean(difference) < 3e-3
    assert np.max(difference) < 1.5e-2


def test_solovev_tokamak():
    """VMEC++ on a Solov'ev boundary with its pressure and iota, against the closed form."""
    r0, a, kappa, b0, q0 = 3.0, 1.0, 1.5, 2.0, 1.5
    solovev = Equilibrium.solovev(
        r0, a, kappa, b0, q0, wall_psi_n=None, points=65, surface_points=64
    )
    c = b0 / (2 * r0**2 * kappa * q0)
    psi_b = c * 0.25 * kappa**2 * ((r0 + a) ** 2 - r0**2) ** 2

    # toroidal flux inside psi: F int dR dZ / R over the surface
    t, w = np.polynomial.legendre.leggauss(48)
    t, w = 0.5 * np.pi * t, 0.5 * np.pi * w

    def toroidal_flux(psi):
        half = 2 * np.sqrt(psi / c) / kappa
        integrand = np.sqrt(psi / c) * half * np.cos(t) ** 2
        return b0 * r0 * np.sum(w * integrand / (r0**2 + half * np.sin(t)) ** 1.5)

    psi_n = np.linspace(0.0, 1.0, 401)
    flux = np.array([toroidal_flux(p * psi_b) for p in psi_n])
    s_of_psi = flux / flux[-1]
    s = np.linspace(0.0, 1.0, 41)
    psi_of_s = np.interp(s, s_of_psi, psi_n) * psi_b
    pressure = 2 * c * (1 + kappa**2) / (4e-7 * np.pi) * (psi_b - psi_of_s)
    iota = 2 * np.pi / np.interp(s, s_of_psi, np.gradient(flux, psi_n * psi_b))

    # the boundary in the polar angle about the axis, from the closed form
    def flux_on_ray(radius, angle):
        r, z = r0 + radius * np.cos(angle), radius * np.sin(angle)
        return c * (r**2 * z**2 + 0.25 * kappa**2 * (r**2 - r0**2) ** 2) - psi_b

    angle = 2 * np.pi * np.arange(256) / 256
    radius = np.array([brentq(flux_on_ray, 1e-6, 2.5, args=(q,)) for q in angle])
    m = np.arange(24)[:, None]
    weight = np.where(m[:, 0] == 0, 1.0, 2.0)
    rbc = weight * np.mean((r0 + radius * np.cos(angle)) * np.cos(m * angle), axis=1)
    zbs = weight * np.mean(radius * np.sin(angle) * np.sin(m * angle), axis=1)

    vmec_input = vmecpp.VmecInput(
        **_defaults() | {"piota_type": "cubic_spline", "ai": np.zeros(0)},
        ai_aux_s=s,
        ai_aux_f=iota,
        lasym=False,
        nfp=1,
        mpol=24,
        ntor=0,
        ntheta=0,
        nzeta=0,
        ns_array=np.array([17, 33]),
        ftol_array=np.array([1e-10, 1e-12]),
        niter_array=np.array([5000, 10000]),
        phiedge=flux[-1],
        ncurr=0,
        pmass_type="cubic_spline",
        am=np.zeros(0),
        am_aux_s=s,
        am_aux_f=pressure,
        pres_scale=1.0,
        pcurr_type="power_series",
        ac=np.zeros(1),
        curtor=0.0,
        lfreeb=False,
        raxis_c=np.array([r0]),
        zaxis_s=np.zeros(1),
        rbc=rbc[:, None],
        zbs=zbs[:, None],
    )
    tokamak = VMECEquilibrium.from_input(vmec_input)

    rng = np.random.default_rng(6)
    q, phi = 2 * np.pi * rng.random((2, 200))
    rho = np.sqrt(rng.random(200))
    big_r = r0 + rho * np.interp(q, angle, radius, period=2 * np.pi) * np.cos(q)
    z = rho * np.interp(q, angle, radius, period=2 * np.pi) * np.sin(q)
    points = np.stack([big_r * np.cos(phi), big_r * np.sin(phi), z], -1)
    # VMEC's s is the normalized toroidal flux, Equilibrium's psi_n the poloidal one;
    # VMEC's radial error: s to 4e-4 on average at 33 surfaces, 1e-4 at 65, 3e-5 at 129
    expected = np.interp(solovev.psi_normalized(big_r, z), psi_n, s_of_psi)
    s_vmec = np.asarray(tokamak.rho_at(points)) ** 2
    assert np.mean(np.abs(s_vmec - expected)) < 1e-3
    assert np.max(np.abs(s_vmec - expected)) < 3e-3
    field, exact = tokamak.field_at(points), solovev.field_at(points)
    difference = np.linalg.norm(field - exact, axis=-1) / np.linalg.norm(exact, axis=-1)
    # 0.04 % on average, 0.24 % at most
    assert np.mean(difference) < 1e-3
    assert np.max(difference) < 5e-3
