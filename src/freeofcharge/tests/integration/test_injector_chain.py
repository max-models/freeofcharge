"""The injector chain end to end: source, slit beamlet, gas neutralizer, duct, plasma.

A small run of ``examples/injector_chain.py`` that checks what crosses between the
packages (SI units with temperatures and energies in eV, array shapes, species names)
and that the results add up (positive powers, shares in [0, 1], the power bookkeeping
from the extracted current to the heating). Each stage is a fixture that names its
stage, package and version when it raises, and each interface assertion starts with the
two stages it joins, so a failed run says which stage and which interface broke.

The ``Chain test`` workflow runs this module against the latest PyPI releases and
against the devel branches. tree-d-3d has no PyPI release: without it, stage 2 is
skipped and the beam line takes a fixed divergence.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import numpy as np
import pytest

import freeofcharge as foc
from freeofcharge import chain

pytest.importorskip("wellbalanced", reason="the chain needs freeofcharge[chain]")

RF_POWER = np.array([3.0e5, 6.0e5])  # W
PERVEANCE = np.array([0.02, 0.15, 0.45])  # j / j_CL of the optics scan
LINE_DENSITY = np.array([0.5e20, 1.4e20, 3.0e20])  # m^-2
OPERATING_LINE_DENSITY = 1.4e20  # m^-2, near the neutralizer's optimum

EXTRACTION_VOLTAGE, GAP = 9.0e3, 6.0e-3  # V, m
APERTURE_AREA = 0.2  # m^2
BEAM_ENERGY = 1.0e6  # eV
ACCELERATOR_TRANSMISSION = 0.7
SHRINK = np.sqrt(EXTRACTION_VOLTAGE / BEAM_ENERGY)  # the angles after the acceleration
SOURCE_WIDTH, DUCT_LENGTH, DUCT_HALF_APERTURE = 0.1, 23.0, 0.25  # m
FIXED_DIVERGENCE = 5.0e-3  # rad, the beam line's divergence without tree-d-3d

# wellbalanced names species by formula; these are the ones deuterium_source builds
SOURCE_SPECIES = {"e", "D2", "D", "D2+"}
SOURCE_VOLUME, SOURCE_WALL_AREA = 0.4, 6.0  # deuterium_source's defaults [m^3, m^2]


class StageError(Exception):
    """A stage of the chain raised; the message names the stage and its package."""


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


@contextmanager
def stage(name: str, package: str):
    """Re-raise any error of the stage with its name, package and version."""
    try:
        yield
    except Exception as error:
        raise StageError(
            f"stage {name} ({package} {package_version(package)}) raised "
            f"{type(error).__name__}: {error}"
        ) from error


# the stages, in the order of the chain; a module fixture that raises fails every test
# that needs it with the same StageError


@pytest.fixture(scope="module")
def sources():
    with stage("1 source", "wellbalanced"):
        return [chain.deuterium_source(p) for p in RF_POWER]


@pytest.fixture(scope="module")
def optics():
    if find_spec("tree_d_3d") is None:
        return None
    with stage("2 slit optics", "tree-d-3d"):
        return chain.slit_divergence(PERVEANCE, rays=12, iterations=6)


@pytest.fixture(scope="module")
def perveance(sources):
    return np.array([s.current_density for s in sources]) / chain.child_langmuir(
        EXTRACTION_VOLTAGE, GAP
    )


@pytest.fixture(scope="module")
def divergence(optics, perveance):
    """The rms beam divergence after the acceleration [rad], per RF power."""
    if optics is None:
        return np.full(RF_POWER.shape, FIXED_DIVERGENCE)
    curve, _ = optics
    return np.interp(perveance, PERVEANCE, curve) * SHRINK


@pytest.fixture(scope="module")
def neutral():
    with stage("3 gas neutralizer", "plasmacoll"):
        return chain.neutral_fraction(LINE_DENSITY)


@pytest.fixture(scope="module")
def transmission(divergence):
    with stage("4 duct", "freeofcharge"):
        return np.array(
            [
                chain.duct_transmission(
                    SOURCE_WIDTH, theta, DUCT_LENGTH, DUCT_HALF_APERTURE
                )
                for theta in divergence
            ]
        )


@pytest.fixture(scope="module")
def accelerated(sources):
    """The beam power that leaves the accelerator [W], per RF power."""
    current = np.array([s.current_density for s in sources]) * APERTURE_AREA
    return current * BEAM_ENERGY * ACCELERATOR_TRANSMISSION


@pytest.fixture(scope="module")
def delivered(accelerated, transmission):
    """The neutral power that enters the torus [W], per RF power."""
    with stage("3 gas neutralizer", "plasmacoll"):
        share = chain.neutral_fraction(OPERATING_LINE_DENSITY)[0]
    return accelerated * transmission * share


@pytest.fixture(scope="module")
def plasma_runs(divergence, delivered):
    """The beam, its deposition and its slowing down, per RF power."""
    with stage("5 plasma", "freeofcharge"):
        equilibrium = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
        origin, direction = foc.tangential_injection(5.3, 12.0)
        plasma = foc.Plasma.parabolic(
            5e19, 2.0e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)}
        )
        rho_edges = np.linspace(0.0, 1.0, 11)
        surfaces = equilibrium.flux_surfaces(rho_edges)
        runs = []
        for theta, power in zip(divergence, delivered, strict=True):
            beam = foc.NeutralBeam.gaussian(
                origin,
                direction,
                BEAM_ENERGY,
                float(power),
                width=float(np.hypot(SOURCE_WIDTH, DUCT_LENGTH * theta)),
                divergence=float(theta),
                markers=200,
                seed=1,
            )
            deposition = foc.deposit(
                beam,
                equilibrium,
                plasma,
                step=0.05,
                births_per_marker=2,
                rho_edges=rho_edges,
                seed=2,
            )
            runs.append((beam, deposition, foc.slow_down(deposition, plasma, surfaces)))
        return plasma, runs


def test_stage_1_source(sources):
    from wellbalanced import ELECTRON

    assert ELECTRON.name == "e", "wellbalanced: the electron is no longer named 'e'"
    for source in sources:
        state = source.state
        assert state.converged
        assert set(state.densities) == SOURCE_SPECIES, (
            f"wellbalanced -> source: species {sorted(state.densities)}, "
            f"expected the formulas {sorted(SOURCE_SPECIES)}"
        )
        # temperatures in eV, not K; densities in m^-3
        assert 1.0 < source.electron_temperature < 30.0, (
            f"wellbalanced -> source: T_e = {source.electron_temperature:g}, not in eV"
        )
        assert state.gas_temperature == pytest.approx(0.05)
        assert 1e16 < source.electron_density < 1e20
        assert source.electron_density == pytest.approx(state.densities["D2+"])
        # the electron power balance closes on the absorbed RF power
        absorbed = 0.5 * source.rf_power
        assert state.power["absorbed"] == pytest.approx(absorbed)
        assert abs(sum(state.power.values())) < 1e-6 * absorbed
        # the chain's Bohm flux (u_B from T_e in eV and the D2+ mass) is the ion wall loss
        assert source.positive_flux * SOURCE_WALL_AREA / SOURCE_VOLUME == pytest.approx(
            state.wall_losses["D2+"], rel=1e-6
        ), "wellbalanced -> source: the D2+ wall loss differs from h n_e u_B"
        assert source.current_density == pytest.approx(
            0.3 * chain.ELEMENTARY_CHARGE * source.positive_flux
        )
        assert source.current_density > 0.0
    assert sources[1].current_density > sources[0].current_density


def test_stage_2_slit_optics(optics):
    if optics is None:
        pytest.skip("tree-d-3d has no PyPI release")
    curve, transmitted = optics
    assert curve.shape == transmitted.shape == PERVEANCE.shape, (
        f"tree-d-3d -> optics: shapes {curve.shape}, {transmitted.shape}"
    )
    assert np.all(np.isfinite(curve)) and np.all((curve > 0.0) & (curve < 0.3)), (
        f"tree-d-3d -> optics: rms divergences {curve} rad"
    )
    assert np.all((transmitted >= 0.0) & (transmitted <= 1.0))
    assert np.all(transmitted == 1.0), "the extraction electrode intercepts the beam"
    assert curve[1] < curve[0] and curve[1] < curve[2], "no V-curve"


def test_interface_source_to_optics(perveance):
    assert perveance.shape == RF_POWER.shape
    assert np.all(perveance > 0.0)
    assert perveance.max() <= PERVEANCE.max(), (
        f"source -> optics: perveance {perveance.max():.2f} beyond the optics scan"
    )


def test_stage_3_neutralizer(neutral):
    from plasmacoll import ChargeStateTransitions, charge_state_fractions

    transitions = ChargeStateTransitions({"gas": dict(chain.GAS_NEUTRALIZER)})
    assert tuple(transitions.states) == ("-", "0", "+"), (
        f"plasmacoll -> neutralizer: states {transitions.states}; the chain reads "
        "the second column as the neutral share"
    )
    path = np.concatenate(([0.0], LINE_DENSITY))
    fractions = charge_state_fractions(transitions, {"gas": path})
    assert fractions.shape == (path.size, 3), (
        f"plasmacoll -> neutralizer: fractions of shape {fractions.shape}"
    )
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((fractions >= -1e-15) & (fractions <= 1.0 + 1e-15))
    assert neutral.shape == LINE_DENSITY.shape
    np.testing.assert_allclose(
        neutral, chain.neutral_fraction_closed_form(LINE_DENSITY), atol=1e-12
    )
    assert np.all((neutral > 0.0) & (neutral < 1.0))


def test_stage_4_duct(divergence, transmission):
    assert transmission.shape == divergence.shape == RF_POWER.shape
    assert np.all((transmission > 0.0) & (transmission <= 1.0))
    order = np.argsort(divergence)
    assert np.all(np.diff(transmission[order]) <= 0.0), "more divergence, more passes"


def test_stage_5_plasma(plasma_runs, delivered):
    plasma, runs = plasma_runs
    assert set(plasma.ions) == {"D", "He"}
    for (beam, deposition, steady), power in zip(runs, delivered, strict=True):
        markers = beam.markers
        n = markers.weights.shape[0]
        # plasmacoll's ParticleArrays: positions (N, 3) m, velocities (N, 3) m/s,
        # weights (N,) atoms/s
        assert markers.positions.shape == (n, 3), "duct -> plasma: marker positions"
        assert markers.velocities.shape == (n, 3), "duct -> plasma: marker velocities"
        np.testing.assert_allclose(
            np.asarray(beam.energies),
            BEAM_ENERGY,
            rtol=1e-9,
            err_msg="duct -> plasma: marker energies are not the beam energy in eV",
        )
        weight_power = float(markers.weights.sum()) * BEAM_ENERGY
        assert weight_power * chain.ELEMENTARY_CHARGE == pytest.approx(power), (
            "duct -> plasma: the marker weights do not carry the delivered power"
        )
        births = deposition.births
        assert births.positions.shape[1] == 3 and births.velocities.shape[1] == 3
        assert births.weights.shape == (births.positions.shape[0],)
        assert deposition.power.shape == deposition.birth_rate.shape == (10,)
        assert 0.0 <= deposition.shine_through_fraction <= 1.0
        assert steady.driven_current > 0.0
        assert steady.stored_energy > 0.0
        assert np.all(steady.electron_heating >= 0.0)
        assert np.all(steady.ion_heating >= 0.0)


def test_power_bookkeeping_end_to_end(
    accelerated, transmission, delivered, plasma_runs
):
    """Every watt from the accelerator to the heating is accounted for."""
    assert np.all(accelerated > 0.0)
    assert np.all((delivered > 0.0) & (delivered < accelerated))
    _, runs = plasma_runs
    for (_, deposition, steady), power in zip(runs, delivered, strict=True):
        assert deposition.injected_power == pytest.approx(power, rel=1e-9), (
            "duct -> plasma: injected power is not the delivered power"
        )
        assert deposition.absorbed_power > 0.0
        assert deposition.absorbed_power + deposition.shine_through_power == (
            pytest.approx(deposition.injected_power, rel=1e-9)
        )
        assert deposition.power.sum() == pytest.approx(deposition.absorbed_power)
        heating = steady.electron_heating.sum() + steady.ion_heating.sum()
        assert heating == pytest.approx(deposition.absorbed_power, rel=1e-6), (
            "deposition -> slowing down: the heating is not the absorbed power"
        )
        if deposition.wall_hits is not None:
            wall = float(deposition.wall_hits.weights.sum()) * BEAM_ENERGY
            assert wall * chain.ELEMENTARY_CHARGE == pytest.approx(
                deposition.shine_through_power, rel=1e-6
            ), "deposition: the wall hits do not carry the shine-through power"
