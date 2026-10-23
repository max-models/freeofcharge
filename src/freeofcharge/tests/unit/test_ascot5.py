from pathlib import Path

import numpy as np
import pytest

from freeofcharge import Equilibrium, NeutralBeam, Plasma, ascot5

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "benchmarks" / "ascot5" / "ascot5_results.json"


@pytest.fixture(scope="module")
def case():
    return ascot5.benchmark_case(markers=40, grid_points=65)


def test_bfield_has_the_flux_and_the_toroidal_field(case):
    eq = case.equilibrium
    b = ascot5.bfield_2ds(eq)
    assert b["psi"].shape == b["bphi"].shape == (b["nr"], b["nz"]) == eq.psi_grid.shape
    assert not b["br"].any() and not b["bz"].any()
    assert b["psi0"] < eq.psi_axis < b["psi1"] == eq.psi_boundary
    # B_phi = F / R with F = B0 R0 for the Solov'ev solution
    p = case.parameters
    expected = p["toroidal_field"] * p["major_radius"] / eq.r[:, None]
    assert b["bphi"] == pytest.approx(np.broadcast_to(expected, b["bphi"].shape))


def test_plasma_is_extended_outside_the_boundary(case):
    plasma = case.plasma
    p = ascot5.plasma_1d(plasma, rho_max=5.0)
    n = len(plasma.rho)
    assert p["nrho"] == n + 2
    assert p["rho"].shape == (n + 2, 1) and p["idensity"].shape == (n + 2, 2)
    assert p["rho"][n:, 0] == pytest.approx([1.0, 5.0], abs=1e-5)
    assert p["edensity"][:n, 0] == pytest.approx(plasma.ne)
    assert p["edensity"][n:, 0] == pytest.approx([1.0, 1.0])
    # quasi-neutral everywhere, also outside
    charge = p["idensity"] @ p["charge"]
    assert charge == pytest.approx(p["edensity"][:, 0], rel=1e-12)
    assert list(p["anum"]) == [2, 12] and list(p["znum"]) == [1, 6]


def test_wall_is_an_open_polygon(case):
    w = ascot5.wall_2d(case.equilibrium)
    assert w["nelements"] == len(w["r"]) == len(w["z"])
    assert not np.allclose([w["r"][0], w["z"][0]], [w["r"][-1], w["z"][-1]])
    with pytest.raises(ValueError, match="no wall"):
        ascot5.wall_2d(Equilibrium.solovev(1.65, 0.5, 1.6, 2.5, wall_psi_n=None))


def test_injector_has_one_beamlet_per_marker(case):
    beam = case.beam
    inj = ascot5.injector(beam)
    positions = np.asarray(beam.markers.positions)
    assert inj["nbeamlet"] == beam.size
    assert inj["beamletx"] == pytest.approx(positions[:, 0])
    assert inj["beamletdz"] == pytest.approx(np.asarray(beam.directions)[:, 2])
    norm = np.hypot(np.hypot(inj["beamletdx"], inj["beamletdy"]), inj["beamletdz"])
    assert norm == pytest.approx(1.0)
    assert inj["power"] == pytest.approx(case.parameters["power"])
    assert inj["energy"] == pytest.approx(case.parameters["energy"])


def test_injector_rejects_beams_bbnbi_cannot_draw(case):
    beam = case.beam
    two = NeutralBeam.concatenate(
        [beam, NeutralBeam.gaussian([3.5, 0, 0], [-1, 0, 0], 30e3, 1e6, markers=5)]
    )
    with pytest.raises(ValueError, match="one energy"):
        ascot5.injector(two)
    beam.markers.weights[0] *= 2.0
    try:
        with pytest.raises(ValueError, match="equal probability"):
            ascot5.injector(beam)
    finally:
        beam.markers.weights[0] /= 2.0


def test_shell_sums_and_their_errors():
    total, error = ascot5.shell_sums(
        [0.1, 0.15, 0.6, 1.2], [1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0]
    )
    assert total == pytest.approx([3.0, 3.0])
    assert error == pytest.approx([np.sqrt(5.0), 3.0])


def test_case_plasma_scales_with_the_density(case):
    low = case.plasma_at(1e19)
    assert isinstance(low, Plasma)
    assert low.ne[0] == pytest.approx(1e19)
    assert case.plasma.ne / low.ne == pytest.approx(5.0)


@pytest.fixture(scope="module")
def ours(case):
    return ascot5.run_freeofcharge(case, step=0.01, births_per_marker=2)


def test_freeofcharge_side_conserves_power(ours):
    absorbed = np.sum(ours["deposited_power"])
    assert absorbed + ours["shine_through_fraction"] * ours[
        "injected_power"
    ] == pytest.approx(ours["injected_power"], rel=1e-9)
    heating = np.sum(ours["electron_heating"]) + np.sum(ours["ion_heating"])
    # the energy below max(2 T_i, 1 keV) goes to the thermal ions
    assert 0.0 < ours["thermalised_power"] < 0.1 * absorbed
    assert heating + ours["thermalised_power"] == pytest.approx(absorbed, rel=1e-9)
    local = ascot5.without_orbits(ours)
    assert local["fast_ions"] == ours["fast_ions"]
    # trapped ions carry no current: the orbits carry less than the local pitch
    assert 0.0 < ours["fast_ion_current"] < local["fast_ion_current"]
    assert 0.0 < ours["trapped_births"] < 1.0
    shine = ours["scan_shine_through_fraction"]
    assert shine[0] > shine[1] > shine[2]


def test_comparison_with_itself_is_exact(ours):
    comparison = ascot5.compare(ours, ours)
    for row in comparison["profiles"].values():
        assert row["ratio"] == pytest.approx(1.0)
        assert row["shape_difference"] == pytest.approx(0.0)
    for row in comparison["totals"].values():
        assert row["ratio"] is None or row["ratio"] == pytest.approx(1.0)
    assert "freeofcharge" in ascot5.format_comparison(comparison)


def test_comparison_with_the_stored_ascot5_results(ours, tmp_path):
    reference = ascot5.load_results(RESULTS)
    assert reference["parameters"]["energy"] == pytest.approx(60e3)
    comparison = ascot5.compare(ours, reference)
    text = ascot5.format_comparison(comparison)
    assert "deposited_power" in text and "shine-through" in text
    # 40 markers: the deposition already agrees to a few percent in total
    assert comparison["profiles"]["birth_rate"]["ratio"] == pytest.approx(1.0, abs=0.05)
    for ours_shine, row in zip(
        ours["scan_shine_through_fraction"], comparison["scan"], strict=True
    ):
        assert row["ours"] == ours_shine
    # the results round-trip through a file
    ascot5.save_results(tmp_path / "ours.json", ours)
    assert ascot5.load_results(tmp_path / "ours.json") == ours


def test_comparison_needs_the_same_shells(ours):
    other = dict(ours, rho_edges=list(np.linspace(0, 1, 5)))
    with pytest.raises(ValueError, match="different shells"):
        ascot5.compare(ours, other)
