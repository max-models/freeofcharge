import dataclasses

import numpy as np
import pytest
from plasmacoll import ParticleArrays

from freeofcharge import (
    ConstantStopping,
    Deposition,
    Equilibrium,
    GriddedEquilibrium,
    NeutralBeam,
    Plasma,
    deposit,
    rho_coordinate,
    slow_down,
)
from freeofcharge.deposition import _step_window
from freeofcharge.gridded_equilibrium import _HelicalTestField
from freeofcharge.injectors import (
    BeamSource,
    InjectorBox,
    W7XInjectors,
    aim_direction,
    deposit_sources,
    w7x_injectors,
)

E = 1.602176634e-19
R0, A = 3.0, 0.9


@pytest.fixture(scope="module")
def walled():
    return Equilibrium.solovev(R0, A, 1.6, 2.5, 1.2, wall_psi_n=1.3)


@pytest.fixture(scope="module")
def open_solovev():
    return Equilibrium.solovev(R0, A, 1.6, 2.5, 1.2, wall_psi_n=None)


def plasma(ne=5e19, **kwargs):
    return Plasma.parabolic(ne, 4e3, ions={"H": (1, 1, 1.0)}, **kwargs)


def source(**kwargs):
    fields = {
        "name": "S",
        "position": [R0 + 3 * A, -2.0, 0.0],
        "direction": [-1.0, 1.0, 0.0],
        "energy": "55 keV",
        "power": "1.8 MW",
        "power_fractions": (0.5, 0.3, 0.2),
        "width": 0.05,
        "divergence": 0.01,
    }
    return BeamSource(**{**fields, **kwargs})


# ----------------------------------------------------------------------
# Sources and boxes
# ----------------------------------------------------------------------


def test_source_components_carry_their_power_at_their_energy():
    beams = source().beams(200, seed=1)
    assert list(beams) == ["full", "half", "third"]
    for k, (name, share) in enumerate(zip(beams, (0.5, 0.3, 0.2), strict=True)):
        beam = beams[name]
        np.testing.assert_allclose(beam.energies, 55e3 / (k + 1), rtol=1e-10)
        assert beam.power() == pytest.approx(share * 1.8e6, rel=1e-10)
        assert beam.size == 200
    assert source().beam(100, seed=1).power() == pytest.approx(1.8e6, rel=1e-10)


def test_source_checks_and_converts():
    s = source(power_fractions=(1.0, 0.0, 0.0), direction=[0.0, 3.0, 0.0])
    assert list(s.beams(10)) == ["full"]
    np.testing.assert_allclose(s.direction, [0.0, 1.0, 0.0])
    assert s.energy == pytest.approx(55e3) and s.power == pytest.approx(1.8e6)
    with pytest.raises(ValueError, match="sum"):
        source(power_fractions=(0.5, 0.3, 0.3))
    with pytest.raises(ValueError, match="negative"):
        source(power_fractions=(1.2, -0.2, 0.0))


def test_converging_sources_cross_at_the_port():
    crossover = np.array([6.0, 1.0, 0.2])
    axis = np.array([-1.0, 0.3, 0.05])
    angles = [(-0.02, 0.03), (0.02, 0.03), (0.0, -0.03)]
    box = InjectorBox.converging(
        "B", crossover, axis, angles, "5 m", energy=50e3, power=1e6
    )
    assert [s.name for s in box.sources] == ["B/S1", "B/S2", "B/S3"]
    assert box.power == pytest.approx(3e6)
    for s, (alpha, beta) in zip(box.sources, angles, strict=True):
        to_port = crossover - s.position
        assert np.linalg.norm(to_port) == pytest.approx(5.0)
        np.testing.assert_allclose(to_port / 5.0, s.direction, atol=1e-12)
        # the angles from the box axis
        a = axis / np.linalg.norm(axis)
        assert np.arccos(np.clip(s.direction @ a, -1, 1)) == pytest.approx(
            np.hypot(alpha, beta), rel=0.02
        )


def test_aim_direction_has_the_tangency_radius():
    target = np.array([5.0 * np.cos(0.4), 5.0 * np.sin(0.4), 0.1])
    for sense in (1, -1):
        d = aim_direction(target, 4.0, sense=sense)
        # distance of the line from the z axis
        assert abs(target[0] * d[1] - target[1] * d[0]) / np.hypot(
            *d[:2]
        ) == pytest.approx(4.0)
        e_phi = np.array([-np.sin(0.4), np.cos(0.4), 0.0])
        assert np.sign(d @ e_phi) == sense
        assert d @ target < 0.0  # inwards
    tilted = aim_direction(target, 4.0, tilt="2 deg")
    assert tilted[2] == pytest.approx(np.sin(np.radians(2.0)))
    with pytest.raises(ValueError, match="tangency"):
        aim_direction(target, 6.0)


def test_w7x_stand_in():
    boxes = w7x_injectors()
    assert [b.name for b in boxes] == ["NI20", "NI21"]
    assert all(len(b.sources) == 4 for b in boxes)
    assert all(b.power == pytest.approx(4 * 1.8e6) for b in boxes)
    first = boxes[0].sources[0]
    assert first.energy == pytest.approx(55e3) and first.mass_number == 1
    # opposite toroidal directions
    senses = []
    for box in boxes:
        s = box.sources[0]
        phi = np.arctan2(s.position[1], s.position[0])
        senses.append(np.sign(s.direction @ [-np.sin(phi), np.cos(phi), 0.0]))
    assert senses[0] == -senses[1]
    (op12b,) = w7x_injectors(sources={"NI21": [1, 2]})
    assert op12b.name == "NI21" and op12b.power == pytest.approx(3.6e6)
    changed = dataclasses.replace(W7XInjectors(), energy=60e3, mass_number=2)
    assert w7x_injectors(changed)[0].sources[0].mass_number == 2


# ----------------------------------------------------------------------
# Deposition per source and component
# ----------------------------------------------------------------------


def test_deposit_sources_per_component_with_a_wall(walled):
    sources = [
        source(name="A"),
        InjectorBox("box", (source(name="B", power_fractions=(1.0, 0.0, 0.0)),)),
    ]
    result = deposit_sources(
        sources,
        walled,
        plasma(),
        markers=50,
        seed=1,
        step=0.01,
        rho_edges=np.linspace(0, 1, 6),
    )
    assert result.sources == ["A", "B"]
    ds = result.to_dataset(walled.flux_surfaces(np.linspace(0, 1, 6)))
    assert ds.birth_rate.dims == ("source", "component", "rho")
    a = ds.sel(source="A")
    np.testing.assert_allclose(
        a.absorbed_power + a.shine_through_power, a.injected_power, rtol=1e-12
    )
    np.testing.assert_allclose(a.injected_power, [0.9e6, 0.54e6, 0.36e6], rtol=1e-10)
    # the lower energies are stopped sooner
    assert float(a.shine_through_fraction[0]) > float(a.shine_through_fraction[2])
    # all of the shine-through ends on the wall
    np.testing.assert_allclose(a.wall_hit_power, a.shine_through_power, rtol=1e-10)
    assert np.isnan(ds.sel(source="B", component="half").injected_power)
    assert ds.rho.attrs["radial_coordinate"] == "poloidal"
    assert "power_density" in ds
    total = result.total()
    assert total.injected_power == pytest.approx(1.8e6 * 2, rel=1e-10)
    np.testing.assert_allclose(
        total.power, np.nansum(ds.power.values, axis=(0, 1)), rtol=1e-12
    )
    assert total.births.weights.shape[0] == 4 * 50  # one birth per absorbing marker
    assert total.wall_hits is not None


def test_deposit_sources_without_a_wall(open_solovev):
    result = deposit_sources(
        [source()], open_solovev, plasma(), markers=20, step=0.02, births_per_marker=0
    )
    assert np.all(np.isnan(result.to_dataset().wall_hit_power))
    assert result.total().wall_hits is None


def test_source_names_must_be_unique(walled):
    with pytest.raises(ValueError, match="unique"):
        deposit_sources([source(), source()], walled, plasma(), markers=2)


def test_combine_checks(walled):
    beam = source().beam(20, seed=1)
    one = deposit(beam, walled, plasma(), step=0.02, rho_edges=np.linspace(0, 1, 5))
    other = deposit(beam, walled, plasma(), step=0.02, rho_edges=np.linspace(0, 1, 3))
    with pytest.raises(ValueError, match="nothing"):
        Deposition.combine([])
    with pytest.raises(ValueError, match="shells"):
        Deposition.combine([one, other])
    with pytest.raises(ValueError, match="isotopes"):
        Deposition.combine([one, dataclasses.replace(one, mass_number=2)])
    with pytest.raises(ValueError, match="radial"):
        Deposition.combine([one, dataclasses.replace(one, radial_coordinate="x")])


# ----------------------------------------------------------------------
# The step window: the same result as the whole line
# ----------------------------------------------------------------------


class _NoShortcuts:
    """An equilibrium without ``rho_and_inside_at`` and without a box: whole lines."""

    has_wall = False

    def __init__(self, equilibrium):
        self.equilibrium = equilibrium
        lower, upper = equilibrium.bounds
        self.bounds = (lower - 100.0, upper + 100.0)

    def __getattr__(self, name):
        if name == "rho_and_inside_at":
            raise AttributeError(name)
        return getattr(self.equilibrium, name)


def test_window_and_one_evaluation_do_not_change_the_result():
    gridded = GriddedEquilibrium(_HelicalTestField(), (32, 16, 32))
    origin = np.array([4.5, -1.5, 0.05])
    beam = NeutralBeam.gaussian(
        origin, [-0.7, 1.0, 0.0], 50e3, 1e6, mass_number=1, width=0.05, markers=40
    )
    kwargs = {"step": 0.01, "births_per_marker": 2, "seed": 3, "chunk_points": 10**8}
    profiles = Plasma.parabolic(3e19, 2e3, ions={"H": (1, 1, 1.0)})
    fast = deposit(beam, gridded, profiles, max_length=12.0, **kwargs)
    whole = deposit(beam, _NoShortcuts(gridded), profiles, max_length=12.0, **kwargs)
    assert 0.0 < fast.shine_through_fraction < 1.0
    np.testing.assert_array_equal(fast.power, whole.power)
    np.testing.assert_array_equal(fast.births.positions, whole.births.positions)
    assert fast.shine_through_power == whole.shine_through_power


def test_window_of_lines_that_miss_or_are_vertical(open_solovev):
    lower, upper = open_solovev.bounds
    positions = np.array(
        [
            [0.0, 0.0, upper[2] + 5.0],  # above, flying up: misses
            [R0, 0.0, upper[2] + 5.0],  # vertical through the plasma
            [R0, 0.0, 0.0],  # horizontal, inside the box
            [3 * upper[0], 0.0, 0.0],  # vertical, outside the cylinder
        ]
    )
    directions = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    first, last = (
        np.asarray(a)
        for a in _step_window(positions, directions, open_solovev, 0.1, 400)
    )
    assert first[0] >= last[0] and first[3] >= last[3]
    assert first[1] < last[1] and first[2] == 0 < last[2]
    # a beam that misses the plasma shines through entirely
    speed = np.sqrt(2 * 50e3 * E / 1.66053906660e-27)
    beam = NeutralBeam(
        ParticleArrays(positions[:1], speed * directions[:1], np.array([1e19])), 1
    )
    result = deposit(beam, open_solovev, plasma(), ConstantStopping(1e-20), step=0.1)
    assert result.shine_through_fraction == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Radial coordinate
# ----------------------------------------------------------------------


def test_radial_coordinate_is_carried_and_checked(walled):
    assert rho_coordinate("toroidal")["long_name"] == r"$\rho_\mathrm{tor}$"
    assert rho_coordinate("other")["definition"] == "unspecified flux label"
    beam = source().beam(20, seed=1)
    edges = np.linspace(0, 1, 5)
    result = deposit(
        beam, walled, plasma(radial_coordinate="poloidal"), rho_edges=edges
    )
    assert result.to_dataset().rho.attrs["definition"].endswith("poloidal flux")
    with pytest.raises(ValueError, match="toroidal flux"):
        deposit(beam, walled, plasma(radial_coordinate="toroidal"))
    surfaces = walled.flux_surfaces(edges)
    assert surfaces.to_dataset().attrs["radial_coordinate"] == "poloidal"
    steady = slow_down(result, plasma(), surfaces)
    assert steady.to_dataset().rho.attrs["radial_coordinate"] == "poloidal"
    toroidal = dataclasses.replace(surfaces, radial_coordinate="toroidal")
    with pytest.raises(ValueError, match="rho"):
        slow_down(result, plasma(), toroidal)
    # unspecified matches anything
    unspecified = dataclasses.replace(result, radial_coordinate="unspecified")
    slow_down(unspecified, plasma(), toroidal)
    gridded = GriddedEquilibrium(_HelicalTestField(), (16, 8, 16))
    assert gridded.radial_coordinate == "unspecified"
