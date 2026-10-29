import numpy as np
import pytest

from freeofcharge import IonSpecies, NeutralBeam, Plasma, tangential_injection


def test_tangential_injection_is_tangent_to_its_radius():
    origin, direction = tangential_injection(5.3, 12.0, height=0.4, toroidal_angle=0.3)
    # distance of the line from the z axis
    closest = (
        origin
        - np.dot(origin[:2], direction[:2])
        / np.dot(direction[:2], direction[:2])
        * direction
    )
    assert np.hypot(closest[0], closest[1]) == pytest.approx(5.3)
    assert np.hypot(origin[0], origin[1]) == pytest.approx(12.0)
    assert origin[2] == 0.4
    phi_hat = np.array([-np.sin(0.3), np.cos(0.3), 0.0])
    assert np.dot(direction, phi_hat) == pytest.approx(1.0)


def test_tangential_injection_rejects_a_start_inside():
    with pytest.raises(ValueError):
        tangential_injection(5.0, 4.0)


def test_gaussian_beam_power_energy_and_divergence():
    beam = NeutralBeam.gaussian(
        [10.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        1e6,
        16.5e6,
        divergence=(0.004, 0.008),
        width=0.2,
        markers=40000,
        seed=1,
    )
    assert beam.power() == pytest.approx(16.5e6, rel=1e-12)
    np.testing.assert_allclose(np.asarray(beam.energies), 1e6, rtol=1e-12)
    d = np.asarray(beam.directions)
    # horizontal plane: y; vertical: z
    assert np.std(np.arctan2(d[:, 1], -d[:, 0])) == pytest.approx(0.004, rel=0.03)
    assert np.std(np.arctan2(d[:, 2], -d[:, 0])) == pytest.approx(0.008, rel=0.03)
    assert np.std(np.asarray(beam.markers.positions)[:, 1]) == pytest.approx(
        0.2, rel=0.03
    )


def test_gaussian_beam_halo():
    beam = NeutralBeam.gaussian(
        [0, 0, 0],
        [0, 0, 1],
        1e5,
        1e6,
        divergence=0.001,
        halo_fraction=0.2,
        halo_divergence=0.03,
        markers=20000,
        seed=2,
    )
    d = np.asarray(beam.directions)
    wide = np.abs(np.arctan2(d[:, 0], d[:, 2])) > 0.006
    assert wide.mean() == pytest.approx(0.2 * 0.84, abs=0.02)


def test_concatenate_needs_one_isotope():
    a = NeutralBeam.gaussian([0, 0, 0], [1, 0, 0], 1e5, 1e6, markers=10, seed=1)
    b = NeutralBeam.gaussian([0, 0, 0], [1, 0, 0], 5e4, 1e6, markers=10, seed=1)
    assert NeutralBeam.concatenate([a, b]).power() == pytest.approx(2e6)
    c = NeutralBeam.gaussian(
        [0, 0, 0], [1, 0, 0], 5e4, 1e6, mass_number=1, markers=10, seed=1
    )
    with pytest.raises(ValueError, match="isotopes"):
        NeutralBeam.concatenate([a, c])


def test_parabolic_plasma_is_quasineutral_with_zeff():
    plasma = Plasma.parabolic(
        1e20, 1e4, ions={"D": (2, 1, 0.9), "C": (12, 6, 1.0 / 60.0)}
    )
    assert float(plasma.zeff(0.3)) == pytest.approx(0.9 + 36 / 60.0)
    assert float(plasma.electron_density(1.2)) == 0.0
    assert float(plasma.electron_density(0.0)) == pytest.approx(1e20)


def test_plasma_rejects_charge_imbalance():
    rho = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="charge"):
        Plasma(
            rho,
            np.ones(2) * 1e19,
            np.ones(2),
            np.ones(2),
            {"D": IonSpecies(2, 1, np.ones(2) * 2e19)},
        )


def beamline_dataset(num=6):
    """Markers in the layout of beamtracer's ``Markers.to_dataset``."""
    import xarray as xr
    from plasmacoll.constants import ATOMIC_MASS

    rng = np.random.default_rng(0)
    position = rng.normal(size=(3, num))
    velocity = np.vstack([np.full(num, 1e7), rng.normal(0, 1e4, (2, num))])
    return xr.Dataset(
        {
            "position": (("component", "marker"), position),
            "velocity": (("component", "marker"), velocity),
            "weight": ("marker", np.arange(1.0, num + 1) * 1e15),
            "state": ("marker", ["0", "+", "0", "-", "0", "0"][:num]),
        },
        coords={"component": ["x", "y", "z"]},
        attrs={"mass": 2.014 * ATOMIC_MASS, "states": ["-", "0", "+"]},
    )


def test_a_beam_from_a_beamline_dataset_keeps_the_neutrals():
    data = beamline_dataset()
    beam = NeutralBeam.from_dataset(data)
    neutral = np.array([0, 2, 4, 5])
    assert beam.mass_number == 2
    assert beam.size == 4
    assert np.allclose(
        np.asarray(beam.markers.positions), data.position.values[:, neutral].T
    )
    assert np.allclose(
        np.asarray(beam.markers.velocities), data.velocity.values[:, neutral].T
    )
    energy = 0.5 * beam.mass * np.sum(data.velocity.values[:, neutral] ** 2, axis=0)
    weights = data.weight.values[neutral]
    assert beam.power() == pytest.approx(np.sum(weights * energy))


def test_a_beam_from_a_dataset_needs_its_mass():
    data = beamline_dataset()
    data.attrs.pop("mass")
    with pytest.raises(ValueError, match="mass_number"):
        NeutralBeam.from_dataset(data)
    assert NeutralBeam.from_dataset(data, mass_number=1).mass_number == 1
    assert NeutralBeam.from_dataset(data.drop_vars("state"), mass_number=2).size == 6
