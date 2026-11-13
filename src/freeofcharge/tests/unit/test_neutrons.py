"""Beam-target and thermonuclear neutron rates.

A mono-energetic deuterium beam in a flat deuterium plasma with one Coulomb
logarithm has constant drags, so its yield has closed forms: with a constant
cross section the integral of `_closed_form_yield`, with the Bosch-Hale cross
section a quadrature of n sigma(E_cm) v dt. The thermonuclear rate is checked by
hand, the reactivities themselves against Bosch and Hale's tables in plasmacoll's
tests, and a JET-like case against the order of magnitude of JET's DD neutron rates.
"""

import dataclasses

import numpy as np
import pytest
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.fusion import FUSION_REACTIONS, NUCLEAR_MASSES, FusionReaction
from scipy.integrate import quad

import freeofcharge as foc
from freeofcharge.neutrons import (
    _closed_form_yield,
    _reactivity_table,
    neutron_rates,
    slowing_down_yield,
    thermonuclear_rate,
)
from freeofcharge.slowing_down import critical_velocity, slowing_down_time

NE, TE, LN = 5e19, 5e3, 17.0
EDGES = np.linspace(0.0, 1.0, 6)


@pytest.fixture(scope="module")
def setup():
    equilibrium = foc.Equilibrium.solovev(3.0, 1.0, 1.5, 3.0, wall_psi_n=1.3)
    surfaces = equilibrium.flux_surfaces(EDGES)
    # flat profiles: every birth sees the same plasma
    plasma = foc.Plasma.parabolic(
        NE, TE, ions={"D": (2, 1, 1.0)}, alpha_n=0.0, alpha_t=0.0
    )
    origin, direction = foc.tangential_injection(3.0, 6.0)
    beam = foc.NeutralBeam.gaussian(origin, direction, 100e3, 1e6, markers=60, seed=1)
    deposition = foc.deposit(
        beam, equilibrium, plasma, step=0.03, rho_edges=EDGES, seed=2
    )
    return equilibrium, surfaces, plasma, deposition


def _drags():
    mass = 2 * ATOMIC_MASS
    tau_s = float(slowing_down_time(mass, 1, NE, TE, LN))
    v_c = float(critical_velocity(NE, TE, [NE], [mass], [1]))
    return 1.0 / tau_s, v_c**3 / tau_s


class ConstantSigma(FusionReaction):
    """D(d,n)3He with a constant cross section and a cold target."""

    def beam_reactivity(self, beam_energy, temperature=0.0, *, beam=None, num=801):
        speed = np.sqrt(
            2 * np.asarray(beam_energy) * ELEMENTARY_CHARGE / NUCLEAR_MASSES["D"]
        )
        return 1e-29 * speed * np.ones_like(np.asarray(temperature, dtype=float))


def test_yield_of_a_constant_cross_section_is_the_closed_form():
    a_e, a_i, v0, v1 = 3.0, 2e19, 3e6, 4e5
    speed = np.exp(np.linspace(np.log(v0), np.log(v1), 201))[None, :]
    value = slowing_down_yield(speed, a_e, a_i, 1e-29 * speed)
    assert value[0] == pytest.approx(
        _closed_form_yield(v0, v1, a_e, a_i, 1e-29), rel=1e-8
    )
    with pytest.raises(ValueError, match="odd"):
        slowing_down_yield(speed[:, :-1], a_e, a_i, speed[:, :-1])


def test_mono_energetic_beam_with_constant_cross_section(setup):
    _, surfaces, plasma, deposition = setup
    reaction = ConstantSigma(
        **{
            field: getattr(FUSION_REACTIONS["D(d,n)3He"], field)
            for field in FusionReaction.__dataclass_fields__
        }
    )
    rates = neutron_rates(deposition, plasma, surfaces, reaction=reaction, ln_lambda=LN)
    a_e, a_i = _drags()
    mass = 2 * ATOMIC_MASS
    v0 = np.sqrt(2 * 100e3 * ELEMENTARY_CHARGE / mass)
    v1 = np.sqrt(2 * 1.5 * TE * ELEMENTARY_CHARGE / mass)
    per_ion = NE * _closed_form_yield(v0, v1, a_e, a_i, 1e-29)
    assert rates.reactions_per_ion == pytest.approx(per_ion, rel=1e-6)
    assert rates.total_beam_target == pytest.approx(
        per_ion * rates.birth_rate, rel=1e-6
    )


def test_mono_energetic_beam_against_quadrature_of_bosch_hale(setup):
    _, surfaces, plasma, deposition = setup
    rates = neutron_rates(
        deposition, plasma, surfaces, ln_lambda=LN, maxwellian_target=False
    )
    dd = FUSION_REACTIONS["D(d,n)3He"]
    a_e, a_i = _drags()
    mass = 2 * ATOMIC_MASS

    def integrand(v):
        e_cm = 0.5 * dd.reduced_mass * v**2 / ELEMENTARY_CHARGE
        return float(dd.sigma(e_cm)) * v / (a_e * v + a_i / v**2)

    v0 = np.sqrt(2 * 100e3 * ELEMENTARY_CHARGE / mass)
    v1 = np.sqrt(2 * 1.5 * TE * ELEMENTARY_CHARGE / mass)
    per_ion = NE * quad(integrand, v1, v0, epsrel=1e-10, limit=200)[0]
    assert rates.reactions_per_ion == pytest.approx(per_ion, rel=1e-4)
    # about 1e-5 reactions per 100 keV deuteron at 5 keV and 5e19 m^-3
    assert 3e-6 < per_ion < 3e-5
    # a warm target adds to it (Gamow), most at low energy
    warm = neutron_rates(deposition, plasma, surfaces, ln_lambda=LN)
    assert 1.0 < warm.total_beam_target / rates.total_beam_target < 1.6


def test_reactivity_table_against_direct_evaluation():
    dd = FUSION_REACTIONS["D(d,n)3He"]
    energy = np.geomspace(5e3, 1e6, 50)
    temperature = np.linspace(2e3, 2e4, 50)
    table = _reactivity_table(dd, "D", energy, temperature, 257)
    direct = dd.beam_reactivity(energy, temperature)
    assert table(energy, temperature) == pytest.approx(direct, rel=2e-3)
    cold = _reactivity_table(dd, "D", energy, None, 257)
    assert cold(energy, 0.0) == pytest.approx(dd.beam_reactivity(energy), rel=1e-3)


def test_thermonuclear_rate_by_hand(setup):
    _, surfaces, plasma, _ = setup
    dd = FUSION_REACTIONS["D(d,n)3He"]
    expected = NE**2 / 2 * dd.reactivity(TE) * surfaces.volume
    assert thermonuclear_rate(plasma, surfaces) == pytest.approx(expected, rel=1e-9)
    dt_plasma = foc.Plasma.parabolic(
        NE, TE, ions={"D": (2, 1, 0.5), "T": (3, 1, 0.5)}, alpha_n=0.0, alpha_t=0.0
    )
    dt = FUSION_REACTIONS["T(d,n)4He"]
    assert thermonuclear_rate(dt_plasma, surfaces, "T(d,n)4He") == pytest.approx(
        NE**2 / 4 * dt.reactivity(TE) * surfaces.volume, rel=1e-9
    )
    # no tritium: no D-T reactions
    assert not np.any(thermonuclear_rate(plasma, surfaces, "T(d,n)4He"))


def test_deuterium_beam_into_tritium(setup):
    equilibrium, surfaces, _, deposition = setup
    plasma = foc.Plasma.parabolic(
        NE, TE, ions={"D": (2, 1, 0.5), "T": (3, 1, 0.5)}, alpha_n=0.0, alpha_t=0.0
    )
    dt = neutron_rates(deposition, plasma, surfaces, reaction="T(d,n)4He")
    dd = neutron_rates(deposition, plasma, surfaces, target="D")
    # 100 keV deuterons react about a hundred times more often with tritons
    assert 50 < dt.total_beam_target / dd.total_beam_target < 300
    assert dt.total_thermonuclear > 0


def test_dataset_and_errors(setup):
    equilibrium, surfaces, plasma, deposition = setup
    rates = neutron_rates(deposition, plasma, surfaces, reaction="D(d,p)T")
    data = rates.to_dataset()
    assert not rates.neutron
    assert data.beam_target_rate.attrs["long_name"] == "beam-target reaction rate"
    assert data.beam_target_emissivity.values == pytest.approx(
        rates.beam_target / surfaces.volume
    )
    assert data.attrs["total_thermonuclear"] == pytest.approx(rates.total_thermonuclear)
    assert data.rho.size == len(EDGES) - 1
    empty = foc.neutrons.NeutronRates(EDGES, np.zeros(5), np.zeros(5), np.ones(5), "x")
    assert empty.reactions_per_ion == 0.0
    with pytest.raises(ValueError, match="unknown reaction"):
        neutron_rates(deposition, plasma, surfaces, reaction="p(p,g)D")
    with pytest.raises(ValueError, match="the target must be 3He"):
        neutron_rates(deposition, plasma, surfaces, reaction="3He(d,p)4He", target="D")
    hydrogen = dataclasses.replace(deposition, mass_number=1)
    with pytest.raises(ValueError, match="not a reactant"):
        neutron_rates(hydrogen, plasma, surfaces)
    with pytest.raises(ValueError, match="no 3He ions"):
        neutron_rates(deposition, plasma, surfaces, reaction="3He(d,p)4He")
    with pytest.raises(ValueError, match="no ion species"):
        neutron_rates(deposition, plasma, surfaces, target="T")
    other = equilibrium.flux_surfaces(np.linspace(0.0, 1.0, 4))
    with pytest.raises(ValueError, match="different shells"):
        neutron_rates(deposition, plasma, other)


def test_jet_like_rate_is_of_the_order_of_jet_neutron_rates():
    # JET deuterium plasmas with ~10 MW of ~100 keV deuterium beams: total DD
    # neutron rates of 1e15 to 1e16 n/s, mostly beam-target
    equilibrium = foc.Equilibrium.solovev(2.96, 1.0, 1.6, 3.0, wall_psi_n=1.3)
    edges = np.linspace(0.0, 1.0, 11)
    surfaces = equilibrium.flux_surfaces(edges)
    plasma = foc.Plasma.parabolic(5e19, 6e3, ions={"D": (2, 1, 1.0)})
    origin, direction = foc.tangential_injection(2.96, 6.0)
    beam = foc.NeutralBeam.gaussian(
        origin, direction, 110e3, 10e6, width=0.15, divergence=0.01, markers=100, seed=1
    )
    deposition = foc.deposit(
        beam, equilibrium, plasma, step=0.03, rho_edges=edges, seed=2
    )
    rates = neutron_rates(deposition, plasma, surfaces)
    assert 1e15 < rates.total_beam_target < 2e16
    assert rates.total_thermonuclear < rates.total_beam_target
