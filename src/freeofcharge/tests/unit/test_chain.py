"""Tests for the injector-chain stages."""

import numpy as np
import pytest

from freeofcharge import chain


def test_child_langmuir_of_a_deuterium_extraction_gap():
    # j = (4/9) eps0 sqrt(2 e / m) V^(3/2) / d^2 for D- across 9 kV and 6 mm
    assert chain.child_langmuir(9.0e3, 6.0e-3) == pytest.approx(913.6, rel=1e-3)
    assert chain.child_langmuir(2 * 9.0e3, 6.0e-3) == pytest.approx(
        2**1.5 * chain.child_langmuir(9.0e3, 6.0e-3)
    )


def test_neutral_fraction_matches_the_closed_form_and_its_maximum():
    pytest.importorskip("plasmacoll")
    pi = np.linspace(0.0, 4.0e20, 201)
    numeric = chain.neutral_fraction(pi)
    np.testing.assert_allclose(
        numeric, chain.neutral_fraction_closed_form(pi), atol=1e-12
    )
    assert numeric[0] == 0.0
    assert 0.55 < numeric.max() < 0.60  # gas neutralizers of 1 MeV D-
    assert 1.2e20 < pi[np.argmax(numeric)] < 1.6e20
    unsorted = chain.neutral_fraction([3e20, 1e20])
    np.testing.assert_allclose(
        unsorted, chain.neutral_fraction([3e20, 1e20][::-1])[::-1]
    )


def test_duct_transmission_limits():
    assert chain.duct_transmission(0.1, 0.0, 23.0, 10.0) == pytest.approx(1.0)
    narrow = chain.duct_transmission(0.1, 5e-3, 23.0, 0.25)
    wide = chain.duct_transmission(0.1, 1e-2, 23.0, 0.25)
    assert 0.0 < wide < narrow < 1.0


def test_source_current_grows_with_rf_power():
    pytest.importorskip("wellbalanced")
    low, high = chain.deuterium_source(2e5), chain.deuterium_source(4e5)
    assert high.current_density > low.current_density > 0.0
    # the same temperature, twice the power: twice the density (a power balance)
    assert high.electron_temperature == pytest.approx(
        low.electron_temperature, rel=1e-3
    )
    assert high.electron_density == pytest.approx(2 * low.electron_density, rel=1e-3)
    assert low.current_density == pytest.approx(
        0.3 * chain.ELEMENTARY_CHARGE * low.positive_flux
    )


def test_slit_divergence_has_a_v_curve():
    pytest.importorskip("tree_d_3d")
    divergence, transmitted = chain.slit_divergence(
        [0.02, 0.15, 0.45], rays=12, iterations=6
    )
    assert np.all(transmitted == 1.0)
    assert divergence[1] < divergence[0] and divergence[1] < divergence[2]
