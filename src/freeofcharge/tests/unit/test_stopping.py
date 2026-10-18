import numpy as np
import pytest

from freeofcharge import ConstantStopping, SuzukiStopping

# ASCOT5's suzuki_sigmav (src/suzuki.c, compiled stand-alone) on the same
# inputs: energy [keV/amu], ne [m^-3], Te [eV], ions (density, mass number,
# charge) -> sigma_s [m^2].
D_ONLY = [(1.0, 2, 1)]
D_C1 = [(0.94, 2, 1), (0.01, 12, 6)]
D_C2 = [(0.88, 2, 1), (0.02, 12, 6)]
H_HE = [(0.90, 1, 1), (0.05, 4, 2)]
DT_HE = [(0.47, 2, 1), (0.47, 3, 1), (0.03, 4, 2)]
ASCOT5_REFERENCE = [
    (50.0, 5e19, 2e3, D_ONLY, 3.806782497586e-20),
    (500.0, 1e20, 1e4, D_ONLY, 6.848872868766e-21),
    (500.0, 1e20, 1e4, D_C1, 8.039822179840e-21),
    (250.0, 3e19, 5e3, D_C2, 1.384044021064e-20),
    (30.0, 2e19, 1e3, H_HE, 5.455243307826e-20),
    (1000.0, 8e19, 2e4, DT_HE, 3.871261187391e-21),
]


@pytest.mark.parametrize(("energy", "ne", "te", "ions", "expected"), ASCOT5_REFERENCE)
def test_suzuki_reproduces_ascot5(energy, ne, te, ions, expected):
    sigma = SuzukiStopping()(
        energy * 1e3,
        ne,
        te,
        [share * ne for share, _, _ in ions],
        [a for _, a, _ in ions],
        [z for _, _, z in ions],
    )
    assert float(sigma) == pytest.approx(expected, rel=1e-10)


def test_suzuki_is_vectorised():
    energies = np.array([50e3, 500e3, 1000e3])
    ne = np.array([5e19, 1e20, 8e19])
    te = np.array([2e3, 1e4, 2e4])
    sigma = SuzukiStopping()(energies, ne, te, [ne], [2], [1])
    single = [
        SuzukiStopping()(e, n, t, [n], [2], [1])
        for e, n, t in zip(energies, ne, te, strict=True)
    ]
    np.testing.assert_allclose(sigma, np.array(single, dtype=float), rtol=1e-14)


def test_suzuki_falls_with_energy():
    energies = np.geomspace(20e3, 5e6, 30)
    sigma = SuzukiStopping()(energies, 5e19, 5e3, [5e19], [2], [1])
    assert np.all(np.diff(sigma) < 0)


def test_suzuki_rejects_energies_outside_the_fit():
    with pytest.raises(ValueError, match="keV/amu"):
        SuzukiStopping()(5e3, 5e19, 2e3, [5e19], [2], [1])


def test_suzuki_rejects_unknown_impurities():
    with pytest.raises(ValueError, match="impurity"):
        SuzukiStopping()(100e3, 5e19, 2e3, [4e19, 1e18], [2, 40], [1, 10])


def test_suzuki_needs_hydrogen():
    with pytest.raises(ValueError, match="hydrogen"):
        SuzukiStopping()(100e3, 5e19, 2e3, [2.5e19], [4], [2])


def test_constant_stopping_broadcasts():
    sigma = ConstantStopping(1e-20)(np.ones(4), 1e19, np.ones((3, 1)), [], [], [])
    assert sigma.shape == (3, 4)
    assert np.all(sigma == 1e-20)
