import numpy as np
import pytest

from freeofcharge import Equilibrium
from freeofcharge.trapping import _shell_modes, pitch_modes, transit_pitch


def circular(epsilon, points=256):
    """``B / B_min`` and field-line weights of a large-aspect-ratio circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    field = 1.0 / (1.0 + epsilon * np.cos(theta))
    return field / field.min(), field / field.sum()


def test_uniform_field_has_the_legendre_eigenvalues():
    b, weight = circular(0.0)
    _, rates, modes, projections = _shell_modes(b, weight, 4, 400)
    # odd Legendre polynomials: l (l + 1) / 2
    assert rates == pytest.approx([1.0, 6.0, 15.0, 28.0], rel=3e-3)
    # the first mode is xi itself and carries all of it
    # normalised with the weight T = 1 / xi: F = sqrt(3/2) xi, (F, xi) = sqrt(3/2) 2/3
    assert abs(projections[0]) == pytest.approx(np.sqrt(1.5) * 2.0 / 3.0, rel=1e-2)
    assert np.abs(projections[1:]).max() < 1e-2 * abs(projections[0])


def test_transit_pitch_is_the_pitch_in_a_uniform_field_and_zero_when_trapped():
    b, weight = circular(0.0)
    xi = np.array([0.1, 0.5, 0.9])
    assert transit_pitch(b, weight, 1.0 - xi**2) == pytest.approx(xi)
    b, weight = circular(0.2)
    lam_c = 1.0 / b.max()
    pitch = transit_pitch(b, weight, np.array([0.0, 0.5 * lam_c, 1.01 * lam_c]))
    assert pitch[0] == pytest.approx(1.0)
    assert 0.0 < pitch[1] < np.sqrt(1.0 - 0.5 * lam_c)
    assert pitch[2] == 0.0


def test_trapping_makes_the_current_decay_faster():
    previous = 1.0
    for epsilon in (0.02, 0.1, 0.3):
        b, weight = circular(epsilon)
        _, rates, _, _ = _shell_modes(b, weight, 2, 400)
        assert rates[0] > previous
        previous = rates[0]


def test_bounce_averaged_modes_match_a_monte_carlo():
    """Lorentz scattering of markers spread along their orbits at each step.

    Between collision steps a marker keeps its ``lambda`` and (if passing) its
    direction, and is put at a random point of its orbit with the time weight
    ``dl / |xi|``: the limit of a transit much faster than the scattering. Its
    mean pitch is the sum of the modes, each decaying at its rate.
    """
    b, weight = circular(0.1, points=128)
    lam, rates, modes, projections = _shell_modes(b, weight, 20, 400)
    xi0 = 0.9
    amplitudes = np.array([np.interp(1.0 - xi0**2, lam, m) for m in modes])
    amplitudes *= projections
    rng = np.random.default_rng(3)
    n, dt = 4000, 4.0e-3
    lam_c = 1.0 / b.max()
    xi = np.full(n, xi0)
    index = np.zeros(n, dtype=int)
    lam_markers = (1.0 - xi**2) / b[index]
    for step in range(int(0.6 / dt) + 1):
        arg = 1.0 - lam_markers[:, None] * b
        density = np.where(arg > 0.0, b / np.sqrt(np.maximum(arg, 1e-12)), 0.0)
        cdf = np.cumsum(density, axis=1)
        index = np.minimum(
            (cdf < rng.random(n)[:, None] * cdf[:, -1:]).sum(1), b.size - 1
        )
        trapped = lam_markers >= lam_c
        sign = np.where(trapped, rng.choice([-1.0, 1.0], n), np.sign(xi))
        xi = sign * np.sqrt(np.maximum(1.0 - lam_markers * b[index], 0.0))
        if step % 50 == 0:
            expected = np.sum(amplitudes * np.exp(-rates * step * dt))
            error = xi.std() / np.sqrt(n)
            # the markers' time step leaves a bias of a few percent (gone at dt = 2.5e-4)
            assert xi.mean() == pytest.approx(
                expected, abs=4.0 * error + 0.03 * expected
            )
        kick = np.sqrt((1.0 - xi**2) * dt) * rng.standard_normal(n)
        xi = np.clip(xi - xi * dt + kick, -1.0, 1.0)
        lam_markers = (1.0 - xi**2) / b[index]
    # against the uniform field's xi0 exp(-t): trapping speeds up the decay
    assert xi.mean() < 0.8 * xi0 * np.exp(-0.6)


def test_amplitudes_sum_to_the_transit_pitch():
    equilibrium = Equilibrium.solovev(1.65, 0.5, 1.6, 2.5, wall_psi_n=None)
    surfaces = equilibrium.flux_surfaces(np.linspace(0.0, 1.0, 6), resolution=100)
    modes = pitch_modes(surfaces)
    assert modes.rates.shape == (5, 9)
    assert np.all(modes.rates[:, 0] > 1.0)  # trapping on every surface
    assert np.all(np.diff(modes.rates, axis=1) > 0.0)
    shell = np.array([0, 2, 4, 4, 4])
    pitch = np.array([0.9, -0.6, 0.95, 0.1, -0.3])
    field = modes.b_birth_scale[shell] * np.array([1.0, 1.05, 1.0, 1.0, 1.1])
    amplitudes = modes.amplitudes(shell, pitch, field)
    lam = (1.0 - pitch**2) / (field / modes.b_birth_scale[shell])
    expected = [
        np.sign(p) * transit_pitch(modes.field[k], modes.line_weight[k], [x])[0]
        for k, p, x in zip(shell, pitch, lam, strict=True)
    ]
    assert amplitudes.sum(axis=1) == pytest.approx(expected, abs=1e-12)
    # strongly passing: mostly the first mode; deeply trapped: nothing
    assert abs(amplitudes[2, 0]) > 5.0 * np.abs(amplitudes[2, 1:]).max()
    assert np.all(amplitudes[3] == 0.0)
