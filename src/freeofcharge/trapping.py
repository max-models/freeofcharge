r"""Orbit-averaged pitch of fast ions: trapping and bounce-averaged scattering.

A fast ion on a flux surface (zero orbit width) keeps its energy and magnetic
moment along its orbit, so ``lambda = (1 - xi^2) B_min / B`` is constant while
its local pitch ``xi = v_par / v = sigma sqrt(1 - lambda b)``, ``b = B /
B_min``, changes. Ions with ``lambda > lambda_c = B_min / B_max`` are trapped:
over a bounce their parallel velocity averages to zero. A passing ion moves
along B at the transit average

.. math::

    \langle\xi\rangle_t = \frac{\oint dl}{\oint dl / |\xi|}
                        = \frac{1}{\overline{T}(\lambda)}, \qquad
    \overline{T}(\lambda) = \overline{(1 - \lambda b)^{-1/2}},

the bar being the average over the length of a field line, ``\overline{A} =
\langle A B\rangle / \langle B\rangle``. That is the parallel velocity its
orbit carries, so the current of the fast ions is ``e v <xi>_t`` per ion, not
``e v xi`` at its birth point.

Pitch-angle scattering, the Lorentz operator ``nu_D / 2 d/dxi (1 - xi^2)
d/dxi``, averaged over the orbit, becomes (Helander and Sigmar, *Collisional
Transport in Magnetized Plasmas*, 2002, ch. 11)

.. math::

    \langle L f\rangle_b = \frac{2 \nu_D}{\overline{T}(\lambda)}
    \frac{\partial}{\partial\lambda}\left(\lambda\,\overline{Q}(\lambda)
    \frac{\partial f}{\partial\lambda}\right), \qquad
    \overline{Q}(\lambda) = \overline{\sqrt{1 - \lambda b}\,/\,b}.

Its odd eigenfunctions ``sigma F_k(lambda)``, zero in the trapped region,
decay as ``exp(-kappa_k int nu_D dt)``; in a uniform field they are the odd
Legendre polynomials with ``kappa = l (l + 1) / 2`` (1, 6, 15, ...), and the
first gives Gaffey's decay of the mean pitch. The expected transit-averaged
pitch of an ion born at ``(lambda_0, sigma_0)`` is then

.. math::

    E[\langle\xi\rangle_t](t) = \sigma_0 \sum_k F_k(\lambda_0)
        \left(F_k, \overline{T}^{-1}\right) e^{-\kappa_k \int \nu_D dt},

with the inner product weighted by ``T``. :func:`pitch_modes` solves the
eigenproblem on each shell by finite volumes in ``lambda``; the slowing down
uses the modes in place of Gaffey's single exponential (RABBIT, Weiland et al.,
Nucl. Fusion 58, 082032, 2018, averages over guiding-centre orbits instead,
with their width).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import eigh_tridiagonal

if TYPE_CHECKING:
    from freeofcharge.equilibrium import FluxSurfaces

__all__ = ["PitchModes", "pitch_modes", "transit_pitch"]

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(4)


def _line_averages(b, line_weight, lam):
    """Return ``T(lambda)`` and ``Q(lambda)``, averages along the field line."""
    lam = np.asarray(lam, dtype=float)
    arg = np.maximum(1.0 - lam[..., None] * b, 1.0e-300)
    root = np.sqrt(arg)
    t = np.sum(line_weight / root, axis=-1)
    q = np.sum(line_weight * root / b, axis=-1)
    return t, q


def transit_pitch(b, line_weight, lam):
    """Return the transit-averaged ``|xi|`` of passing ions, zero for trapped ones.

    Args:
        b: ``B / B_min`` at points along a field line.
        line_weight: Their share of the field-line length (sum one).
        lam: ``lambda = (1 - xi^2) / b`` of the ions.
    """
    lam = np.asarray(lam, dtype=float)
    lam_c = 1.0 / np.max(b)
    passing = lam < lam_c
    t, _ = _line_averages(b, line_weight, np.where(passing, lam, 0.0))
    return np.where(passing, 1.0 / t, 0.0)


@dataclass(frozen=True)
class PitchModes:
    """Odd eigenmodes of the bounce-averaged pitch-angle scattering, per shell."""

    lam: np.ndarray
    rates: np.ndarray
    modes: np.ndarray
    projections: np.ndarray
    b_birth_scale: np.ndarray
    field: np.ndarray
    line_weight: np.ndarray
    """Nodes in ``lambda`` from 0 to ``lambda_c``, shape ``(shells, nodes)``."""
    """Decay rates ``kappa_k`` in units of ``nu_D``, shape ``(shells, modes)``;
    the last is the rate given to the remainder (see :meth:`amplitudes`)."""
    """``F_k`` on the nodes, normalised with the weight ``T``, zero at
    ``lambda_c``; shape ``(shells, modes - 1, nodes)``."""
    """``(F_k, 1 / T)``, shape ``(shells, modes - 1)``."""
    """``B_min`` of each shell [T], to turn the field at a birth into ``b``."""
    """``B / B_min`` along the field line of each shell."""
    """Shares of the field-line length of those points."""

    def amplitudes(self, shell, pitch, field_strength):
        """Return the amplitude of each mode for births, shape ``(births, modes)``.

        The sum of the amplitudes is the birth's transit-averaged pitch (signed
        along B). The modes beyond those kept are lumped into one remainder,
        given the first rate left out: an upper bound on how long it lasts.

        Args:
            shell: Shell of each birth.
            pitch: Local ``xi`` at each birth.
            field_strength: ``|B|`` at each birth [T].
        """
        shell = np.asarray(shell)
        pitch = np.asarray(pitch, dtype=float)
        b = np.asarray(field_strength, dtype=float) / self.b_birth_scale[shell]
        # a birth slightly inside the shell centre's surface may see b < 1
        lam = np.clip((1.0 - pitch**2) / np.maximum(b, 1.0), 0.0, None)
        sign = np.sign(pitch)
        n_modes = self.rates.shape[1]
        out = np.zeros((len(pitch), n_modes))
        for k in np.unique(shell):
            here = shell == k
            nodes = self.lam[k]
            values = np.stack(
                [
                    np.interp(lam[here], nodes, mode, right=0.0)
                    for mode in self.modes[k]
                ],
                axis=-1,
            )
            kept = values * self.projections[k]
            total = transit_pitch(self.field[k], self.line_weight[k], lam[here])
            out[here, :-1] = kept
            out[here, -1] = total - kept.sum(axis=-1)
        return sign[:, None] * out


def _shell_modes(b, line_weight, n_modes: int, nodes: int):
    lam_c = 1.0 / np.max(b)
    x = np.linspace(0.0, 1.0, nodes)
    # nodes crowd towards lambda_c, where T has its singularity
    lam = lam_c * (1.0 - (1.0 - x) ** 2)
    mid = 0.5 * (lam[1:] + lam[:-1])
    _, q_mid = _line_averages(b, line_weight, mid)
    stiffness = mid * q_mid / np.diff(lam)  # (nodes - 1,)
    # control volumes [c_{i-1}, c_i] of the unknown nodes 0 .. nodes - 2
    edges = np.concatenate([[0.0], mid])
    lo, hi = edges[:-1], edges[1:]
    points = 0.5 * (hi + lo)[:, None] + 0.5 * (hi - lo)[:, None] * _GL_NODES
    t_points, _ = _line_averages(b, line_weight, points)
    mass = 0.5 * (hi - lo) * np.sum(_GL_WEIGHTS * t_points, axis=-1)
    unknown = nodes - 1
    k_left = np.concatenate([[0.0], stiffness[: unknown - 1]])
    diagonal = (stiffness[:unknown] + k_left) / mass
    off = -stiffness[: unknown - 1] / np.sqrt(mass[:-1] * mass[1:])
    eigenvalues, vectors = eigh_tridiagonal(
        diagonal, off, select="i", select_range=(0, n_modes - 1)
    )
    modes = np.zeros((n_modes, nodes))
    modes[:, :unknown] = (vectors / np.sqrt(mass)[:, None]).T
    t_nodes, _ = _line_averages(b, line_weight, lam[:unknown])
    projections = modes[:, :unknown] @ (mass / t_nodes)
    return lam, 2.0 * eigenvalues, modes, projections


def pitch_modes(
    surfaces: FluxSurfaces, *, modes: int = 8, nodes: int = 200
) -> PitchModes:
    """Return the odd eigenmodes of the bounce-averaged Lorentz operator per shell.

    Args:
        surfaces: Shells with their field points (:meth:`Equilibrium.flux_surfaces`).
        modes: Modes kept; one more rate is computed for the remainder.
        nodes: Finite-volume nodes in ``lambda``.
    """
    if surfaces.field_strength is None or surfaces.surface_weight is None:
        raise ValueError("the surfaces have no field points")
    strength = surfaces.field_strength
    b_min = strength.min(axis=1)
    b = strength / b_min[:, None]
    # the average along the field line weights the surface average by B
    line_weight = surfaces.surface_weight * b
    line_weight /= line_weight.sum(axis=1, keepdims=True)
    shells = strength.shape[0]
    lam = np.empty((shells, nodes))
    rates = np.empty((shells, modes + 1))
    shapes = np.empty((shells, modes, nodes))
    projections = np.empty((shells, modes))
    for k in range(shells):
        lam[k], rates[k], shape, projection = _shell_modes(
            b[k], line_weight[k], modes + 1, nodes
        )
        shapes[k] = shape[:modes]
        projections[k] = projection[:modes]
    return PitchModes(
        lam=lam,
        rates=rates,
        modes=shapes,
        projections=projections,
        b_birth_scale=b_min,
        field=b,
        line_weight=line_weight,
    )
