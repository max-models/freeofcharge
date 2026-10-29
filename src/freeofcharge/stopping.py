"""Beam-stopping cross sections of fast hydrogen atoms in a plasma.

A neutral beam is attenuated as ``dI/ds = -n_e sigma_s I``, with the effective
stopping cross section ``sigma_s`` of all ionising channels (ionisation by
electrons and ions, charge exchange, and their multi-step paths through excited
states). A stopping model is a callable

``sigma_s(energy_per_amu, ne, te, ion_densities, ion_masses, ion_charges)``

of the beam energy per nucleon [eV], the electron density [m^-3] and
temperature [eV], and the density [m^-3], mass number and charge number of
each ion species, returning ``sigma_s`` [m^2] on the broadcast shape of the
arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cunumpy as xp
import numpy as np

__all__ = [
    "ConstantStopping",
    "ElectronCRStopping",
    "StoppingModel",
    "SuzukiStopping",
]


class StoppingModel(Protocol):
    """The call signature of a beam-stopping model."""

    def __call__(
        self,
        energy_per_amu,
        ne,
        te,
        ion_densities: Sequence,
        ion_mass_numbers: Sequence[int],
        ion_charges: Sequence[int],
    ) -> Any:
        """Return ``sigma_s`` [m^2]."""
        ...


@dataclass(frozen=True)
class ConstantStopping:
    """A constant stopping cross section, for tests and estimates."""

    sigma: float
    """The cross section [m^2]."""

    def __call__(
        self, energy_per_amu, ne, te, ion_densities, ion_mass_numbers, ion_charges
    ) -> Any:
        """Return ``sigma`` on the broadcast shape of the arguments."""
        shape = np.broadcast_shapes(
            np.shape(energy_per_amu), np.shape(ne), np.shape(te)
        )
        return xp.full(shape, self.sigma)


# Suzuki et al., Plasma Phys. Control. Fusion 40, 2097 (1998), equation 28: the
# coefficients A_ijk of tables 2a (100 to 10000 keV/amu) and 3a (10 to 100
# keV/amu) for plasmas of H, D and T, and B_ijk of tables 2b,c and 3b,c for the
# impurities, without the magnetic-field column (a negligible effect). As
# implemented in ASCOT5 (src/suzuki.c, LGPL-3.0), whose results the tests
# reproduce; the Li and Be rows are left out there, being equal in the paper.
# STELLOPT (LIBSTELL suzuki_sigma.f90) has the same hydrogen coefficients.
_A_HIGH = np.array(
    [
        [12.7, 1.25, 0.452, 0.0105, 0.547, -0.102, 0.360, -0.0298, -0.0959, 4.21e-3],
        [14.1, 1.11, 0.408, 0.0105, 0.547, -0.0403, 0.345, -0.0288, -0.0971, 4.74e-3],
        [12.7, 1.26, 0.449, 0.0105, 0.547, -0.00577, 0.336, -0.0282, -0.0974, 4.87e-3],
    ]
)
_A_LOW = np.array(
    [
        [-52.9, -1.36, 0.0719, 0.0137, 0.454, 0.403, -0.220, 0.0666, -0.0677, -1.48e-3],
        [-67.9, -1.22, 0.0814, 0.0139, 0.454, 0.465, -0.273, 0.0751, -0.0630, -5.08e-4],
        [-74.2, -1.18, 0.0843, 0.0139, 0.453, 0.491, -0.294, 0.0788, -0.0612, -1.85e-4],
    ]
)
#: Charge number and the Z_eff range (exclusive) of each impurity row.
_IMPURITIES = (
    (2, 1.0, 2.1),
    (6, 1.0, 5.0),
    (6, 5.0, 6.0),
    (8, 1.0, 5.0),
    (7, 1.0, 5.0),
    (5, 1.0, 5.0),
    (26, 1.0, 5.0),
)
_B_HIGH = np.array(
    [
        [
            0.231,
            0.343,
            -0.185,
            -0.162e-1,
            0.105,
            -0.703e-1,
            0.531e-1,
            0.342e-2,
            -0.838e-2,
            0.415e-2,
            -0.335e-2,
            -0.221e-3,
        ],
        [
            -0.101e1,
            -0.865e-2,
            -0.124,
            -0.145e-1,
            0.391,
            0.161e-1,
            0.298e-1,
            0.332e-2,
            -0.248e-1,
            -0.104e-2,
            -0.152e-2,
            -0.189e-3,
        ],
        [
            -0.100e1,
            -0.255e-1,
            -0.125,
            -0.142e-1,
            0.388,
            0.206e-1,
            0.297e-1,
            0.326e-2,
            -0.246e-1,
            -0.131e-2,
            -0.148e-2,
            -0.180e-3,
        ],
        [
            -0.102e1,
            -0.148e-1,
            -0.674e-1,
            -0.917e-2,
            0.359,
            0.143e-1,
            0.139e-1,
            0.184e-2,
            -0.209e-1,
            -0.732e-3,
            -0.502e-3,
            -0.949e-4,
        ],
        [
            -0.102e1,
            -0.139e-1,
            -0.979e-1,
            -0.117e-1,
            0.375,
            0.156e-1,
            0.224e-1,
            0.254e-2,
            -0.226e-1,
            -0.889e-3,
            -0.104e-2,
            -0.139e-3,
        ],
        [
            -0.732,
            0.183e-1,
            -0.155,
            -0.172e-1,
            0.321,
            0.946e-2,
            0.397e-1,
            0.420e-2,
            -0.204e-1,
            -0.619e-3,
            -0.224e-2,
            -0.254e-3,
        ],
        [
            -0.820,
            -0.636e-2,
            0.542e-1,
            0.395e-2,
            0.202,
            0.806e-3,
            -0.200e-2,
            -0.178e-2,
            -0.610e-2,
            0.651e-3,
            0.175e-2,
            0.146e-3,
        ],
    ]
)
_B_LOW = np.array(
    [
        [
            -0.792,
            0.420e-1,
            0.530e-1,
            -0.139e-1,
            0.301,
            -0.264e-1,
            -0.299e-1,
            0.607e-2,
            0.272e-3,
            0.611e-2,
            0.347e-2,
            -0.919e-3,
        ],
        [
            0.161,
            0.598e-1,
            -0.336e-2,
            -0.426e-2,
            -0.157,
            -0.396e-1,
            0.460e-2,
            0.219e-2,
            0.391e-1,
            0.711e-2,
            -0.144e-2,
            -0.385e-3,
        ],
        [
            0.158,
            0.554e-1,
            -0.431e-2,
            -0.335e-2,
            -0.155,
            -0.374e-1,
            0.537e-2,
            0.174e-2,
            0.388e-1,
            0.683e-2,
            -0.160e-2,
            -0.322e-3,
        ],
        [
            0.111,
            0.541e-1,
            -0.346e-3,
            -0.368e-2,
            -0.108,
            -0.347e-1,
            0.193e-2,
            0.181e-2,
            0.280e-1,
            0.604e-2,
            -0.841e-3,
            -0.317e-3,
        ],
        [
            0.139,
            0.606e-1,
            -0.306e-2,
            -0.455e-2,
            -0.133,
            -0.394e-1,
            0.399e-2,
            0.236e-2,
            0.335e-1,
            0.690e-2,
            -0.124e-2,
            -0.405e-3,
        ],
        [
            0.122,
            0.527e-1,
            -0.430e-3,
            -0.318e-2,
            -0.151,
            -0.364e-1,
            0.343e-2,
            0.151e-2,
            0.420e-1,
            0.692e-2,
            -0.141e-2,
            -0.290e-3,
        ],
        [
            -0.110e-1,
            0.202e-1,
            0.946e-3,
            -0.409e-2,
            -0.666e-2,
            -0.117e-1,
            -0.236e-3,
            0.202e-2,
            0.408e-2,
            0.185e-2,
            -0.648e-4,
            -0.313e-3,
        ],
    ]
)
#: Lower bound of the low-energy table and upper bound of the high-energy one
#: [keV/amu] (ASCOT5 starts the low-energy table at 9 keV/amu, below the 10 of
#: the paper).
_E_MIN_KEV = 9.0
_E_SPLIT_KEV = 100.0
_E_MAX_KEV = 10000.0


@dataclass(frozen=True)
class SuzukiStopping:
    """Suzuki's fit of the stopping cross section of H, D, T beams.

    Valid for 10 keV/amu to 10 MeV/amu, plasmas of H, D or T with He, B, C, N,
    O or Fe impurities in the Z_eff ranges of the fit. A hydrogen species is
    picked by its mass number (1 to 3); with several, their cross sections are
    averaged with their densities.

    The paper fits 10 to 100 and 100 to 10000 keV/amu separately (its tables 3
    and 2), and the two fits do not meet at 100 keV/amu: they differ there by a
    few per cent, with a sign that depends on T_e (+1.7 % at 1 keV and -4.4 %
    at 10 keV, for 5e19 m^-3). The coefficients agree with ASCOT5 and STELLOPT,
    so the step is in the published fit. Within a factor ``blend`` of 100
    keV/amu, ``log sigma`` is interpolated linearly in ``log E`` between the
    two fits, which makes the cross section continuous; ``blend=1`` gives the
    fits as published.
    """

    blend: float = 1.1
    """Half-width of the blend window, as a factor on 100 keV/amu (1: none)."""

    def __post_init__(self) -> None:
        """Check the blend window."""
        if not 1.0 <= self.blend <= 2.0:
            raise ValueError("blend must be between 1 (none) and 2")

    @staticmethod
    def _impurity_coefficients(z: int, zeff, low) -> Any:
        """Return the B row of impurity ``z`` for each point (last axis 12)."""
        zeff = xp.asarray(zeff, dtype=float)
        shape = np.broadcast_shapes(zeff.shape, np.shape(low))
        rows = xp.full(shape, -1, dtype=int)
        for row, (z_row, zmin, zmax) in enumerate(_IMPURITIES):
            if z_row == z:
                rows = xp.where((zeff > zmin) & (zeff < zmax), row, rows)
        if xp.any(rows < 0):
            raise ValueError(
                f"Suzuki's fit has no data for impurity Z = {z} at "
                f"Z_eff = {zeff.min()} to {zeff.max()}"
            )
        return xp.where(
            low[..., None], xp.asarray(_B_LOW)[rows], xp.asarray(_B_HIGH)[rows]
        )

    def _fit(
        self, e_kev, ne, te, ion_densities, ion_mass_numbers, ion_charges, low
    ) -> Any:
        """Return ``sigma_s`` [m^2] from the low-energy table where ``low``."""
        ne = xp.asarray(ne, dtype=float)
        log_e = xp.log(e_kev)
        n19 = ne * 1.0e-19
        log_n = xp.log(xp.maximum(n19, 1.0e-30))
        u = xp.log(xp.asarray(te, dtype=float) * 1.0e-3)

        hydrogen = [i for i, z in enumerate(ion_charges) if z == 1]
        impurities = [i for i, z in enumerate(ion_charges) if z != 1]
        if not hydrogen:
            raise ValueError("Suzuki's fit needs a hydrogen ion species")
        densities = [xp.asarray(n, dtype=float) for n in ion_densities]

        sigma_h = 0.0
        n_h = 0.0
        for i in hydrogen:
            a = xp.where(
                low[..., None],
                xp.asarray(_A_LOW[ion_mass_numbers[i] - 1]),
                xp.asarray(_A_HIGH[ion_mass_numbers[i] - 1]),
            )
            a = xp.moveaxis(a, -1, 0)
            sigma = (
                (a[0] * 1.0e-16 / e_kev)
                * (1.0 + a[1] * log_e + a[2] * log_e**2)
                * (
                    1.0
                    + (1.0 - xp.exp(-a[3] * n19)) ** a[4]
                    * (a[5] + a[6] * log_e + a[7] * log_e**2)
                )
                * (1.0 + a[8] * u + a[9] * u**2)
            )
            sigma_h = sigma_h + densities[i] * sigma
            n_h = n_h + densities[i]
        sigma_h = sigma_h / n_h

        charge_sum = sum(densities[i] * ion_charges[i] for i in range(len(densities)))
        zeff = (
            sum(densities[i] * ion_charges[i] ** 2 for i in range(len(densities)))
            / charge_sum
        )
        numerator = 0.0
        denominator = 0.0
        for i in impurities:
            z = ion_charges[i]
            b = self._impurity_coefficients(z, zeff, low)
            b = xp.moveaxis(b, -1, 0)
            s_z = (
                b[0]
                + b[1] * u
                + b[2] * log_n
                + b[3] * log_n * u
                + b[4] * log_e
                + b[5] * log_e * u
                + b[6] * log_e * log_n
                + b[7] * log_e * log_n * u
                + b[8] * log_e**2
                + b[9] * log_e**2 * u
                + b[10] * log_e**2 * log_n
                + b[11] * log_e**2 * log_n * u
            )
            weight = z * (z - 1) * densities[i]
            numerator = numerator + weight * s_z
            denominator = denominator + weight
        if impurities:
            positive = denominator > 0
            sigma_z = xp.where(
                positive, numerator / xp.where(positive, denominator, 1.0), 0.0
            )
        else:
            sigma_z = 0.0
        return sigma_h * (1.0 + (zeff - 1.0) * sigma_z) * 1.0e-4

    def __call__(
        self, energy_per_amu, ne, te, ion_densities, ion_mass_numbers, ion_charges
    ) -> Any:
        """Return ``sigma_s`` [m^2]; raise ValueError outside the fit."""
        e_kev = xp.asarray(energy_per_amu, dtype=float) * 1.0e-3
        if xp.any((e_kev < _E_MIN_KEV) | (e_kev >= _E_MAX_KEV)):
            raise ValueError(
                f"Suzuki's fit covers {_E_MIN_KEV} to {_E_MAX_KEV} keV/amu, "
                f"got {e_kev.min()} to {e_kev.max()}"
            )
        args = (e_kev, ne, te, ion_densities, ion_mass_numbers, ion_charges)
        low = e_kev < _E_SPLIT_KEV
        sigma = self._fit(*args, low)
        if self.blend > 1.0:
            lower, upper = _E_SPLIT_KEV / self.blend, _E_SPLIT_KEV * self.blend
            window = (e_kev > lower) & (e_kev < upper)
            if xp.any(window):
                other = self._fit(*args, ~low)
                sigma_low = xp.where(low, sigma, other)
                sigma_high = xp.where(low, other, sigma)
                w = (xp.log(e_kev) - np.log(lower)) / (np.log(upper) - np.log(lower))
                blended = xp.exp((1.0 - w) * xp.log(sigma_low) + w * xp.log(sigma_high))
                sigma = xp.where(window, blended, sigma)
        return sigma


#: The atomic mass unit [kg] and the elementary charge [C].
_AMU = 1.66053906660e-27
_E = 1.602176634e-19


class ElectronCRStopping:
    """The electron-impact part of the stopping, from OpenCR's collisional-radiative model.

    A fast atom is ionized by the plasma electrons, directly and through its
    excited levels, at OpenCR's collisional-radiative ionization coefficient
    ``S_CR(T_e, n_e)`` of ``hydrogen/atomic/basic`` (``opencr``, the ``cr``
    extra). The electrons are much faster than the beam, so the cross section
    per electron is ``sigma_e = S_CR / v_beam``. The excited levels here are
    populated by electrons only: proton impact, which populates them in a
    fusion plasma and dominates the stopping above about 30 keV/amu, needs
    proton-impact data for H(n) that OpenCR does not have yet. So this is the
    electron share and a check of Suzuki's fit (:class:`SuzukiStopping`, which
    includes everything), not a replacement for it.

    Args:
        temperatures: The electron temperatures (eV) of the table.
        densities: The electron densities (m^-3) of the table.
        n_max: The levels of the atomic model.
    """

    def __init__(
        self,
        temperatures: Sequence[float] = tuple(np.geomspace(10.0, 2.0e4, 20)),
        densities: Sequence[float] = tuple(np.geomspace(1e18, 1e21, 7)),
        n_max: int = 10,
    ) -> None:
        """Tabulate ``S_CR`` with OpenCR."""
        try:
            import opencr
            from opencr.tables import effective_rate_table
        except ImportError as error:  # pragma: no cover - the cr extra is in CI
            raise ImportError(
                "OpenCR is needed: pip install 'freeofcharge[cr]'"
            ) from error
        # cross sections to 1 MeV: the Maxwellian average at keV temperatures
        model = opencr.load_model(
            "hydrogen/atomic/basic", n_max=n_max, max_energy=1.0e6
        )
        conditions = opencr.Conditions(1.0, 1e19, densities={"H(n=1)": 1.0, "H+": 1.0})
        self.name = "effective_rate:H(n=1)>H+"
        self.table = effective_rate_table(
            model, conditions, [self.name], temperatures, densities
        )

    def __call__(
        self, energy_per_amu, ne, te, ion_densities, ion_mass_numbers, ion_charges
    ) -> Any:
        """Return ``sigma_e = S_CR(T_e, n_e) / v_beam`` [m^2]."""
        energy = np.asarray(energy_per_amu, dtype=float)
        speed = np.sqrt(2.0 * energy * _E / _AMU)
        rate = self.table(
            self.name, np.asarray(te, dtype=float), np.asarray(ne, dtype=float)
        )
        return xp.asarray(rate / speed)
