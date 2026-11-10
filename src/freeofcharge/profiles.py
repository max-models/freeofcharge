r"""Plasma profiles as functions of the normalised poloidal flux radius.

The densities and temperatures take astropy Quantities (``1e20 * u.m**-3``,
``20 * u.keV``, also arrays), strings such as ``"20 keV"`` or plain numbers in
$\mathrm{m^{-3}}$ and eV; temperatures may be given in K as well. They are kept as
plain numbers in $\mathrm{m^{-3}}$ and eV.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cunumpy as xp
import numpy as np
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS
from plasmacoll.units import to_array, to_unit

__all__ = ["IonSpecies", "Plasma"]


@dataclass(frozen=True)
class IonSpecies:
    r"""A thermal ion species; ``density`` takes quantities ($\mathrm{m^{-3}}$)."""

    mass_number: int
    charge: int
    density: np.ndarray
    """Atomic mass number (1 to 3 for hydrogen isotopes)."""
    """Charge number."""
    r"""Density on `Plasma.rho` [$\mathrm{m^{-3}}$]."""

    def __post_init__(self) -> None:
        r"""Convert a density given as a quantity to $\mathrm{m^{-3}}$."""
        object.__setattr__(self, "density", to_array(self.density, "m^-3", "density"))

    @property
    def mass(self) -> float:
        """Return the mass [kg] (mass number times the atomic mass unit)."""
        return self.mass_number * ATOMIC_MASS


@dataclass(frozen=True)
class Plasma:
    r"""Electron and ion profiles on a grid of $\rho = \sqrt{\psi_n}$.

    Between the grid points the profiles are linear; outside the last closed
    flux surface ($\rho > 1$) every density is zero. ``ne``, ``te`` and ``ti``
    take quantities (``[5, 2] * u.keV`` or a temperature in K).
    """

    rho: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ti: np.ndarray
    ions: Mapping[str, IonSpecies] = field(default_factory=dict)
    """Increasing radius grid, from 0 to 1."""
    r"""Electron density [$\mathrm{m^{-3}}$]."""
    """Electron temperature [eV]."""
    """Ion temperature [eV]."""
    """Ion species by name."""

    def __post_init__(self) -> None:
        """Convert the quantities; check the shapes and quasi-neutrality (to 1%)."""
        for name, unit in (("ne", "m^-3"), ("te", "eV"), ("ti", "eV")):
            object.__setattr__(self, name, to_array(getattr(self, name), unit, name))
        n = len(self.rho)
        for name in ("ne", "te", "ti"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} points, rho {n}"
                )
        for name, ion in self.ions.items():
            if len(ion.density) != n:
                raise ValueError(f"ion {name} has {len(ion.density)} points, rho {n}")
        if self.ions:
            charge = sum(
                ion.charge * np.asarray(ion.density) for ion in self.ions.values()
            )
            if not np.allclose(charge, self.ne, rtol=1.0e-2, atol=0.0):
                raise ValueError(
                    "the ion charge density differs from ne by more than 1%"
                )

    @classmethod
    @method(
        equations={
            "profile": r"x(\rho) = x_0 \left[ (1 - f)(1 - \rho^2)^\alpha + f \right]",
        },
        assumptions=(
            r"each ion density a fixed share of $n_e$",
            r"linear between the grid points, zero density outside $\rho = 1$ (temperatures keep the edge value)",
            r"quasi-neutral to 1% at every grid point",
        ),
    )
    def parabolic(
        cls,
        ne0: float,
        te0: float,
        ti0: float | None = None,
        *,
        ions: Mapping[str, tuple[int, int, float]],
        alpha_n: float = 0.5,
        alpha_t: float = 1.5,
        edge_fraction: float = 0.05,
        points: int = 101,
    ) -> Plasma:
        r"""Return profiles $x_0\,[(1 - f)(1 - \rho^2)^\alpha + f]$.

        ``ne0``, ``te0`` and ``ti0`` take quantities (``"1e20 m^-3"``,
        ``20 * u.keV``).

        Args:
            ne0: Central electron density [$\mathrm{m^{-3}}$].
            te0: Central electron temperature [eV].
            ti0: Central ion temperature [eV]; ``te0`` if None.
            ions: ``name -> (mass number, charge, share of ne)``, the shares
                being fractions of the electron density ($n_j = \text{share} \cdot n_e$).
            alpha_n: Peaking exponent of the densities.
            alpha_t: Peaking exponent of the temperatures.
            edge_fraction: Value at $\rho = 1$ relative to the centre.
            points: Number of grid points.
        """
        ne0 = to_unit(ne0, "m^-3", "ne0")
        te0 = to_unit(te0, "eV", "te0")
        ti0 = to_unit(ti0, "eV", "ti0")
        rho = np.linspace(0.0, 1.0, points)

        def shape(alpha: float) -> np.ndarray:
            return (1.0 - edge_fraction) * (1.0 - rho**2) ** alpha + edge_fraction

        ne = ne0 * shape(alpha_n)
        return cls(
            rho=rho,
            ne=ne,
            te=te0 * shape(alpha_t),
            ti=(te0 if ti0 is None else ti0) * shape(alpha_t),
            ions={
                name: IonSpecies(mass_number, charge, share * ne)
                for name, (mass_number, charge, share) in ions.items()
            },
        )

    def _interp(self, values: np.ndarray, rho):
        rho = xp.asarray(rho, dtype=float)
        return xp.interp(rho, xp.asarray(self.rho), xp.asarray(values))

    def _at(self, values: np.ndarray, rho):
        rho = xp.asarray(rho, dtype=float)
        return xp.where(rho <= 1.0, self._interp(values, rho), 0.0)

    def electron_density(self, rho):
        r"""Return $n_e$ [$\mathrm{m^{-3}}$] at ``rho``."""
        return self._at(self.ne, rho)

    def electron_temperature(self, rho):
        """Return $T_e$ [eV] at ``rho`` (the edge value outside)."""
        return self._interp(self.te, rho)

    def ion_temperature(self, rho):
        """Return $T_i$ [eV] at ``rho`` (the edge value outside)."""
        return self._interp(self.ti, rho)

    def ion_densities(self, rho) -> list:
        """Return the density of each ion species at ``rho``, in order."""
        return [self._at(ion.density, rho) for ion in self.ions.values()]

    @property
    def ion_mass_numbers(self) -> list[int]:
        """Return the mass number of each ion species, in order."""
        return [ion.mass_number for ion in self.ions.values()]

    @property
    def ion_masses(self) -> list[float]:
        """Return the mass [kg] of each ion species, in order."""
        return [ion.mass for ion in self.ions.values()]

    @property
    def ion_charges(self) -> list[int]:
        """Return the charge number of each ion species, in order."""
        return [ion.charge for ion in self.ions.values()]

    @method(
        equations={
            "zeff": r"Z_\mathrm{eff} = \frac{\sum_j n_j Z_j^2}{n_e}",
        },
        assumptions=(r"profiles interpolated without the cut at $\rho = 1$",),
    )
    def zeff(self, rho):
        r"""Return $Z_\mathrm{eff} = \sum_j n_j Z_j^2 / n_e$ at ``rho`` (1 outside)."""
        z2 = sum(
            ion.charge**2 * self._interp(ion.density, rho) for ion in self.ions.values()
        )
        return z2 / self._interp(self.ne, rho)
