r"""Beam-stopping cross sections from tables, ADAS adf21 files or a sampled model.

`TabulatedStopping` is a `freeofcharge.stopping.StoppingModel` that interpolates
the stopping cross section $\sigma_i(E, n, T)$ of each receiver species $i$ (the
plasma ion the beam collides with) on a grid of beam energy per nucleon, density
and temperature, and combines the species of the plasma by the composite rule
of the ADAS beam-stopping coefficients (Anderson et al., Plasma Phys. Control.
Fusion 42, 781, 2000):

$$\sigma_s = \sum_i f_i\,\sigma_i(E, n_i^\star, T), \qquad
f_i = \frac{Z_i n_i}{\sum_j Z_j n_j},$$

the stopping in a pure plasma of each species, weighted with its share of the
electrons. With ``density_rule="equivalent"`` (ADAS) each species' table is read
at the equivalent electron density $n_i^\star = \sum_j Z_j^2 n_j / Z_i =
Z_\mathrm{eff} n_e / Z_i$ (ADAS adas304, as CHERAB's beam attenuator does), with
``density_rule="electron"`` at $n_i^\star = n_e$, which is the form of Suzuki's fit
(Suzuki et al., Plasma Phys. Control. Fusion 40, 2097, 1998, eq. 28): there
$\sigma_s = \sigma_H [1 + (Z_\mathrm{eff} - 1) \bar S_Z]$ with
$Z_\mathrm{eff} - 1 = \sum_Z Z (Z - 1) n_Z / n_e$, which is the rule above with
$\sigma_Z = \sigma_H [1 + (Z - 1) S_Z]$ for a pure plasma of impurity $Z$.

The tables come from

- ADAS adf21 files (`TabulatedStopping.from_adf21`, `read_adf21`): the stopping
  coefficient $S$ [$\mathrm{cm^3/s}$] of one receiver on an energy-density grid at a
  reference temperature, and its temperature dependence at the reference energy
  and density, combined as the separable product $S(E, n, T) = S_{En}(E, n)\,
  S_T(T) / S_\mathrm{ref}$; $\sigma = S / v_\mathrm{beam}$;
- any stopping model sampled on a grid (`TabulatedStopping.from_model`), so a slow
  model is tabulated once, saved to netCDF (`TabulatedStopping.save`) and
  reused (`TabulatedStopping.load`).

Between the nodes $\ln \sigma$ is interpolated linearly in $\ln E$, $\ln n$ and
$\ln T$; outside the grid the table raises, clamps to the edge or extrapolates
(``outside``, for all axes or per axis). ADAS data are licensed: none is shipped
with freeofcharge; `write_adf21` writes a file in the format, e.g. for tests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import to_unit

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.stopping import StoppingModel

__all__ = [
    "Adf21",
    "TabulatedStopping",
    "read_adf21",
    "write_adf21",
]

#: The axes of a table, in the order of its dimensions after ``species``.
AXES = ("energy", "density", "temperature")
_OUTSIDE = ("raise", "clamp", "extrapolate")
_DENSITY_RULES = ("electron", "equivalent")
#: Slack of the range check, in the logarithm (nodes count as inside).
_LOG_SLACK = 1.0e-9
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?")


def _beam_speed(energy_per_amu) -> Any:
    """Return the speed [m/s] of a beam of ``energy_per_amu`` [eV/amu]."""
    return np.sqrt(2.0 * np.asarray(energy_per_amu) * ELEMENTARY_CHARGE / ATOMIC_MASS)


# ----------------------------------------------------------------------
# ADAS adf21
# ----------------------------------------------------------------------


class Adf21:
    r"""The contents of an ADAS adf21 file (effective beam-stopping coefficients).

    In the file's units: energies [eV/amu], densities [$\mathrm{cm^{-3}}$],
    temperatures [eV], coefficients [$\mathrm{cm^3/s}$].
    """

    def __init__(
        self,
        charge: int,
        element: str,
        energies,
        densities,
        coefficients,
        temperatures,
        temperature_coefficients,
        reference: float,
        reference_energy: float,
        reference_density: float,
        reference_temperature: float,
    ) -> None:
        """Store and check the blocks of the file."""
        self.charge = int(charge)
        """Charge number of the receiver (target) ion."""
        self.element = element.strip()
        """Element symbol of the receiver."""
        self.energies = np.asarray(energies, dtype=float)
        """Beam energies of the energy-density table [eV/amu]."""
        self.densities = np.asarray(densities, dtype=float)
        r"""Target densities of the energy-density table [$\mathrm{cm^{-3}}$]."""
        self.coefficients = np.asarray(coefficients, dtype=float)
        r"""$S_{En}$ at the reference temperature, ``(energy, density)`` [$\mathrm{cm^3/s}$]."""
        self.temperatures = np.asarray(temperatures, dtype=float)
        """Target temperatures of the temperature table [eV]."""
        self.temperature_coefficients = np.asarray(
            temperature_coefficients, dtype=float
        )
        r"""$S_T$ at the reference energy and density [$\mathrm{cm^3/s}$]."""
        self.reference = float(reference)
        r"""$S_\mathrm{ref}$ at the reference energy, density and temperature."""
        self.reference_energy = float(reference_energy)
        self.reference_density = float(reference_density)
        self.reference_temperature = float(reference_temperature)
        shape = (len(self.energies), len(self.densities))
        if self.coefficients.shape != shape:
            raise ValueError(
                f"adf21 table has shape {self.coefficients.shape}, the grid {shape}"
            )
        if self.temperature_coefficients.shape != self.temperatures.shape:
            raise ValueError("adf21 temperature table and grid differ in length")
        for name in ("energies", "densities", "temperatures"):
            values = getattr(self, name)
            if np.any(values <= 0.0) or np.any(np.diff(values) <= 0.0):
                raise ValueError(f"adf21 {name} must be positive and increasing")
        if (
            np.any(self.coefficients <= 0.0)
            or np.any(self.temperature_coefficients <= 0.0)
            or self.reference <= 0.0
        ):
            raise ValueError("adf21 coefficients must be positive")


def _values(lines: list[str]) -> list[float]:
    """Return the numbers of ``lines`` (Fortran ``D`` exponents allowed)."""
    return [
        float(token.replace("D", "E").replace("d", "e"))
        for line in lines
        for token in _NUMBER.findall(line)
    ]


def _field(text: str, name: str) -> float:
    """Return the number after ``/NAME=`` in ``text``."""
    match = re.search(rf"/\s*{name}\s*=\s*({_NUMBER.pattern})", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"adf21 header has no {name}= field: {text.strip()!r}")
    return float(match.group(1).replace("D", "E").replace("d", "e"))


def read_adf21(path: str | Path) -> Adf21:
    r"""Read an ADAS adf21 file of effective beam-stopping coefficients.

    The layout, as documented by ADAS (adf21, written by adas312) and read by
    CHERAB (``cherab.openadas.parse``): a header line with the receiver's charge
    and ``/SVREF=`` (the coefficient at the reference conditions) and ``/SPEC=``
    (its element); a line with the numbers of energies and densities and
    ``/TREF=``; the energies [eV/amu] and densities [$\mathrm{cm^{-3}}$]; the
    coefficients [$\mathrm{cm^3/s}$], energy fastest, one density after another; a
    line with the number of temperatures, ``/EREF=`` and ``/NREF=``; the
    temperatures [eV] and the coefficients along them. Blocks are separated by
    lines of hyphens, values 8 per line with ``D`` or ``E`` exponents, and
    comment lines (starting with ``C``) follow the data.
    """
    text = Path(path).read_text()
    blocks: list[list[str]] = [[]]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("---"):
            blocks.append([])
        elif stripped and not stripped.upper().startswith("C"):
            blocks[-1].append(line)
    blocks = [block for block in blocks if block]
    if len(blocks) < 7:
        raise ValueError(f"{path}: not an adf21 file ({len(blocks)} blocks, need 7)")
    header, sizes, grid, table, t_header, t_grid, t_table = blocks[:7]
    first = header[0]
    charge = int(first.split("/")[0])
    element = re.search(r"/\s*SPEC\s*=\s*([A-Za-z]+)", first, re.IGNORECASE)
    n_energy, n_density = (int(v) for v in _values([sizes[0].split("/")[0]]))
    n_temperature = int(_values([t_header[0].split("/")[0]])[0])
    grid_values = _values(grid)
    table_values = _values(table)
    t_values = _values(t_grid)
    t_coefficients = _values(t_table)
    if len(grid_values) != n_energy + n_density:
        raise ValueError(
            f"{path}: {len(grid_values)} grid values, expected {n_energy + n_density}"
        )
    if len(table_values) != n_energy * n_density:
        raise ValueError(
            f"{path}: {len(table_values)} coefficients, expected {n_energy * n_density}"
        )
    if len(t_values) != n_temperature or len(t_coefficients) != n_temperature:
        raise ValueError(f"{path}: expected {n_temperature} temperatures")
    return Adf21(
        charge=charge,
        element="" if element is None else element.group(1),
        energies=grid_values[:n_energy],
        densities=grid_values[n_energy:],
        coefficients=np.reshape(table_values, (n_density, n_energy)).T,
        temperatures=t_values,
        temperature_coefficients=t_coefficients,
        reference=_field(first, "SVREF"),
        reference_energy=_field(t_header[0], "EREF"),
        reference_density=_field(t_header[0], "[ND]REF"),
        reference_temperature=_field(sizes[0], "TREF"),
    )


def _fortran(value: float) -> str:
    """Return ``value`` in the 9 characters of a Fortran ``1PD9.3`` field."""
    return f"{value:9.3E}".replace("E", "D")


def _format_block(values) -> list[str]:
    """Return ``values`` as lines of 8 Fortran ``1PD10.3`` fields."""
    fields = [" " + _fortran(v) for v in np.ravel(values)]
    return ["".join(fields[i : i + 8]) for i in range(0, len(fields), 8)]


def write_adf21(path: str | Path, data: Adf21, comment: str = "") -> None:
    """Write ``data`` as an adf21 file, in the fixed columns `read_adf21` and CHERAB read."""
    rule = "-" * 79
    lines = [
        f"{data.charge:5d} /SVREF={_fortran(data.reference)} /SPEC={data.element:<2}"
        " /DATE=        /CODE=freeofcharge",
        rule,
        f" {len(data.energies):4d} {len(data.densities):4d}"
        f" /TREF={_fortran(data.reference_temperature)}",
        rule,
        *_format_block(data.energies),
        *_format_block(data.densities),
        rule,
        *_format_block(data.coefficients.T),
        rule,
        f" {len(data.temperatures):4d} /EREF={_fortran(data.reference_energy)}"
        f" /NREF={_fortran(data.reference_density)}",
        rule,
        *_format_block(data.temperatures),
        rule,
        *_format_block(data.temperature_coefficients),
        rule,
        *[f"C {line}" for line in comment.splitlines()],
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _union_grid(grids: Sequence[np.ndarray], name: str) -> np.ndarray:
    """Return the nodes of all ``grids`` inside the range they share."""
    low = max(grid[0] for grid in grids)
    high = min(grid[-1] for grid in grids)
    if low >= high:
        raise ValueError(f"the adf21 files share no {name} range")
    nodes = np.unique(np.concatenate(grids))
    return nodes[(nodes >= low * (1 - 1e-12)) & (nodes <= high * (1 + 1e-12))]


def _interp_log(nodes: np.ndarray, values: np.ndarray, x: np.ndarray, axis: int):
    """Interpolate ``ln values`` linearly in ``ln nodes`` along ``axis`` at ``x``."""
    log_nodes = np.log(nodes)
    index = np.clip(np.searchsorted(log_nodes, np.log(x)) - 1, 0, len(nodes) - 2)
    w = (np.log(x) - log_nodes[index]) / (log_nodes[index + 1] - log_nodes[index])
    shape = [1] * values.ndim
    shape[axis] = len(x)
    w = w.reshape(shape)
    lower = np.log(np.take(values, index, axis=axis))
    upper = np.log(np.take(values, index + 1, axis=axis))
    return np.exp((1.0 - w) * lower + w * upper)


# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------


class _Axis:
    """The nodes of one axis (logarithms) and a lookup of their cells.

    A uniform grid of bins over the axis holds the cell of each bin's lower
    edge; a point then needs at most as many comparisons as a bin has nodes,
    instead of a binary search.
    """

    def __init__(self, nodes: np.ndarray) -> None:
        """Build the lookup."""
        self.nodes = nodes
        self.size = len(nodes)
        self.low = float(nodes[0])
        self.high = float(nodes[-1])
        if self.size == 1:
            return
        bins = int(min(max(4 * (self.size - 1), 16), 1 << 16))
        self.scale = bins / (self.high - self.low)
        edges = self.low + np.arange(bins) / self.scale
        self.lookup = np.clip(
            np.searchsorted(nodes, edges, side="right") - 1, 0, self.size - 2
        )
        # the most nodes inside one bin: the comparisons a point may need
        upper = np.clip(
            np.searchsorted(nodes, edges + 1.0 / self.scale, side="right") - 1,
            0,
            self.size - 2,
        )
        self.steps = int(np.max(upper - self.lookup))
        self.bins = bins

    def locate(self, x) -> tuple[Any, Any]:
        """Return the cell index and the upper node's weight of ``x`` (logarithms)."""
        nodes = self.nodes
        bin_ = ((x - self.low) * self.scale).astype(np.intp)
        index = self.lookup.take(np.clip(bin_, 0, self.bins - 1))
        for _ in range(self.steps):
            index = index + (x >= nodes.take(index + 1)) * (index < self.size - 2)
        lower = nodes.take(index)
        weight = (x - lower) / (nodes.take(index + 1) - lower)
        return index, weight


@method(
    equations={
        "composite": r"\sigma_s = \sum_i \frac{Z_i n_i}{\sum_j Z_j n_j}\, \sigma_i(E, n_i^\star, T), \qquad n_i^\star = \begin{cases} n_e & \text{electron} \\ \sum_j Z_j^2 n_j / Z_i & \text{equivalent (ADAS)} \end{cases}",
        "interpolation": r"\ln \sigma_i = \sum_{a, b, c \in \{0, 1\}} w^E_a\, w^n_b\, w^T_c\, \ln \sigma_i(E_{k+a}, n_{l+b}, T_{m+c}), \quad w_1 = \frac{\ln x - \ln x_k}{\ln x_{k+1} - \ln x_k}",
        "adf21": r"\sigma_i(E, n, T) = \frac{S_{En}(E, n)\, S_T(T)}{S_\mathrm{ref}\, v_\mathrm{beam}(E)}, \qquad v_\mathrm{beam} = \sqrt{2 E / m_u}",
        "suzuki": r"\sigma_Z = \frac{\sigma_\mathrm{mix} - (1 - f)\,\sigma_H}{f}, \quad \sigma_\mathrm{mix}: \text{the model with } Z n_Z = f n_e,\ n_H = (1 - f) n_e",
    },
    assumptions=(
        r"the plasma's stopping is the electron-weighted sum of pure-plasma stopping coefficients of each receiver (Anderson et al. 2000)",
        r"$T$ is the electron temperature of the call (ADAS tabulates the target ion temperature; the beam energy is not shifted by plasma rotation)",
        r"weights normalised with $\sum_j Z_j n_j$ (quasi-neutral plasma)",
        r"`from_model` extracts an impurity's pure-plasma $\sigma_Z$ from a mixture with the first hydrogen species, which is exact for a model of the composite form (Suzuki's fit, within one row of its $Z_\mathrm{eff}$ table)",
        r"$\ln\sigma$ multilinear in $\ln E$, $\ln n$, $\ln T$; exact for the separable adf21 product",
    ),
    references=(
        "Anderson et al., Plasma Phys. Control. Fusion 42, 781 (2000): composite beam-stopping coefficients of mixed plasmas",
        "ADAS adf21 format and adas304 composite rule; as read and applied by [CHERAB](https://github.com/cherab/core) (`openadas/parse/utility.py`, `model/attenuator/singleray.pyx`)",
        "[Suzuki et al., Plasma Phys. Control. Fusion 40, 2097 (1998)](https://doi.org/10.1088/0741-3335/40/12/009), eq. 28",
    ),
)
class TabulatedStopping:
    r"""Stopping cross sections interpolated from per-species tables.

    The table is an `xarray.Dataset` with the variable ``sigma``
    [$\mathrm{m^2}$] on the dimensions ``species``, ``energy`` [eV/amu],
    ``density`` [$\mathrm{m^{-3}}$] and ``temperature`` [eV], and the coordinates
    ``charge`` and ``mass_number`` along ``species`` (mass number 0: any isotope).
    An axis with one node is a quantity the stopping does not depend on.

    A plasma ion is matched to the table species of its charge; when several
    have that charge, to the one of its mass number. Ions without a table raise
    a ValueError, unless their density is zero. Without ions (empty lists), a
    table of one species is that species' cross section.

    Args:
        table: The tables.
        density_rule: ``"electron"`` (each species at $n_e$, Suzuki's form) or
            ``"equivalent"`` (at $\sum_j Z_j^2 n_j / Z_i$, ADAS); the table's
            ``density_rule`` attribute if None, else ``"electron"``.
        outside: What to do outside the grid: ``"raise"`` a ValueError,
            ``"clamp"`` to the edge or ``"extrapolate"`` linearly in the
            logarithms; a string for all axes or a mapping from ``"energy"``,
            ``"density"``, ``"temperature"`` (missing ones raise).
    """

    def __init__(
        self,
        table: xr.Dataset,
        *,
        density_rule: str | None = None,
        outside: str | Mapping[str, str] = "raise",
    ) -> None:
        """Check the table and prepare the interpolation."""
        if density_rule is None:
            density_rule = str(table.attrs.get("density_rule", "electron"))
        if density_rule not in _DENSITY_RULES:
            raise ValueError(f"density_rule must be one of {_DENSITY_RULES}")
        if isinstance(outside, str):
            outside = dict.fromkeys(AXES, outside)
        self.outside = {axis: outside.get(axis, "raise") for axis in AXES}
        """What happens outside the grid, per axis."""
        for axis, mode in self.outside.items():
            if mode not in _OUTSIDE:
                raise ValueError(f"outside[{axis!r}] must be one of {_OUTSIDE}")
        unknown = set(outside) - set(AXES)
        if unknown:
            raise ValueError(f"outside has unknown axes {sorted(unknown)}")
        sigma = table["sigma"].transpose("species", *AXES)
        values = np.asarray(sigma.values, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("the table's cross sections must be positive and finite")
        self.table = table.assign(sigma=sigma)
        """The tables (``sigma`` and the grid)."""
        self.table.attrs["density_rule"] = density_rule
        self.density_rule = density_rule
        """The density each species' table is read at."""
        self.charges = np.asarray(table["charge"].values, dtype=int)
        self.mass_numbers = np.asarray(table["mass_number"].values, dtype=int)
        self._axes = []
        for axis in AXES:
            nodes = np.asarray(table[axis].values, dtype=float)
            if np.any(nodes <= 0.0) or np.any(np.diff(nodes) <= 0.0):
                raise ValueError(f"the {axis} grid must be positive and increasing")
            self._axes.append(_Axis(np.log(nodes)))
        # one flat (energy, density, temperature) table of ln sigma per species
        shape = values.shape[1:]
        self._log_sigma = [np.log(v).ravel() for v in values]
        self._strides = (shape[1] * shape[2], shape[2], 1)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def dataset(
        sigma,
        energies,
        densities,
        temperatures,
        charges: Sequence[int],
        mass_numbers: Sequence[int],
        names: Sequence[str] | None = None,
        **attrs,
    ) -> xr.Dataset:
        r"""Return a table dataset from ``sigma`` [$\mathrm{m^2}$] of shape (species, energy, density, temperature)."""
        names = (
            [f"Z={z},A={a}" for z, a in zip(charges, mass_numbers, strict=True)]
            if names is None
            else list(names)
        )
        coords = {
            "species": ("species", names),
            "charge": ("species", np.asarray(charges, dtype=int)),
            "mass_number": ("species", np.asarray(mass_numbers, dtype=int)),
            "energy": (
                "energy",
                np.asarray(energies, dtype=float),
                {"long_name": "beam energy", "units": "eV/amu"},
            ),
            "density": (
                "density",
                np.asarray(densities, dtype=float),
                {"long_name": "density", "units": "m^-3"},
            ),
            "temperature": (
                "temperature",
                np.asarray(temperatures, dtype=float),
                {"long_name": "temperature", "units": "eV"},
            ),
        }
        sigma_attrs = {"long_name": r"$\sigma_s$", "units": "m^2"}
        return xr.Dataset(
            {"sigma": (("species", *AXES), np.asarray(sigma, float), sigma_attrs)},
            coords=coords,
            attrs=attrs,
        )

    @classmethod
    @sp.profile("freeofcharge.stopping_tabulate", tags=PROFILE_TAGS)
    def from_model(
        cls,
        model: StoppingModel,
        energies,
        densities,
        temperatures,
        ions: Sequence[tuple[int, int]],
        *,
        impurity_fraction: float = 0.1,
        outside: str | Mapping[str, str] = "raise",
    ) -> TabulatedStopping:
        r"""Sample ``model`` on a grid, one table per receiver species.

        A hydrogen species ($Z = 1$) is sampled in a pure plasma of it. An
        impurity is sampled in a mixture with the first hydrogen species of
        ``ions``, carrying ``impurity_fraction`` of the electrons; its pure-plasma
        cross section follows from the composite rule (``density_rule`` electron),
        $\sigma_Z = [\sigma_\mathrm{mix} - (1 - f)\,\sigma_H] / f$, which is exact for
        a model of that form such as `freeofcharge.SuzukiStopping` (in one row
        of its $Z_\mathrm{eff}$ table; $Z_\mathrm{eff} = 1 + f (Z - 1)$ here).

        Args:
            model: The stopping model.
            energies: Beam energies per nucleon [eV/amu] (or quantities).
            densities: Electron densities [$\mathrm{m^{-3}}$] (or quantities).
            temperatures: Temperatures [eV] (or quantities).
            ions: ``(mass number, charge)`` of each receiver species.
            impurity_fraction: Share $f$ of the electrons from the impurity.
            outside: See `TabulatedStopping`.
        """
        energies = np.asarray(to_unit(energies, "eV", "energies"), float).ravel()
        densities = np.asarray(to_unit(densities, "m^-3", "densities"), float).ravel()
        temperatures = np.asarray(
            to_unit(temperatures, "eV", "temperatures"), float
        ).ravel()
        if not 0.0 < impurity_fraction < 1.0:
            raise ValueError("impurity_fraction must be between 0 and 1")
        e, n, t = np.meshgrid(energies, densities, temperatures, indexing="ij")
        hydrogen = [(a, z) for a, z in ions if z == 1]

        def sample(densities_, masses, charges) -> np.ndarray:
            sigma = model(e, n, t, densities_, masses, charges)
            return np.broadcast_to(np.asarray(sigma, dtype=float), e.shape)

        sigma = []
        reference = None
        for mass, charge in ions:
            if charge == 1:
                sigma.append(sample([n], [mass], [1]))
            else:
                if not hydrogen:
                    raise ValueError(
                        "from_model needs a hydrogen species to sample an impurity in"
                    )
                if reference is None:
                    reference = sample([n], [hydrogen[0][0]], [1])
                f = impurity_fraction
                mix = sample(
                    [(1.0 - f) * n, f * n / charge],
                    [hydrogen[0][0], mass],
                    [1, charge],
                )
                pure = (mix - (1.0 - f) * reference) / f
                if np.any(pure <= 0.0):
                    raise ValueError(
                        f"the impurity Z = {charge} gives a non-positive pure-plasma "
                        "cross section; the model is not of the composite form"
                    )
                sigma.append(pure)
        table = cls.dataset(
            np.stack(sigma),
            energies,
            densities,
            temperatures,
            [z for _, z in ions],
            [a for a, _ in ions],
            density_rule="electron",
            source=f"sampled from {type(model).__name__}",
        )
        return cls(table, outside=outside)

    @classmethod
    @sp.profile("freeofcharge.stopping_read_adf21", tags=PROFILE_TAGS)
    def from_adf21(
        cls,
        files: Sequence[str | Path | Adf21] | Mapping[str, str | Path | Adf21],
        *,
        mass_numbers: Mapping[str, int] | Sequence[int] | None = None,
        outside: str | Mapping[str, str] = "raise",
    ) -> TabulatedStopping:
        r"""Return the table of one or more adf21 files (one per receiver).

        The cross section is $\sigma = S_{En}(E, n) S_T(T) / (S_\mathrm{ref}
        v_\mathrm{beam})$ on the grid of the file's energies, densities and
        temperatures. Files on different grids are combined on the union of
        their nodes inside the range they share (exact: the piecewise
        log-linear interpolation of each file is linear on the finer cells).
        ``density_rule`` is ``"equivalent"``, the ADAS rule.

        Args:
            files: Paths or `Adf21` contents; a mapping names the species.
            mass_numbers: Mass number of each species (0, any isotope, if None).
            outside: See `TabulatedStopping`.
        """
        if isinstance(files, Mapping):
            names = list(files)
            items = list(files.values())
        else:
            items = list(files)
            names = None
        data = [item if isinstance(item, Adf21) else read_adf21(item) for item in items]
        if names is None:
            names = [f"{d.element or 'Z'}{d.charge}+" for d in data]
        if mass_numbers is None:
            masses = [0] * len(data)
        elif isinstance(mass_numbers, Mapping):
            masses = [int(mass_numbers.get(name, 0)) for name in names]
        else:
            masses = [int(m) for m in mass_numbers]
        energies = _union_grid([d.energies for d in data], "energy")
        densities = _union_grid([d.densities for d in data], "density")
        temperatures = _union_grid([d.temperatures for d in data], "temperature")
        sigma = []
        for d in data:
            s_en = _interp_log(
                d.densities,
                _interp_log(d.energies, d.coefficients, energies, 0),
                densities,
                1,
            )
            s_t = _interp_log(
                d.temperatures, d.temperature_coefficients, temperatures, 0
            )
            s = s_en[:, :, None] * s_t[None, None, :] / d.reference  # cm^3/s
            sigma.append(s * 1.0e-6 / _beam_speed(energies)[:, None, None])
        table = cls.dataset(
            np.stack(sigma),
            energies,
            densities * 1.0e6,
            temperatures,
            [d.charge for d in data],
            masses,
            names,
            density_rule="equivalent",
            source="ADAS adf21",
        )
        return cls(table, outside=outside)

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the table to a netCDF file."""
        self.table.to_netcdf(path)

    @classmethod
    def load(
        cls, path: str | Path, *, outside: str | Mapping[str, str] = "raise"
    ) -> TabulatedStopping:
        """Return the table saved in ``path``."""
        with xr.open_dataset(path) as dataset:
            table = dataset.load()
        return cls(table, outside=outside)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _locate(self, axis: int, values) -> tuple[Any, Any]:
        """Return the cell index and the weight of the upper node along ``axis``."""
        grid = self._axes[axis]
        x = np.log(values)
        if grid.size == 1:
            return 0, 0.0
        mode = self.outside[AXES[axis]]
        if mode == "raise" and (
            np.min(x) < grid.low - _LOG_SLACK or np.max(x) > grid.high + _LOG_SLACK
        ):
            unit = self.table[AXES[axis]].attrs.get("units", "")
            raise ValueError(
                f"the stopping table covers {AXES[axis]} {np.exp(grid.low):.4g} to "
                f"{np.exp(grid.high):.4g} {unit}, got {np.exp(np.min(x)):.4g} to "
                f"{np.exp(np.max(x)):.4g}; pass outside='clamp' or "
                "'extrapolate' to allow it"
            )
        if mode != "extrapolate":
            x = np.clip(x, grid.low, grid.high)
        return grid.locate(x)

    def _cells(self, energy, density, temperature) -> tuple[Any, list]:
        """Return the flat index of each point's cell and the axis weights."""
        base: Any = 0
        weights = []
        for axis, values in enumerate((energy, density, temperature)):
            index, weight = self._locate(axis, values)
            base = base + index * self._strides[axis]
            weights.append(weight)
        return base, weights

    def _log_sigma_at(self, cells: tuple[Any, list], species: int) -> Any:
        r"""Return $\ln \sigma$ of ``species`` in ``cells`` (trilinear)."""
        base, (w_e, w_n, w_t) = cells
        table = self._log_sigma[species]
        s_e, s_n, s_t = (
            stride if grid.size > 1 else 0
            for stride, grid in zip(self._strides, self._axes, strict=True)
        )

        def along_t(offset: int) -> Any:
            lower = table.take(base + offset)
            if not s_t:
                return lower
            return lower + w_t * (table.take(base + offset + s_t) - lower)

        def along_n(offset: int) -> Any:
            lower = along_t(offset)
            if not s_n:
                return lower
            return lower + w_n * (along_t(offset + s_n) - lower)

        lower = along_n(0)
        if not s_e:
            return lower
        return lower + w_e * (along_n(s_e) - lower)

    def _match(self, mass_number: int, charge: int) -> int:
        """Return the table species of a plasma ion."""
        same = np.flatnonzero(self.charges == charge)
        if len(same) > 1:
            same = same[
                (self.mass_numbers[same] == mass_number)
                | (self.mass_numbers[same] == 0)
            ]
        if len(same) == 0:
            raise ValueError(
                f"the stopping table has no species for the ion A = {mass_number}, "
                f"Z = {charge} (it has charges {sorted(set(self.charges.tolist()))})"
            )
        return int(same[0])

    def species_sigma(self, energy_per_amu, density, temperature) -> np.ndarray:
        r"""Return $\sigma_i$ [$\mathrm{m^2}$] of every species at one density, ``(..., species)``."""
        e, n, t = np.broadcast_arrays(
            np.asarray(energy_per_amu, float),
            np.asarray(density, float),
            np.asarray(temperature, float),
        )
        cells = self._cells(e, n, t)
        return np.stack(
            [
                np.broadcast_to(np.exp(self._log_sigma_at(cells, i)), e.shape)
                for i in range(len(self.charges))
            ],
            axis=-1,
        )

    def _has_charge(self, charge: int) -> bool:
        """Whether a table species has ``charge``."""
        return bool(np.any(self.charges == charge))

    def __call__(
        self, energy_per_amu, ne, te, ion_densities, ion_mass_numbers, ion_charges
    ) -> Any:
        r"""Return $\sigma_s$ [$\mathrm{m^2}$] of the plasma's composite (see the module)."""
        energy = np.asarray(energy_per_amu, dtype=float)
        ne = np.asarray(ne, dtype=float)
        te = np.asarray(te, dtype=float)
        densities = [np.asarray(n, dtype=float) for n in ion_densities]
        shape = np.broadcast_shapes(
            energy.shape, ne.shape, te.shape, *(n.shape for n in densities)
        )
        energy, ne, te = (np.broadcast_to(v, shape) for v in (energy, ne, te))
        if not densities:
            if len(self.charges) != 1:
                raise ValueError(
                    "a table of several species needs the ion densities to combine them"
                )
            cells = self._cells(energy, ne, te)
            return xp.asarray(np.exp(self._log_sigma_at(cells, 0)))
        ions = [
            (self._match(a, z), z, n)
            for n, a, z in zip(densities, ion_mass_numbers, ion_charges, strict=True)
            if np.any(n != 0.0) or self._has_charge(z)
        ]
        if len(ions) == 1:  # a pure plasma: its table
            weights = [1.0]
            ions_z = [ions[0][1]]
        else:
            electrons = sum(z * n for _, z, n in ions)
            electrons = np.where(electrons > 0.0, electrons, 1.0)
            weights = [z * n / electrons for _, z, n in ions]
            ions_z = [z for _, z, _ in ions]
        if self.density_rule == "electron":
            cells = self._cells(energy, ne, te)
            total = sum(
                w * np.exp(self._log_sigma_at(cells, i))
                for w, (i, _, _) in zip(weights, ions, strict=True)
            )
        else:
            charge_squared = sum(z * z * n for _, z, n in ions)
            # where no ions: the electron density (the weights vanish there)
            charge_squared = np.where(charge_squared > 0.0, charge_squared, ne)
            total = sum(
                w
                * np.exp(
                    self._log_sigma_at(self._cells(energy, charge_squared / z, te), i)
                )
                for w, z, (i, _, _) in zip(weights, ions_z, ions, strict=True)
            )
        return xp.asarray(np.broadcast_to(total, shape))
