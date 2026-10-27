"""The ASCOT5 benchmark: one case in both codes, and its comparison.

ASCOT5 (Hirvijoki et al., Comput. Phys. Commun. 185, 1310, 2014; Varje et al.,
arXiv:1908.02482, 2019; https://github.com/ascot4fusion/ascot5)
is the reference: BBNBI5 for the deposition and shine-through, ``ascot5_main``
(guiding centres with Monte Carlo Coulomb collisions) for the slowing down.

The functions ``bfield_2ds``, ``plasma_1d``, ``wall_2d`` and ``injector``
return keyword arguments of a5py's ``Ascot.data.create_input`` for the inputs
``B_2DS``, ``plasma_1D``, ``wall_2D`` and the ``Injector`` of ``nbi``: plain
NumPy arrays, so neither a5py nor h5py is needed here. The script
``benchmarks/ascot5/run_ascot5.py`` writes them with a5py, runs BBNBI5 and
ASCOT5 and stores what they give, per shell of ``rho``, in a JSON file of the
format of :func:`run_freeofcharge`; :func:`compare` sets the two side by side.

The case (:func:`benchmark_case`) is a Solov'ev equilibrium of ASDEX Upgrade's
size with one 60 keV deuterium beam, so that ASCOT5 slows the ions down in
minutes on a laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cunumpy as xp
import numpy as np

from freeofcharge.beam import NeutralBeam, tangential_injection
from freeofcharge.deposition import deposit
from freeofcharge.equilibrium import Equilibrium, FluxSurfaces
from freeofcharge.profiles import Plasma
from freeofcharge.slowing_down import SlowingDown, slow_down

__all__ = [
    "BenchmarkCase",
    "benchmark_case",
    "bfield_2ds",
    "compare",
    "format_comparison",
    "injector",
    "load_results",
    "plasma_1d",
    "run_freeofcharge",
    "save_results",
    "shell_amounts",
    "shell_sums",
    "wall_2d",
    "without_orbits",
]


# ----------------------------------------------------------------------
# Inputs in a5py's format
# ----------------------------------------------------------------------


def bfield_2ds(equilibrium: Equilibrium, *, psi_axis_margin: float = 1.0e-6) -> dict:
    """Return the ``B_2DS`` input: ``psi`` and ``B_phi`` on the equilibrium grid.

    ASCOT5 computes ``B_R`` and ``B_Z`` from ``psi`` with the signs of
    :mod:`freeofcharge.equilibrium`, so ``br`` and ``bz`` are zero. It stops a
    marker where its splines of ``psi`` dip below ``psi0``, so ``psi0`` is
    moved past the axis value by ``psi_axis_margin`` of the flux difference;
    ``rho^2`` shifts by as much.
    """
    r, z = equilibrium.r, equilibrium.z
    rr, zz = np.meshgrid(r, z, indexing="ij")
    _, _, b_phi = equilibrium.magnetic_field(rr.ravel(), zz.ravel())
    delta = equilibrium.psi_boundary - equilibrium.psi_axis
    zeros = np.zeros((len(r), len(z)))
    return {
        "rmin": float(r[0]),
        "rmax": float(r[-1]),
        "nr": len(r),
        "zmin": float(z[0]),
        "zmax": float(z[-1]),
        "nz": len(z),
        "axisr": equilibrium.axis[0],
        "axisz": equilibrium.axis[1],
        "psi": np.array(equilibrium.psi_grid),
        "psi0": equilibrium.psi_axis - psi_axis_margin * delta,
        "psi1": equilibrium.psi_boundary,
        "br": zeros,
        "bphi": xp.to_numpy(b_phi).reshape(rr.shape),
        "bz": zeros.copy(),
    }


def plasma_1d(plasma: Plasma, *, rho_max: float = 10.0) -> dict:
    """Return the ``plasma_1D`` input on the grid of ``plasma``, extended outside.

    ASCOT5 needs profiles wherever a marker goes, and stops it beyond the last
    point of the grid; freeofcharge's densities are zero outside ``rho = 1``.
    Two points are added, just outside 1 and at ``rho_max``, with the
    composition of the edge scaled to ``n_e = 1 m^-3`` and the edge
    temperatures.
    """
    rho = np.asarray(plasma.rho, dtype=float)
    ions = list(plasma.ions.values())
    if not ions:
        raise ValueError("the plasma needs at least one ion species")
    outside = np.array([1.0 + 1.0e-6, rho_max])
    scale = 1.0 / plasma.ne[-1]

    def extend(values, edge):
        return np.concatenate([np.asarray(values, dtype=float), [edge, edge]])

    ne = extend(plasma.ne, 1.0)
    idensity = np.stack(
        [extend(ion.density, ion.density[-1] * scale) for ion in ions], axis=1
    )
    n = len(rho) + 2
    return {
        "nrho": n,
        "nion": len(ions),
        "anum": np.array([ion.mass_number for ion in ions]),
        "znum": np.array([ion.charge for ion in ions]),
        "mass": np.array([float(ion.mass_number) for ion in ions]),
        "charge": np.array([ion.charge for ion in ions]),
        "rho": np.concatenate([rho, outside]).reshape(n, 1),
        "vtor": np.zeros((n, 1)),
        "edensity": ne.reshape(n, 1),
        "etemperature": extend(plasma.te, plasma.te[-1]).reshape(n, 1),
        "idensity": idensity,
        "itemperature": extend(plasma.ti, plasma.ti[-1]).reshape(n, 1),
    }


def wall_2d(equilibrium: Equilibrium) -> dict:
    """Return the ``wall_2D`` input: the first wall as an open (R, Z) polygon."""
    if equilibrium.wall is None:
        raise ValueError("the equilibrium has no wall")
    vertices = np.asarray(equilibrium.wall.vertices, dtype=float)
    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    return {"nelements": len(vertices), "r": vertices[:, 0], "z": vertices[:, 1]}


def injector(beam: NeutralBeam, *, ids: int = 1, divergence: float = 1.0e-10) -> dict:
    """Return the ``Injector`` of the ``nbi`` input: one beamlet per marker.

    BBNBI5 draws a beamlet at random for each of its markers and adds a
    Gaussian divergence of standard deviation ``divergence / sqrt(2)`` in
    two planes; with the default (nothing) it flies along the freeofcharge
    lines themselves. The markers must have one energy and equal weights.
    """
    beam_energies = xp.to_numpy(beam.energies)
    weights = xp.to_numpy(beam.markers.weights)
    if beam.size == 0:
        raise ValueError("the beam has no markers")
    if not np.allclose(beam_energies, beam_energies[0], rtol=1.0e-9):
        raise ValueError("BBNBI5 injectors have one energy (and its 1/2 and 1/3)")
    if not np.allclose(weights, weights[0], rtol=1.0e-9):
        raise ValueError("BBNBI5 draws beamlets with equal probability")
    positions = xp.to_numpy(beam.markers.positions)
    directions = xp.to_numpy(beam.directions)
    return {
        "ids": ids,
        "anum": beam.mass_number,
        "znum": 1,
        "mass": float(beam.mass_number),
        "energy": float(beam_energies[0]),
        "efrac": np.array([1.0, 0.0, 0.0]),
        "power": beam.power(),
        "divh": divergence,
        "divv": divergence,
        "divhalofrac": 0.0,
        "divhaloh": divergence,
        "divhalov": divergence,
        "nbeamlet": beam.size,
        "beamletx": positions[:, 0].copy(),
        "beamlety": positions[:, 1].copy(),
        "beamletz": positions[:, 2].copy(),
        "beamletdx": directions[:, 0].copy(),
        "beamletdy": directions[:, 1].copy(),
        "beamletdz": directions[:, 2].copy(),
    }


def shell_sums(rho, values, rho_edges) -> tuple[np.ndarray, np.ndarray]:
    """Return the sum of ``values`` in each shell and its Monte Carlo error.

    ``values`` are per marker (weights, powers); the error is the standard
    error of the sum, ``sqrt(sum v^2)``.
    """
    rho = np.asarray(rho, dtype=float)
    values = np.asarray(values, dtype=float)
    total, _ = np.histogram(rho, bins=rho_edges, weights=values)
    square, _ = np.histogram(rho, bins=rho_edges, weights=values**2)
    return total, np.sqrt(square)


def _parabolic_plasma(parameters: dict, ne0: float) -> Plasma:
    p = parameters
    return Plasma.parabolic(
        ne0, p["te0"], p["ti0"], ions=p["ions"], points=p["profile_points"]
    )


# ----------------------------------------------------------------------
# The case and the freeofcharge side
# ----------------------------------------------------------------------


@dataclass
class BenchmarkCase:
    """The equilibrium, plasma and beam that both codes are given."""

    equilibrium: Equilibrium
    plasma: Plasma
    beam: NeutralBeam
    rho_edges: np.ndarray
    densities: np.ndarray
    parameters: dict = field(default_factory=dict)
    """Central electron densities of the shine-through scan [m^-3]."""

    def plasma_at(self, ne0: float) -> Plasma:
        """Return the case's plasma with the central density ``ne0``."""
        return _parabolic_plasma(self.parameters, ne0)


def benchmark_case(
    *,
    markers: int = 2000,
    shells: int = 20,
    grid_points: int = 129,
    seed: int = 1,
) -> BenchmarkCase:
    """Return the benchmark case.

    A Solov'ev equilibrium of ASDEX Upgrade's size (R0 = 1.65 m, outboard
    minor radius 0.5 m, elongation 1.6, 2.5 T, q = 1 on the axis, a wall on
    ``psi_n = 1.3``); deuterium with 1% carbon (``Z_eff = 1.3``), parabolic
    profiles with 5e19 m^-3 and 2.5 keV in the centre; one 2.5 MW beam of 60
    keV deuterium tangent to R = 1.0 m from R = 3.5 m, 10 cm wide with 10
    mrad divergence, injected along the field. The shine-through is also
    scanned over the central density. The fast ions are slowed down until
    ``max(2 T_i, 1 keV)``.

    Args:
        markers: Beam markers (BBNBI5 beamlets).
        shells: Shells of ``rho`` of the profiles.
        grid_points: Points of the (R, Z) grid each way.
        seed: Seed of the beam markers.
    """
    parameters = {
        "major_radius": 1.65,
        "minor_radius": 0.5,
        "elongation": 1.6,
        "toroidal_field": 2.5,
        "q_axis": 1.0,
        "wall_psi_n": 1.3,
        "ne0": 5.0e19,
        "te0": 2.5e3,
        "ti0": 2.5e3,
        "ions": {"D": (2, 1, 0.94), "C": (12, 6, 0.01)},
        "profile_points": 101,
        "energy": 60.0e3,
        "power": 2.5e6,
        "tangency_radius": 1.0,
        "start_radius": 3.5,
        "width": 0.1,
        "divergence": 0.01,
        "markers": markers,
        "seed": seed,
        # the fast ions end at max(2 T_i, 1 keV), ASCOT5's end conditions
        "thermal_factor": 2.0,
        "minimum_energy": 1.0e3,
    }
    p = parameters
    equilibrium = Equilibrium.solovev(
        p["major_radius"],
        p["minor_radius"],
        p["elongation"],
        p["toroidal_field"],
        p["q_axis"],
        wall_psi_n=p["wall_psi_n"],
        points=grid_points,
    )
    origin, direction = tangential_injection(p["tangency_radius"], p["start_radius"])
    beam = NeutralBeam.gaussian(
        origin,
        direction,
        p["energy"],
        p["power"],
        width=p["width"],
        divergence=p["divergence"],
        markers=markers,
        seed=seed,
    )
    return BenchmarkCase(
        equilibrium=equilibrium,
        plasma=_parabolic_plasma(parameters, p["ne0"]),
        beam=beam,
        rho_edges=np.linspace(0.0, 1.0, shells + 1),
        densities=np.array([1.0e19, 2.0e19, 5.0e19]),
        parameters=parameters,
    )


def _slowing_down_results(steady: SlowingDown, surfaces: FluxSurfaces) -> dict:
    return {
        "electron_heating": steady.electron_heating.tolist(),
        "ion_heating": steady.ion_heating.tolist(),
        "fast_ion_density": steady.fast_ion_density.tolist(),
        "fast_ion_current_density": steady.fast_ion_current_density.tolist(),
        "fast_ion_current": float(
            np.sum(steady.fast_ion_current_density * surfaces.area)
        ),
        "current_density": steady.current_density.tolist(),
        "driven_current": steady.driven_current,
        "fast_ions": float(np.sum(steady.fast_ion_density * surfaces.volume)),
        "stored_energy": steady.stored_energy,
        "thermalised_power": steady.thermalised_power,
        "trapped_births": steady.trapped_births,
    }


def run_freeofcharge(
    case: BenchmarkCase,
    *,
    step: float = 0.005,
    births_per_marker: int = 4,
    seed: int = 2,
    surfaces: FluxSurfaces | None = None,
) -> dict:
    """Return freeofcharge's results for ``case`` in the format of the benchmark file.

    Per shell: the birth rate [1/s] and deposited power [W] (no sampling
    noise), the heating of electrons and ions [W], the fast-ion density
    [m^-3] and parallel current density [A/m^2]; the shine-through fraction
    at each central density of the scan; the totals. The slowing down with
    the current of the orbits (``orbit_average``) is at the top level, the
    one with the local pitch under ``"local"``, as in the ASCOT5 file.
    """
    rho_edges = case.rho_edges
    if surfaces is None:
        surfaces = case.equilibrium.flux_surfaces(rho_edges)
    scan = []
    for ne0 in case.densities:
        result = deposit(
            case.beam,
            case.equilibrium,
            case.plasma_at(float(ne0)),
            step=step,
            births_per_marker=0,
            rho_edges=rho_edges,
        )
        scan.append(result.shine_through_fraction)
    deposition = deposit(
        case.beam,
        case.equilibrium,
        case.plasma,
        step=step,
        births_per_marker=births_per_marker,
        rho_edges=rho_edges,
        seed=seed,
    )
    end = {
        "thermal_factor": case.parameters.get("thermal_factor", 0.0),
        "minimum_energy": case.parameters.get("minimum_energy", 0.0),
    }
    steady = slow_down(deposition, case.plasma, surfaces, **end)
    local = slow_down(deposition, case.plasma, surfaces, orbit_average=False, **end)
    return {
        "code": "freeofcharge",
        "rho_edges": rho_edges.tolist(),
        "volume": surfaces.volume.tolist(),
        "area": surfaces.area.tolist(),
        "scan_ne0": case.densities.tolist(),
        "scan_shine_through_fraction": [float(s) for s in scan],
        "injected_power": deposition.injected_power,
        "shine_through_fraction": deposition.shine_through_fraction,
        "birth_rate": deposition.birth_rate.tolist(),
        "deposited_power": deposition.power.tolist(),
        **_slowing_down_results(steady, surfaces),
        "local": _slowing_down_results(local, surfaces),
    }


# ----------------------------------------------------------------------
# Results files and the comparison
# ----------------------------------------------------------------------


def save_results(path: str | Path, results: dict) -> None:
    """Write a results dictionary as JSON."""
    Path(path).write_text(json.dumps(results, indent=1) + "\n")


def load_results(path: str | Path) -> dict:
    """Read a results dictionary written by :func:`save_results`."""
    return json.loads(Path(path).read_text())


#: Profiles compared shell by shell: per shell [1/s], [W] or densities.
PROFILES = (
    "birth_rate",
    "deposited_power",
    "electron_heating",
    "ion_heating",
    "fast_ion_density",
    "fast_ion_current_density",
    "current_density",
)

#: Scalars compared.
TOTALS = (
    "shine_through_fraction",
    "thermalised_power",
    "fast_ions",
    "stored_energy",
    "fast_ion_current",
    "driven_current",
)

#: What turns a density into the amount in a shell.
_MEASURE = {
    "fast_ion_density": "volume",
    "fast_ion_current_density": "area",
    "current_density": "area",
}


def _ratio(a: float, b: float) -> float | None:
    return None if b == 0.0 else float(a / b)


def shell_amounts(results: dict, name: str) -> np.ndarray:
    """Return profile ``name`` of ``results`` per shell.

    Densities are multiplied by the shell volumes or areas of the same results.
    """
    values = np.asarray(results[name], dtype=float)
    if name in _MEASURE:
        values = values * np.asarray(results[_MEASURE[name]], dtype=float)
    return values


def without_orbits(results: dict) -> dict:
    """Return the results with the slowing down of the run without orbits.

    The deposition is the same; the slowing-down profiles and totals are those
    stored under ``"local"``: for ASCOT5 the markers stay at their birth
    points, for freeofcharge the current is that of the local pitch.
    """
    return {**results, **results["local"]}


def compare(ours: dict, reference: dict) -> dict:
    """Return the agreement of ``ours`` with ``reference`` (ASCOT5).

    For each profile, the totals over the shells, their ratio and the shape
    difference: the largest difference of the shell values normalised to
    their totals, relative to the largest normalised reference value. For
    each scalar, both values and their ratio (None where the reference is
    zero). For the scan, the shine-through of both at each density.
    """
    edges_a, edges_b = np.asarray(ours["rho_edges"]), np.asarray(reference["rho_edges"])
    if edges_a.shape != edges_b.shape or not np.allclose(edges_a, edges_b):
        raise ValueError("the results have different shells")
    out: dict = {"profiles": {}, "totals": {}, "scan": []}
    for name in PROFILES:
        if name not in ours or name not in reference:
            continue
        a, b = shell_amounts(ours, name), shell_amounts(reference, name)
        total_a, total_b = float(a.sum()), float(b.sum())
        scale = np.abs(b / total_b).max()
        out["profiles"][name] = {
            "ours": total_a,
            "reference": total_b,
            "ratio": _ratio(total_a, total_b),
            "shape_difference": float(
                np.max(np.abs(a / total_a - b / total_b)) / scale
            ),
        }
    for name in TOTALS:
        if name in ours and name in reference:
            out["totals"][name] = {
                "ours": float(ours[name]),
                "reference": float(reference[name]),
                "ratio": _ratio(ours[name], reference[name]),
            }
    if "scan_ne0" in ours and "scan_ne0" in reference:
        errors = reference.get(
            "scan_shine_through_error", [0.0] * len(reference["scan_ne0"])
        )
        for ne0, a, b, error in zip(
            ours["scan_ne0"],
            ours["scan_shine_through_fraction"],
            reference["scan_shine_through_fraction"],
            errors,
            strict=True,
        ):
            out["scan"].append(
                {"ne0": ne0, "ours": a, "reference": b, "reference_error": error}
            )
    return out


def format_comparison(comparison: dict, reference: str = "ASCOT5") -> str:
    """Return the comparison as a plain-text table."""
    lines = [f"{'quantity':<28}{'freeofcharge':>14}{reference:>14}{'ratio':>8}"]
    for name, row in {**comparison["profiles"], **comparison["totals"]}.items():
        ratio = "-" if row["ratio"] is None else f"{row['ratio']:.3f}"
        lines.append(
            f"{name:<28}{row['ours']:>14.4g}{row['reference']:>14.4g}{ratio:>8}"
        )
    for row in comparison["scan"]:
        lines.append(
            f"{'shine-through ' + format(row['ne0'], '.0e'):<28}{row['ours']:>14.4f}"
            f"{row['reference']:>14.4f}  +- {row['reference_error']:.4f}"
        )
    return "\n".join(lines)
