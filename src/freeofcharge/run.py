r"""One call from beams to profiles: `run_nbi`, its parameter file and the command line.

`run_nbi` chains the steps of a neutral-beam calculation on one set of flux
shells and returns a single `xarray.Dataset` over $\rho$:

1. deposition: `freeofcharge.deposit` for each `NeutralBeam`,
   `freeofcharge.deposit_sources` for the energy components of each
   `BeamSource` (and the sources of each `InjectorBox`): birth and power
   density, shine-through per source;
2. slowing down: `freeofcharge.slow_down` of all births of one isotope
   together: electron and ion heating, fast-ion density and pressure, driven
   current;
3. torque: `freeofcharge.torque.beam_torque`, collisional and
   $\mathbf{j}\times\mathbf{B}$ in an axisymmetric `freeofcharge.Equilibrium`,
   collisional only on the shells of a 3D equilibrium;
4. neutrons: `freeofcharge.neutrons.neutron_rates` for each neutron-producing
   reaction of `plasmacoll.fusion.FUSION_REACTIONS` whose reactants are a beam
   isotope and an ion species of the plasma (a D beam in a D plasma: D(d,n)3He;
   in a DT plasma also T(d,n)4He); skipped when there is none, e.g. a hydrogen
   beam.

Beams of different isotopes are slowed down separately (`Deposition.combine`
joins one isotope only) and their profiles added. Each step is the function
named above with its own defaults, so the numbers are those of calling them one
by one (the tests check that the totals agree exactly); the physics and its
assumptions are in their docstrings: zero orbit width, no losses, no charge
exchange of the fast ions.

The parameter file is TOML (`read_params`), with the sections
``[equilibrium]``, ``[plasma]``, ``[[beams]]``, ``[numerics]`` and the optional
``[slowing_down]``, ``[torque]`` and ``[neutrons]``; all numbers in SI units, eV
for energies and temperatures and rad for angles. ``freeofcharge run params.toml
-o out.nc`` runs it and writes the dataset, ``freeofcharge summary out.nc``
prints its totals (`main`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr
from maybempi import MPI
from plasmacoll.fusion import FUSION_REACTIONS

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.beam import NeutralBeam, tangential_injection
from freeofcharge.deposition import Deposition, deposit
from freeofcharge.equilibrium import (
    Equilibrium,
    FluxSurfaces,
    MagneticEquilibrium,
    radial_coordinate_of,
    rho_coordinate,
)
from freeofcharge.injectors import COMPONENTS, BeamSource, InjectorBox, deposit_sources
from freeofcharge.neutrons import _species, neutron_rates
from freeofcharge.profiles import IonSpecies, Plasma
from freeofcharge.slowing_down import SlowingDown, slow_down
from freeofcharge.stopping import StoppingModel
from freeofcharge.torque import beam_torque

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

__all__ = [
    "ParameterError",
    "RunInputs",
    "build_inputs",
    "main",
    "read_params",
    "run_nbi",
    "run_params",
    "summary",
]

Beams = NeutralBeam | BeamSource | InjectorBox
"""What `run_nbi` deposits: markers, or sources with energy components."""

#: The isotopes of the beam that take part in fusion reactions.
_REACTANTS = {2: "D", 3: "T"}

#: The 0-d totals of the dataset, in the order `summary` prints them.
_TOTALS = (
    "injected_power",
    "absorbed_power",
    "shine_through_power",
    "shine_through_fraction",
    "electron_heating_power",
    "ion_heating_power",
    "thermalised_power",
    "stored_energy",
    "driven_current",
    "trapped_births",
    "torque_collisional_electrons_total",
    "torque_collisional_ions_total",
    "torque_jxb_total",
    "torque_total",
    "birth_angular_momentum",
    "thermalised_angular_momentum",
    "beam_target_neutron_rate",
    "thermonuclear_neutron_rate",
    "neutron_rate",
)


def _variable(dims, values, long_name: str, units: str) -> tuple:
    return (dims, values, {"long_name": long_name, "units": units})


def _scalar(value: float, long_name: str, units: str) -> tuple:
    return _variable((), float(value), long_name, units)


def _named(beams) -> list[tuple[str, Beams]]:
    """Return ``(name, item)`` for a beam, a sequence or a mapping of them."""
    if isinstance(beams, (NeutralBeam, BeamSource, InjectorBox)):
        items: list[tuple[str | None, Any]] = [(None, beams)]
    elif isinstance(beams, Mapping):
        items = list(beams.items())
    else:
        items = [(None, item) for item in beams]
    if not items:
        raise ValueError("no beams to deposit")
    named: list[tuple[str, Beams]] = []
    for i, (name, item) in enumerate(items):
        if isinstance(item, NeutralBeam):
            named.append((f"beam{i + 1}" if name is None else str(name), item))
        elif isinstance(item, (BeamSource, InjectorBox)):
            named.append((item.name, item))
        else:
            raise TypeError(
                "beams are NeutralBeam, BeamSource or InjectorBox, "
                f"not {type(item).__name__}"
            )
    names = [name for name, _ in named]
    if len(set(names)) != len(names):
        raise ValueError(f"beam names must be unique: {names}")
    return named


def _deposit_all(
    named: list[tuple[str, Beams]],
    equilibrium: MagneticEquilibrium,
    plasma: Plasma,
    *,
    markers: int,
    seed: int | None,
    comm: MPI.Comm | None,
    options: dict,
) -> tuple[dict[tuple[str, str], Deposition], bool]:
    """Return the depositions by ``(source, component)`` and whether there are sources."""
    beams = [(name, item) for name, item in named if isinstance(item, NeutralBeam)]
    sources = [item for _, item in named if not isinstance(item, NeutralBeam)]
    depositions: dict[tuple[str, str], Deposition] = {}
    for i, (name, beam) in enumerate(beams):
        depositions[(name, COMPONENTS[0])] = deposit(
            beam,
            equilibrium,
            plasma,
            seed=None if seed is None else seed + i,
            comm=comm,
            **options,
        )
    if sources:
        tabulated = deposit_sources(
            sources,
            equilibrium,
            plasma,
            markers=markers,
            seed=None if seed is None else seed + 10 * len(beams),
            comm=comm,
            **options,
        )
        depositions.update(tabulated.depositions)
        if len(depositions) != len(beams) + len(tabulated.depositions):
            raise ValueError(
                f"beam names {[name for name, _ in beams]} and source names "
                f"{tabulated.sources} must differ"
            )
    return depositions, bool(sources)


def _per_source(
    depositions: dict[tuple[str, str], Deposition],
    surfaces: FluxSurfaces,
    with_components: bool,
) -> dict[str, tuple]:
    """Return the profiles and powers of each source (and energy component)."""
    sources = list(dict.fromkeys(source for source, _ in depositions))
    components: list[str] = list(COMPONENTS) if with_components else [COMPONENTS[0]]
    n_shells = len(surfaces.rho_edges) - 1
    shape = (len(sources), len(components))
    birth = np.full((*shape, n_shells), np.nan)
    power = np.full((*shape, n_shells), np.nan)
    scalars = {
        name: np.full(shape, np.nan)
        for name in ("injected", "absorbed", "shine_through")
    }
    for (source, component), d in depositions.items():
        index = (sources.index(source), components.index(component))
        birth[index] = d.birth_rate
        power[index] = d.power
        scalars["injected"][index] = d.injected_power
        scalars["absorbed"][index] = d.absorbed_power
        scalars["shine_through"][index] = d.shine_through_power
    fraction = scalars["shine_through"] / scalars["injected"]
    dims: tuple[str, ...] = ("source", "component")
    if not with_components:
        birth, power, fraction = birth[:, 0], power[:, 0], fraction[:, 0]
        scalars = {name: values[:, 0] for name, values in scalars.items()}
        dims = ("source",)
    return {
        "source_birth_density": _variable(
            (*dims, "rho"),
            birth / surfaces.volume,
            "birth rate density per source",
            "1/(m^3 s)",
        ),
        "source_power_density": _variable(
            (*dims, "rho"),
            power / surfaces.volume,
            "deposited power density per source",
            "W/m^3",
        ),
        "source_injected_power": _variable(
            dims, scalars["injected"], "injected power per source", "W"
        ),
        "source_absorbed_power": _variable(
            dims, scalars["absorbed"], "absorbed power per source", "W"
        ),
        "source_shine_through_power": _variable(
            dims, scalars["shine_through"], "shine-through power per source", "W"
        ),
        "source_shine_through_fraction": _variable(
            dims, fraction, "shine-through fraction per source", ""
        ),
    }


def _add(first: dict, second: dict) -> dict:
    """Return the element-wise sums of two dicts of arrays and numbers."""
    return {key: first[key] + second[key] for key in first}


def _slowing_down_sums(results: list[tuple[SlowingDown, float]]) -> dict:
    """Return the additive parts of the slowing down of each isotope, summed."""
    total: dict | None = None
    births = 0.0
    trapped = 0.0
    for result, birth_rate in results:
        parts = {
            name: getattr(result, name)
            for name in (
                "electron_heating",
                "ion_heating",
                "fast_ion_density",
                "fast_ion_pressure",
                "fast_ion_current_density",
                "current_density",
                "driven_current",
                "stored_energy",
                "thermalised_power",
                "birth_power",
            )
        }
        total = parts if total is None else _add(total, parts)
        births += birth_rate
        trapped += result.trapped_births * birth_rate
    assert total is not None
    total["trapped_births"] = (
        results[0][0].trapped_births
        if len(results) == 1
        else (trapped / births if births > 0.0 else 0.0)
    )
    return total


def _torque_sums(torques: list[xr.Dataset], jxb: bool) -> dict:
    """Return the shell torques and the angular momenta of each isotope, summed."""
    total: dict | None = None
    for torque in torques:
        parts = {
            "shell_torque": torque["shell_torque"].values,
            **{
                name: float(torque.attrs[name])
                for name in (
                    "birth_angular_momentum",
                    "guiding_centre_angular_momentum",
                    "thermalised_angular_momentum",
                    "jxb_outside",
                )
            },
        }
        total = parts if total is None else _add(total, parts)
    assert total is not None
    if len(torques) == 1:
        total["balance"] = float(torques[0].attrs["balance"])
    else:
        deposited = (
            float(np.sum(total["shell_torque"]))
            + total["jxb_outside"]
            + total["thermalised_angular_momentum"]
        )
        reference = total[
            "birth_angular_momentum" if jxb else "guiding_centre_angular_momentum"
        ]
        total["balance"] = deposited / reference - 1.0 if reference != 0.0 else 0.0
    return total


def _reactions(mass_numbers: Sequence[int], plasma: Plasma) -> dict[int, list[str]]:
    """Return the neutron reactions of each beam isotope with the plasma's ions."""
    found: dict[int, list[str]] = {}
    for mass_number in mass_numbers:
        beam = _REACTANTS.get(mass_number)
        if beam is None:
            continue
        for key, fusion in FUSION_REACTIONS.items():
            if not fusion.neutron or beam not in (fusion.target, fusion.projectile):
                continue
            other = fusion.projectile if beam == fusion.target else fusion.target
            if other in ("D", "T", "3He") and _species(plasma, other) is not None:
                found.setdefault(mass_number, []).append(key)
    return found


def _describe(
    equilibrium,
    plasma: Plasma,
    named: list[tuple[str, Beams]],
    comm: MPI.Comm | None,
    options: dict,
) -> dict:
    """Return what the run was made of, for the provenance attributes."""
    beams = []
    for name, item in named:
        if isinstance(item, NeutralBeam):
            beams.append(
                {
                    "name": name,
                    "type": "NeutralBeam",
                    "mass_number": item.mass_number,
                    "power": item.power(comm),
                    "markers": item.size if comm is None else comm.allreduce(item.size),
                }
            )
        else:
            sources = item.sources if isinstance(item, InjectorBox) else (item,)
            beams.append(
                {
                    "name": name,
                    "type": type(item).__name__,
                    "sources": [
                        {
                            key: value.tolist()
                            if isinstance(value, np.ndarray)
                            else value
                            for key, value in asdict(source).items()
                        }
                        for source in sources
                    ],
                }
            )
    return {
        "equilibrium": {
            "type": type(equilibrium).__name__,
            "radial_coordinate": radial_coordinate_of(equilibrium),
        },
        "plasma": {
            "ne0": float(plasma.ne[0]),
            "te0": float(plasma.te[0]),
            "ti0": float(plasma.ti[0]),
            "points": len(plasma.rho),
            "ions": {
                name: {"mass_number": ion.mass_number, "charge": ion.charge}
                for name, ion in plasma.ions.items()
            },
            "radial_coordinate": plasma.radial_coordinate or "unset",
        },
        "beams": beams,
        "options": options,
    }


def _json(value) -> str:
    return json.dumps(value, default=str, sort_keys=False)


@sp.profile("freeofcharge.run_nbi", tags=PROFILE_TAGS)
def run_nbi(
    equilibrium: MagneticEquilibrium,
    plasma: Plasma,
    beams: Beams | Sequence[Beams] | Mapping[str, Beams],
    *,
    rho_edges=None,
    surfaces: FluxSurfaces | None = None,
    surface_options: Mapping[str, Any] | None = None,
    stopping: StoppingModel | None = None,
    step: float = 0.01,
    births_per_marker: int = 1,
    markers: int = 1000,
    seed: int | None = None,
    slowing_down_options: Mapping[str, Any] | None = None,
    torque: bool = True,
    torque_options: Mapping[str, Any] | None = None,
    neutrons: bool | None = None,
    neutron_options: Mapping[str, Any] | None = None,
    comm: MPI.Comm | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    r"""Deposit, slow down, and return heating, current, torque and neutrons in one dataset.

    Args:
        equilibrium: Flux, boundary and wall: an axisymmetric
            `freeofcharge.Equilibrium` or a 3D one (e.g. a
            `freeofcharge.GriddedEquilibrium`).
        plasma: Profiles.
        beams: A `NeutralBeam`, `BeamSource` or `InjectorBox`, a sequence of
            them, or a mapping from names to them. Markers are named by the
            mapping or ``beam1``, ``beam2``, ... in order; sources and boxes
            by their own names.
        rho_edges: Shell edges; 50 equal shells if None (`freeofcharge.deposit`).
        surfaces: The shells, if already computed; else
            ``equilibrium.flux_surfaces(rho_edges, **surface_options)``.
        surface_options: Passed to ``flux_surfaces``.
        stopping: Stopping model of the deposition; Suzuki's if None.
        step: Step along the lines [m].
        births_per_marker: Births sampled per marker.
        markers: Markers per energy component of each `BeamSource`.
        seed: The ``i``-th `NeutralBeam` (from 0) is deposited with
            ``seed + i``, the sources by `freeofcharge.deposit_sources` with
            ``seed + 10 * (number of NeutralBeams)``; random if None.
        slowing_down_options: Passed to `freeofcharge.slow_down`
            (``thermal_factor``, ``minimum_energy``, ``ln_lambda``,
            ``speed_points``, ...); ``orbit_average`` defaults to whether the
            shells are axisymmetric (False in 3D: the local pitch).
        torque: Compute the torque. The $\mathbf{j}\times\mathbf{B}$ part is
            included for an axisymmetric `freeofcharge.Equilibrium` only.
        torque_options: Passed to `freeofcharge.torque.beam_torque`.
        neutrons: Compute the neutron rates: None for those reactions that
            apply (none for a hydrogen beam: skipped), True to require one,
            False to skip.
        neutron_options: Passed to `freeofcharge.neutrons.neutron_rates`, but
            for ``reactions``: the keys of `plasmacoll.fusion.FUSION_REACTIONS`
            to compute instead of those that apply.
        comm: Communicator over which the markers are split; serial if None.
        inputs: What the inputs were made from (e.g. the parameter file), kept
            as JSON in the attribute ``parameters``.

    Returns:
        On the shell centres ``rho`` (and the edges ``rho_edge``): the shells'
        ``volume`` and ``area``; ``birth_density``, ``power_density`` and per
        source (and energy component for `BeamSource`) the
        ``source_birth_density``, ``source_power_density``,
        ``source_injected_power``, ``source_absorbed_power``,
        ``source_shine_through_power`` and ``source_shine_through_fraction``;
        ``electron_heating``, ``ion_heating`` [W/m^3], ``fast_ion_density``,
        ``fast_ion_pressure``, ``fast_ion_current_density``,
        ``current_density`` (beam-driven, with the electron shielding);
        ``torque_collisional_electrons``, ``torque_collisional_ions``,
        ``torque_jxb`` (axisymmetric only) and their sum ``torque`` [N m/m^3];
        ``beam_target_emissivity`` and ``thermonuclear_emissivity`` by
        ``reaction`` and their sum over the reactions, ``neutron_emissivity``
        [1/(m^3 s)]. The totals are 0-d variables (``injected_power``,
        ``absorbed_power``, ``shine_through_fraction``,
        ``electron_heating_power``, ``driven_current``, ``torque_total``,
        ``neutron_rate``, ...; `summary` lists them). The attributes hold the
        provenance: ``inputs`` (JSON: the equilibrium's type, the central
        plasma values, the beams, the options), ``parameters`` (JSON of
        ``inputs``), the version, which torque and reactions were computed,
        and the torque's angular momentum ``torque_balance``.
    """
    import freeofcharge

    named = _named(beams)
    if surfaces is None:
        if rho_edges is None:
            rho_edges = np.linspace(0.0, 1.0, 51)
        surfaces = equilibrium.flux_surfaces(
            np.asarray(rho_edges, dtype=float), **dict(surface_options or {})
        )
    rho_edges = np.asarray(surfaces.rho_edges, dtype=float)
    deposit_options: dict[str, Any] = {
        "step": step,
        "births_per_marker": births_per_marker,
        "rho_edges": rho_edges,
    }
    if stopping is not None:
        deposit_options["stopping"] = stopping
    depositions, with_components = _deposit_all(
        named,
        equilibrium,
        plasma,
        markers=markers,
        seed=seed,
        comm=comm,
        options=deposit_options,
    )

    # one Deposition per isotope, slowed down together
    groups: dict[int, list[Deposition]] = {}
    for d in depositions.values():
        groups.setdefault(d.mass_number, []).append(d)
    combined = {
        mass_number: parts[0] if len(parts) == 1 else Deposition.combine(parts)
        for mass_number, parts in sorted(groups.items())
    }
    sd_options = {
        "orbit_average": surfaces.axisymmetric,
        **(slowing_down_options or {}),
    }
    slowed = [
        (
            slow_down(d, plasma, surfaces, comm=comm, **sd_options),
            float(np.sum(d.birth_rate)),
        )
        for d in combined.values()
    ]
    sd = _slowing_down_sums(slowed)

    volume = np.asarray(surfaces.volume, dtype=float)
    rho = surfaces.rho
    birth_rate = np.sum([d.birth_rate for d in depositions.values()], axis=0)
    power = np.sum([d.power for d in depositions.values()], axis=0)
    injected = float(sum(d.injected_power for d in depositions.values()))
    absorbed = float(sum(d.absorbed_power for d in depositions.values()))
    shine = float(sum(d.shine_through_power for d in depositions.values()))
    variables: dict[str, tuple] = {
        "volume": _variable("rho", volume, "shell volume", "m^3"),
        "area": _variable("rho", np.asarray(surfaces.area), "shell area", "m^2"),
        "birth_density": _variable(
            "rho", birth_rate / volume, "birth rate density", "1/(m^3 s)"
        ),
        "power_density": _variable(
            "rho", power / volume, "deposited power density", "W/m^3"
        ),
        **_per_source(depositions, surfaces, with_components),
        "electron_heating": _variable(
            "rho", sd["electron_heating"] / volume, "electron heating", "W/m^3"
        ),
        "ion_heating": _variable(
            "rho", sd["ion_heating"] / volume, "ion heating", "W/m^3"
        ),
        "fast_ion_density": _variable(
            "rho", sd["fast_ion_density"], "fast-ion density", "1/m^3"
        ),
        "fast_ion_pressure": _variable(
            "rho", sd["fast_ion_pressure"], "fast-ion pressure", "Pa"
        ),
        "fast_ion_current_density": _variable(
            "rho",
            sd["fast_ion_current_density"],
            "fast-ion current density",
            "A/m^2",
        ),
        "current_density": _variable(
            "rho", sd["current_density"], "beam-driven current density", "A/m^2"
        ),
        "injected_power": _scalar(injected, "injected power", "W"),
        "absorbed_power": _scalar(absorbed, "absorbed power", "W"),
        "shine_through_power": _scalar(shine, "shine-through power", "W"),
        "shine_through_fraction": _scalar(
            shine / injected if injected > 0.0 else 0.0, "shine-through fraction", ""
        ),
        "electron_heating_power": _scalar(
            np.sum(sd["electron_heating"]), "electron heating power", "W"
        ),
        "ion_heating_power": _scalar(
            np.sum(sd["ion_heating"]), "ion heating power", "W"
        ),
        "thermalised_power": _scalar(
            sd["thermalised_power"], "power thermalised with the fast ions", "W"
        ),
        "stored_energy": _scalar(sd["stored_energy"], "fast-ion energy", "J"),
        "driven_current": _scalar(sd["driven_current"], "beam-driven current", "A"),
        "trapped_births": _scalar(
            sd["trapped_births"], "share of births on trapped orbits", ""
        ),
    }

    attrs: dict[str, Any] = {
        "title": "freeofcharge run_nbi",
        "freeofcharge_version": freeofcharge.__version__,
        "radial_coordinate": surfaces.radial_coordinate,
        "geometry": "axisymmetric" if surfaces.axisymmetric else "3D",
        "isotopes": ",".join(str(m) for m in combined),
        "power_balance": (
            (
                float(np.sum(sd["electron_heating"]) + np.sum(sd["ion_heating"]))
                + sd["thermalised_power"]
            )
            / sd["birth_power"]
            - 1.0
            if sd["birth_power"] > 0.0
            else 0.0
        ),
    }

    # torque
    jxb = False
    if torque:
        t_options = dict(torque_options or {})
        jxb = bool(t_options.pop("jxb", isinstance(equilibrium, Equilibrium)))
        torques = [
            beam_torque(
                d, plasma, surfaces, equilibrium, jxb=jxb, comm=comm, **t_options
            )
            for d in combined.values()
        ]
        components = list(torques[0]["component"].values)
        sums = _torque_sums(torques, jxb)
        shell_torque = sums["shell_torque"]
        for k, name in enumerate(components):
            long_name = torques[0][name].attrs["long_name"]
            variables[f"torque_{name}"] = _variable(
                "rho", shell_torque[k] / volume, long_name, "N m/m^3"
            )
            variables[f"torque_{name}_total"] = _scalar(
                np.sum(shell_torque[k]), long_name, "N m"
            )
        variables["torque"] = _variable(
            "rho", np.sum(shell_torque, axis=0) / volume, "beam torque", "N m/m^3"
        )
        variables["torque_total"] = _scalar(np.sum(shell_torque), "beam torque", "N m")
        variables["birth_angular_momentum"] = _scalar(
            sums["birth_angular_momentum"],
            "angular momentum rate of the births",
            "N m",
        )
        variables["thermalised_angular_momentum"] = _scalar(
            sums["thermalised_angular_momentum"],
            "angular momentum thermalised with the fast ions",
            "N m",
        )
        if jxb:
            variables["jxb_torque_outside"] = _scalar(
                sums["jxb_outside"],
                r"$j \times B$ torque beyond the outermost edge",
                "N m",
            )
        attrs["torque_balance"] = sums["balance"]
    attrs["torque"] = (
        ("collisional and j x B" if jxb else "collisional only")
        if torque
        else "not computed"
    )

    # neutrons
    n_options = dict(neutron_options or {})
    requested = n_options.pop("reactions", None)
    by_isotope: dict[int, list[str]] = {}
    if neutrons is not False:
        if requested is None:
            by_isotope = _reactions(list(combined), plasma)
        else:
            requested = [requested] if isinstance(requested, str) else list(requested)
            unknown = [r for r in requested if r not in FUSION_REACTIONS]
            if unknown:
                raise ValueError(
                    f"unknown reactions {unknown}; known: {sorted(FUSION_REACTIONS)}"
                )
            for mass_number in combined:
                beam = _REACTANTS.get(mass_number)
                keys = [
                    key
                    for key in requested
                    if beam
                    in (FUSION_REACTIONS[key].target, FUSION_REACTIONS[key].projectile)
                ]
                if keys:
                    by_isotope[mass_number] = keys
        if (neutrons or requested is not None) and not by_isotope:
            raise ValueError(
                "no neutron reaction between the beam isotopes "
                f"{sorted(combined)} and the plasma's ions {list(plasma.ions)}"
            )
    reactions = list(dict.fromkeys(r for keys in by_isotope.values() for r in keys))
    if reactions:
        n_shells = len(rho_edges) - 1
        beam_target = np.zeros((len(reactions), n_shells))
        thermonuclear = np.zeros((len(reactions), n_shells))
        neutron = np.zeros(len(reactions), dtype=bool)
        for mass_number, keys in by_isotope.items():
            for key in keys:
                rates = neutron_rates(
                    combined[mass_number],
                    plasma,
                    surfaces,
                    reaction=key,
                    comm=comm,
                    **n_options,
                )
                k = reactions.index(key)
                beam_target[k] = beam_target[k] + rates.beam_target
                thermonuclear[k] = rates.thermonuclear
                neutron[k] = rates.neutron
        what = np.where(neutron, 1.0, 0.0)[:, None]
        variables["beam_target_emissivity"] = _variable(
            ("reaction", "rho"),
            beam_target / volume,
            "beam-target reaction emissivity",
            "1/(m^3 s)",
        )
        variables["thermonuclear_emissivity"] = _variable(
            ("reaction", "rho"),
            thermonuclear / volume,
            "thermonuclear reaction emissivity",
            "1/(m^3 s)",
        )
        variables["neutron_emissivity"] = _variable(
            "rho",
            np.sum(what * (beam_target + thermonuclear), axis=0) / volume,
            "neutron emissivity",
            "1/(m^3 s)",
        )
        bt = float(np.sum(what * beam_target))
        th = float(np.sum(what * thermonuclear))
        variables["beam_target_neutron_rate"] = _scalar(
            bt, "beam-target neutron rate", "1/s"
        )
        variables["thermonuclear_neutron_rate"] = _scalar(
            th, "thermonuclear neutron rate", "1/s"
        )
        variables["neutron_rate"] = _scalar(bt + th, "neutron rate", "1/s")
    attrs["neutron_reactions"] = ",".join(reactions) if reactions else "none"

    options = {
        "rho_edges": rho_edges.tolist(),
        "step": step,
        "births_per_marker": births_per_marker,
        "markers": markers,
        "seed": seed,
        "stopping": type(stopping).__name__
        if stopping is not None
        else "SuzukiStopping",
        "slowing_down": sd_options,
        "torque": dict(torque_options or {}) if torque else False,
        "neutrons": {"reactions": reactions, **n_options},
    }
    attrs["inputs"] = _json(_describe(equilibrium, plasma, named, comm, options))
    if inputs is not None:
        attrs["parameters"] = _json(dict(inputs))

    coords: dict[str, Any] = {
        "rho": ("rho", rho, rho_coordinate(surfaces.radial_coordinate)),
        "rho_edge": (
            "rho_edge",
            rho_edges,
            {**rho_coordinate(surfaces.radial_coordinate), "long_name": "shell edge"},
        ),
        "source": list(dict.fromkeys(source for source, _ in depositions)),
    }
    if with_components:
        coords["component"] = list(COMPONENTS)
    if reactions:
        coords["reaction"] = reactions
    return xr.Dataset(variables, coords=coords, attrs=attrs)


def summary(dataset: xr.Dataset) -> str:
    """Return the totals of a `run_nbi` dataset, one per line with their units."""
    lines = [
        f"{dataset.attrs.get('title', 'freeofcharge')}: "
        f"{dataset.attrs.get('geometry', '?')} geometry, "
        f"{dataset.sizes.get('rho', 0)} shells, sources "
        f"{', '.join(str(s) for s in dataset['source'].values)}",
        f"torque: {dataset.attrs.get('torque', '?')}; "
        f"neutron reactions: {dataset.attrs.get('neutron_reactions', '?')}",
    ]
    for name in _TOTALS:
        if name not in dataset:
            continue
        variable = dataset[name]
        units = variable.attrs.get("units", "")
        label = str(variable.attrs.get("long_name", name))
        label = label.replace("$", "").replace(r"\times", "x")
        value = float(variable.values)
        lines.append(
            f"  {label:48s} "
            f"{value:12.4g} {units}".rstrip()
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# The parameter file
# ----------------------------------------------------------------------


class ParameterError(ValueError):
    """A parameter file that does not describe a run, with where and why."""


_NUMBER = "a number"
_INT = "an integer"
_STR = "a string"
_BOOL = "true or false"
_NUMBERS = "a list of numbers"
_VECTOR = "a list of three numbers"
_PAIR = "a number or a list of two numbers"
_TABLE = "a table"
_TABLES = "a list of tables"
_INTS = "a list of integers"
_GRID = "a list of three integers"
_STRS = "a string or a list of strings"
_EDGES = "a number of shells or a list of edges"
_AUTO = 'true, false or "auto"'


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers(value, length: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and all(_is_number(v) for v in value)
        and (length is None or len(value) == length)
    )


_CHECKS = {
    _NUMBER: _is_number,
    _INT: lambda v: isinstance(v, int) and not isinstance(v, bool),
    _STR: lambda v: isinstance(v, str),
    _BOOL: lambda v: isinstance(v, bool),
    _NUMBERS: lambda v: _numbers(v) and len(v) > 0,
    _VECTOR: lambda v: _numbers(v, 3),
    _PAIR: lambda v: _is_number(v) or _numbers(v, 2),
    _TABLE: lambda v: isinstance(v, dict),
    _TABLES: lambda v: isinstance(v, list) and all(isinstance(t, dict) for t in v),
    _INTS: lambda v: (
        isinstance(v, list)
        and all(isinstance(i, int) and not isinstance(i, bool) for i in v)
    ),
    _GRID: lambda v: (
        isinstance(v, list)
        and len(v) == 3
        and all(isinstance(i, int) and not isinstance(i, bool) and i > 0 for i in v)
    ),
    _STRS: lambda v: (
        isinstance(v, str)
        or (isinstance(v, list) and all(isinstance(s, str) for s in v))
    ),
    _EDGES: lambda v: (
        (isinstance(v, int) and not isinstance(v, bool) and v > 0)
        or (isinstance(v, list) and _numbers(v) and len(v) > 1)
    ),
    _AUTO: lambda v: isinstance(v, bool) or v == "auto",
}


def _fields(
    table, where: str, required: Mapping[str, str], optional: Mapping[str, str]
) -> dict:
    """Return ``table`` after checking its keys and the kinds of their values."""
    if not isinstance(table, dict):
        raise ParameterError(f"{where}: expected a table")
    unknown = sorted(set(table) - set(required) - set(optional))
    if unknown:
        allowed = ", ".join(sorted({*required, *optional}))
        raise ParameterError(
            f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {allowed}"
        )
    for key in required:
        if key not in table:
            raise ParameterError(f"{where}.{key}: missing, {required[key]} is required")
    for key, value in table.items():
        kind = required.get(key, optional.get(key))
        assert kind is not None
        if not _CHECKS[kind](value):
            raise ParameterError(
                f"{where}.{key}: expected {kind}, got {value!r} ({type(value).__name__})"
            )
    return dict(table)


def _kind(table, where: str, kinds: Sequence[str]) -> str:
    if not isinstance(table, dict):
        raise ParameterError(f"{where}: expected a table")
    kind = table.get("kind")
    if kind not in kinds:
        raise ParameterError(
            f"{where}.kind: expected one of {', '.join(repr(k) for k in kinds)}, "
            f"got {kind!r}"
        )
    return str(kind)


_OPTICS = {
    "width": _PAIR,
    "divergence": _PAIR,
    "halo_fraction": _NUMBER,
    "halo_divergence": _PAIR,
}

_SCHEMA: dict[str, dict[str, tuple[dict[str, str], dict[str, str]]]] = {
    "equilibrium": {
        "solovev": (
            {"major_radius": _NUMBER, "minor_radius": _NUMBER},
            {
                "elongation": _NUMBER,
                "toroidal_field": _NUMBER,
                "q_axis": _NUMBER,
                "wall_psi_n": _NUMBER,
                "wall": _BOOL,
                "points": _INT,
            },
        ),
        "geqdsk": ({"path": _STR}, {"method": _STR}),
        "vmec": (
            {},
            {
                "path": _STR,
                "boundary": _STR,
                "field_strength": _NUMBER,
                "mpol": _INT,
                "ntor": _INT,
                "ns": _INTS,
                "grid": _GRID,
            },
        ),
        "desc": ({"path": _STR}, {"grid": _GRID}),
    },
    "plasma": {
        "parabolic": (
            {"ne0": _NUMBER, "te0": _NUMBER, "ions": _TABLE},
            {
                "ti0": _NUMBER,
                "alpha_n": _NUMBER,
                "alpha_t": _NUMBER,
                "edge_fraction": _NUMBER,
                "points": _INT,
                "radial_coordinate": _STR,
            },
        ),
        "tabulated": (
            {
                "rho": _NUMBERS,
                "ne": _NUMBERS,
                "te": _NUMBERS,
                "ti": _NUMBERS,
                "ions": _TABLE,
            },
            {"radial_coordinate": _STR},
        ),
    },
    "beams": {
        "tangential": (
            {
                "energy": _NUMBER,
                "power": _NUMBER,
                "mass_number": _INT,
                "tangency_radius": _NUMBER,
                "start_radius": _NUMBER,
            },
            {
                "name": _STR,
                "height": _NUMBER,
                "toroidal_angle": _NUMBER,
                "sense": _INT,
                "markers": _INT,
                **_OPTICS,
            },
        ),
        "gaussian": (
            {
                "energy": _NUMBER,
                "power": _NUMBER,
                "mass_number": _INT,
                "origin": _VECTOR,
                "direction": _VECTOR,
            },
            {"name": _STR, "markers": _INT, **_OPTICS},
        ),
        "source": (
            {
                "name": _STR,
                "energy": _NUMBER,
                "power": _NUMBER,
                "mass_number": _INT,
                "position": _VECTOR,
                "direction": _VECTOR,
            },
            {"power_fractions": _VECTOR, **_OPTICS},
        ),
        "box": ({"name": _STR, "sources": _TABLES}, {}),
        "markers": ({"path": _STR}, {"name": _STR, "mass_number": _INT}),
    },
}

_SECTIONS = {
    "numerics": {
        "rho_edges": _EDGES,
        "step": _NUMBER,
        "births_per_marker": _INT,
        "markers": _INT,
        "seed": _INT,
        "surfaces": _TABLE,
    },
    "slowing_down": {
        "thermal_factor": _NUMBER,
        "minimum_energy": _NUMBER,
        "ln_lambda": _NUMBER,
        "orbit_average": _BOOL,
        "modes": _INT,
        "speed_points": _INT,
    },
    "torque": {
        "enabled": _BOOL,
        "jxb": _BOOL,
        "thermal_factor": _NUMBER,
        "minimum_energy": _NUMBER,
        "ln_lambda": _NUMBER,
        "speed_points": _INT,
    },
    "neutrons": {
        "enabled": _AUTO,
        "reactions": _STRS,
        "maxwellian_target": _BOOL,
        "thermal_factor": _NUMBER,
        "minimum_energy": _NUMBER,
        "ln_lambda": _NUMBER,
        "speed_points": _INT,
        "table_points": _INT,
    },
}

_TOP = ("title", "equilibrium", "plasma", "beams", *_SECTIONS)


def _check_params(params: dict) -> dict:
    """Check every section of a parsed parameter file; return it."""
    unknown = sorted(set(params) - set(_TOP))
    if unknown:
        raise ParameterError(
            f"unknown section(s) {', '.join(unknown)}; allowed: {', '.join(_TOP)}"
        )
    for section in ("equilibrium", "plasma", "beams"):
        if section not in params:
            raise ParameterError(f"[{section}] is missing")
    if "title" in params and not isinstance(params["title"], str):
        raise ParameterError("title: expected a string")
    for section in ("equilibrium", "plasma"):
        table = params[section]
        kind = _kind(table, section, list(_SCHEMA[section]))
        required, optional = _SCHEMA[section][kind]
        _fields(table, section, required, {"kind": _STR, **optional})
    eq = params["equilibrium"]
    if eq["kind"] == "vmec" and ("path" in eq) == ("boundary" in eq):
        raise ParameterError(
            "equilibrium: a VMEC equilibrium takes either path (a wout file) "
            "or boundary (a ConStellaration boundary JSON), not both or neither"
        )
    plasma = params["plasma"]
    ion_keys = (
        {"mass_number": _INT, "charge": _INT, "fraction": _NUMBER}
        if plasma["kind"] == "parabolic"
        else {"mass_number": _INT, "charge": _INT, "density": _NUMBERS}
    )
    if not plasma["ions"]:
        raise ParameterError("plasma.ions: at least one ion species is required")
    for name, ion in plasma["ions"].items():
        _fields(ion, f"plasma.ions.{name}", ion_keys, {})
    beams = params["beams"]
    if not isinstance(beams, list) or not beams:
        raise ParameterError("beams: expected one or more [[beams]] tables")
    for i, beam in enumerate(beams):
        where = f"beams[{i}]"
        kind = _kind(beam, where, list(_SCHEMA["beams"]))
        required, optional = _SCHEMA["beams"][kind]
        _fields(beam, where, required, {"kind": _STR, **optional})
        if kind == "box":
            if not beam["sources"]:
                raise ParameterError(f"{where}.sources: a box needs sources")
            s_required, s_optional = _SCHEMA["beams"]["source"]
            for j, source in enumerate(beam["sources"]):
                _fields(source, f"{where}.sources[{j}]", s_required, s_optional)
    for section, keys in _SECTIONS.items():
        if section in params:
            _fields(params[section], section, {}, keys)
    return params


def read_params(path: str | Path) -> dict:
    """Return the checked contents of a TOML parameter file.

    Raises:
        ParameterError: The file is not TOML, or a section or key is unknown,
            missing or of the wrong kind; the message names it.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            params = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ParameterError(f"{path}: not valid TOML: {error}") from error
    return _check_params(params)


@dataclass
class RunInputs:
    """What a parameter file describes, ready for `run_nbi`."""

    equilibrium: MagneticEquilibrium
    plasma: Plasma
    beams: dict[str, Beams]
    options: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments of `run_nbi`."""
    """The parameter file's contents."""


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _build_equilibrium(table: dict, base: Path) -> MagneticEquilibrium:
    kind = table["kind"]
    if kind == "solovev":
        options = {
            key: table[key]
            for key in ("elongation", "toroidal_field", "q_axis", "points")
            if key in table
        }
        wall = table.get("wall_psi_n", 1.2) if table.get("wall", True) else None
        return Equilibrium.solovev(
            table["major_radius"], table["minor_radius"], wall_psi_n=wall, **options
        )
    if kind == "geqdsk":
        options = {"method": table["method"]} if "method" in table else {}
        return Equilibrium.from_geqdsk(_path(table["path"], base), **options)
    if kind == "vmec":
        from freeofcharge.vmec_equilibrium import VMECEquilibrium

        if "path" in table:
            equilibrium: Any = VMECEquilibrium.from_wout(_path(table["path"], base))
        else:
            boundary = json.loads(_path(table["boundary"], base).read_text())
            options: dict[str, Any] = {
                key: tuple(table[key]) if key == "ns" else table[key]
                for key in ("field_strength", "mpol", "ntor", "ns")
                if key in table
            }
            equilibrium = VMECEquilibrium.from_boundary(boundary, **options)
    else:
        from freeofcharge.desc_equilibrium import DESCEquilibrium

        equilibrium = DESCEquilibrium.from_file(_path(table["path"], base))
    if "grid" in table:
        from freeofcharge.gridded_equilibrium import GriddedEquilibrium

        n_r, n_phi, n_z = (int(n) for n in table["grid"])
        equilibrium = GriddedEquilibrium(equilibrium, (n_r, n_phi, n_z))
    return equilibrium


def _build_plasma(table: dict) -> Plasma:
    coordinate = table.get("radial_coordinate")
    if table["kind"] == "parabolic":
        options = {
            key: table[key]
            for key in ("alpha_n", "alpha_t", "edge_fraction", "points")
            if key in table
        }
        return Plasma.parabolic(
            table["ne0"],
            table["te0"],
            table.get("ti0"),
            ions={
                name: (ion["mass_number"], ion["charge"], ion["fraction"])
                for name, ion in table["ions"].items()
            },
            radial_coordinate=coordinate,
            **options,
        )
    return Plasma(
        rho=np.asarray(table["rho"], dtype=float),
        ne=np.asarray(table["ne"], dtype=float),
        te=np.asarray(table["te"], dtype=float),
        ti=np.asarray(table["ti"], dtype=float),
        ions={
            name: IonSpecies(
                ion["mass_number"], ion["charge"], np.asarray(ion["density"], float)
            )
            for name, ion in table["ions"].items()
        },
        radial_coordinate=coordinate,
    )


def _optics(table: dict) -> dict:
    return {
        key: tuple(table[key]) if isinstance(table[key], list) else table[key]
        for key in _OPTICS
        if key in table
    }


def _source(table: dict) -> BeamSource:
    return BeamSource(
        name=table["name"],
        position=np.asarray(table["position"], dtype=float),
        direction=np.asarray(table["direction"], dtype=float),
        energy=table["energy"],
        power=table["power"],
        power_fractions=tuple(table.get("power_fractions", (1.0, 0.0, 0.0))),
        mass_number=table["mass_number"],
        **_optics(table),
    )


def _build_beam(
    table: dict, i: int, base: Path, markers: int, seed: int | None, comm
) -> Beams:
    kind = table["kind"]
    if kind in ("tangential", "gaussian"):
        if kind == "tangential":
            origin, direction = tangential_injection(
                table["tangency_radius"],
                table["start_radius"],
                height=table.get("height", 0.0),
                toroidal_angle=table.get("toroidal_angle", 0.0),
                sense=table.get("sense", 1),
            )
        else:
            origin, direction = table["origin"], table["direction"]
        return NeutralBeam.gaussian(
            origin,
            direction,
            table["energy"],
            table["power"],
            mass_number=table["mass_number"],
            markers=table.get("markers", markers),
            seed=None if seed is None else seed + 500 + i,
            comm=comm,
            **_optics(table),
        )
    if kind == "source":
        return _source(table)
    if kind == "box":
        return InjectorBox(
            table["name"], tuple(_source(source) for source in table["sources"])
        )
    return NeutralBeam.from_dataset(
        xr.load_dataset(_path(table["path"], base)),
        mass_number=table.get("mass_number"),
        comm=comm,
    )


def build_inputs(
    params: dict, base: str | Path = ".", comm: MPI.Comm | None = None
) -> RunInputs:
    """Return the equilibrium, plasma, beams and options of checked parameters.

    Relative paths are taken from ``base`` (the parameter file's folder).
    Beam markers (``tangential`` and ``gaussian``) are drawn with the seed
    ``numerics.seed + 500 + i`` for the ``i``-th beam (from 0). Errors of the
    constructors (e.g. a start radius inside the tangency radius) are raised as
    `ParameterError` naming the section.
    """
    base = Path(base)
    numerics = dict(params.get("numerics", {}))
    markers = numerics.pop("markers", 1000)
    seed = numerics.get("seed")

    def build(where: str, make):
        try:
            return make()
        except (ValueError, TypeError, OSError, KeyError) as error:
            raise ParameterError(f"{where}: {error}") from error

    equilibrium = build(
        "equilibrium", lambda: _build_equilibrium(params["equilibrium"], base)
    )
    plasma = build("plasma", lambda: _build_plasma(params["plasma"]))
    beams: dict[str, Beams] = {}
    for i, table in enumerate(params["beams"]):
        item = build(
            f"beams[{i}]",
            lambda table=table, i=i: _build_beam(table, i, base, markers, seed, comm),
        )
        name = item.name if not isinstance(item, NeutralBeam) else table.get("name")
        name = f"beam{i + 1}" if name is None else name
        if name in beams:
            raise ParameterError(f"beams[{i}].name: {name!r} is used twice")
        beams[name] = item
    edges = numerics.pop("rho_edges", 50)
    rho_edges = (
        np.linspace(0.0, 1.0, edges + 1)
        if isinstance(edges, int)
        else np.asarray(edges)
    )
    if np.any(np.diff(rho_edges) <= 0) or rho_edges[0] < 0 or rho_edges[-1] > 1:
        raise ParameterError("numerics.rho_edges: increasing edges within [0, 1]")
    torque = dict(params.get("torque", {}))
    neutrons = dict(params.get("neutrons", {}))
    enabled = neutrons.pop("enabled", "auto")
    options: dict[str, Any] = {
        "rho_edges": rho_edges,
        "surface_options": numerics.pop("surfaces", None),
        "markers": markers,
        "slowing_down_options": dict(params.get("slowing_down", {})),
        "torque": torque.pop("enabled", True),
        "torque_options": torque,
        "neutrons": None if enabled == "auto" else enabled,
        "neutron_options": neutrons,
        **numerics,
    }
    return RunInputs(equilibrium, plasma, beams, options, params)


def run_params(path: str | Path, comm: MPI.Comm | None = None) -> xr.Dataset:
    """Read a parameter file and return the `run_nbi` dataset it describes."""
    path = Path(path)
    params = read_params(path)
    inputs = build_inputs(params, path.parent, comm)
    try:
        dataset = run_nbi(
            inputs.equilibrium,
            inputs.plasma,
            inputs.beams,
            comm=comm,
            inputs=params,
            **inputs.options,
        )
    except TypeError as error:
        # an option one of the steps does not take
        raise ParameterError(str(error)) from error
    dataset.attrs["parameter_file"] = str(path)
    if "title" in params:
        dataset.attrs["title"] = params["title"]
    return dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line: ``freeofcharge run`` and ``freeofcharge summary``.

    Returns:
        The exit status: 0, or 2 for a bad parameter file.
    """
    parser = argparse.ArgumentParser(
        prog="freeofcharge",
        description="Neutral beams in tokamak and stellarator plasmas.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run a TOML parameter file and write the profiles (netCDF)"
    )
    run.add_argument("params", type=Path, help="the parameter file (TOML)")
    run.add_argument(
        "-o", "--output", type=Path, default=None, help="netCDF file to write"
    )
    run.add_argument(
        "-q", "--quiet", action="store_true", help="do not print the summary"
    )
    show = commands.add_parser("summary", help="print the totals of a netCDF file")
    show.add_argument("dataset", type=Path, help="a file written by freeofcharge run")
    args = parser.parse_args(argv)
    if args.command == "summary":
        with xr.open_dataset(args.dataset) as dataset:
            print(summary(dataset))
        return 0
    try:
        dataset = run_params(args.params)
    except ParameterError as error:
        print(f"freeofcharge: {args.params}: {error}", file=sys.stderr)
        return 2
    if args.output is not None:
        dataset.to_netcdf(args.output)
    if not args.quiet:
        print(summary(dataset))
        if args.output is not None:
            print(f"written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
