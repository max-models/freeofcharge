"""Run the freeofcharge benchmark case in BBNBI5 and ASCOT5 and store the results.

Needs ASCOT5's ``bbnbi5`` and ``ascot5_main`` built, and a5py (with
``libascot.so`` for the moments of the distribution) in the same environment as
freeofcharge; see the documentation page "ASCOT5 benchmark" for the commands.
Writes ``ascot5_results.json`` next to this script (or ``--out``) in the format
of ``freeofcharge.ascot5.run_freeofcharge``, with the slowing down of a second
run without orbits (the markers only collide) under ``"local"``; the ASCOT5
HDF5 files go to ``--work``.

    python benchmarks/ascot5/run_ascot5.py --ascot5 /path/to/ascot5/build
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import unyt
from a5py import Ascot
from a5py.ascot5io.nbi import Injector
from a5py.ascot5io.options import Opt
from a5py.ascot5io.state import State
from scipy import constants

from freeofcharge import ascot5

HERE = Path(__file__).resolve().parent


def write_inputs(path: Path, case: ascot5.BenchmarkCase, ne0: float) -> Ascot:
    """Write the case with the central density ``ne0`` to a new HDF5 file."""
    path.unlink(missing_ok=True)
    a5 = Ascot(str(path), create=True)
    a5.data.create_input("B_2DS", **ascot5.bfield_2ds(case.equilibrium), desc="SOLOVEV")
    a5.data.create_input(
        "plasma_1D", **ascot5.plasma_1d(case.plasma_at(ne0)), desc="PARABOLIC"
    )
    a5.data.create_input("wall_2D", **ascot5.wall_2d(case.equilibrium), desc="WALL")
    a5.data.create_input(
        "nbi",
        ninj=1,
        injectors=[Injector(**ascot5.injector(case.beam))],
        desc="BEAMLETS",
    )
    a5.data.create_input("E_TC", exyz=np.zeros((3, 1)), desc="ZERO")
    a5.data.create_input("N0_1D", desc="ZERO")
    a5.data.create_input("Boozer", desc="DUMMY")
    a5.data.create_input("MHD_STAT", desc="DUMMY")
    a5.data.create_input("asigma_loc", desc="DUMMY")
    a5.data.create_input("opt", **Opt.get_default(), desc="BBNBI")
    return a5


def run(command: list[str]) -> float:
    """Run a command, fail on a non-zero exit, return the wall time [s]."""
    start = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - start


def has(endcond, name: str) -> np.ndarray:
    """Return which markers have the end condition ``name``."""
    return np.array([State.endcond_check(e, name) for e in endcond], dtype=bool)


def bbnbi_run(a5: Ascot):
    """Return the BBNBI5 run group of a file (None if there is none)."""
    with h5py.File(a5.file_getpath(), "r") as f:
        names = [name for name in f.get("results", {}) if name.startswith("bbnbi")]
    return a5.data[names[-1]] if names else None


def deposition(path: Path, rho_edges) -> dict:
    """Return the birth profile and shine-through of the BBNBI5 run in ``path``."""
    a5 = Ascot(str(path))
    run_group = bbnbi_run(a5)
    a5.input_init(bfield=True)
    weight, energy, rho = run_group.getstate("weight", "ekin", "rho", mode="prt")
    a5.input_free()
    weight = weight.v
    power = weight * energy.to("J").v
    endcond = run_group.getstate("endcond", mode="prt")
    ionized = has(endcond, "IONIZED")
    wall = has(endcond, "WALL")
    birth, birth_error = ascot5.shell_sums(rho.v[ionized], weight[ionized], rho_edges)
    deposited, deposited_error = ascot5.shell_sums(
        rho.v[ionized], power[ionized], rho_edges
    )
    injected = power.sum()
    shine = power[wall].sum() / injected
    return {
        "markers": int(weight.size),
        "ionized_markers": int(ionized.sum()),
        "wall_markers": int(wall.sum()),
        "other_markers": int((~ionized & ~wall).sum()),
        "injected_power": float(injected),
        "shine_through_fraction": float(shine),
        "shine_through_error": float(np.sqrt(shine * (1 - shine) / weight.size)),
        "birth_rate": birth.tolist(),
        "birth_rate_error": birth_error.tolist(),
        "deposited_power": deposited.tolist(),
        "deposited_power_error": deposited_error.tolist(),
    }


def slowing_down_options(
    case: ascot5.BenchmarkCase,
    cpu_time: float,
    orbits: bool = True,
    local_step: float = 5.0e-6,
) -> dict:
    """Return the options of the slowing-down run.

    Guiding centres with collisions until 2 T_i (or 1 keV), with the
    adaptive step. Without ``orbits`` the markers stay where they were born
    and only collide, as in freeofcharge's local model; then a fixed step of
    ``local_step`` resolves the slowing down, which the collision tolerance of
    the adaptive step alone does not do finely enough for the distribution.
    """
    rho_edges = case.rho_edges
    mass = case.beam.mass_number * constants.atomic_mass
    p_max = 1.5 * np.sqrt(2 * mass * case.parameters["energy"] * constants.e)
    opt = Opt.get_default()
    opt.update(
        {
            "SIM_MODE": 2,
            "ENABLE_ADAPTIVE": int(orbits),
            # the adaptive step would also start from the user-defined one
            "FIXEDSTEP_USE_USERDEFINED": int(not orbits),
            "FIXEDSTEP_USERDEFINED": local_step,
            "ENDCOND_SIMTIMELIM": 1,
            "ENDCOND_MAX_MILEAGE": 2.0,
            "ENDCOND_CPUTIMELIM": 1,
            "ENDCOND_MAX_CPUTIME": cpu_time,
            "ENDCOND_ENERGYLIM": 1,
            "ENDCOND_MIN_ENERGY": 1.0e3,
            "ENDCOND_MIN_THERMAL": 2.0,
            "ENDCOND_WALLHIT": 1,
            "ENABLE_ORBIT_FOLLOWING": int(orbits),
            "ENABLE_COULOMB_COLLISIONS": 1,
            "ENABLE_DIST_RHO5D": 1,
            "DIST_MIN_RHO": float(rho_edges[0]),
            "DIST_MAX_RHO": float(rho_edges[-1]),
            "DIST_NBIN_RHO": len(rho_edges) - 1,
            "DIST_MIN_THETA": 0.0,
            "DIST_MAX_THETA": 360.0,
            "DIST_NBIN_THETA": 8,
            "DIST_MIN_PHI": 0.0,
            "DIST_MAX_PHI": 360.0,
            "DIST_NBIN_PHI": 1,
            "DIST_MIN_PPA": -p_max,
            "DIST_MAX_PPA": p_max,
            "DIST_NBIN_PPA": 80,
            "DIST_MIN_PPE": 0.0,
            "DIST_MAX_PPE": p_max,
            "DIST_NBIN_PPE": 40,
            "DIST_MIN_TIME": 0.0,
            "DIST_MAX_TIME": 10.0,
            "DIST_NBIN_TIME": 1,
            "DIST_MIN_CHARGE": 0.5,
            "DIST_MAX_CHARGE": 1.5,
            "DIST_NBIN_CHARGE": 1,
        }
    )
    return opt


def slowing_down(path: Path, markers: int, seed: int) -> Ascot:
    """Add ``markers`` of the ions BBNBI5 left in ``path`` as the next run's markers.

    A random subset, weighted up to the whole birth rate, starting at t = 0.
    """
    a5 = Ascot(str(path))
    bbnbi = bbnbi_run(a5)
    names = ["r", "phi", "z", "weight", "vr", "vphi", "vz", "mass", "charge"]
    names += ["anum", "znum", "ids", "endcond"]
    state = dict(zip(names, bbnbi.getstate(*names, mode="prt"), strict=True))
    ionized = np.flatnonzero(has(state.pop("endcond"), "IONIZED"))
    rng = np.random.default_rng(seed)
    chosen = np.sort(
        rng.choice(ionized, size=min(markers, ionized.size), replace=False)
    )
    mrk = {name: value[chosen] for name, value in state.items()}
    mrk["weight"] = mrk["weight"] * (ionized.size / chosen.size)
    mrk["time"] = np.zeros(chosen.size) * unyt.s
    mrk["n"] = chosen.size
    a5.data.create_input("prt", **mrk, desc="BBNBI", activate=True)
    return a5


def moments(path: Path, case: ascot5.BenchmarkCase) -> dict:
    """Return the slowing-down profiles of the ASCOT5 run in ``path``."""
    a5 = Ascot(str(path))
    run_group = a5.data.active
    a5.input_init(run=True, bfield=True, plasma=True)
    dist = run_group.getdist("rho5d")
    mom = run_group.getdist_moments(
        dist,
        "density",
        "energydensity",
        "electronpowerdep",
        "ionpowerdep",
        "parallelcurrent",
        "currentdrive",
    )
    volume = np.sum(mom.volume, axis=(1, 2))
    area = np.sum(mom.area, axis=(1, 2)) / mom.area.shape[2]

    def shells(name, units):
        values = mom.ordinate(name, toravg=True, polavg=True).to(units).v
        return np.asarray(values).ravel()

    density = shells("density", "m**-3")
    energy_density = shells("energydensity", "J/m**3")
    electron = shells("electronpowerdep", "W/m**3") * volume
    ion = shells("ionpowerdep", "W/m**3") * volume
    current = shells("parallelcurrent", "A/m**2")
    driven = shells("currentdrive", "A/m**2")
    a5.input_free()

    endcond = run_group.getstate("endcond", state="end")
    names = np.asarray([State.endcond_tostring(e) for e in endcond])
    weight = run_group.getstate("weight").v
    energy_end = run_group.getstate("ekin", state="end").to("J").v
    energy_ini = run_group.getstate("ekin", state="ini").to("J").v
    thermal = has(endcond, "EMIN") | has(endcond, "THERMAL")
    lost = has(endcond, "WALL")
    summary = {name: int(np.sum(names == name)) for name in np.unique(names)}
    return {
        "volume": volume.tolist(),
        "area": area.tolist(),
        "electron_heating": electron.tolist(),
        "ion_heating": ion.tolist(),
        "fast_ion_density": density.tolist(),
        "fast_ion_current_density": current.tolist(),
        "fast_ion_current": float(np.sum(current * area)),
        "current_density": driven.tolist(),
        "driven_current": float(np.sum(driven * area)),
        "fast_ions": float(np.sum(density * volume)),
        "stored_energy": float(np.sum(energy_density * volume)),
        "slowing_down_markers": int(weight.size),
        "end_conditions": summary,
        "birth_power": float(np.sum(weight * energy_ini)),
        "thermalised_power": float(np.sum((weight * energy_end)[thermal])),
        "lost_power": float(np.sum((weight * energy_ini)[lost])),
    }


def main(argv: list[str] | None = None) -> None:
    """Run both stages of the benchmark and write the results file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ascot5", type=Path, required=True, help="the build folder")
    parser.add_argument("--work", type=Path, default=HERE / "work")
    parser.add_argument("--out", type=Path, default=HERE / "ascot5_results.json")
    parser.add_argument("--beamlets", type=int, default=2000)
    parser.add_argument("--bbnbi-markers", type=int, default=50000)
    parser.add_argument("--slowing-down-markers", type=int, default=2000)
    parser.add_argument("--cpu-time", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--reuse", action="store_true", help="keep the BBNBI5 runs found in --work"
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    args.work.mkdir(parents=True, exist_ok=True)
    bbnbi5 = str(args.ascot5.resolve() / "bbnbi5")
    ascot5_main = str(args.ascot5.resolve() / "ascot5_main")

    case = ascot5.benchmark_case(markers=args.beamlets)
    scan = []
    timing = {}
    for ne0 in case.densities:
        path = args.work / f"bbnbi_{ne0:.0e}.h5"
        if not (args.reuse and path.exists() and bbnbi_run(Ascot(str(path)))):
            write_inputs(path, case, float(ne0))
            timing[f"bbnbi5 {ne0:.0e}"] = run(
                [bbnbi5, f"--in={path}", f"--n={args.bbnbi_markers}", "--d=BENCHMARK"]
            )
        scan.append(deposition(path, case.rho_edges))
        print(
            f"BBNBI5 ne0 = {ne0:.1e}: shine-through {scan[-1]['shine_through_fraction']:.4f}"
        )

    main_density = case.parameters["ne0"]
    path = args.work / f"bbnbi_{main_density:.0e}.h5"
    a5 = slowing_down(path, args.slowing_down_markers, args.seed)
    slowing = {}
    for orbits in (True, False):
        name = "SLOWINGDOWN" if orbits else "LOCAL"
        a5.data.create_input(
            "opt",
            **slowing_down_options(case, args.cpu_time, orbits),
            desc=name,
            activate=True,
        )
        timing[f"ascot5_main {name}"] = run(
            [ascot5_main, f"--in={path}", f"--d={name}"]
        )
        print(f"ASCOT5 {name}: {timing[f'ascot5_main {name}']:.0f} s")
        slowing[orbits] = moments(path, case)

    results = {
        "code": "ASCOT5 (BBNBI5 and ascot5_main)",
        "version": subprocess.run(
            ["git", "-C", str(args.ascot5.resolve().parent), "describe", "--always"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "rho_edges": case.rho_edges.tolist(),
        "parameters": {k: v for k, v in case.parameters.items() if k != "ions"}
        | {"ions": {k: list(v) for k, v in case.parameters["ions"].items()}},
        "bbnbi_markers": args.bbnbi_markers,
        "omp_threads": int(os.environ["OMP_NUM_THREADS"]),
        "wall_time": timing,
        "scan_ne0": case.densities.tolist(),
        "scan_shine_through_fraction": [s["shine_through_fraction"] for s in scan],
        "scan_shine_through_error": [s["shine_through_error"] for s in scan],
    }
    main_index = int(np.argmin(np.abs(case.densities - main_density)))
    results |= scan[main_index]
    results |= slowing[True]
    results["local"] = slowing[False]
    ascot5.save_results(args.out, results)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
