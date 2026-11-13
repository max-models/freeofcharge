"""run_nbi against the steps one by one, the parameter file and the command line."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from freeofcharge import (
    BeamSource,
    Deposition,
    Equilibrium,
    GriddedEquilibrium,
    InjectorBox,
    NeutralBeam,
    Plasma,
    deposit,
    deposit_sources,
    slow_down,
    tangential_injection,
)
from freeofcharge.gridded_equilibrium import _HelicalTestField
from freeofcharge.neutrons import neutron_rates
from freeofcharge.run import (
    ParameterError,
    build_inputs,
    main,
    read_params,
    run_nbi,
    run_params,
    summary,
)
from freeofcharge.torque import beam_torque

EDGES = np.linspace(0.0, 1.0, 11)
DATA = Path(__file__).parents[1] / "data" / "coilstellaration"


@pytest.fixture(scope="module")
def equilibrium():
    return Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3, points=97)


@pytest.fixture(scope="module")
def plasma():
    return Plasma.parabolic(1e20, 2e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)})


@pytest.fixture(scope="module")
def surfaces(equilibrium):
    return equilibrium.flux_surfaces(EDGES)


def _beam(sense=1, mass_number=2, energy=1.0e6, markers=150):
    origin, direction = tangential_injection(5.3, 12.0, sense=sense)
    return NeutralBeam.gaussian(
        origin,
        direction,
        energy,
        16.5e6,
        mass_number=mass_number,
        width=0.3,
        markers=markers,
        seed=1,
    )


def _source(name="S1", mass_number=1):
    return BeamSource(
        name,
        [12.0, -3.0, 0.0],
        [-0.6, 1.0, 0.0],
        1.0e5,
        2.0e6,
        (0.7, 0.2, 0.1),
        mass_number=mass_number,
        width=0.1,
    )


@pytest.fixture(scope="module")
def single(equilibrium, plasma, surfaces):
    beam = _beam()
    result = run_nbi(
        equilibrium,
        plasma,
        beam,
        surfaces=surfaces,
        step=0.03,
        births_per_marker=2,
        seed=7,
    )
    return beam, result


def test_totals_equal_the_steps_one_by_one(single, equilibrium, plasma, surfaces):
    beam, ds = single
    deposition = deposit(
        beam,
        equilibrium,
        plasma,
        step=0.03,
        births_per_marker=2,
        rho_edges=EDGES,
        seed=7,
    )
    sd = slow_down(deposition, plasma, surfaces)
    torque = beam_torque(deposition, plasma, surfaces, equilibrium)
    rates = neutron_rates(deposition, plasma, surfaces, reaction="D(d,n)3He")

    def value(name):
        return float(ds[name].values)

    assert value("injected_power") == deposition.injected_power
    assert value("absorbed_power") == deposition.absorbed_power
    assert value("shine_through_fraction") == deposition.shine_through_fraction
    assert value("electron_heating_power") == float(np.sum(sd.electron_heating))
    assert value("ion_heating_power") == float(np.sum(sd.ion_heating))
    assert value("stored_energy") == sd.stored_energy
    assert value("driven_current") == sd.driven_current
    assert value("trapped_births") == sd.trapped_births
    for name in ("collisional_electrons", "collisional_ions", "jxb"):
        assert value(f"torque_{name}_total") == torque.attrs[f"{name}_total"]
    assert value("torque_total") == torque.attrs["total"]
    assert value("birth_angular_momentum") == torque.attrs["birth_angular_momentum"]
    assert value("beam_target_neutron_rate") == rates.total_beam_target
    assert value("thermonuclear_neutron_rate") == rates.total_thermonuclear
    # profiles
    np.testing.assert_array_equal(
        ds.power_density.values, deposition.power / surfaces.volume
    )
    np.testing.assert_array_equal(
        ds.electron_heating.values, sd.electron_heating / surfaces.volume
    )
    np.testing.assert_array_equal(ds.fast_ion_pressure.values, sd.fast_ion_pressure)
    np.testing.assert_array_equal(ds.current_density.values, sd.current_density)
    np.testing.assert_array_equal(ds.torque_jxb.values, torque.jxb.values)
    np.testing.assert_array_equal(
        ds.beam_target_emissivity.sel(reaction="D(d,n)3He").values,
        rates.to_dataset().beam_target_emissivity.values,
    )
    np.testing.assert_allclose(
        ds.neutron_emissivity.values,
        (rates.beam_target + rates.thermonuclear) / surfaces.volume,
        rtol=1e-15,
    )
    assert ds.attrs["torque_balance"] == torque.attrs["balance"]
    assert abs(ds.attrs["power_balance"]) < 1e-12


def test_dataset_layout_and_units(single):
    _, ds = single
    assert list(ds.source.values) == ["beam1"]
    assert "component" not in ds.dims
    assert ds.sizes["rho"] == 10 and ds.sizes["rho_edge"] == 11
    for name, variable in ds.data_vars.items():
        assert "units" in variable.attrs and "long_name" in variable.attrs, name
    assert ds.electron_heating.attrs["units"] == "W/m^3"
    assert ds.torque.attrs["units"] == "N m/m^3"
    assert ds.rho.attrs["radial_coordinate"] == "poloidal"
    assert ds.attrs["torque"] == "collisional and j x B"
    assert ds.attrs["neutron_reactions"] == "D(d,n)3He"
    inputs = json.loads(ds.attrs["inputs"])
    assert inputs["equilibrium"]["type"] == "Equilibrium"
    assert inputs["beams"][0]["power"] == pytest.approx(16.5e6)
    assert inputs["options"]["seed"] == 7
    assert inputs["plasma"]["ions"]["D"] == {"mass_number": 2, "charge": 1}
    # the volume integral of the heating density is the heating power
    heating = float((ds.electron_heating * ds.volume).sum())
    assert heating == pytest.approx(float(ds.electron_heating_power), rel=1e-12)


@pytest.mark.parametrize("engine", ["scipy", "netcdf4"])
def test_netcdf_round_trip(single, tmp_path, engine):
    pytest.importorskip("netCDF4" if engine == "netcdf4" else "scipy")
    _, ds = single
    path = tmp_path / "out.nc"
    ds.to_netcdf(path, engine=engine)
    back = xr.load_dataset(path)
    xr.testing.assert_allclose(back, ds)
    assert back.attrs["inputs"] == ds.attrs["inputs"]


def test_sources_and_isotopes_add_up(equilibrium, plasma, surfaces):
    beam = _beam(markers=100)
    box = InjectorBox("box", (_source("box/S1"), _source("box/S2")))
    ds = run_nbi(
        equilibrium,
        plasma,
        [beam, box],
        surfaces=surfaces,
        step=0.03,
        markers=60,
        seed=3,
    )
    assert list(ds.source.values) == ["beam1", "box/S1", "box/S2"]
    assert list(ds.component.values) == ["full", "half", "third"]
    # a NeutralBeam is one mono-energetic component
    assert np.isnan(ds.source_injected_power.sel(source="beam1", component="half"))
    assert ds.attrs["isotopes"] == "1,2"
    # the same, step by step: the beam with seed + 0, the sources with seed + 10
    d_beam = deposit(beam, equilibrium, plasma, step=0.03, rho_edges=EDGES, seed=3)
    tab = deposit_sources(
        [box], equilibrium, plasma, markers=60, seed=13, step=0.03, rho_edges=EDGES
    )
    d_h = tab.total()
    injected = d_beam.injected_power + d_h.injected_power
    assert float(ds.injected_power) == pytest.approx(injected, rel=1e-14)
    np.testing.assert_allclose(
        ds.source_power_density.sel(source="box/S2", component="half").values,
        tab.depositions[("box/S2", "half")].power / surfaces.volume,
        rtol=1e-15,
    )
    sd_d = slow_down(d_beam, plasma, surfaces)
    sd_h = slow_down(d_h, plasma, surfaces)
    assert float(ds.driven_current) == pytest.approx(
        sd_d.driven_current + sd_h.driven_current, rel=1e-12
    )
    assert float(ds.electron_heating_power) == pytest.approx(
        np.sum(sd_d.electron_heating) + np.sum(sd_h.electron_heating), rel=1e-12
    )
    t_d = beam_torque(d_beam, plasma, surfaces, equilibrium)
    t_h = beam_torque(d_h, plasma, surfaces, equilibrium)
    assert float(ds.torque_total) == pytest.approx(
        t_d.attrs["total"] + t_h.attrs["total"], rel=1e-12
    )
    assert abs(ds.attrs["torque_balance"]) < 1e-12
    # only the D beam makes neutrons
    rates = neutron_rates(d_beam, plasma, surfaces)
    assert float(ds.beam_target_neutron_rate) == rates.total_beam_target
    assert float(ds.trapped_births) == pytest.approx(
        (
            sd_d.trapped_births * d_beam.birth_rate.sum()
            + sd_h.trapped_births * d_h.birth_rate.sum()
        )
        / (d_beam.birth_rate.sum() + d_h.birth_rate.sum())
    )


def test_hydrogen_beam_skips_neutrons(equilibrium, surfaces):
    hydrogen = Plasma.parabolic(5e19, 5e3, ions={"H": (1, 1, 1.0)})
    ds = run_nbi(
        equilibrium,
        hydrogen,
        {"nb": _beam(mass_number=1, energy=1e5, markers=50)},
        surfaces=surfaces,
        step=0.05,
        torque=False,
        seed=1,
    )
    assert ds.attrs["neutron_reactions"] == "none"
    assert ds.attrs["torque"] == "not computed"
    assert "neutron_rate" not in ds and "torque" not in ds
    assert "neutron rate" not in summary(ds)
    with pytest.raises(ValueError, match="no neutron reaction"):
        run_nbi(
            equilibrium,
            hydrogen,
            _beam(mass_number=1, energy=1e5, markers=20),
            surfaces=surfaces,
            step=0.05,
            neutrons=True,
        )


def test_dt_plasma_counts_both_reactions(equilibrium, surfaces):
    dt = Plasma.parabolic(1e20, 2e4, ions={"D": (2, 1, 0.5), "T": (3, 1, 0.5)})
    ds = run_nbi(
        equilibrium,
        dt,
        _beam(markers=50),
        surfaces=surfaces,
        step=0.05,
        torque=False,
        seed=1,
        neutron_options={"maxwellian_target": False},
    )
    assert set(ds.reaction.values) == {"T(d,n)4He", "D(d,n)3He"}
    dt_rate = ds.beam_target_emissivity.sel(reaction="T(d,n)4He")
    dd_rate = ds.beam_target_emissivity.sel(reaction="D(d,n)3He")
    # at 1 MeV the D-T cross section is far above the D-D one
    assert float((dt_rate * ds.volume).sum()) > 10 * float((dd_rate * ds.volume).sum())
    total = (ds.beam_target_emissivity + ds.thermonuclear_emissivity).sum("reaction")
    np.testing.assert_allclose(ds.neutron_emissivity, total, rtol=1e-14)


def test_requested_reactions(equilibrium, plasma, surfaces):
    ds = run_nbi(
        equilibrium,
        plasma,
        _beam(markers=30),
        surfaces=surfaces,
        step=0.05,
        torque=False,
        neutron_options={"reactions": "D(d,n)3He"},
        seed=1,
    )
    assert list(ds.reaction.values) == ["D(d,n)3He"]
    with pytest.raises(ValueError, match="no T ions"):
        run_nbi(
            equilibrium,
            plasma,
            _beam(markers=30),
            surfaces=surfaces,
            torque=False,
            neutron_options={"reactions": ["T(d,n)4He"]},
        )
    with pytest.raises(ValueError, match="unknown reactions"):
        run_nbi(
            equilibrium,
            plasma,
            _beam(markers=30),
            surfaces=surfaces,
            neutron_options={"reactions": ["X(y,z)W"]},
        )
    with pytest.raises(ValueError, match="no neutron reaction"):
        run_nbi(
            equilibrium,
            plasma,
            _beam(markers=30, mass_number=1),
            surfaces=surfaces,
            neutron_options={"reactions": ["D(d,n)3He"]},
        )


def test_bad_beams(equilibrium, plasma, surfaces):
    with pytest.raises(ValueError, match="no beams"):
        run_nbi(equilibrium, plasma, [], surfaces=surfaces)
    with pytest.raises(TypeError, match="NeutralBeam"):
        run_nbi(equilibrium, plasma, ["beam"], surfaces=surfaces)
    with pytest.raises(ValueError, match="must differ"):
        run_nbi(
            equilibrium,
            plasma,
            {"B/S1": _beam(markers=10), "box": InjectorBox("B", (_source("B/S1"),))},
            surfaces=surfaces,
            markers=10,
        )
    with pytest.raises(ValueError, match="unique"):
        run_nbi(equilibrium, plasma, [_source("S1"), _source("S1")], surfaces=surfaces)


def test_3d_torque_is_collisional_only():
    helical = _HelicalTestField()
    gridded = GriddedEquilibrium(helical, (48, 32, 48))
    edges = np.linspace(0.0, 1.0, 6)
    surfaces = helical.flux_surfaces(edges, field_points=(32, 16))
    plasma = Plasma.parabolic(5e19, 3e3, ions={"D": (2, 1, 1.0)})
    beam = NeutralBeam.gaussian(
        [4.0, -2.5, 0.0],
        [-0.55, 1.0, 0.0],
        55e3,
        1.8e6,
        mass_number=2,
        width=0.05,
        divergence=0.01,
        markers=100,
        seed=1,
    )
    with pytest.warns(UserWarning, match="trapped-electron correction"):
        ds = run_nbi(gridded, plasma, beam, surfaces=surfaces, step=0.01, seed=2)
    assert ds.attrs["geometry"] == "3D"
    assert ds.attrs["torque"] == "collisional only"
    assert "torque_jxb" not in ds and "jxb_torque_outside" not in ds
    assert ds.rho.attrs["radial_coordinate"] == "toroidal"
    assert np.isnan(float(ds.driven_current))
    deposition = deposit(beam, gridded, plasma, step=0.01, rho_edges=edges, seed=2)
    torque = beam_torque(deposition, plasma, surfaces, gridded, jxb=False)
    assert float(ds.torque_total) == torque.attrs["total"]
    assert abs(ds.attrs["torque_balance"]) < 1e-12
    with pytest.warns(UserWarning):
        sd = slow_down(deposition, plasma, surfaces, orbit_average=False)
    assert float(ds.ion_heating_power) == float(np.sum(sd.ion_heating))
    assert float(ds.beam_target_neutron_rate) > 0.0


def test_combined_deposition_is_one_isotope(equilibrium, plasma):
    # what run_nbi relies on to group the births by isotope
    d1 = deposit(_beam(markers=10), equilibrium, plasma, rho_edges=EDGES, step=0.1)
    d2 = deposit(
        _beam(markers=10, mass_number=1), equilibrium, plasma, rho_edges=EDGES, step=0.1
    )
    with pytest.raises(ValueError, match="isotopes"):
        Deposition.combine([d1, d2])


# ----------------------------------------------------------------------
# parameter file and command line
# ----------------------------------------------------------------------

PARAMS = """
title = "test run"

[equilibrium]
kind = "solovev"
major_radius = 6.2
minor_radius = 1.8
elongation = 1.7
toroidal_field = 5.3
wall_psi_n = 1.3
points = 97

[plasma]
kind = "parabolic"
ne0 = 1.0e20
te0 = 2.0e4
radial_coordinate = "poloidal"
[plasma.ions.D]
mass_number = 2
charge = 1
fraction = 0.96
[plasma.ions.He]
mass_number = 4
charge = 2
fraction = 0.02

[[beams]]
kind = "tangential"
name = "NB1"
energy = 1.0e6
power = 16.5e6
mass_number = 2
tangency_radius = 5.3
start_radius = 12.0
width = 0.3
divergence = 0.005
markers = 100

[numerics]
rho_edges = 10
step = 0.03
births_per_marker = 2
seed = 4

[slowing_down]
thermal_factor = 0.0

[torque]
enabled = true

[neutrons]
enabled = "auto"
"""


def _write(tmp_path, text, name="params.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_cli_round_trip(tmp_path, capsys):
    params = _write(tmp_path, PARAMS)
    out = tmp_path / "out.nc"
    assert main(["run", str(params), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "test run" in printed and "beam torque" in printed
    assert "j x B torque" in printed
    assert f"written to {out}" in printed
    ds = xr.load_dataset(out)
    assert list(ds.source.values) == ["NB1"]
    assert ds.attrs["parameter_file"] == str(params)
    assert json.loads(ds.attrs["parameters"])["beams"][0]["name"] == "NB1"
    assert main(["summary", str(out)]) == 0
    totals = capsys.readouterr().out
    assert "shine-through fraction" in totals and "neutron rate" in totals
    # the same as the Python call with the same objects
    inputs = build_inputs(read_params(params), tmp_path)
    direct = run_nbi(inputs.equilibrium, inputs.plasma, inputs.beams, **inputs.options)
    assert float(ds.torque_total) == float(direct.torque_total)
    assert float(ds.neutron_rate) == float(direct.neutron_rate)


def test_cli_entry_point_in_a_subprocess(tmp_path):
    params = _write(
        tmp_path,
        PARAMS.replace("markers = 100", "markers = 20").replace(
            'enabled = "auto"', "enabled = false"
        ),
    )
    out = tmp_path / "out.nc"
    src = str(Path(__file__).parents[3])
    env_path = ":".join([src, *sys.path])
    done = subprocess.run(
        [sys.executable, "-m", "freeofcharge.run", "run", str(params), "-o", str(out)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_path, "PATH": ""},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "injected power" in done.stdout
    assert "neutron_rate" not in xr.load_dataset(out)


def test_cli_reports_bad_parameters(tmp_path, capsys):
    params = _write(tmp_path, PARAMS.replace("start_radius = 12.0", "start_radius = 5"))
    assert main(["run", str(params), "-q"]) == 2
    error = capsys.readouterr().err
    assert "beams[0]" in error and "start radius must exceed" in error


def _replace(old, new):
    return PARAMS.replace(old, new, 1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (PARAMS + "\n[extra]\na = 1\n", "unknown section"),
        (
            _replace("[numerics]", "[numerics]\nfoo = 1"),
            r"numerics: unknown key\(s\) foo",
        ),
        (_replace("energy = 1.0e6\n", ""), "beams\\[0\\].energy: missing"),
        (_replace("energy = 1.0e6", 'energy = "1 MeV"'), "expected a number"),
        (_replace('kind = "tangential"', 'kind = "laser"'), "beams\\[0\\].kind"),
        (_replace('kind = "solovev"', 'kind = "efit"'), "equilibrium.kind"),
        (_replace("ne0 = 1.0e20", "ne0 = [1.0e20]"), "plasma.ne0: expected a number"),
        (_replace("fraction = 0.96", "share = 0.96"), "plasma.ions.D"),
        (_replace("rho_edges = 10", "rho_edges = [0.0, 0.5, 0.4]"), "rho_edges"),
        (_replace("rho_edges = 10", "rho_edges = true"), "rho_edges"),
        (_replace("seed = 4", "seed = 4.5"), "numerics.seed: expected an integer"),
        (_replace('enabled = "auto"', 'enabled = "yes"'), "neutrons.enabled"),
        (_replace("title = ", "title = 3 #"), "title"),
        (PARAMS.split("[[beams]]")[0], r"\[beams\] is missing"),
        ("beams = []\n" + PARAMS.split("[[beams]]")[0], "beams: expected one"),
        ("x = = 1", "not valid TOML"),
        (_replace("fraction = 0.96", "fraction = 0.5"), "plasma: the ion charge"),
        (
            _replace("[numerics]", "[numerics.surfaces]\nno_such = 1\n[numerics]"),
            "no_such",
        ),
        (
            "equilibrium = 3\n[plasma" + PARAMS.split("[plasma", 1)[1],
            "equilibrium: expected a table",
        ),
        (PARAMS.replace("[plasma.ions.He]", "[plasma.ions.He.x]"), "plasma.ions.He"),
        (
            PARAMS.split("[plasma.ions.D]")[0]
            + "ions = {}\n[[beams]]"
            + PARAMS.split("[[beams]]")[1],
            "at least one ion",
        ),
        (_replace("width = 0.3", "width = {a = 1}"), "expected a number or a list"),
        (
            '[equilibrium]\nkind = "desc"\npath = "x.h5"\ngrid = [1, 2]\n[plasma'
            + PARAMS.split("[plasma", 1)[1],
            "three integers",
        ),
    ],
)
def test_bad_parameters(tmp_path, text, message):
    path = _write(tmp_path, text)
    with pytest.raises(ParameterError, match=message):
        run_params(path)


def test_options_reach_the_steps(tmp_path):
    text = (
        _replace('kind = "solovev"', 'kind = "solovev"\nwall = false')
        .replace("[torque]\nenabled = true", "[torque]\njxb = false")
        .replace('enabled = "auto"', 'reactions = ["D(d,n)3He"]\nthermal_factor = 2.0')
    )
    inputs = build_inputs(read_params(_write(tmp_path, text)), tmp_path)
    assert inputs.equilibrium.wall is None
    assert inputs.options["torque"] is True
    assert inputs.options["torque_options"] == {"jxb": False}
    assert inputs.options["neutrons"] is None
    assert inputs.options["neutron_options"]["thermal_factor"] == 2.0
    assert inputs.options["slowing_down_options"] == {"thermal_factor": 0.0}
    np.testing.assert_allclose(inputs.options["rho_edges"], np.linspace(0, 1, 11))


def test_other_kinds_build(tmp_path, equilibrium):
    equilibrium.to_geqdsk(tmp_path / "g.eqdsk")
    beam = _beam(markers=20)
    markers = xr.Dataset(
        {
            "position": (("component", "marker"), np.asarray(beam.markers.positions).T),
            "velocity": (
                ("component", "marker"),
                np.asarray(beam.markers.velocities).T,
            ),
            "weight": ("marker", np.asarray(beam.markers.weights)),
        },
        attrs={"mass": 2 * 1.66053906660e-27},
    )
    markers.to_netcdf(tmp_path / "markers.nc")
    rho = np.linspace(0, 1, 11)
    ne = 1e20 * (1 - 0.9 * rho**2)
    text = f"""
[equilibrium]
kind = "geqdsk"
path = "g.eqdsk"

[plasma]
kind = "tabulated"
rho = {rho.tolist()}
ne = {ne.tolist()}
te = {(2e4 * (1 - 0.9 * rho**2)).tolist()}
ti = {(2e4 * (1 - 0.9 * rho**2)).tolist()}
[plasma.ions.D]
mass_number = 2
charge = 1
density = {ne.tolist()}

[[beams]]
kind = "markers"
path = "markers.nc"

[[beams]]
kind = "gaussian"
energy = 5e5
power = 1e6
mass_number = 2
origin = [12.0, -5.3, 0.0]
direction = [-1.0, 0.0, 0.0]
divergence = [0.01, 0.02]
markers = 20

[[beams]]
kind = "source"
name = "S"
energy = 1e5
power = 1e6
mass_number = 1
position = [12.0, -3.0, 0.0]
direction = [-0.6, 1.0, 0.0]
power_fractions = [0.8, 0.1, 0.1]

[[beams]]
kind = "box"
name = "B"
[[beams.sources]]
name = "B/S1"
energy = 1e5
power = 1e6
mass_number = 1
position = [12.0, 3.0, 0.0]
direction = [-0.6, -1.0, 0.0]

[numerics]
rho_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
step = 0.05
markers = 20
"""
    ds = run_params(_write(tmp_path, text))
    assert list(ds.source.values) == ["beam1", "beam2", "S", "B/S1"]
    assert float(ds.injected_power) == pytest.approx(16.5e6 + 3e6, rel=1e-9)
    assert ds.sizes["rho"] == 4
    params = read_params(_write(tmp_path, text))
    params["beams"][1]["name"] = "beam1"
    with pytest.raises(ParameterError, match="used twice"):
        build_inputs(params, tmp_path)
    params["beams"][3]["sources"] = []
    with pytest.raises(ParameterError, match="needs sources"):
        from freeofcharge.run import _check_params

        _check_params(params)


def test_vmec_parameters_run_3d(tmp_path):
    pytest.importorskip("vmecpp")
    boundary = DATA / "coilstellaration_boundary.json"
    rest = "[plasma]" + PARAMS.split("[plasma]")[1]
    both = f'[equilibrium]\nkind = "vmec"\nboundary = "{boundary}"\npath = "w.nc"\n'
    with pytest.raises(ParameterError, match="either path"):
        run_params(_write(tmp_path, both + rest))
    text = f"""
[equilibrium]
kind = "vmec"
boundary = "{boundary}"
mpol = 5
ntor = 4
ns = [9, 17]
grid = [24, 16, 24]

[plasma]
kind = "parabolic"
ne0 = 5e19
te0 = 3e3
[plasma.ions.D]
mass_number = 2
charge = 1
fraction = 1.0

[[beams]]
kind = "gaussian"
energy = 3e4
power = 1e6
mass_number = 2
origin = [2.0, -0.3, 0.0]
direction = [-1.0, 0.3, 0.0]
markers = 50

[numerics]
rho_edges = 4
step = 0.005
seed = 1
"""
    with pytest.warns(UserWarning, match="trapped-electron correction"):
        ds = run_params(_write(tmp_path, text))
    assert ds.attrs["geometry"] == "3D"
    assert ds.attrs["torque"] == "collisional only"
    assert json.loads(ds.attrs["inputs"])["equilibrium"]["type"] == "GriddedEquilibrium"
    assert "torque_jxb" not in ds
    assert float(ds.absorbed_power) > 0.0
    assert abs(ds.attrs["torque_balance"]) < 1e-12
    assert float(ds.neutron_rate) > 0.0


def test_default_shells_and_stopping(equilibrium, plasma):
    from freeofcharge import ConstantStopping

    ds = run_nbi(
        equilibrium,
        plasma,
        _beam(markers=10),
        stopping=ConstantStopping(sigma=1e-20),
        step=0.1,
        torque=False,
        neutrons=False,
    )
    assert ds.sizes["rho"] == 50
    assert json.loads(ds.attrs["inputs"])["options"]["stopping"] == "ConstantStopping"
    assert ds.attrs["neutron_reactions"] == "none"
