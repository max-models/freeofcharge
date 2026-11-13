r"""Shine-through heat loads on the first wall: the beam-dump or armour footprint.

`freeofcharge.deposit` attenuates every marker of a beam on its straight line and
keeps the share $T = e^{-\tau_\mathrm{total}}$ of its stream that is not ionised
(``Deposition.transmission``). `shine_through_loads` follows that surviving stream
on the same line to its first hit on the wall, a simgeometry surface: an
axisymmetric `freeofcharge.Equilibrium`'s wall polygon revolved about $z$
(`freeofcharge.orbits.revolved_wall`), a `TriangleMesh` read from CAD or a
`FourierSurface` (a 3D wall such as W7-X's stand-in). The hit is exact for the
triangulation (``first_hit``); on a closed wall a line that starts outside first
enters it, and the hit is where it leaves again, as in `deposit`.

The power of the stream, $P_b = w_b T_b E_b$, is tallied on the face it hits
(`freeofcharge.orbits.face_loads`, the tally of the orbit losses): the heat flux
of face $f$ is $q_f = \sum_{b \to f} P_b / A_f$. The angle of incidence $\alpha$
(between the line and the face normal) enters through the footprint, which is
stretched by $1 / \cos\alpha$: a parallel beam of power $P$ and cross section
$A$ puts $q = P\cos\alpha / A$ on a plate (Lambert's cosine law of the projected
area, as for the heat flux along field lines onto a target, Stangeby, *The Plasma
Boundary of Magnetic Fusion Devices*, IOP 2000, ch. 5). The power-weighted mean
$\alpha$ of each face is returned too.

A marker can stand for a beamlet with a divergence that its direction does not
sample: ``divergence`` spreads each stream over a Gaussian cone about its
direction, and ``resample`` draws that many lines per stream (the weight shared
equally), as BBNBI describes a beamlet (Asunta et al., Comput. Phys. Commun.
188, 33, 2015). With the markers of `freeofcharge.NeutralBeam.gaussian`, whose
directions already sample the beam's divergence, the cone widens the footprint
further (the angular variances add): there ``resample`` with a small
``divergence`` only smooths the Monte Carlo noise. The exact alternative is a
beam with more markers, deposited with ``births_per_marker=0`` (only the profiles
and the transmission).
"""

from __future__ import annotations

from typing import Any

import cunumpy as xp
import numpy as np
import scope_profiler as sp
import xarray as xr
from maybempi import MPI
from methodcard import method
from plasmacoll import make_rng
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.units import to_unit

from freeofcharge._profiling import PROFILE_TAGS
from freeofcharge.beam import NeutralBeam
from freeofcharge.deposition import Deposition, allreduce
from freeofcharge.orbits import face_loads, load_map, revolved_wall

__all__ = ["shine_through_loads", "wall_surface"]


def wall_surface(wall, n_phi: int = 256) -> tuple[Any, np.ndarray | None]:
    """Return a simgeometry surface for ``wall`` and its $(R, Z)$ outline if revolved.

    Args:
        wall: A surface with ``first_hit`` (`TriangleMesh`, `FourierSurface`),
            returned as it is; an axisymmetric equilibrium with a wall polygon
            (`freeofcharge.Equilibrium`, ``wall.vertices``); or the outline
            ``(n, 2)`` $(R, Z)$ [m] itself, revolved about $z$.
        n_phi: Toroidal segments of a revolved wall.
    """
    if hasattr(wall, "first_hit"):
        return wall, None
    polygon = getattr(wall, "wall", None)
    if polygon is not None:
        outline = np.asarray(polygon.vertices, dtype=float)
    elif hasattr(wall, "has_wall"):
        raise ValueError("the equilibrium has no axisymmetric wall: pass a surface")
    else:
        outline = np.asarray(wall, dtype=float)
    if outline.ndim != 2 or outline.shape[1] != 2 or outline.shape[0] < 3:
        raise ValueError(f"a wall outline is (n, 2) (R, Z), got {outline.shape}")
    if np.allclose(outline[0], outline[-1]):
        outline = outline[:-1]
    return revolved_wall(outline, n_phi), outline


def _arc_length(outline: np.ndarray, r, z) -> tuple[np.ndarray, float]:
    r"""Return the arc length $s$ along ``outline`` of the points nearest to $(R, Z)$, and its length.

    $s = 0$ at the lowest vertex, increasing counter-clockwise in the $(R, Z)$
    plane: up the outboard side, over the top and down the inboard side, so
    that neither midplane lies at the ends of the map.
    """
    r_o, z_o = outline[:, 0], outline[:, 1]
    if np.sum(r_o * np.roll(z_o, -1) - np.roll(r_o, -1) * z_o) < 0.0:
        outline = outline[::-1]
    outline = np.roll(outline, -int(np.argmin(outline[:, 1])), axis=0)
    a = outline
    b = np.roll(outline, -1, axis=0)
    edge = b - a
    length = np.linalg.norm(edge, axis=1)
    start = np.concatenate([[0.0], np.cumsum(length)[:-1]])
    # the faces of a revolved wall share few (R, Z): project each once
    points, back = np.unique(
        np.round(np.stack([np.ravel(r), np.ravel(z)], axis=-1), 12),
        axis=0,
        return_inverse=True,
    )
    s = np.empty(len(points))
    for lo in range(0, len(points), 4096):
        p = points[lo : lo + 4096, None, :]
        t = np.clip(np.sum((p - a) * edge, axis=-1) / length**2, 0.0, 1.0)
        gap = np.sum((a + t[..., None] * edge - p) ** 2, axis=-1)
        k = np.argmin(gap, axis=1)
        s[lo : lo + 4096] = start[k] + t[np.arange(len(k)), k] * length[k]
    return s[np.ravel(back)], float(length.sum())


def _spread(directions: np.ndarray, divergence, copies: int, rng) -> np.ndarray:
    r"""Return ``copies`` directions per line, Gaussian in two angles about each.

    $\hat d' \propto \hat d + \tan\theta_h \hat h + \tan\theta_v \hat v$ with
    $\hat h = \hat d \times \hat z$, $\hat v = \hat h \times \hat d$, as
    `freeofcharge.NeutralBeam.gaussian`; ``(copies * n, 3)``, copy-major.
    """
    sigma_h, sigma_v = np.broadcast_to(np.asarray(divergence, dtype=float), (2,))
    horizontal = np.cross(directions, [0.0, 0.0, 1.0])
    norm = np.linalg.norm(horizontal, axis=1)
    horizontal[norm < 1e-12] = [1.0, 0.0, 0.0]
    horizontal /= np.linalg.norm(horizontal, axis=1)[:, None]
    vertical = np.cross(horizontal, directions)
    n = len(directions)
    angles = rng.standard_normal((copies, n, 2)) * np.array([sigma_h, sigma_v])
    spread = (
        directions[None]
        + np.tan(angles[..., :1]) * horizontal[None]
        + np.tan(angles[..., 1:]) * vertical[None]
    )
    spread /= np.linalg.norm(spread, axis=-1, keepdims=True)
    return spread.reshape(copies * n, 3)


def _crossing(surface, starts: np.ndarray, ends: np.ndarray, closed: bool):
    """First hits of ``starts -> ends``, after entering a closed ``surface``.

    Returns ``(hit, point, normal, face, entered)``; ``entered`` is whether the
    line starts inside or crosses into the wall (always true if not closed).
    """
    hits = surface.first_hit(starts, ends)
    hit = np.asarray(hits.hit, dtype=bool).copy()
    point = np.asarray(hits.point, dtype=float).copy()
    normal = np.asarray(hits.normal, dtype=float).copy()
    face = np.asarray(hits.face, dtype=np.int64).copy()
    entered = np.ones(len(starts), dtype=bool)
    if closed and len(starts):
        outside = ~np.asarray(xp.to_numpy(surface.inside_at(starts)), dtype=bool)
        entered = ~outside | hit
        entering = np.nonzero(outside & hit)[0]
        if len(entering):
            again = surface.first_hit(point[entering], ends[entering])
            hit[entering] = np.asarray(again.hit, dtype=bool)
            point[entering] = np.asarray(again.point, dtype=float)
            normal[entering] = np.asarray(again.normal, dtype=float)
            face[entering] = np.asarray(again.face, dtype=np.int64)
    return hit, point, normal, face, entered


def _wall_hits(surface, starts, directions, length, rng, *, tries: int = 4):
    r"""Return ``(hit, point, normal, face)`` of the lines on ``surface``, and their directions.

    On a closed wall the inside must lie just before a hit, and a line inside
    the wall must hit it; a line that passed through an edge or a vertex
    without a hit (the triangle tests round off there) fails this and is traced
    again with its direction turned by about $10^{-9}$ rad. On an open wall
    every line that misses is traced again that way, since a line through an
    edge can slip between two faces; one that passes beside the wall misses again.
    """
    mesh = getattr(surface, "mesh", surface)
    closed = bool(getattr(mesh, "closed", False))
    directions = directions.copy()
    hit, point, normal, face, entered = _crossing(
        surface, starts, starts + length[:, None] * directions, closed
    )
    if not closed:
        # a line through an edge or vertex can slip between two faces: retrace the
        # misses turned slightly; a line that really passes beside the wall misses again
        lines = np.nonzero(~hit)[0]
        for _ in range(tries):
            if not len(lines):
                break
            turned = directions[lines] + 1e-9 * rng.standard_normal((len(lines), 3))
            turned /= np.linalg.norm(turned, axis=1)[:, None]
            ends = starts[lines] + length[lines, None] * turned
            found = _crossing(surface, starts[lines], ends, closed)
            now = np.asarray(found[0], dtype=bool)
            caught = lines[now]
            directions[caught] = turned[now]
            for array, value in zip(
                (hit, point, normal, face, entered), found, strict=True
            ):
                array[caught] = np.asarray(value)[now]
            lines = lines[~now]
        return hit, point, normal, face, directions
    lines = np.arange(len(starts))
    for _ in range(tries):
        before = point[lines] - 1e-6 * directions[lines]
        inside = np.ones(len(lines), dtype=bool)
        hits = np.nonzero(hit[lines])[0]
        inside[hits] = np.asarray(
            xp.to_numpy(surface.inside_at(before[hits])), dtype=bool
        )
        lines = lines[~inside | (~hit[lines] & entered[lines])]
        if not len(lines):
            break
        turned = directions[lines] + 1e-9 * rng.standard_normal((len(lines), 3))
        directions[lines] = turned / np.linalg.norm(turned, axis=1)[:, None]
        ends = starts[lines] + length[lines, None] * directions[lines]
        found = _crossing(surface, starts[lines], ends, closed)
        for array, value in zip(
            (hit, point, normal, face, entered), found, strict=True
        ):
            array[lines] = value
    return hit, point, normal, face, directions


@method(
    equations={
        "stream": r"P_b = w_b\, e^{-\tau_b}\, E_b",
        "heat_flux": r"q_f = \frac{1}{A_f} \sum_{b \to f} P_b",
        "pencil": r"q = \frac{P \cos\alpha}{A}, \qquad \cos\alpha = |\hat d \cdot \hat n|",
        "cone": r"\hat d' \propto \hat d + \tan\theta_h\, \hat h + \tan\theta_v\, \hat v, \qquad \theta_{h,v} \sim \mathcal{N}(0, \sigma_{h,v}^2), \qquad w' = \frac{w}{n_r}",
    },
    assumptions=(
        r"straight lines from the markers, the attenuation of `deposit`; nothing is absorbed between the plasma and the wall",
        r"the first exit through the wall's triangulation (`first_hit`); on a closed wall a line that starts outside first enters it",
        r"all the power deposited where it hits (no reflection, no sputtering); a steady state",
        r"the lines of a cone keep the transmission of their marker",
    ),
)
@sp.profile("freeofcharge.shine_through_loads", tags=PROFILE_TAGS)
def shine_through_loads(
    beam: NeutralBeam,
    deposition: Deposition,
    wall,
    *,
    n_phi: int = 256,
    divergence: float | tuple[float, float] = 0.0,
    resample: int = 1,
    seed: int | None = None,
    max_length: float | None = None,
    map_bins: tuple[int, int] = (64, 180),
    comm: MPI.Comm | None = None,
) -> xr.Dataset:
    r"""Return the heat flux of the shine-through of ``beam`` on each face of ``wall``.

    Dataset over ``face`` (`freeofcharge.orbits.face_loads`): ``power`` [W],
    ``area`` [m^2], ``wall_load`` (the heat flux $q_f$) [W/m^2], ``markers``
    (lines), ``incidence`` (the power-weighted mean angle $\alpha$ to the face
    normal) [rad], the face centres ``X``, ``Y``, ``Z``, and ``vertices`` and
    ``faces``; the footprint ``wall_load_map`` over ``(theta, phi)`` on a
    `FourierSurface`, over ``(s, phi)`` on a revolved wall ($s$ the arc length
    along the outline from its lowest point, counter-clockwise); and this rank's
    lines that hit, over ``hit``: ``hit_position``, ``hit_power``,
    ``hit_incidence``, ``hit_face``. Attributes: ``injected_power``,
    ``shine_through_power`` (the deposition's), ``wall_power`` (onto the
    wall), ``missed_power`` (lines that leave without a hit, e.g. past an open
    panel), the ``peak_wall_load`` [W/m^2] and its face centre
    ``peak_position`` [m]. ``wall_power + missed_power`` is
    ``shine_through_power`` to roundoff.

    Args:
        beam: This rank's markers, as deposited.
        deposition: `freeofcharge.deposit` of ``beam`` (its ``transmission``).
        wall: A surface or an axisymmetric wall (`wall_surface`).
        n_phi: Toroidal segments of a revolved wall.
        divergence: Standard deviation [rad] of the cone over which each stream
            is spread, a scalar or ``(horizontal, vertical)``; 0 for none.
        resample: Lines drawn per stream in the cone.
        seed: Seed of the cone's directions (per rank).
        max_length: Length of the lines [m]; to beyond the wall's box if None.
        map_bins: Bins of the footprint map, ``(theta or s, phi)``.
        comm: Communicator over which the markers are split; serial if None.
    """
    transmission = deposition.transmission
    if transmission is None:
        raise ValueError("the deposition has no transmission: deposit the beam again")
    transmission = np.asarray(xp.to_numpy(transmission), dtype=float)
    if transmission.shape != (beam.size,):
        raise ValueError(
            f"the deposition has {transmission.shape[0]} markers, the beam {beam.size}"
        )
    divergence = to_unit(divergence, "rad", "divergence")
    max_length = to_unit(max_length, "m", "max_length")
    resample = int(resample)
    if resample < 1:
        raise ValueError("resample must be at least 1")
    if resample > 1 and not np.any(np.asarray(divergence) > 0.0):
        raise ValueError("resampling needs a divergence to spread the lines")
    surface, outline = wall_surface(wall, n_phi)
    mesh = getattr(surface, "mesh", surface)

    origins = np.asarray(xp.to_numpy(beam.markers.positions), dtype=float)
    directions = np.asarray(xp.to_numpy(beam.directions), dtype=float)
    power = (
        np.asarray(xp.to_numpy(beam.markers.weights), dtype=float)
        * transmission
        * np.asarray(xp.to_numpy(beam.energies), dtype=float)
        * ELEMENTARY_CHARGE
    )
    streaming = power > 0.0
    origins, directions, power = (
        origins[streaming],
        directions[streaming],
        power[streaming],
    )
    rank = 0 if comm is None else comm.Get_rank()
    if np.any(np.asarray(divergence) > 0.0):
        directions = _spread(directions, divergence, resample, make_rng(seed, rank))
        origins = np.tile(origins, (resample, 1))
        power = np.tile(power / resample, resample)

    vertices = np.asarray(mesh.vertices, dtype=float)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if max_length is None:
        reach = np.linalg.norm(origins - 0.5 * (lower + upper), axis=1)
        length = reach + float(np.linalg.norm(upper - lower)) + 1.0
    else:
        length = np.full(len(origins), max_length)
    hit, point, normal, face, directions = _wall_hits(
        surface,
        origins,
        directions,
        length,
        make_rng(None if seed is None else seed + 1, rank),
    )

    cos_alpha = np.abs(np.sum(directions[hit] * normal[hit], axis=1))
    alpha = np.arccos(np.clip(cos_alpha, 0.0, 1.0))
    on_wall = power[hit]
    loads = face_loads(
        surface,
        face[hit],
        on_wall,
        theta_bins=map_bins[0],
        phi_bins=map_bins[1],
        comm=comm,
    )
    n_faces = loads.sizes["face"]
    weighted = allreduce(
        np.bincount(face[hit], weights=on_wall * alpha, minlength=n_faces), comm
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        incidence = np.where(loads.power > 0, weighted / loads.power, np.nan)
    loads["incidence"] = (
        "face",
        incidence,
        {"long_name": "angle of incidence", "units": "rad"},
    )
    loads["wall_load"].attrs["long_name"] = "heat flux"

    if outline is not None:
        centres = np.stack([loads.X.values, loads.Y.values, loads.Z.values], axis=-1)
        s_face, perimeter = _arc_length(
            outline, np.hypot(centres[:, 0], centres[:, 1]), centres[:, 2]
        )
        phi_face = np.mod(np.arctan2(centres[:, 1], centres[:, 0]), 2 * np.pi)
        s_edges = np.linspace(0.0, perimeter, map_bins[0] + 1)
        p_edges = np.linspace(0.0, 2 * np.pi, map_bins[1] + 1)
        loads["wall_load_map"] = (
            ("s", "phi"),
            load_map(
                s_face,
                phi_face,
                loads.area.values,
                loads.power.values,
                s_edges,
                p_edges,
            ),
            {"long_name": "heat flux", "units": "W/m^2"},
        )
        loads.coords["s"] = (
            "s",
            0.5 * (s_edges[1:] + s_edges[:-1]),
            {"long_name": "$s$", "units": "m"},
        )
        loads.coords["phi"] = (
            "phi",
            0.5 * (p_edges[1:] + p_edges[:-1]),
            {"long_name": r"$\varphi$", "units": "rad"},
        )
    elif "wall_load_map" in loads:
        loads["wall_load_map"].attrs["long_name"] = "heat flux"

    loads["hit_position"] = (("hit", "xyz"), point[hit], {"units": "m"})
    loads["hit_power"] = ("hit", on_wall, {"long_name": "power", "units": "W"})
    loads["hit_incidence"] = (
        "hit",
        alpha,
        {"long_name": "angle of incidence", "units": "rad"},
    )
    loads["hit_face"] = ("hit", face[hit])

    missed = allreduce(np.array([float(np.sum(power[~hit]))]), comm)[0]
    peak = int(np.argmax(loads.wall_load.values))
    loads.attrs.update(
        injected_power=deposition.injected_power,
        shine_through_power=deposition.shine_through_power,
        missed_power=float(missed),
        peak_position=[float(loads[c].values[peak]) for c in ("X", "Y", "Z")],
    )
    return loads
