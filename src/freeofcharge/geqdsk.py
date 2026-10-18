"""Read and write G-EQDSK equilibrium files.

The layout (format ``(6a8, 3i4)`` then ``(5e16.9)``): a header line ending in
``nw nh``; 20 scalars (``rdim zdim rcentr rleft zmid``, ``rmaxis zmaxis simag
sibry bcentr``, ``current simag - rmaxis -``, ``zmaxis - sibry - -``); the
profiles ``fpol pres ffprim pprime`` on ``nw`` points uniform in flux from the
axis to the boundary; ``psirz`` with R varying fastest; ``qpsi``; a line
``nbbbs limitr``; the boundary and the limiter as interleaved ``(R, Z)`` pairs.
Numbers may run together (``1.0e+00-2.0e+00``), so they are read by pattern,
not by splitting on spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["GEqdsk", "read", "write"]

_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")


@dataclass
class GEqdsk:
    """The parts of a G-EQDSK file that :class:`~freeofcharge.Equilibrium` uses."""

    r: np.ndarray
    z: np.ndarray
    psi: np.ndarray
    """Flux, shape ``(len(r), len(z))``."""
    psi_axis: float
    psi_boundary: float
    fpol: np.ndarray
    axis: tuple[float, float]
    boundary: np.ndarray
    """Vertices ``(n, 2)``."""
    limiter: np.ndarray
    """Vertices ``(n, 2)``, possibly empty."""
    description: str = ""
    b_centre: float = 0.0
    r_centre: float = 0.0
    current: float = 0.0
    pressure: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _numbers(lines: list[str], count: int, start: int) -> tuple[np.ndarray, int]:
    """Read ``count`` numbers from ``lines[start:]``; return them and the next line."""
    values: list[float] = []
    line = start
    while len(values) < count:
        if line >= len(lines):
            raise ValueError(
                f"G-EQDSK file ends after {len(values)} of {count} numbers"
            )
        values.extend(
            float(v.replace("D", "E").replace("d", "e"))
            for v in _NUMBER.findall(lines[line])
        )
        line += 1
    if len(values) != count:
        raise ValueError("a G-EQDSK block does not end at a line end")
    return np.asarray(values), line


def read(path: str | Path) -> GEqdsk:
    """Read a G-EQDSK file."""
    lines = Path(path).read_text().splitlines()
    header = lines[0]
    nw, nh = (int(v) for v in header.split()[-2:])
    scalars, line = _numbers(lines, 20, 1)
    rdim, zdim, rcentr, rleft, zmid = scalars[0:5]
    rmaxis, zmaxis, simag, sibry, bcentr = scalars[5:10]
    current = scalars[10]
    profiles, line = _numbers(lines, 4 * nw, line)
    fpol, pres = profiles[:nw], profiles[nw : 2 * nw]
    psirz, line = _numbers(lines, nw * nh, line)
    qpsi, line = _numbers(lines, nw, line)
    nbbbs, limitr = 0, 0
    while line < len(lines) and not lines[line].strip():
        line += 1
    if line < len(lines):
        nbbbs, limitr = (int(v) for v in lines[line].split()[:2])
        line += 1
    boundary, line = _numbers(lines, 2 * nbbbs, line)
    limiter, line = _numbers(lines, 2 * limitr, line)
    return GEqdsk(
        r=rleft + np.linspace(0.0, rdim, nw),
        z=zmid - 0.5 * zdim + np.linspace(0.0, zdim, nh),
        psi=psirz.reshape(nh, nw).T,
        psi_axis=float(simag),
        psi_boundary=float(sibry),
        fpol=fpol,
        axis=(float(rmaxis), float(zmaxis)),
        boundary=boundary.reshape(-1, 2),
        limiter=limiter.reshape(-1, 2),
        description=header[:48].strip(),
        b_centre=float(bcentr),
        r_centre=float(rcentr),
        current=float(current),
        pressure=pres,
        q=qpsi,
    )


def _block(values) -> list[str]:
    values = np.asarray(values, dtype=float).ravel()
    return [
        "".join(f"{v:16.9e}" for v in values[i : i + 5])
        for i in range(0, len(values), 5)
    ]


def write(path: str | Path, data: GEqdsk) -> None:
    """Write a G-EQDSK file."""
    nw, nh = len(data.r), len(data.z)
    rdim = data.r[-1] - data.r[0]
    zdim = data.z[-1] - data.z[0]
    zmid = 0.5 * (data.z[-1] + data.z[0])
    rmaxis, zmaxis = data.axis
    simag, sibry = data.psi_axis, data.psi_boundary
    zeros = np.zeros(nw)
    pressure = data.pressure if len(data.pressure) == nw else zeros
    q = data.q if len(data.q) == nw else zeros
    lines = [f"{data.description[:48]:<48}{0:4d}{nw:4d}{nh:4d}"]
    lines += _block([rdim, zdim, data.r_centre, data.r[0], zmid])
    lines += _block([rmaxis, zmaxis, simag, sibry, data.b_centre])
    lines += _block([data.current, simag, 0.0, rmaxis, 0.0])
    lines += _block([zmaxis, 0.0, sibry, 0.0, 0.0])
    for profile in (data.fpol, pressure, zeros, zeros):
        lines += _block(profile)
    lines += _block(np.asarray(data.psi).T)
    lines += _block(q)
    lines.append(f"{len(data.boundary):5d}{len(data.limiter):5d}")
    lines += _block(np.asarray(data.boundary).ravel())
    lines += _block(np.asarray(data.limiter).ravel())
    Path(path).write_text("\n".join(lines) + "\n")
