"""The ``biot_savart_polylines`` and ``segment_distance`` kernels.

Their Pyccel (compiled, or Python), NumPy and CUDA versions against each other and
against the plain NumPy Biot–Savart sum; the CUDA version runs on the CPU through
cunumpy's emulator (skipped without a C++ compiler).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from cunumpy.kernel_testing import emulate_cuda_kernel, emulation_compiler

from freeofcharge.coils import MU_0, CoilSet
from freeofcharge.kernels.catalog import KERNEL_NAMES, KERNELS

DATA = Path(__file__).parents[1] / "data" / "coilstellaration"
PYCCEL = pytest.mark.skipif(
    importlib.util.find_spec("pyccel") is None, reason="Pyccel is not installed"
)


def _coils(rng, n_coils=3, n_points=17):
    """Wobbly closed loops with different currents."""
    s = 2 * np.pi * np.arange(n_points) / n_points
    loops = []
    for c in range(n_coils):
        radius = 1.0 + 0.1 * c + 0.05 * rng.normal(size=n_points)
        loops.append(
            np.stack(
                [radius * np.cos(s), radius * np.sin(s), 0.2 * c + 0.1 * np.sin(3 * s)],
                -1,
            )
        )
    return CoilSet(np.array(loops), rng.normal(size=n_coils) * 1e6)


def _points(rng, coils, n=97):
    """Points around the coils, with some on the nodes and on a segment's line."""
    points = rng.uniform(-1.5, 1.5, (n, 3))
    nodes = coils.filaments.reshape(-1, 3)
    points[:3] = nodes[:3]  # on the filaments: those segments add nothing
    points[3] = 2 * nodes[1] - nodes[0]  # on the line beyond a segment's end
    return points


def _cases(name, rng):
    """Arguments of kernel ``name`` and the index of its output array."""
    coils = _coils(rng)
    points = np.array(_points(rng, coils).T, order="C")
    n = points.shape[1]
    if name == "biot_savart_polylines":
        args = [
            points,
            coils._nodes,
            coils.filaments.shape[1],
            coils.currents,
            np.zeros((3, n)),
        ]
        return args, 4
    args = [points, coils._nodes, coils._edges, coils._inverse_length2, np.zeros(n)]
    return args, 4


def _run(implementation, args):
    args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    implementation(*args)
    return args


def _reference_field(coils, points):
    """The Biot–Savart sum over all segments at once, as the method card writes it."""
    ra = points[:, None, :] - coils.starts[None]
    rb = points[:, None, :] - coils.ends[None]
    na = np.linalg.norm(ra, axis=-1)
    nb = np.linalg.norm(rb, axis=-1)
    denom = na * nb * (na * nb + np.sum(ra * rb, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(denom > 0, coils.segment_currents * (na + nb) / denom, 0.0)
    return MU_0 / (4 * np.pi) * np.sum(factor[..., None] * np.cross(ra, rb), axis=1)


def _reference_distance(coils, points):
    edge = coils.ends - coils.starts
    w = points[:, None, :] - coils.starts[None]
    t = np.clip(np.sum(w * edge, -1) / np.sum(edge * edge, -1), 0.0, 1.0)
    return np.linalg.norm(w - t[..., None] * edge, axis=-1).min(axis=1)


def _close(result, expected, rtol=1e-13):
    """Agreement relative to the largest value (single components may cancel)."""
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=rtol * scale)


def test_the_catalog_has_every_kernel():
    assert set(KERNEL_NAMES) == {"biot_savart_polylines", "segment_distance"}
    KERNELS.check_signatures()


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_host_versions_agree(name):
    args, out = _cases(name, np.random.default_rng(4))
    host = KERNELS[name].host_kernel.kernel
    expected = _run(host.python, args)[out]
    for implementation in (host.get("numpy"), KERNELS[name]):
        _close(_run(implementation, args)[out], expected)


@PYCCEL
@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_the_compiled_version_runs(name):
    # with Pyccel, a call with NumPy arrays runs the compiled kernel, not a fallback
    assert KERNELS[name].host_kernel.kernel.available("pyccel")
    assert KERNELS[name].selected() == "pyccel"


@pytest.mark.skipif(emulation_compiler() is None, reason="no C++ compiler")
@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_cuda_version_agrees(name):
    args, out = _cases(name, np.random.default_rng(5))
    expected = _run(KERNELS[name].host_kernel.kernel.python, args)[out]
    result = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    emulate_cuda_kernel(KERNELS[name].cuda_kernel, *result, n_threads=args[0].shape[1])
    _close(result[out], expected)


def test_field_and_distance_match_numpy():
    rng = np.random.default_rng(6)
    coils = _coils(rng)
    points = _points(rng, coils)
    _close(coils.field_at(points), _reference_field(coils, points))
    _close(coils.distance_at(points), _reference_distance(coils, points))
    # shapes: one point, a grid, no point
    assert coils.field_at(points[0]).shape == (3,)
    assert coils.field_at(points.reshape(97, 1, 3)).shape == (97, 1, 3)
    assert coils.distance_at(points[5]).shape == ()
    assert coils.field_at(np.zeros((0, 3))).shape == (0, 3)
    assert coils.distance_at(np.zeros((2, 0, 3))).shape == (2, 0)


def test_field_of_the_coilstellaration_sample_matches_numpy():
    pytest.importorskip("plasma_plots")
    coilset = json.loads((DATA / "coilstellaration_coilset.json").read_text())
    coils = CoilSet.from_coilset(coilset, n_points=128)
    assert coils.starts.shape[0] == 3840
    rng = np.random.default_rng(7)
    r, phi = rng.uniform(0.6, 1.4, 64), rng.uniform(0, 2 * np.pi, 64)
    points = np.stack(
        [r * np.cos(phi), r * np.sin(phi), rng.uniform(-0.3, 0.3, 64)], -1
    )
    _close(coils.field_at(points), _reference_field(coils, points))
    _close(coils.distance_at(points), _reference_distance(coils, points))
