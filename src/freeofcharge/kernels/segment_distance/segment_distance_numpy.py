"""The NumPy version of the ``segment_distance`` kernel, for machines without Pyccel.

Vectorized over chunks of points times all segments; the same arguments and results as
``segment_distance_pyccel.py``.
"""

from __future__ import annotations

import numpy as np

_MAX_PAIRS = 2_000_000
"""Most point–segment pairs held at once."""


def segment_distance(
    points: np.ndarray,
    starts: np.ndarray,
    edges: np.ndarray,
    inverse_length2: np.ndarray,
    distance: np.ndarray,
) -> None:
    """The distance of each point to the nearest of the segments ``starts + u edges``, ``u`` in [0, 1]."""
    a, edge = starts.T, edges.T
    size = max(1, _MAX_PAIRS // max(1, a.shape[0]))
    for lo in range(0, points.shape[1], size):
        w = points[:, lo : lo + size].T[:, None, :] - a[None]
        t = np.clip(np.sum(w * edge[None], axis=-1) * inverse_length2, 0.0, 1.0)
        gap = w - t[..., None] * edge[None]
        distance[lo : lo + size] = np.sqrt(np.min(np.sum(gap * gap, axis=-1), axis=1))
