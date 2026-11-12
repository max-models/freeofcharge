"""The NumPy version of the ``biot_savart_polylines`` kernel, for machines without Pyccel.

Vectorized over chunks of points times all segments; the same arguments and results as
``biot_savart_polylines_pyccel.py``.
"""

from __future__ import annotations

import numpy as np

_MAX_PAIRS = 2_000_000
"""Most point–segment pairs held at once."""


def biot_savart_polylines(
    points: np.ndarray,
    nodes: np.ndarray,
    n_per_coil: int,
    currents: np.ndarray,
    field: np.ndarray,
) -> None:
    """The sum over straight segments of ``I (r_a + r_b) / (r_a r_b (r_a r_b + r_a.r_b)) r_a x r_b``."""
    polylines = nodes.T.reshape(currents.shape[0], n_per_coil, 3)
    a = polylines.reshape(-1, 3)
    b = np.roll(polylines, -1, axis=1).reshape(-1, 3)
    segment_currents = np.repeat(currents, n_per_coil)
    size = max(1, _MAX_PAIRS // max(1, a.shape[0]))
    for lo in range(0, points.shape[1], size):
        p = points[:, lo : lo + size].T
        ra = p[:, None, :] - a[None]
        rb = p[:, None, :] - b[None]
        na = np.sqrt(np.sum(ra * ra, axis=-1))
        nb = np.sqrt(np.sum(rb * rb, axis=-1))
        denom = na * nb * (na * nb + np.sum(ra * rb, axis=-1))
        factor = np.where(
            denom > 0.0,
            segment_currents * (na + nb) / np.where(denom > 0.0, denom, 1.0),
            0.0,
        )
        field[:, lo : lo + size] = np.sum(
            factor[..., None] * np.cross(ra, rb), axis=1
        ).T
