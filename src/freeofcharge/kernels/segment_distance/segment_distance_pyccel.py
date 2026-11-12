"""The Pyccel version (CPU) of the ``segment_distance`` kernel.

``segment_distance_cuda.cu`` next to this file is the CUDA version (GPU) and
``segment_distance_numpy.py`` the version for machines without Pyccel, all with the same
arguments; `freeofcharge.coils.CoilSet.distance_at` calls it.
"""

# mypy: ignore-errors

from math import sqrt


def segment_distance(
    points: "float[:,:]",
    starts: "float[:,:]",
    edges: "float[:,:]",
    inverse_length2: "float[:]",
    distance: "float[:]",
) -> None:
    """The distance of each point to the nearest of the segments ``starts + u edges``, ``u`` in [0, 1].

    ``points`` ``(3, n)``, ``starts`` and ``edges`` ``(3, m)``, ``inverse_length2``
    ``(m,)`` the inverse squared lengths of the edges (finite also for an edge of zero
    length); ``distance`` ``(n,)``.
    """
    n = points.shape[1]
    m = starts.shape[1]
    # fmt: off
    #$ omp parallel for private(x, y, z, best, j, wx, wy, wz, t, gx, gy, gz, gap2) schedule(static)
    # fmt: on
    for k in range(n):
        x = points[0, k]
        y = points[1, k]
        z = points[2, k]
        best = 1.0e300
        for j in range(m):
            wx = x - starts[0, j]
            wy = y - starts[1, j]
            wz = z - starts[2, j]
            t = (wx * edges[0, j] + wy * edges[1, j] + wz * edges[2, j]) * (
                inverse_length2[j]
            )
            # if, not min and max: Pyccel's C min and max are variadic function calls
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            gx = wx - t * edges[0, j]
            gy = wy - t * edges[1, j]
            gz = wz - t * edges[2, j]
            gap2 = gx * gx + gy * gy + gz * gz
            if gap2 < best:
                best = gap2
        distance[k] = sqrt(best)
