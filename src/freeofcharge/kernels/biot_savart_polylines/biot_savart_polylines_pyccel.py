"""The Pyccel version (CPU) of the ``biot_savart_polylines`` kernel.

``biot_savart_polylines_cuda.cu`` next to this file is the CUDA version (GPU) and
``biot_savart_polylines_numpy.py`` the version for machines without Pyccel, all with the
same arguments; `freeofcharge.coils.CoilSet.field_at` calls it.
"""

# mypy: ignore-errors

from math import sqrt


def biot_savart_polylines(
    points: "float[:,:]",
    nodes: "float[:,:]",
    n_per_coil: int,
    currents: "float[:]",
    field: "float[:,:]",
) -> None:
    """The sum over straight segments of ``I (r_a + r_b) / (r_a r_b (r_a r_b + r_a.r_b)) r_a x r_b``.

    ``points`` ``(3, n)``; ``nodes`` ``(3, n_coils * n_per_coil)`` holds the nodes of each
    closed polyline in turn, its segments joining every node to the next and the last
    back to the first; ``currents`` ``(n_coils,)``. ``field`` ``(3, n)`` gets the sum,
    without the factor ``mu_0 / (4 pi)``. A segment adds nothing where the denominator
    vanishes (on its end points and on its line beyond them). The end of a segment is the
    start of the next, so each node's distance is computed once per point.
    """
    n = points.shape[1]
    n_coils = currents.shape[0]
    # fmt: off
    #$ omp parallel for private(x, y, z, bx, by, bz, c, first, current, ax, ay, az, na, i, j, cx, cy, cz, nb, denom, factor) schedule(static)
    # fmt: on
    for k in range(n):
        x = points[0, k]
        y = points[1, k]
        z = points[2, k]
        bx = 0.0
        by = 0.0
        bz = 0.0
        for c in range(n_coils):
            first = c * n_per_coil
            current = currents[c]
            ax = x - nodes[0, first]
            ay = y - nodes[1, first]
            az = z - nodes[2, first]
            na = sqrt(ax * ax + ay * ay + az * az)
            for i in range(n_per_coil):
                j = first + i + 1 if i + 1 < n_per_coil else first
                cx = x - nodes[0, j]
                cy = y - nodes[1, j]
                cz = z - nodes[2, j]
                nb = sqrt(cx * cx + cy * cy + cz * cz)
                denom = na * nb * (na * nb + (ax * cx + ay * cy + az * cz))
                if denom > 0.0:
                    factor = current * (na + nb) / denom
                    bx += factor * (ay * cz - az * cy)
                    by += factor * (az * cx - ax * cz)
                    bz += factor * (ax * cy - ay * cx)
                ax = cx
                ay = cy
                az = cz
                na = nb
        field[0, k] = bx
        field[1, k] = by
        field[2, k] = bz
