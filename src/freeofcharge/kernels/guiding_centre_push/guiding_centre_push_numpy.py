"""The NumPy version of the ``guiding_centre_push`` kernel, for machines without Pyccel.

Vectorized over the markers, step by step; the same arguments and results as
``guiding_centre_push_pyccel.py`` (to roundoff), and much slower.
"""

from __future__ import annotations

import numpy as np


def spline_weights(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cubic B-spline weights ``(n, 4)`` of the nodes around ``t`` and their derivatives."""
    u = 1.0 - t
    t2 = t * t
    t3 = t2 * t
    w = np.stack(
        [
            u**3 / 6.0,
            (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
            (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
            t3 / 6.0,
        ],
        axis=-1,
    )
    d = np.stack([-0.5 * u * u, 1.5 * t2 - 2.0 * t, -1.5 * t2 + t + 0.5, 0.5 * t2], -1)
    return w, d


def _locate(grid, shape, x, y, z):
    """Fractional grid indices and whether the points are on the grid."""
    r = np.hypot(x, y)
    fi = (r - grid[0]) / grid[2]
    fk = (z - grid[1]) / grid[4]
    ok = (fi >= 0.0) & (fi <= shape[0] - 1) & (fk >= 0.0) & (fk <= shape[2] - 1)
    phi = np.arctan2(y, x)
    fj = np.mod(phi, grid[5]) / grid[3]
    # off the grid: any node, the result is discarded
    fi = np.where(ok, fi, 0.0)
    fk = np.where(ok, fk, 0.0)
    return r, phi, fi, fj, fk, ok


def _blocks(c, grid, shape, fi, fj, fk, components, n_components=4):
    """The 4 x 4 x 4 coefficients around each point, ``(n, len(components), 4, 4, 4)``."""
    pad = int(grid[6])
    c = c.reshape(*shape[3:6], n_components)
    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64)
    k0 = np.floor(fk).astype(np.int64)
    offsets = np.arange(4)
    ii = (i0 + pad - 1)[:, None] + offsets
    jj = (j0 + pad - 1)[:, None] + offsets
    kk = (k0 + pad - 1)[:, None] + offsets
    block = c[ii[:, :, None, None], jj[:, None, :, None], kk[:, None, None, :]][
        ..., components
    ]
    return np.moveaxis(block, -1, 1), fi - i0, fj - j0, fk - k0


def scalar_at(c, n_components, component, grid, shape, x, y, z):
    """Return one interpolated component at points (0 off the grid) and whether they are on it."""
    _, _, fi, fj, fk, ok = _locate(grid, shape, x, y, z)
    block, ti, tj, tk = _blocks(c, grid, shape, fi, fj, fk, [component], n_components)
    wi, _ = spline_weights(ti)
    wj, _ = spline_weights(tj)
    wk, _ = spline_weights(tk)
    value = np.einsum("na,nb,nc,nabc->n", wi, wj, wk, block[:, 0])
    return np.where(ok, value, 0.0), ok


def guiding_centre_rhs(c, grid, shape, x, y, z, vpar, mu, qm):
    """Guiding-centre velocity (x, y, z), parallel acceleration, rho^2 and on-grid flags."""
    r, _, fi, fj, fk, ok = _locate(grid, shape, x, y, z)
    r = np.where(ok, r, 1.0)
    block, ti, tj, tk = _blocks(c, grid, shape, fi, fj, fk, [0, 1, 2, 3])
    wi, di = spline_weights(ti)
    wj, dj = spline_weights(tj)
    wk, dk = spline_weights(tk)
    rho2 = np.einsum("na,nb,nc,nabc->n", wi, wj, wk, block[:, 0])
    field = block[:, 1:]
    b = np.einsum("na,nb,nc,nqabc->qn", wi, wj, wk, field)
    d_r = np.einsum("na,nb,nc,nqabc->qn", di, wj, wk, field) / grid[2]
    d_p = np.einsum("na,nb,nc,nqabc->qn", wi, dj, wk, field) / grid[3]
    d_z = np.einsum("na,nb,nc,nqabc->qn", wi, wj, dk, field) / grid[4]
    br, bp, bz = b
    babs = np.sqrt(br * br + bp * bp + bz * bz)
    u = b / babs
    g = np.stack(
        [np.sum(u * d_r, axis=0), np.sum(u * d_p, axis=0) / r, np.sum(u * d_z, axis=0)]
    )
    curl = np.stack(
        [
            d_p[2] / r - d_z[1],
            d_z[0] - d_r[2],
            d_r[1] + bp / r - d_p[0] / r,
        ]
    )
    u_x_g = np.cross(u, g, axis=0)
    kappa = (curl + u_x_g) / babs
    star = b + (vpar / qm) * kappa
    spar = np.sum(u * star, axis=0)
    velocity = (vpar * star + (mu / qm) * u_x_g) / spar
    acc = -mu * np.sum(star * g, axis=0) / spar
    cos_p, sin_p = np.cos(np.arctan2(y, x)), np.sin(np.arctan2(y, x))
    vx = cos_p * velocity[0] - sin_p * velocity[1]
    vy = sin_p * velocity[0] + cos_p * velocity[1]
    return vx, vy, velocity[2], acc, rho2, ok


def guiding_centre_push(
    state: np.ndarray,
    mu: np.ndarray,
    duration: np.ndarray,
    max_step: np.ndarray,
    c: np.ndarray,
    grid: np.ndarray,
    shape: np.ndarray,
    qm: float,
    rho2_stop: float,
    wall: np.ndarray,
    clearance_stop: float,
    clearance_high: float,
    rho_edges: np.ndarray,
    previous: np.ndarray,
    elapsed: np.ndarray,
    status: np.ndarray,
    shell_time: np.ndarray,
    shell_vpar: np.ndarray,
    path: np.ndarray,
    taken: np.ndarray,
) -> None:
    """RK4 guiding-centre push, see ``guiding_centre_push_pyccel.guiding_centre_push``."""
    previous[:] = state[:3]
    elapsed[:] = 0.0
    status[:] = 0
    taken[:] = 0
    record = path.shape[0]
    positive = duration > 0.0
    steps = np.where(
        positive, np.ceil(duration / np.where(positive, max_step, 1.0)), 0
    ).astype(np.int64)
    dt = np.where(positive, duration / np.maximum(steps, 1), 0.0)
    h = 0.5 * dt
    n_shells = rho_edges.shape[0] - 1
    done = np.zeros(state.shape[1], dtype=np.int64)
    active = steps > 0
    while np.any(active):
        idx = np.nonzero(active)[0]
        x, y, z, v = (state[q, idx] for q in range(4))
        m, d, hh = mu[idx], dt[idx], h[idx]
        x1, y1, z1, v1, rho2, ok1 = guiding_centre_rhs(
            c, grid, shape, x, y, z, v, m, qm
        )
        x2, y2, z2, v2, _, ok2 = guiding_centre_rhs(
            c, grid, shape, x + hh * x1, y + hh * y1, z + hh * z1, v + hh * v1, m, qm
        )
        x3, y3, z3, v3, _, ok3 = guiding_centre_rhs(
            c, grid, shape, x + hh * x2, y + hh * y2, z + hh * z2, v + hh * v2, m, qm
        )
        x4, y4, z4, v4, _, ok4 = guiding_centre_rhs(
            c, grid, shape, x + d * x3, y + d * y3, z + d * z3, v + d * v3, m, qm
        )
        good = ok1 & ok2 & ok3 & ok4
        # the steps taken: score their start
        rho = np.sqrt(np.maximum(rho2, 0.0))
        shell = np.searchsorted(rho_edges, rho, side="right") - 1
        scored = good & (shell >= 0) & (shell < n_shells)
        np.add.at(shell_time, (shell[scored], idx[scored]), d[scored])
        np.add.at(shell_vpar, (shell[scored], idx[scored]), (v * d)[scored])
        status[idx[~good]] = 2
        active[idx[~good]] = False
        g = idx[good]
        sixth = d[good] / 6.0
        new = np.stack(
            [
                x[good] + sixth * (x1 + 2 * x2 + 2 * x3 + x4)[good],
                y[good] + sixth * (y1 + 2 * y2 + 2 * y3 + y4)[good],
                z[good] + sixth * (z1 + 2 * z2 + 2 * z3 + z4)[good],
                v[good] + sixth * (v1 + 2 * v2 + 2 * v3 + v4)[good],
            ]
        )
        previous[:, g] = state[:3, g]
        if record > 1:
            room = taken[g] < record
            path[taken[g][room], :, g[room]] = state[:3, g[room]].T
        taken[g] += 1
        state[:, g] = new
        elapsed[g] += d[good]
        done[g] += 1
        rho2_new, on_grid = scalar_at(c, 4, 0, grid, shape, new[0], new[1], new[2])
        stop = ~on_grid | (rho2_new >= rho2_stop)
        if wall.shape[0] > 1:
            clearance, _ = scalar_at(wall, 1, 0, grid, shape, new[0], new[1], new[2])
            stop |= (clearance < clearance_stop) | (clearance > clearance_high)
        status[g[stop]] = 1
        active[g[stop]] = False
        active[g[done[g] >= steps[g]]] = False
