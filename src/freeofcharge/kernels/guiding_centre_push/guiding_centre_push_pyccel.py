r"""The Pyccel version (CPU) of the ``guiding_centre_push`` kernel.

``guiding_centre_push_numpy.py`` next to this file is the version for machines without
Pyccel, with the same arguments; `freeofcharge.orbits` calls it. The field is the
cubic B-spline interpolant of `freeofcharge.GriddedEquilibrium` (its padded
coefficients of $\rho^2$, $B_R$, $B_\varphi$, $B_Z$ on a grid in $(R, \varphi, Z)$ over
one field period), differentiated exactly, so the drifts are those of the
interpolated field.
"""

# mypy: ignore-errors

from math import atan2, ceil, floor, sqrt


def spline_weights(t: float):
    """The cubic B-spline weights of the four nodes around a point and their derivatives."""
    u = 1.0 - t
    t2 = t * t
    t3 = t2 * t
    w0 = u * u * u / 6.0
    w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    w3 = t3 / 6.0
    d0 = -0.5 * u * u
    d1 = 1.5 * t2 - 2.0 * t
    d2 = -1.5 * t2 + t + 0.5
    d3 = 0.5 * t2
    return w0, w1, w2, w3, d0, d1, d2, d3


def scalar_at(
    c: "float[:]",
    n_components: int,
    component: int,
    grid: "float[:]",
    shape: "int[:]",
    x: float,
    y: float,
    z: float,
):
    """Return one interpolated component at a point and 1, or 0 and 0 off the grid."""
    r = sqrt(x * x + y * y)
    fi = (r - grid[0]) / grid[2]
    fk = (z - grid[1]) / grid[4]
    if fi < 0.0 or fi > shape[0] - 1 or fk < 0.0 or fk > shape[2] - 1:
        return 0.0, 0
    phi = atan2(y, x)
    phi = phi - grid[5] * floor(phi / grid[5])
    fj = phi / grid[3]
    pad = int(grid[6])
    i0 = int(floor(fi))
    j0 = int(floor(fj))
    k0 = int(floor(fk))
    a0, a1, a2, a3, da0, da1, da2, da3 = spline_weights(fi - i0)
    b0, b1, b2, b3, db0, db1, db2, db3 = spline_weights(fj - j0)
    e0, e1, e2, e3, de0, de1, de2, de3 = spline_weights(fk - k0)
    i0 = i0 + pad - 1
    j0 = j0 + pad - 1
    k0 = k0 + pad - 1
    sj = shape[5] * n_components
    si = shape[4] * sj
    nc = n_components
    value = 0.0
    for a in range(4):
        wa = a0 if a == 0 else (a1 if a == 1 else (a2 if a == 2 else a3))
        for b in range(4):
            wb = b0 if b == 0 else (b1 if b == 1 else (b2 if b == 2 else b3))
            q = (i0 + a) * si + (j0 + b) * sj + k0 * nc + component
            value += (
                wa
                * wb
                * (e0 * c[q] + e1 * c[q + nc] + e2 * c[q + 2 * nc] + e3 * c[q + 3 * nc])
            )
    return value, 1


def guiding_centre_rhs(
    c: "float[:]",
    grid: "float[:]",
    shape: "int[:]",
    x: float,
    y: float,
    z: float,
    vpar: float,
    mu: float,
    qm: float,
):
    """The guiding-centre velocity and parallel acceleration, rho^2, and 1 (0 off the grid).

    ``dX/dt = (v_par B* + (mu / qm) b x grad|B|) / B*_par``,
    ``dv_par/dt = -mu B* . grad|B| / B*_par``, ``B* = B + (v_par / qm) curl b``, with
    ``mu = v_perp^2 / (2 |B|)`` per unit mass and ``qm = q / m``.
    """
    r = sqrt(x * x + y * y)
    fi = (r - grid[0]) / grid[2]
    fk = (z - grid[1]) / grid[4]
    if fi < 0.0 or fi > shape[0] - 1 or fk < 0.0 or fk > shape[2] - 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    phi = atan2(y, x)
    cos_p = x / r
    sin_p = y / r
    phi = phi - grid[5] * floor(phi / grid[5])
    fj = phi / grid[3]
    pad = int(grid[6])
    i0 = int(floor(fi))
    j0 = int(floor(fj))
    k0 = int(floor(fk))
    a0, a1, a2, a3, da0, da1, da2, da3 = spline_weights(fi - i0)
    b0, b1, b2, b3, db0, db1, db2, db3 = spline_weights(fj - j0)
    e0, e1, e2, e3, de0, de1, de2, de3 = spline_weights(fk - k0)
    i0 = i0 + pad - 1
    j0 = j0 + pad - 1
    k0 = k0 + pad - 1
    rho2 = 0.0
    br = 0.0
    bp = 0.0
    bz = 0.0
    br_r = 0.0
    br_p = 0.0
    br_z = 0.0
    bp_r = 0.0
    bp_p = 0.0
    bp_z = 0.0
    bz_r = 0.0
    bz_p = 0.0
    bz_z = 0.0
    sj = shape[5] * 4
    si = shape[4] * sj
    for a in range(4):
        wa = a0 if a == 0 else (a1 if a == 1 else (a2 if a == 2 else a3))
        va = da0 if a == 0 else (da1 if a == 1 else (da2 if a == 2 else da3))
        for b in range(4):
            wb = b0 if b == 0 else (b1 if b == 1 else (b2 if b == 2 else b3))
            vb = db0 if b == 0 else (db1 if b == 1 else (db2 if b == 2 else db3))
            q = (i0 + a) * si + (j0 + b) * sj + k0 * 4
            # the four nodes along Z: value and Z derivative of each component
            s0 = e0 * c[q] + e1 * c[q + 4] + e2 * c[q + 8] + e3 * c[q + 12]
            s1 = e0 * c[q + 1] + e1 * c[q + 5] + e2 * c[q + 9] + e3 * c[q + 13]
            s2 = e0 * c[q + 2] + e1 * c[q + 6] + e2 * c[q + 10] + e3 * c[q + 14]
            s3 = e0 * c[q + 3] + e1 * c[q + 7] + e2 * c[q + 11] + e3 * c[q + 15]
            t1 = de0 * c[q + 1] + de1 * c[q + 5] + de2 * c[q + 9] + de3 * c[q + 13]
            t2 = de0 * c[q + 2] + de1 * c[q + 6] + de2 * c[q + 10] + de3 * c[q + 14]
            t3 = de0 * c[q + 3] + de1 * c[q + 7] + de2 * c[q + 11] + de3 * c[q + 15]
            wab = wa * wb
            dab_r = va * wb
            dab_p = wa * vb
            rho2 += wab * s0
            br += wab * s1
            bp += wab * s2
            bz += wab * s3
            br_r += dab_r * s1
            bp_r += dab_r * s2
            bz_r += dab_r * s3
            br_p += dab_p * s1
            bp_p += dab_p * s2
            bz_p += dab_p * s3
            br_z += wab * t1
            bp_z += wab * t2
            bz_z += wab * t3
    # index derivatives to (d/dR, d/dphi, d/dZ)
    ir = 1.0 / grid[2]
    ip = 1.0 / grid[3]
    iz = 1.0 / grid[4]
    br_r *= ir
    bp_r *= ir
    bz_r *= ir
    br_p *= ip
    bp_p *= ip
    bz_p *= ip
    br_z *= iz
    bp_z *= iz
    bz_z *= iz
    babs = sqrt(br * br + bp * bp + bz * bz)
    inv_b = 1.0 / babs
    ur = br * inv_b
    up = bp * inv_b
    uz = bz * inv_b
    # grad|B| in the orthonormal cylindrical basis
    gr = ur * br_r + up * bp_r + uz * bz_r
    gp = (ur * br_p + up * bp_p + uz * bz_p) / r
    gz = ur * br_z + up * bp_z + uz * bz_z
    # curl B
    cr = bz_p / r - bp_z
    cp = br_z - bz_r
    cz = bp_r + bp / r - br_p / r
    # curl b = curl B / |B| + b x grad|B| / |B|
    kr = (cr + (up * gz - uz * gp)) * inv_b
    kp = (cp + (uz * gr - ur * gz)) * inv_b
    kz = (cz + (ur * gp - up * gr)) * inv_b
    rho_par = vpar / qm
    sr = br + rho_par * kr
    sp = bp + rho_par * kp
    sz = bz + rho_par * kz
    spar = ur * sr + up * sp + uz * sz
    mq = mu / qm
    vr = (vpar * sr + mq * (up * gz - uz * gp)) / spar
    vp = (vpar * sp + mq * (uz * gr - ur * gz)) / spar
    vz = (vpar * sz + mq * (ur * gp - up * gr)) / spar
    acc = -mu * (sr * gr + sp * gp + sz * gz) / spar
    return cos_p * vr - sin_p * vp, sin_p * vr + cos_p * vp, vz, acc, rho2, 1


def push_one(
    k: int,
    state: "float[:,:]",
    mu: "float[:]",
    duration: "float[:]",
    max_step: "float[:]",
    c: "float[:]",
    grid: "float[:]",
    shape: "int[:]",
    qm: float,
    rho2_stop: float,
    wall: "float[:]",
    clearance_stop: float,
    rho_edges: "float[:]",
    previous: "float[:,:]",
    elapsed: "float[:]",
    status: "int[:]",
    shell_time: "float[:,:]",
    shell_vpar: "float[:,:]",
) -> None:
    """Push marker ``k`` (see `guiding_centre_push`)."""
    x = state[0, k]
    y = state[1, k]
    z = state[2, k]
    v = state[3, k]
    m = mu[k]
    previous[0, k] = x
    previous[1, k] = y
    previous[2, k] = z
    elapsed[k] = 0.0
    status[k] = 0
    if duration[k] <= 0.0:
        return
    steps = int(ceil(duration[k] / max_step[k]))
    dt = duration[k] / steps
    h = 0.5 * dt
    n_shells = rho_edges.shape[0] - 1
    t = 0.0
    for _ in range(steps):
        x1, y1, z1, v1, rho2, ok = guiding_centre_rhs(c, grid, shape, x, y, z, v, m, qm)
        if ok == 0:
            status[k] = 2
            break
        rho = sqrt(rho2) if rho2 > 0.0 else 0.0
        x2, y2, z2, v2, rho2, ok = guiding_centre_rhs(
            c, grid, shape, x + h * x1, y + h * y1, z + h * z1, v + h * v1, m, qm
        )
        if ok == 0:
            status[k] = 2
            break
        x3, y3, z3, v3, rho2, ok = guiding_centre_rhs(
            c, grid, shape, x + h * x2, y + h * y2, z + h * z2, v + h * v2, m, qm
        )
        if ok == 0:
            status[k] = 2
            break
        x4, y4, z4, v4, rho2, ok = guiding_centre_rhs(
            c, grid, shape, x + dt * x3, y + dt * y3, z + dt * z3, v + dt * v3, m, qm
        )
        if ok == 0:
            status[k] = 2
            break
        # the step is taken: score its start
        for s in range(n_shells):
            if rho >= rho_edges[s] and rho < rho_edges[s + 1]:
                shell_time[s, k] += dt
                shell_vpar[s, k] += v * dt
        previous[0, k] = x
        previous[1, k] = y
        previous[2, k] = z
        x = x + dt / 6.0 * (x1 + 2.0 * x2 + 2.0 * x3 + x4)
        y = y + dt / 6.0 * (y1 + 2.0 * y2 + 2.0 * y3 + y4)
        z = z + dt / 6.0 * (z1 + 2.0 * z2 + 2.0 * z3 + z4)
        v = v + dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        t += dt
        rho2, ok = scalar_at(c, 4, 0, grid, shape, x, y, z)
        if ok == 0 or rho2 >= rho2_stop:
            status[k] = 1
            break
        if wall.shape[0] > 1:
            clearance, ok = scalar_at(wall, 1, 0, grid, shape, x, y, z)
            if clearance < clearance_stop:
                status[k] = 1
                break
    state[0, k] = x
    state[1, k] = y
    state[2, k] = z
    state[3, k] = v
    elapsed[k] = t


def guiding_centre_push(
    state: "float[:,:]",
    mu: "float[:]",
    duration: "float[:]",
    max_step: "float[:]",
    c: "float[:]",
    grid: "float[:]",
    shape: "int[:]",
    qm: float,
    rho2_stop: float,
    wall: "float[:]",
    clearance_stop: float,
    rho_edges: "float[:]",
    previous: "float[:,:]",
    elapsed: "float[:]",
    status: "int[:]",
    shell_time: "float[:,:]",
    shell_vpar: "float[:,:]",
) -> None:
    """Push guiding centres by RK4 for their ``duration``, in equal steps up to ``max_step``.

    ``state`` ``(4, n)``: ``x, y, z`` [m] and ``v_par`` [m/s], updated in place; ``mu``
    ``(n,)`` the magnetic moment per mass [m^2 s^-2 T^-1]; ``c`` the padded B-spline
    coefficients of ``rho^2, B_R, B_phi, B_Z``, flattened from ``(n_R', n_phi', n_Z', 4)``; ``grid`` ``(R_0, Z_0, dR,
    dphi, dZ, period, pad)``; ``shape`` the nodes ``(n_R, n_phi, n_Z)``
    and the padded ``(n_R', n_phi', n_Z')``; ``qm`` = q/m.
    A marker stops early after a step that ends at ``rho^2 >= rho2_stop``, closer to
    the wall than ``clearance_stop`` (``wall``: the flattened padded coefficients of
    the wall clearance on the same grid; one entry: no wall) or off the
    grid (``status`` 1; ``previous`` holds the start of that step) or when a stage
    leaves the grid (``status`` 2, the state of the step's start); else ``status`` 0.
    ``elapsed`` gets the time pushed; ``shell_time`` and ``shell_vpar``
    ``(n_shells, n)`` add the time and ``v_par`` times the time spent in each shell
    of ``rho_edges`` (scored at the start of each step).
    """
    n = state.shape[1]
    # $ omp parallel for schedule(dynamic, 4)
    for k in range(n):
        push_one(
            k,
            state,
            mu,
            duration,
            max_step,
            c,
            grid,
            shape,
            qm,
            rho2_stop,
            wall,
            clearance_stop,
            rho_edges,
            previous,
            elapsed,
            status,
            shell_time,
            shell_vpar,
        )
