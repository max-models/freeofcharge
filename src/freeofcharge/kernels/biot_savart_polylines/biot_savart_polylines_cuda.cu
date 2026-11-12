// The CUDA version (GPU) of the biot_savart_polylines kernel; biot_savart_polylines_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of points: one thread per point, summing over all segments.

#include "cunumpy/array_view.cuh"

extern "C" __global__
void biot_savart_polylines(
    Array2D<double> points,
    Array2D<double> nodes,
    long long n_per_coil,
    Array1D<double> currents,
    Array2D<double> field
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= points.shape[1]) return;
    const long long n_coils = currents.shape[0];
    const double x = points(0, k);
    const double y = points(1, k);
    const double z = points(2, k);
    double bx = 0.0, by = 0.0, bz = 0.0;
    for (long long c = 0; c < n_coils; ++c) {
        const long long first = c * n_per_coil;
        const double current = currents(c);
        double ax = x - nodes(0, first);
        double ay = y - nodes(1, first);
        double az = z - nodes(2, first);
        double na = sqrt(ax * ax + ay * ay + az * az);
        for (long long i = 0; i < n_per_coil; ++i) {
            const long long j = i + 1 < n_per_coil ? first + i + 1 : first;
            const double cx = x - nodes(0, j);
            const double cy = y - nodes(1, j);
            const double cz = z - nodes(2, j);
            const double nb = sqrt(cx * cx + cy * cy + cz * cz);
            const double denom = na * nb * (na * nb + (ax * cx + ay * cy + az * cz));
            if (denom > 0.0) {
                const double factor = current * (na + nb) / denom;
                bx += factor * (ay * cz - az * cy);
                by += factor * (az * cx - ax * cz);
                bz += factor * (ax * cy - ay * cx);
            }
            ax = cx;
            ay = cy;
            az = cz;
            na = nb;
        }
    }
    field(0, k) = bx;
    field(1, k) = by;
    field(2, k) = bz;
}
