// The CUDA version (GPU) of the segment_distance kernel; segment_distance_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of points: one thread per point, the minimum over all segments.

#include "cunumpy/array_view.cuh"

extern "C" __global__
void segment_distance(
    Array2D<double> points,
    Array2D<double> starts,
    Array2D<double> edges,
    Array1D<double> inverse_length2,
    Array1D<double> distance
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= points.shape[1]) return;
    const long long m = starts.shape[1];
    const double x = points(0, k);
    const double y = points(1, k);
    const double z = points(2, k);
    double best = 1.0e300;
    for (long long j = 0; j < m; ++j) {
        const double wx = x - starts(0, j);
        const double wy = y - starts(1, j);
        const double wz = z - starts(2, j);
        double t = (wx * edges(0, j) + wy * edges(1, j) + wz * edges(2, j))
                   * inverse_length2(j);
        t = fmin(fmax(t, 0.0), 1.0);
        const double gx = wx - t * edges(0, j);
        const double gy = wy - t * edges(1, j);
        const double gz = wz - t * edges(2, j);
        best = fmin(best, gx * gx + gy * gy + gz * gz);
    }
    distance(k) = sqrt(best);
}
