"""The guiding_centre_push kernel: RK4 guiding-centre orbits on a gridded field.

Pyccel and NumPy versions with the same arguments (no CUDA version yet: on CuPy
arrays the call falls back to the host). ``kernel(...)`` runs the compiled Pyccel
version, or the NumPy version if Pyccel is unavailable (see
``cunumpy.kernels.set_host_kernel_implementation``). `freeofcharge.orbits` calls it.
"""

from freeofcharge.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__, missing_cuda="fallback")
