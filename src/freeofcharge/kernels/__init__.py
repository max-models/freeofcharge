"""The compiled kernels of freeofcharge, one folder per kernel (see `freeofcharge.kernels.catalog`).

Every kernel folder builds its kernel with `KERNEL_PACKAGE`::

    kernel = KERNEL_PACKAGE.from_folder(__name__)

with the defaults of `cunumpy.kernels.KernelPackage`: the Pyccel version is compiled on
its first host call and kept under ``FREEOFCHARGE_KERNEL_CACHE`` (default
``~/.cache/freeofcharge/kernels``; ``FREEOFCHARGE_OPENMP=0`` builds serial), the CUDA
version runs for CuPy arguments and the host version for NumPy ones, one CUDA thread per
entry of the last axis of the first array (the points, stored component-major as
``(3, n)``). Without Pyccel the NumPy versions run. Importing a kernel needs neither
Pyccel nor CuPy.
"""

import cunumpy as xp

#: The build cache and launch options of freeofcharge's kernels.
KERNEL_PACKAGE = xp.kernels.KernelPackage("freeofcharge")
