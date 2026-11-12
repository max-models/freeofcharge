"""freeofcharge's compiled kernels: one folder per kernel.

Every kernel ``<name>`` lives in its own folder::

    freeofcharge/kernels/<name>/__init__.py        kernel = KERNEL_PACKAGE.from_folder(__name__)
    freeofcharge/kernels/<name>/<name>_pyccel.py   def <name>(...)    CPU, compiled with Pyccel
    freeofcharge/kernels/<name>/<name>_cuda.cu     __global__ void <name>(...)    GPU (NVRTC)
    freeofcharge/kernels/<name>/<name>_numpy.py    def <name>(...)    CPU, without Pyccel

All versions take the same arguments: a call runs the CUDA version on CuPy arrays and the
compiled Pyccel version on NumPy arrays (the NumPy version if Pyccel is unavailable;
``cunumpy.kernels.set_host_kernel_implementation`` chooses).
"""

from __future__ import annotations

import importlib

import cunumpy as xp

__all__ = ["KERNELS", "KERNEL_NAMES"]

#: The kernels, by name: the Biot–Savart field of closed polylines and the distance
#: to the nearest segment (`freeofcharge.coils.CoilSet`).
KERNEL_NAMES = ("biot_savart_polylines", "segment_distance")

_PACKAGE = __name__.rpartition(".")[0]

#: Every kernel of freeofcharge, by name.
KERNELS = xp.kernels.KernelCatalog(
    {
        name: importlib.import_module(f"{_PACKAGE}.{name}").kernel
        for name in KERNEL_NAMES
    }
)
