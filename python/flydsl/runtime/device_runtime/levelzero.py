# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Level Zero device runtime (FlyDSL xegpu backend)."""

from __future__ import annotations

import ctypes
import os
from typing import ClassVar

from .base import DeviceRuntime

_ZE_LIB = None
_ZE_LIB_TRIED = False


def _load_ze_loader():
    global _ZE_LIB, _ZE_LIB_TRIED
    if _ZE_LIB_TRIED:
        return _ZE_LIB
    _ZE_LIB_TRIED = True
    for soname in ("libze_loader.so.1", "libze_loader.so"):
        try:
            _ZE_LIB = ctypes.CDLL(soname)
            return _ZE_LIB
        except OSError:
            continue
    return None


def _ze_device_count() -> int:
    """Best-effort GPU device count via Level Zero zeInit/zeDeviceGet."""
    lib = _load_ze_loader()
    if lib is None:
        return 0
    try:
        lib.zeInit.argtypes = [ctypes.c_uint32]
        lib.zeInit.restype = ctypes.c_uint32
        if lib.zeInit(0) != 0:
            return 0

        # zeDriverGet
        n_drivers = ctypes.c_uint32(0)
        if lib.zeDriverGet(ctypes.byref(n_drivers), None) != 0 or n_drivers.value == 0:
            return 0
        drivers = (ctypes.c_void_p * n_drivers.value)()
        if lib.zeDriverGet(ctypes.byref(n_drivers), drivers) != 0:
            return 0

        total = 0
        for i in range(n_drivers.value):
            n_devs = ctypes.c_uint32(0)
            if lib.zeDeviceGet(drivers[i], ctypes.byref(n_devs), None) != 0:
                continue
            total += n_devs.value
        return total
    except Exception:
        return 0


class LevelZeroDeviceRuntime(DeviceRuntime):
    """Level Zero-based runtime for the xegpu compile backend.

    Phase 1 stub: enough surface (kind, device_count, current_device_id) for
    `get_device_runtime()` to instantiate. Kernel launch / module load go
    through `libmlir_levelzero_runtime.so` via the JIT executor.
    """

    kind: ClassVar[str] = "xegpu"

    def device_count(self) -> int:
        return _ze_device_count()

    def current_device_id(self) -> int:
        # Honour Level Zero's standard selector; default to device 0.
        raw = os.environ.get("ZE_AFFINITY_MASK") or os.environ.get("ONEAPI_DEVICE_SELECTOR") or ""
        first = raw.split(",", 1)[0].split(":", 1)[-1].strip()
        try:
            return int(first) if first else 0
        except ValueError:
            return 0
