# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Intel GPU compile backend (XeGPU / XeVM via Level Zero)."""

from typing import List

from ...runtime.device import get_intel_arch, is_intel_arch
from ...utils import env
from .base import BaseBackend, GPUTarget

# Map Intel chip -> XeVM zebin chip token for `gpu-lower-to-xevm-pipeline`.
# Add entries here as we support more sub-archs. The zebin-chip default in
# upstream is `bmg` (Battlemage); `pvc` is Ponte Vecchio (Xe-HPC).
_ZEBIN_CHIP_MAP = {
    "pvc": "pvc",
    "bmg": "bmg",
}

_DEFAULT_ARCH = "bmg"


def _zebin_chip_for(arch: str) -> str:
    return _ZEBIN_CHIP_MAP.get(arch.lower(), _DEFAULT_ARCH)


class XegpuBackend(BaseBackend):
    """Intel GPU compile backend.

    Phase 1 ships a stub pipeline that wraps upstream MLIR's
    `gpu-lower-to-xevm-pipeline`. The Fly-side `convert-fly-to-xegpu` pass
    is a no-op today; Phase 2 will populate it with patterns analogous to
    convert-fly-to-rocdl.
    """

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "xegpu"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or get_intel_arch()
        # SIMD16 is the bring-up width; Intel Xe-HPC also supports SIMD8/SIMD32
        # but we defer those until SIMD16 is end-to-end.
        return GPUTarget(backend="xegpu", arch=arch, warp_size=16)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="xegpu", arch=arch, warp_size=16)

    # -- compile pipeline ------------------------------------------------

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        chip = _zebin_chip_for(self.target.arch)
        # Phase 1: register the entry point; Phase 2 will plug in
        # gpu-lower-to-xevm-pipeline + binary-fragment options.
        pre_binary_fragments = [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "fly-int-swizzle-simplify",
            "canonicalize",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            "convert-fly-to-xegpu",
            "canonicalize",
            (
                "gpu-lower-to-xevm-pipeline{"
                f'xegpu-op-level=workgroup zebin-chip={chip} '
                'use-64bit-index=true binary-format=fatbin'
                "}"
            ),
        ]
        return pre_binary_fragments

    def gpu_module_targets(self) -> List[str]:
        chip = _zebin_chip_for(self.target.arch)
        # gpu-lower-to-xevm-pipeline attaches the target itself via
        # xevm-attach-target; the listed module attributes are still useful
        # if the host harness needs to introspect the target before lowering.
        return [f'#xevm.target<O = 2, chip = "{chip}">']

    # -- cache / fingerprint ---------------------------------------------

    def native_lib_patterns(self) -> List[str]:
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libmlir_levelzero_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        # Mirror rocm.py shape: shared MLIR runner utils + the per-vendor
        # GPU runner. libfly_jit_runtime.so is omitted while we rely on
        # mlir_levelzero_runtime's mgpuModuleLoad/Launch/Unload directly.
        return [
            "libmlir_c_runner_utils.so",
            "libmlir_levelzero_runtime.so",
        ]
