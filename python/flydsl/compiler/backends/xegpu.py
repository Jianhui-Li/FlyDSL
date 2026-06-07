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
        # The pipeline shape mirrors rocm.py rather than upstream's aggregate
        # gpu-lower-to-xevm-pipeline, because:
        #   1. We need bare-ptr calling convention on `gpu-to-llvm` to marshal
        #      struct-typed kernel args (e.g. lowered fly.layout) through
        #      mgpuLaunchKernel.
        #   2. FlyDSL kernels are authored at the lane level, so we must NOT
        #      run XeGPUWgToSgDistribute / XeGPUSgToLaneDistribute (those
        #      wrap the kernel body in gpu.warp_execute_on_lane_0).
        # We invoke the per-stage upstream passes individually for full
        # control. xevm-attach-target runs at module scope to attach
        # #xevm.target<...>; the gpu.module() nest does the device-side
        # SPV/XeVM lowering; gpu-to-llvm then handles the host launch.
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
            f'xevm-attach-target{{chip={chip} O=2}}',
            (
                "gpu.module("
                "convert-scf-to-cf,cse,"
                "convert-math-to-xevm,"
                "convert-xegpu-to-xevm,"
                "convert-gpu-to-llvm-spv{use-64bit-index=true},"
                "cse,"
                "reconcile-unrealized-casts"
                ")"
            ),
        ]
        binary_prep_fragments = [
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-vector-to-llvm",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            "gpu.module(convert-xevm-to-llvm)",
        ]
        binary_fragment = f'gpu-module-to-binary{{format=fatbin}}'
        return [*pre_binary_fragments, *binary_prep_fragments, binary_fragment]

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
            "libmlir_sycl_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        # First entry is the per-vendor GPU runtime; jit_executor.py:92-94
        # explicitly dlopen's basenames[0] and looks up `mgpuModuleUnload`
        # for cleanup.
        #
        # We use libmlir_sycl_runtime.so (NOT libmlir_levelzero_runtime.so)
        # because the SYCL wrapper takes a `sycl::queue *` directly as the
        # stream argument to mgpuLaunchKernel. torch.xpu.Stream exposes the
        # underlying SYCL queue address via .sycl_queue, so we can hand the
        # caller's torch stream straight to the runtime. The Level Zero
        # variant uses its own StreamWrapper * (only obtainable via
        # mgpuStreamCreate) and would crash on a torch-supplied integer.
        return [
            "libmlir_sycl_runtime.so",
            "libmlir_c_runner_utils.so",
        ]
