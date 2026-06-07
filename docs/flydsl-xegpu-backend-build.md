# FlyDSL XeGPU Backend — Build & Verify Guide

This document captures the exact commands to reproduce the FlyDSL `xegpu`
backend build through Phase 2d on a Linux host with an Intel GPU and Level
Zero.

State after Phase 2d:

- **Phase 1 (skeleton)**: `FLYDSL_BACKENDS=xegpu` builds end-to-end without
  ROCm; `fly-opt` registers the `fly_xegpu` dialect and the
  `convert-fly-to-xegpu` pass; `XegpuBackend` reports SIMD16 and a pipeline
  that delegates GPU-side lowering to upstream MLIR's
  `gpu-lower-to-xevm-pipeline`.
- **Phase 2c (real conversion)**: `convert-fly-to-xegpu` ports the universal
  subset of `convert-fly-to-rocdl` (drops AMD buffer / MFMA atoms; remaps
  address spaces for SPIR-V). The Fly→XeGPU/XeVM/SPIR-V pipeline is now
  per-pass instead of the upstream aggregate so we can pass bare-pointer
  conv on `gpu-to-llvm`.
- **Phase 2d (compile-only)**: `examples/01-vectorAdd-xegpu.py` (a
  backend-neutral copy of `01-vectorAdd.py`) compiles end-to-end to a
  SPIR-V/XeVM fatbin under `COMPILE_ONLY=1`.

Phase 2e (actually launching the kernel on Intel GPU via Level Zero) is in
progress; see `## Known issues` at the bottom.

Tested on this host (Intel-only, no ROCm).

## One-time prerequisites

```bash
# Python deps for the LLVM build (nanobind etc.)
pip install nanobind numpy pybind11

# patchelf — copies runtime libs into the wheel rpath; CMake errors out without it
pip install patchelf

# torch-cpu (host-side; Phase 2e will need an Intel-XPU torch build)
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Host packages (already present on this Jupyter image; install if missing):

- `level-zero-dev` (ze_api.h)
- `libze1` / `libze-loader` (libze_loader.so.1)

```bash
ls /usr/include/level_zero/ze_api.h /lib/x86_64-linux-gnu/libze_loader.so.1
```

## Build LLVM/MLIR (~30 min, run once)

```bash
cd /home/jovyan/workspace3/FlyDSL

# Clones llvm-project at the pinned hash from thirdparty/llvm-hash.txt
# into ../llvm-project/, configures into ../llvm-project/build-flydsl/,
# installs to ../llvm-project/mlir_install/.
bash scripts/build_llvm.sh -j64
```

`scripts/build_llvm.sh` is configured for the xegpu path:

- `LLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU;SPIRV"` — SPIRV is required;
  without it `gpu-module-to-binary{format=fatbin}` fails late in the
  pipeline with `Cannot run TargetRegistry::lookupTarget() for SPIRV`.
- `MLIR_ENABLE_LEVELZERO_RUNNER=1` — produces
  `libmlir_levelzero_runtime.so` exposing the upstream `mgpu*` ABI
  (`mgpuModuleLoad/Launch/Unload`, `mgpuMem*`, `mgpuStream*`).

`scripts/build.sh` auto-detects the install via its `MLIR_PATH` candidate
list — no env var needed afterwards.

## Build FlyDSL with the xegpu backend (~5 min)

```bash
cd /home/jovyan/workspace3/FlyDSL

# FLYDSL_BACKENDS selects which backend descriptors load.
# "xegpu" only — no rocdl, so no HIP requirement.
FLYDSL_BACKENDS=xegpu bash scripts/build.sh -j64
```

Build outputs land in `build-fly/`. The interesting ones:

- `build-fly/bin/fly-opt` — the optimizer driver
- `build-fly/python_packages/flydsl/` — the importable package
- `build-fly/python_packages/flydsl/_mlir/_mlir_libs/libmlir_levelzero_runtime.so`
  — staged here by the build so the JIT executor can dlopen it via
  `_resolve_runtime_libs` (`jit_executor.py:67`).

## Verify Phase 1 success criteria

```bash
cd /home/jovyan/workspace3/FlyDSL
export PYTHONPATH="$PWD/build-fly/python_packages:$PWD:$PYTHONPATH"

# 1. fly-opt registers fly_xegpu dialect and convert-fly-to-xegpu pass
./build-fly/bin/fly-opt --help 2>&1 | grep -E "fly_xegpu|convert-fly-to-xegpu"

# 2. XegpuBackend works and reports SIMD16
FLYDSL_COMPILE_BACKEND=xegpu FLYDSL_RUNTIME_KIND=xegpu python3 -c "
from flydsl.compiler.backends.xegpu import XegpuBackend
import flydsl.compiler.backends as b
print('registered:', sorted(b._registry.keys()))
target = XegpuBackend.detect_target()
print('target:', target)
backend = XegpuBackend(target)
print('warp_size:', backend.target.warp_size)
print('runtime libs:', backend.jit_runtime_lib_basenames())
print('pipeline (last 2):')
for s in backend.pipeline_fragments(compile_hints={})[-2:]:
    print('  ', s)
from flydsl.runtime.device import is_intel_arch, get_intel_arch
print('intel arch:', get_intel_arch(), 'is_intel:', is_intel_arch())
print('OK')
"
```

Expected output:

- `Available Dialects: ... fly, fly_xegpu, ... xegpu, xevm`
- `--convert-fly-to-xegpu  Lower Fly to MLIR upstream and xegpu/xevm dialects`
- `target: GPUTarget(backend='xegpu', arch=<your-arch>, warp_size=16)`
  where `<your-arch>` is auto-detected from
  `/sys/class/drm/card*/device/device`: `pvc` for PCI IDs `0x0BD0..0x0BDF`
  (Ponte Vecchio / Data Center GPU Max), `bmg` for `0xE200..0xE2FF`
  (Battlemage / Arc B). Override with `FLYDSL_GPU_ARCH=pvc` (or `bmg`).
- last pipeline fragment is
  `gpu-module-to-binary{format=fatbin}`
- runtime libs:
  `['libmlir_levelzero_runtime.so', 'libmlir_c_runner_utils.so']`

## Verify Phase 2d — vectorAdd compiles end-to-end

```bash
cd /home/jovyan/workspace3/FlyDSL

PYTHONPATH=$PWD/build-fly/python_packages:$PWD \
FLYDSL_COMPILE_BACKEND=xegpu \
FLYDSL_RUNTIME_KIND=xegpu \
COMPILE_ONLY=1 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 examples/01-vectorAdd-xegpu.py
```

Expected output:

```
> vectorAdd: n=128, grid_x=2
[flydsl] COMPILE_ONLY=1, compilation succeeded (arch=pvc)
[Eager] Result correct: False   ← expected: COMPILE_ONLY skips the kernel launch
```

`arch=pvc` here reflects the auto-detected device — your output will show
whatever sub-arch the sysfs probe matched.

`Result correct: False` is **expected** under `COMPILE_ONLY=1` — kernel
execution is skipped, so `C` stays zeroed and `torch.allclose(C, A+B)`
fails. The compile bar is what matters at this phase.

The full pipeline that runs:

| Stage | Pass(es) |
|---|---|
| Fly canonicalization | `fly-rewrite-func-signature` → `fly-canonicalize` → `fly-layout-lowering` → `fly-int-swizzle-simplify` → `canonicalize` |
| Atom-call SSA + reg promotion | `fly-convert-atom-call-to-ssa-form` → `fly-promote-regmem-to-vectorssa` |
| Fly → XeGPU stubs (universal) | `convert-fly-to-xegpu` → `canonicalize` |
| Target attach | `xevm-attach-target{chip=bmg O=2}` |
| Device lowering (inside `gpu.module`) | `convert-scf-to-cf` → `cse` → `convert-math-to-xevm` → `convert-xegpu-to-xevm` → `convert-gpu-to-llvm-spv{use-64bit-index=true}` → `cse` → `reconcile-unrealized-casts` |
| Host lowering | `convert-scf-to-cf` → `convert-cf-to-llvm` → `gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}` → `convert-vector-to-llvm` → `convert-arith-to-llvm` → `convert-func-to-llvm` → `reconcile-unrealized-casts` |
| Final XeVM emit | `gpu.module(convert-xevm-to-llvm)` → `gpu-module-to-binary{format=fatbin}` |

To inspect the IR after every pass (useful when debugging the pipeline):

```bash
PYTHONPATH=$PWD/build-fly/python_packages:$PWD \
FLYDSL_COMPILE_BACKEND=xegpu FLYDSL_RUNTIME_KIND=xegpu \
COMPILE_ONLY=1 \
FLYDSL_DEBUG_PRINT_AFTER_ALL=1 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 examples/01-vectorAdd-xegpu.py >/tmp/xegpu-pipeline.log 2>&1

grep -nE "IR Dump After|error:" /tmp/xegpu-pipeline.log
```

## Iterating

For incremental rebuilds (after editing C++):

```bash
cd /home/jovyan/workspace3/FlyDSL/build-fly
ninja -j64
```

After editing only Python sources (the build copies them into
`build-fly/python_packages/flydsl/`), refresh the copy without a full
rebuild:

```bash
cd /home/jovyan/workspace3/FlyDSL/build-fly
ninja CopyFlyPythonSources
```

For a clean rebuild (after editing CMake or after rebuilding LLVM):

```bash
rm -rf /home/jovyan/workspace3/FlyDSL/build-fly
FLYDSL_BACKENDS=xegpu bash scripts/build.sh -j64
```

## Notes / gotchas hit during bring-up

- **Don't skip `pip install patchelf`** — the build silently completes most
  targets, then dies at the `CopyFlyPythonSources` step setting rpath on
  copied runtime libs.
- **`FLYDSL_BACKENDS` must be exported on the build line** —
  `scripts/build.sh` reads it as an env var and forwards to
  `cmake -DFLYDSL_BACKENDS=...`.
- **Don't reuse a stale `build-fly/`** if you flip `FLYDSL_BACKENDS` — wipe
  it first; CMake's cache-variable semantics around `STRINGS` allowed values
  can cause confusing reconfigure errors.
- **Don't add `useDefaultTypePrinterParser` to a dialect with zero types** —
  the vtable references unresolved tablegen-generated symbols. Re-enable
  when the first xegpu atom type lands.
- **`FLYDSL_RUNTIME_KIND=xegpu` must match `FLYDSL_COMPILE_BACKEND=xegpu`**
  at runtime — there's a hard pairing check in
  `device_runtime/__init__.py:ensure_compile_runtime_pairing_from_env`.
- **After editing `python/flydsl/...`**, re-run `ninja CopyFlyPythonSources`
  in `build-fly/` — the JIT loads from `build-fly/python_packages/`, not
  the source tree, so a stale copy will silently use the old pipeline.
- **Pin `xegpu-op-level=lane`, not `workgroup`** — FlyDSL kernels are
  authored at the lane level. Workgroup mode runs `XeGPUWgToSgDistribute`
  + `XeGPUSgToLaneDistribute`, which wraps the body in
  `gpu.warp_execute_on_lane_0` to drive distribution we've already done.
  Lane mode skips both passes.
- **Use the per-pass shape, not `gpu-lower-to-xevm-pipeline`** — the
  upstream aggregate doesn't expose `host-bare-ptr-calling-convention=true`
  / `kernel-bare-ptr-calling-convention=true` on its embedded `gpu-to-llvm`
  step. Without bare-ptr conv, struct-typed kernel args (lowered
  `fly.layout` becomes `!llvm.struct<packed (struct<packed (i32)>)>`) fail
  to legalize through `mgpuLaunchKernel`.
- **SPIRV must be in `LLVM_TARGETS_TO_BUILD`** — without it the pipeline
  succeeds through dialect conversion and dies at code-gen with
  `Cannot run TargetRegistry::lookupTarget() for SPIRV without having the
  target built.`
- **`MLIR_ENABLE_LEVELZERO_RUNNER=1` is non-default** in upstream MLIR;
  `scripts/build_llvm.sh` sets it explicitly.

## Known issues (Phase 2e — not yet working)

Dropping `COMPILE_ONLY=1` segfaults inside the JIT'd code at the
`mgpuLaunchKernel` call:

```
Fatal Python error: Segmentation fault
Current thread 0x... (most recent call first):
  File ".../jit_function.py", line 1215 in __call__   ← self._func_exe(self._tls.packed)
```

**`gpu.launch_func` surviving in the final IR is expected, not a bug.**
Both the upstream xevm integration tests (e.g.
`mlir/test/Integration/Dialect/XeGPU/WG/simple_gemm.mlir`, verified on
this host with `bin/mlir-runner`) and our FlyDSL pipeline leave
`gpu.launch_func <%stream : !llvm.ptr>` in place at the end of the pass
pipeline. The op gets translated to `mgpuLaunchKernel` calls **at
MLIR→LLVM-IR translation time**, by either:

- upstream `SelectObjectAttr::launchKernel`
  (`mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp:256`), the
  default offloading translator registered via
  `mlirRegisterAllLLVMTranslations` — already wired in
  `python/mlir_flydsl/FlyRegisterEverything.cpp:32-33`; or
- FlyDSL's `LaunchKernel::createKernelLaunch` for `#fly.explicit_module`
  (`lib/Dialect/Fly/IR/FlyLLVMTranslation.cpp:206`), used only when the
  kernel has `link_extern` — not the case for vectorAdd.

So the launch IS being translated correctly inside `ExecutionEngine`. The
segfault happens later — at runtime, inside the emitted
`mgpuStreamCreate` / `mgpuLaunchKernel` / `mgpuStreamSynchronize` chain
or inside the kernel itself.

**Confirmed working on this host** (so the Level Zero stack is fine):

```
cd /home/jovyan/workspace2/llvm-project/build
./bin/mlir-opt mlir/test/Integration/Dialect/XeGPU/WG/simple_gemm.mlir \
  --gpu-lower-to-xevm-pipeline="xegpu-op-level=workgroup" \
| ./bin/mlir-runner \
    --shared-libs=lib/libmlir_levelzero_runtime.so \
    --shared-libs=lib/libmlir_runner_utils.so \
    --entry-point-result=void
# Produces a real 256x256 GEMM result on Intel(R) Data Center GPU Max 1100
```

Suspects in order of likelihood:

1. **Argument packing mismatch** between FlyDSL's `from_dlpack` /
   `_ArgPacker` and what upstream `SelectObjectAttr::launchKernel`'s
   `createKernelArgArray`
   (`mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp:357`)
   expects. Our kernel signature is
   `(ptr<1>, struct<packed (struct<packed (i32)>)>, ptr<1>, …)`; the
   struct is an i32-tuple lowered from `fly.layout`. Upstream tests use
   bare `memref` operands that expand under bare-ptr conv to a single
   pointer; our nested-struct kernel arg may need a different packing
   for the SPIR-V kernel ABI.

2. **Block dim 64 on PVC SIMD16**. `01-vectorAdd-xegpu.py` inherited
   `block_dim = 64` from the rocdl wave64 vectorAdd. Intel PVC subgroup
   is 16; `xegpu-op-level=lane` doesn't redistribute, so a 64-thread
   block at runtime may stride past the kernel's actual SIMD16 width.

3. **Bare-ptr conv vs torch DLPack**: with `use-bare-pointers-for-host`
   the host ABI expects `(ptr, …)` per memref instead of
   `(allocated_ptr, aligned_ptr, offset, sizes…, strides…)`. Confirm
   `from_dlpack(...).mark_layout_dynamic(...)` produces the right packing
   on the xegpu path.

**Next concrete step**: run
`tests/mlir/xegpu/vectorAdd-reproducer.mlir` (a hand-authored host
wrapper that mimics what FlyDSL emits, but uses upstream `gpu.alloc` /
`gpu.memcpy` for known-good data marshalling) under upstream
`mlir-runner`. If that succeeds, the bug is in FlyDSL's argument
packing. If it segfaults the same way, the bug is in our kernel-side IR
or the lowering pipeline.

This investigation continues on the `xegpu-debug` branch of the fork.
The `xegpu-bringup` branch is parked at `b98544f3` (Phase 1 + Phase 2c/2d
+ auto-detect) so the Phase-2e dive doesn't pollute the milestone log.
