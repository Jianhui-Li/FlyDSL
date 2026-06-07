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

Diagnosed root cause: **`gpu.launch_func` is not being lowered to mgpu
runtime calls by the `gpu-to-llvm` pass.** Inspecting the IR after
`gpu-to-llvm`:

```mlir
gpu.launch_func <%arg7 : !llvm.ptr> @kernels::@vectorAddKernel_0
                blocks in (%22, %1, %1) threads in (%0, %1, %1) : i64
                args(%arg0 : !llvm.ptr<1>, %arg1 : !llvm.struct<...>, ...)
llvm.return
```

The launch op survives `gpu-to-llvm` — the upstream `LegalizeLaunchFuncOp`
pattern (`mlir/lib/Conversion/GPUCommon/GPUToLLVMConversion.cpp:948`)
treats a `gpu.launch_func` as legal once all operand types are LLVM-
compatible, which is true after the fly-side `GpuLaunchFuncOpLowering`
runs. The op then never reaches the rewrite path that would emit
`mgpuStreamCreate` / `mgpuLaunchKernel` / `mgpuStreamSynchronize`.

Phase 2e is paused while we figure out the right combination of
`gpu-async-region` ordering, `convert-async-to-llvm`, and
`addDynamicallyLegalOp<gpu::LaunchFuncOp>` legality so that the existing
upstream pattern actually fires. The rocdl pipeline appears to handle
this differently (or its `from_dlpack`-fed launches end up synchronous
without an async-token result, sidestepping the issue) — unclear without
a side-by-side comparison run.

Sub-issues worth keeping in mind once the launch lowering works:

1. **Device pointers**: `examples/01-vectorAdd-xegpu.py` now uses
   `torch.tensor(...).xpu()` (requires
   `pip install torch --index-url https://download.pytorch.org/whl/xpu`)
   so DLPack hands the kernel real Level-Zero device pointers via
   `flyc.from_dlpack`. Verified: `torch.xpu.is_available()` reports
   `Intel(R) Data Center GPU Max 1100`.

2. **Block dim**: `01-vectorAdd-xegpu.py` sets `block_dim = 64` (inherited
   from the rocdl-shape vectorAdd, which is wave64). On Intel SIMD16 we
   probably want 16 threads/block; lane-mode codegen doesn't redistribute
   so the runtime dims may need to drop to 16.

3. **Bare-ptr conv vs torch DLPack**: with `use-bare-pointers-for-host`
   the host ABI expects `(ptr, …)` per memref instead of
   `(allocated_ptr, aligned_ptr, offset, sizes…, strides…)`. Confirm
   `from_dlpack(...).mark_layout_dynamic(...)` produces the right packing
   for this convention on the xegpu path.
