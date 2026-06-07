# FlyDSL XeGPU Backend — Build & Verify Guide

This document captures the exact commands to reproduce the FlyDSL `xegpu`
backend build through Phase 2e on a Linux host with an Intel GPU.

State after Phase 2e:

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
- **Phase 2e (end-to-end)**: vectorAdd actually launches on Intel PVC and
  produces correct results. The xegpu backend now uses upstream MLIR's
  **SYCL** runtime wrapper (`libmlir_sycl_runtime.so`), not the Level Zero
  one — see "Why SYCL, not Level Zero" below.

Tested on this host (Intel PVC / Data Center GPU Max 1100, no ROCm).

## Why SYCL, not Level Zero

Both upstream wrappers expose the same `mgpu*` ABI
(`mgpuLaunchKernel` / `mgpuStreamCreate` / ...), but they differ in what a
"stream" means. `libmlir_levelzero_runtime.so` allocates and dereferences
its own `StreamWrapper *` (`LevelZeroRuntimeWrappers.cpp:571`), so passing
in a NULL stream — or any pointer not produced by `mgpuStreamCreate` —
segfaults. We need to feed the runtime a real stream that already exists,
namely the one torch is using, and torch.xpu exposes it as a SYCL queue
(`torch.xpu.Stream.sycl_queue` is the address of a `sycl::queue`).
`libmlir_sycl_runtime.so` (`SyclRuntimeWrappers.cpp`) takes a
`sycl::queue *` directly, so we can hand torch's queue straight through.

## One-time prerequisites

```bash
# Python deps for the LLVM build (nanobind etc.)
pip install nanobind numpy pybind11

# patchelf — copies runtime libs into the wheel rpath; CMake errors out without it
pip install patchelf

# torch with Intel-XPU support (provides torch.xpu.Stream and SYCL queue access).
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Host packages (already present on this Jupyter image; install if missing):

- Intel oneAPI compiler (`icpx`, `libsycl.so`) at `/opt/intel/oneapi/`.
  Required to build `libmlir_sycl_runtime.so`. Install via the standalone
  Intel DPC++ Compiler component.
- `level-zero-dev` (`ze_api.h`) and `libze_loader.so.1` — Level Zero is
  still a transitive dep of `libmlir_sycl_runtime.so` (for module loading).

```bash
ls /opt/intel/oneapi/compiler/2025.3/bin/icpx          # SYCL compiler
ls /opt/intel/oneapi/compiler/2025.3/lib/libsycl.so    # SYCL runtime
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
  `libmlir_levelzero_runtime.so`. Not on the runtime path anymore (we use
  SYCL instead), but Level Zero CMake config is needed so MLIRConfig
  exposes the LevelZero target our SYCL runtime wrapper transitively
  depends on.

## Build the SYCL runtime wrapper (~10 sec, one-time)

`libmlir_sycl_runtime.so` requires Intel's `icpx` SYCL compiler. The rest
of LLVM happily builds with gcc, so we don't switch the global compiler;
instead we compile this single .cpp standalone with `icpx -fsycl`:

```bash
cd /home/jovyan/workspace3/FlyDSL
bash scripts/build_sycl_runtime.sh
```

This drops `libmlir_sycl_runtime.so` directly into
`/home/jovyan/workspace3/llvm-project/mlir_install/lib/`, where the
FlyDSL CMake build will pick it up.

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
- `build-fly/python_packages/flydsl/_mlir/_mlir_libs/libmlir_sycl_runtime.so`
  — staged here by the build (rpath includes
  `/opt/intel/oneapi/compiler/.../lib`) so the JIT executor can dlopen it
  via `_resolve_runtime_libs` (`jit_executor.py:67`).

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
  `['libmlir_sycl_runtime.so', 'libmlir_c_runner_utils.so']`

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

## Verify Phase 2e — vectorAdd runs end-to-end on Intel PVC

```bash
cd /home/jovyan/workspace3/FlyDSL

PYTHONPATH=$PWD/build-fly/python_packages:$PWD \
FLYDSL_COMPILE_BACKEND=xegpu \
FLYDSL_RUNTIME_KIND=xegpu \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python3 examples/01-vectorAdd-xegpu.py
```

Expected output:

```
[Eager] Result correct: True
```

The example allocates `A`, `B`, `C` on Intel XPU via
`torch.tensor(...).xpu()`, passes the current torch XPU stream through
`fx.Stream(torch.xpu.current_stream())`, launches the kernel via the
SYCL runtime, and confirms `torch.allclose(C, A + B)`.

### How the launch path works

1. Python packs the kernel args into a packed-pointer array. The stream
   slot reads `torch.xpu.Stream.sycl_queue` (the SYCL queue address) via
   `Stream._extract_stream_value` (`expr/typing.py:1126`).
2. The compiled host wrapper passes this queue through to
   `gpu.launch_func`'s `<%stream : !llvm.ptr>` operand.
3. At MLIR→LLVM-IR translation time, upstream's
   `SelectObjectAttr::launchKernel`
   (`mlir/lib/Target/LLVMIR/Dialect/GPU/SelectObjectAttr.cpp:256`) emits
   a call to `mgpuLaunchKernel(kernel, gx, gy, gz, bx, by, bz, smem,
   stream, args, nullptr, nargs)` with our SYCL queue as the `stream`
   arg.
4. The SYCL runtime
   (`mlir/lib/ExecutionEngine/SyclRuntimeWrappers.cpp:194`) does
   `queue->submit([&](sycl::handler &cgh) { ...; cgh.parallel_for(...); })`
   on torch's queue — so the kernel actually runs on the same SYCL queue
   torch is using, and `torch.xpu.synchronize()` waits for it correctly.

### Phase 2e debug history (kept for archaeology)

The original Phase-2e segfault was tracked through three wrong
hypotheses before the SYCL switch worked:

- **Wrong hypothesis 1**: `gpu.launch_func` not lowered by `gpu-to-llvm`.
  `gpu.launch_func` surviving the pass pipeline is expected — both the
  upstream xevm integration tests
  (`mlir/test/Integration/Dialect/XeGPU/WG/simple_gemm.mlir`, verified
  on this host with `bin/mlir-runner`) and our pipeline leave it in
  place. It gets translated to `mgpuLaunchKernel` at MLIR→LLVM-IR
  translation time via `SelectObjectAttr::launchKernel`, which is
  registered in `python/mlir_flydsl/FlyRegisterEverything.cpp`.

- **Wrong hypothesis 2**: argument packing / kernel-side ABI bug. To
  isolate, `tests/mlir/xegpu/vectorAdd-reproducer.mlir` reproduces the
  exact `!llvm.ptr<1>` kernel ABI FlyDSL emits, with a hand-authored
  host wrapper that uses `gpu.alloc` / `gpu.memcpy`. It runs cleanly
  under upstream `mlir-runner` and prints `C = A+B = 3` across all 128
  elements on PVC — proving the kernel IR + lowering pipeline + Level
  Zero runtime are all fine. So the bug had to be in FlyDSL's Python
  driver path.

- **Actual root cause**: `Stream(None)` →
  `Stream._extract_stream_value` returned `0` (NULL) →
  `mgpuLaunchKernel(... stream=NULL)` → `LevelZeroRuntimeWrappers.cpp:571`
  did `nullptr->enqueueOp(...)`. The Level Zero runtime requires a
  stream produced by `mgpuStreamCreate`, but FlyDSL's launch wrapper
  expects the caller (Python) to provide the stream. Switching to the
  SYCL runtime fixed this because torch.xpu's `sycl_queue` IS a valid
  `sycl::queue *`.
