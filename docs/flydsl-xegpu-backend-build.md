# FlyDSL XeGPU Backend — Build & Verify Guide

This document captures the exact commands to reproduce the Phase 1 build of
the FlyDSL `xegpu` backend on a Linux host with an Intel GPU and Level Zero.

The Phase 1 backend is a **stub**: `FLYDSL_BACKENDS=xegpu` builds end-to-end
without ROCm, `fly-opt` registers the `fly_xegpu` dialect and the
`convert-fly-to-xegpu` pass (no patterns yet), and `XegpuBackend` reports
SIMD16 with a pipeline that ends in upstream MLIR's
`gpu-lower-to-xevm-pipeline`. Phase 2 will populate the real conversions and
get `examples/01-vectorAdd.py` running on Intel GPUs via Level Zero.

Tested on this host (Intel-only, no ROCm).

## One-time prerequisites

```bash
# Python deps for the LLVM build (nanobind etc.)
pip install nanobind numpy pybind11

# patchelf — copies runtime libs into the wheel rpath; CMake errors out without it
pip install patchelf

# (Optional) torch-cpu for the import smoke test at the end
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Build LLVM/MLIR (~30 min, run once)

```bash
cd /home/jovyan/workspace3/FlyDSL

# Builds llvm-project at the pinned hash from thirdparty/llvm-hash.txt
# into ../llvm-project/build-flydsl/, installs to ../llvm-project/mlir_install/
bash scripts/build_llvm.sh -j64
```

This populates `/home/jovyan/workspace3/llvm-project/mlir_install/`.
`scripts/build.sh` auto-detects this path via its `MLIR_PATH` candidate list
— no env var needed afterwards.

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
- `target: GPUTarget(backend='xegpu', arch='bmg', warp_size=16)`
- pipeline ends in `gpu-lower-to-xevm-pipeline{xegpu-op-level=workgroup zebin-chip=bmg use-64bit-index=true binary-format=fatbin}`

## Iterating

For incremental rebuilds (after editing C++):

```bash
cd /home/jovyan/workspace3/FlyDSL/build-fly
ninja -j64
```

For a clean rebuild (after editing CMake):

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
  the vtable references unresolved tablegen-generated symbols. Phase 2 will
  re-enable when the first xegpu atom type lands.
- **`FLYDSL_RUNTIME_KIND=xegpu` must match `FLYDSL_COMPILE_BACKEND=xegpu`**
  at runtime — there's a hard pairing check in
  `device_runtime/__init__.py:ensure_compile_runtime_pairing_from_env`.
