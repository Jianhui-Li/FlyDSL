#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Build libmlir_sycl_runtime.so out of upstream MLIR's SyclRuntimeWrappers.cpp.
#
# This is split out of scripts/build_llvm.sh because the file requires the
# Intel icpx (DPC++) SYCL compiler -- the rest of LLVM happily builds with
# gcc, so we don't want to globally switch CMAKE_CXX_COMPILER. Compiling
# this single file with icpx and dropping the resulting shared object into
# mlir_install/lib/ is the smallest viable integration.
#
# Why we need it: FlyDSL's xegpu backend launches kernels through the
# upstream `mgpu*` ABI (mgpuLaunchKernel / mgpuStreamCreate / ...). The
# Level Zero variant of that ABI in libmlir_levelzero_runtime.so requires
# a `StreamWrapper *` it allocates itself; there's no way to feed in a
# torch.xpu.Stream's underlying queue. The SYCL variant uses a
# `sycl::queue *` directly, which torch.xpu.Stream exposes via .sycl_queue
# (an int that is the queue address) -- so we can pass torch's stream
# straight into the runtime.
#
# Prerequisites:
#   - Intel oneAPI installed at /opt/intel/oneapi (provides icpx + libsycl).
#   - LLVM/MLIR install built by scripts/build_llvm.sh (we read its headers).
#   - libze_loader (Level Zero loader) on the system; SyclRuntimeWrappers.cpp
#     uses ze_module_handle_t for module loading.
#
# Usage:
#   bash scripts/build_sycl_runtime.sh
#
# Output:
#   ${LLVM_INSTALL_DIR}/lib/libmlir_sycl_runtime.so

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"

LLVM_SRC_DIR="${LLVM_SRC_DIR:-${BASE_DIR}/llvm-project}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${LLVM_SRC_DIR}/mlir_install}"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
CMPLR_ROOT="${CMPLR_ROOT:-${ONEAPI_ROOT}/compiler/2025.3}"

WRAPPER_SRC="${LLVM_SRC_DIR}/mlir/lib/ExecutionEngine/SyclRuntimeWrappers.cpp"
ICPX="${CMPLR_ROOT}/bin/icpx"
OUT="${LLVM_INSTALL_DIR}/lib/libmlir_sycl_runtime.so"

echo "LLVM_SRC_DIR:     ${LLVM_SRC_DIR}"
echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"
echo "CMPLR_ROOT:       ${CMPLR_ROOT}"
echo "Wrapper source:   ${WRAPPER_SRC}"
echo "Output:           ${OUT}"

[[ -f "${WRAPPER_SRC}" ]] || { echo "Error: ${WRAPPER_SRC} not found." >&2; exit 1; }
[[ -x "${ICPX}" ]]        || { echo "Error: ${ICPX} not found. Install Intel oneAPI." >&2; exit 1; }
[[ -d "${LLVM_INSTALL_DIR}/include" ]] || { echo "Error: ${LLVM_INSTALL_DIR}/include missing. Run scripts/build_llvm.sh first." >&2; exit 1; }

mkdir -p "$(dirname "${OUT}")"

"${ICPX}" \
  -fsycl \
  -shared -fPIC -fexceptions -frtti -O2 \
  -I "${LLVM_INSTALL_DIR}/include" \
  -I "${LLVM_INSTALL_DIR}/include/llvm" \
  "${WRAPPER_SRC}" \
  -lze_loader \
  -Wl,-soname,libmlir_sycl_runtime.so \
  -o "${OUT}"

echo ""
echo "Built ${OUT}"
nm -D "${OUT}" | grep -E "^[0-9a-f]+ T mgpu" | sed 's/^/  /' | head -10
