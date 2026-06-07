// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Stub xegpu lowering pass — Phase 1 skeleton.
// For Phase 2 (vectorAdd) we plan to delegate the bulk of GPU lowering to
// upstream MLIR's `gpu-lower-to-xevm-pipeline`; this pass exists today so
// `XegpuBackend.pipeline_fragments` can name a stable per-backend entry
// point and so `fly-opt --help` lists the pass.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Conversion/FlyToXeGPU/FlyToXeGPU.h"
#include "flydsl/Dialect/FlyXeGPU/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOXEGPUCONVERSIONPASS
#include "flydsl/Conversion/FlyToXeGPU/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly_xegpu;

namespace {

class FlyToXeGPUConversionPass
    : public mlir::impl::FlyToXeGPUConversionPassBase<FlyToXeGPUConversionPass> {
public:
  using mlir::impl::FlyToXeGPUConversionPassBase<
      FlyToXeGPUConversionPass>::FlyToXeGPUConversionPassBase;

  void runOnOperation() override {
    // Phase 1: no-op. Phase 2 will populate this with patterns analogous to
    // FlyToROCDL (MakePtrOpLowering / PtrLoad/StoreOpLowering /
    // GpuLaunchFuncOpLowering / ...) plus xegpu-specific copy/MMA atoms.
  }
};

} // namespace
