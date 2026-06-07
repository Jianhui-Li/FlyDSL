// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyToXeGPUConversionPass : Pass<"convert-fly-to-xegpu"> {
  let summary = "Lower Fly to MLIR upstream and xegpu/xevm dialects";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "xegpu::XeGPUDialect"
  ];
}
