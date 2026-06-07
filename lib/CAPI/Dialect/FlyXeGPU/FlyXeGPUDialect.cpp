// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl-c/FlyXeGPUDialect.h"

#include "flydsl/Conversion/FlyToXeGPU/FlyToXeGPU.h"
#include "flydsl/Dialect/FlyXeGPU/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/FlyToXeGPU/Passes.h.inc"
} // namespace mlir

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyXeGPU, fly_xegpu, mlir::fly_xegpu::FlyXeGPUDialect)

void mlirRegisterFlyToXeGPUConversionPass(void) { mlir::registerFlyToXeGPUConversionPass(); }

void flydsl_register_xegpu_dialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::fly_xegpu::FlyXeGPUDialect>();
}

void flydsl_register_xegpu_passes(void) {
  mlirRegisterFlyToXeGPUConversionPass();
}
