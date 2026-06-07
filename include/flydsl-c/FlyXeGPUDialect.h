// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_C_FLYXEGPUDIALECT_H
#define FLYDSL_C_FLYXEGPUDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyXeGPU, fly_xegpu);

MLIR_CAPI_EXPORTED void mlirRegisterFlyToXeGPUConversionPass(void);

/// Backend plugin registration: insert all XeGPU dialects into \p registry.
MLIR_CAPI_EXPORTED void flydsl_register_xegpu_dialects(MlirDialectRegistry registry);
/// Backend plugin registration: register all XeGPU passes.
MLIR_CAPI_EXPORTED void flydsl_register_xegpu_passes(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYXEGPUDIALECT_H
