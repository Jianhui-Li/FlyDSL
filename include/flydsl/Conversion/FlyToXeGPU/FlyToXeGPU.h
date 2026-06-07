// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef CONVERSION_FLYTOXEGPU_FLYTOXEGPU_H
#define CONVERSION_FLYTOXEGPU_FLYTOXEGPU_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYTOXEGPUCONVERSIONPASS
#include "flydsl/Conversion/FlyToXeGPU/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYTOXEGPU_FLYTOXEGPU_H
