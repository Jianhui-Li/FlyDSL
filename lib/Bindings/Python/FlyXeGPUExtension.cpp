// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Phase 1 stub: an empty nanobind module so the
// `_mlirDialectsFlyXeGPU` extension target builds and importable Python
// surface exists. Concrete atom types (DPAS, 2D-block, SLM copy) will be
// bound here in later phases as they are added to FlyXeGPU/IR/Atom.td.

#include "BindingUtils.h"

namespace nb = nanobind;

NB_MODULE(_mlirDialectsFlyXeGPU, m) {
  m.doc() = "MLIR Python FlyXeGPU Extension";
}
