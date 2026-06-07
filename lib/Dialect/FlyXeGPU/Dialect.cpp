// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyXeGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_xegpu;

#include "flydsl/Dialect/FlyXeGPU/IR/AtomStateEnums.cpp.inc"
#include "flydsl/Dialect/FlyXeGPU/IR/AttrEnums.cpp.inc"
#include "flydsl/Dialect/FlyXeGPU/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyXeGPU/IR/Atom.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/FlyXeGPU/IR/AttrDefs.cpp.inc"

void FlyXeGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyXeGPU/IR/Atom.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "flydsl/Dialect/FlyXeGPU/IR/AttrDefs.cpp.inc"
      >();
}
