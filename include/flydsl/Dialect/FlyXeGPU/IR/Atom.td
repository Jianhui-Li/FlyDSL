// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYXEGPU_ATOM
#define FLYXEGPU_ATOM

include "flydsl/Dialect/FlyXeGPU/IR/Dialect.td"

def FlyXeGPU_AtomStateField : I32EnumAttr<"AtomStateField", "", [
  I32EnumAttrCase<"Reserved",            0, "reserved">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyXeGPU_Dialect.cppNamespace;
}

// No concrete atom types yet — Phase 2+ will add DPAS / 2D-block-load /
// SLM copy types here.

#endif // FLYXEGPU_ATOM
