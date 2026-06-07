// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYXEGPU_DIALECT
#define FLYXEGPU_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def FlyXeGPU_Dialect : Dialect {
  let name = "fly_xegpu";
  let cppNamespace = "::mlir::fly_xegpu";

  let dependentDialects = [
    "xegpu::XeGPUDialect"
  ];

  // useDefault*PrinterParser is intentionally left off in the Phase 1
  // skeleton: with no concrete types/attrs the tablegen generators emit
  // empty inc files and the default printer/parser entries on the dialect
  // vtable would be unresolved at link time. Re-enable when we add the
  // first xegpu atom type / attribute.
}

class FlyXeGPU_I32EnumAttr<string enumName, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<enumName, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyXeGPU_Dialect.cppNamespace;
}

class FlyXeGPU_EnumAttr<EnumInfo enumInfo, string attrMnemonic, list<Trait> traits = []>
    : EnumAttr<FlyXeGPU_Dialect, enumInfo, attrMnemonic, traits>;

class FlyXeGPU_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<FlyXeGPU_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class FlyXeGPU_MmaOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyXeGPU_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyXeGPU_CopyOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyXeGPU_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyXeGPU_StatefulCopyOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyXeGPU_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_StatefulOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
  let extraClassDeclaration = [{
    static std::optional<unsigned> getFieldIndex(AtomStateField field);
  }];
}

class FlyXeGPU_StatefulMmaOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyXeGPU_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_StatefulOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
  let extraClassDeclaration = [{
    static std::optional<unsigned> getFieldIndex(AtomStateField field);
  }];
}

#endif // FLYXEGPU_DIALECT
