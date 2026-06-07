set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Atom.td)
mlir_tablegen(Atom.h.inc -gen-typedef-decls -typedefs-dialect=fly_xegpu)
mlir_tablegen(Atom.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_xegpu)
mlir_tablegen(AtomStateEnums.h.inc -gen-enum-decls -typedefs-dialect=fly_xegpu)
mlir_tablegen(AtomStateEnums.cpp.inc -gen-enum-defs -typedefs-dialect=fly_xegpu)

set(LLVM_TARGET_DEFINITIONS AttrDefs.td)
mlir_tablegen(AttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=fly_xegpu)
mlir_tablegen(AttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=fly_xegpu)
mlir_tablegen(AttrEnums.h.inc -gen-enum-decls -attrdefs-dialect=fly_xegpu)
mlir_tablegen(AttrEnums.cpp.inc -gen-enum-defs -attrdefs-dialect=fly_xegpu)

add_public_tablegen_target(MLIRFlyXeGPUIncGen)
