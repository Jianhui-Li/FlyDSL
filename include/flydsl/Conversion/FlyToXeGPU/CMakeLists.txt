set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlyToXeGPU)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix FlyToXeGPU)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix FlyToXeGPU)

add_mlir_generic_tablegen_target(FlyToXeGPUPassIncGen)
