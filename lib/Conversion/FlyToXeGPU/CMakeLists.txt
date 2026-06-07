add_mlir_conversion_library(MLIRFlyToXeGPU
  FlyToXeGPU.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyXeGPUIncGen
  FlyToXeGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyXeGPUDialect

  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRXeGPUDialect
)
