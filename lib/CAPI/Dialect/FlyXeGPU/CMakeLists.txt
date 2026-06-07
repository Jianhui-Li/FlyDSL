add_mlir_public_c_api_library(MLIRCPIFlyXeGPU
  FlyXeGPUDialect.cpp
  LINK_LIBS PUBLIC
  MLIRFlyXeGPUDialect
  MLIRFlyToXeGPU
)
