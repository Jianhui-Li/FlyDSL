# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# XeGPU backend descriptor (Intel GPUs via upstream MLIR xegpu / xevm dialects).
# Self-registers into global properties consumed by downstream CMakeLists.txt.

# TableGen / header subdirectories under include/flydsl/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS "FlyXeGPU")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_CONVERSION_SUBDIRS "FlyToXeGPU")

# C++ library subdirectories under lib/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS "FlyXeGPU")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS "FlyToXeGPU")

# CAPI wrapper subdirectory under lib/CAPI/Dialect/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS "FlyXeGPU")

# CAPI link targets for _mlirRegisterEverything (EMBED_CAPI_LINK_LIBS)
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_EMBED_CAPI_LIBS "MLIRCPIFlyXeGPU")

# Link targets for fly-opt
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_FLYOPT_LINK_LIBS "MLIRCPIFlyXeGPU")

# Upstream MLIR dialect Python sources needed by this backend.
# Note: upstream MLIR does not ship a top-level MLIRPythonSources.Dialects.xegpu
# target (only the transform-ops xegpu module). Phase 1 doesn't need upstream
# xegpu Python bindings; revisit when we want xegpu IR builders in Python.

# Stubgen modules for this backend
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_STUBGEN_MODULES
  "flydsl._mlir._mlir_libs._mlirDialectsFlyXeGPU")

# Convenience boolean for Python CMakeLists gating of XeGPU-specific bindings.
set(FLYDSL_HAS_XEGPU ON)
