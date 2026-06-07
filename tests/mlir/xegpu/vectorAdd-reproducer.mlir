// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Phase 2e debug reproducer for the FlyDSL xegpu backend.
//
// This file mimics the IR FlyDSL emits for examples/01-vectorAdd-xegpu.py
// after the convert-fly-to-xegpu pass, but with two changes that let it
// run under upstream mlir-runner without FlyDSL's Python harness:
//
//   1. Host data lives in normal `memref<128xf32>` and is marshalled to
//      the device with `gpu.alloc` + `gpu.memcpy` (the same pattern the
//      upstream simple_gemm test uses), instead of being passed in as
//      pre-allocated XPU device pointers from `flyc.from_dlpack`.
//   2. The kernel takes plain `!llvm.ptr<1>` operands without the
//      `!llvm.struct<packed (struct<packed (i32)>)>` layout-tuple operands
//      FlyDSL inserts after `fly-rewrite-func-signature`. Compare this run
//      against a variant that DOES include those operands to test whether
//      the nested-struct kernel ABI is what's segfaulting in Phase 2e.
//
// Usage:
//
//   cd /home/jovyan/workspace2/llvm-project/build
//   ./bin/mlir-opt $FLYDSL/tests/mlir/xegpu/vectorAdd-reproducer.mlir \
//     --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
//   | ./bin/mlir-runner \
//       --shared-libs=lib/libmlir_levelzero_runtime.so \
//       --shared-libs=lib/libmlir_runner_utils.so \
//       --entry-point-result=void
//
// Expected: prints "0 1 2 3 4 5 6 7 ..." (A[i] = i, B[i] = i, C = A+B).
//
// If this passes but the FlyDSL Python launch still segfaults, the bug
// is in argument packing inside flydsl.compiler.jit_executor; if it
// fails the same way, the bug is in our kernel-side IR or the pipeline
// (and the FlyDSL Python layer is innocent).

module @reproducer attributes {gpu.container_module} {
  gpu.module @kernels [#xevm.target<chip = "pvc">] {
    gpu.func @vectorAddKernel_0(%A: !llvm.ptr<1>,
                                %B: !llvm.ptr<1>,
                                %C: !llvm.ptr<1>,
                                %n: i32) kernel {
      %c1_i32 = arith.constant 1 : i32
      %block_id_x = gpu.block_id x
      %bid_i32 = arith.index_cast %block_id_x : index to i32
      %thread_id_x = gpu.thread_id x
      %tid_i32 = arith.index_cast %thread_id_x : index to i32
      %block_dim_i32 = arith.constant 64 : i32
      %offset = arith.muli %bid_i32, %block_dim_i32 : i32
      %idx = arith.addi %offset, %tid_i32 : i32
      // Bounds check
      %in_bounds = arith.cmpi slt, %idx, %n : i32
      scf.if %in_bounds {
        %a_ptr = llvm.getelementptr %A[%idx]
            : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %b_ptr = llvm.getelementptr %B[%idx]
            : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %c_ptr = llvm.getelementptr %C[%idx]
            : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %a = llvm.load %a_ptr : !llvm.ptr<1> -> f32
        %b = llvm.load %b_ptr : !llvm.ptr<1> -> f32
        %sum = arith.addf %a, %b : f32
        llvm.store %sum, %c_ptr : f32, !llvm.ptr<1>
      }
      gpu.return
    }
  }

  // TODO(Phase 2e): port the host wrapper. The skeleton below uses the
  // upstream gpu.alloc / gpu.memcpy pattern; once it works, copy it back
  // and replace the device pointers with the bare !llvm.ptr<1> args
  // FlyDSL emits, then add the !llvm.struct<packed (struct<packed (i32)>)>
  // layout-tuple operands to bisect which change triggers the segfault.
  func.func @main() attributes {llvm.emit_c_interface} {
    // Stub: prints a placeholder until the host wrapper is filled in.
    // Replace with: gpu.alloc / gpu.memcpy / gpu.launch_func / gpu.memcpy
    // / gpu.dealloc / printMemrefF32 — see
    // mlir/test/Integration/Dialect/XeGPU/WG/simple_gemm.mlir lines 14-50
    // for a working template.
    return
  }
}
