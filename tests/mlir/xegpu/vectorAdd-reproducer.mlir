// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Phase 2e debug reproducer for the FlyDSL xegpu backend.
//
// Modeled on
//   mlir/test/Integration/Dialect/XeVM/GPU/xevm_store_cst.mlir
// which proves the upstream xevm pipeline can launch a kernel that takes
// !llvm.ptr<1> args directly (the exact ABI FlyDSL emits after
// convert-fly-to-xegpu). The host wrapper uses gpu.alloc / gpu.memcpy +
// memref.extract_aligned_pointer_as_index + llvm.inttoptr +
// llvm.addrspacecast to feed the kernel a raw global pointer.
//
// What this isolates: if vectorAdd works here under mlir-runner but
// segfaults from the FlyDSL JIT executor with the same kernel signature,
// the bug is in FlyDSL's argument packing / JIT path -- NOT in our
// kernel-side IR or in the lowering pipeline.
//
// Usage:
//
//   cd /home/jovyan/workspace2/llvm-project/build
//   ./bin/mlir-opt $FLYDSL/tests/mlir/xegpu/vectorAdd-reproducer.mlir \
//     --gpu-lower-to-xevm-pipeline="xegpu-op-level=lane" \
//   | ./bin/mlir-runner \
//       --shared-libs=lib/libmlir_levelzero_runtime.so \
//       --shared-libs=lib/libmlir_runner_utils.so \
//       --shared-libs=lib/libmlir_c_runner_utils.so \
//       --entry-point-result=void
//
// Expected: prints A as [1, 1, ...], B as [2, 2, ...], C as [3, 3, ...].

module @reproducer attributes {gpu.container_module} {

  gpu.module @kernel {
    // Kernel signature mimics what FlyDSL emits after convert-fly-to-xegpu:
    // bare !llvm.ptr<1> for each global memref.
    gpu.func @vectorAdd(%A: !llvm.ptr<1>, %B: !llvm.ptr<1>, %C: !llvm.ptr<1>) kernel {
      %tid = gpu.thread_id x
      %tid_i64 = arith.index_cast %tid : index to i64
      %a_ptr = llvm.getelementptr %A[%tid_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %b_ptr = llvm.getelementptr %B[%tid_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %c_ptr = llvm.getelementptr %C[%tid_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %a = llvm.load %a_ptr : !llvm.ptr<1> -> f32
      %b = llvm.load %b_ptr : !llvm.ptr<1> -> f32
      %sum = arith.addf %a, %b : f32
      llvm.store %sum, %c_ptr : f32, !llvm.ptr<1>
      gpu.return
    }
  }

  func.func @test(%A: memref<128xf32>, %B: memref<128xf32>) -> memref<128xf32>
      attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index

    // Allocate device buffers and copy host data in.
    %A_gpu = gpu.alloc() : memref<128xf32>
    gpu.memcpy %A_gpu, %A : memref<128xf32>, memref<128xf32>
    %B_gpu = gpu.alloc() : memref<128xf32>
    gpu.memcpy %B_gpu, %B : memref<128xf32>, memref<128xf32>
    %C_gpu = gpu.alloc() : memref<128xf32>

    // Convert each device memref to a raw global !llvm.ptr<1>.
    %A_idx = memref.extract_aligned_pointer_as_index %A_gpu : memref<128xf32> -> index
    %A_i64 = arith.index_cast %A_idx : index to i64
    %A_p0 = llvm.inttoptr %A_i64 : i64 to !llvm.ptr
    %A_p1 = llvm.addrspacecast %A_p0 : !llvm.ptr to !llvm.ptr<1>

    %B_idx = memref.extract_aligned_pointer_as_index %B_gpu : memref<128xf32> -> index
    %B_i64 = arith.index_cast %B_idx : index to i64
    %B_p0 = llvm.inttoptr %B_i64 : i64 to !llvm.ptr
    %B_p1 = llvm.addrspacecast %B_p0 : !llvm.ptr to !llvm.ptr<1>

    %C_idx = memref.extract_aligned_pointer_as_index %C_gpu : memref<128xf32> -> index
    %C_i64 = arith.index_cast %C_idx : index to i64
    %C_p0 = llvm.inttoptr %C_i64 : i64 to !llvm.ptr
    %C_p1 = llvm.addrspacecast %C_p0 : !llvm.ptr to !llvm.ptr<1>

    // Launch: 1 block, 128 threads (PVC SIMD16 will run this as 8 subgroups
    // of 16 lanes; safe for vectorAdd since each lane is independent).
    gpu.launch_func @kernel::@vectorAdd
        blocks in (%c1, %c1, %c1)
        threads in (%c128, %c1, %c1)
        args(%A_p1 : !llvm.ptr<1>, %B_p1 : !llvm.ptr<1>, %C_p1 : !llvm.ptr<1>)

    // Copy result back to host.
    %C_host = memref.alloc() : memref<128xf32>
    gpu.memcpy %C_host, %C_gpu : memref<128xf32>, memref<128xf32>
    gpu.dealloc %A_gpu : memref<128xf32>
    gpu.dealloc %B_gpu : memref<128xf32>
    gpu.dealloc %C_gpu : memref<128xf32>
    return %C_host : memref<128xf32>
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.alloc() : memref<128xf32>
    %B = memref.alloc() : memref<128xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1_f32 = arith.constant 1.0 : f32
    %c2_f32 = arith.constant 2.0 : f32
    scf.for %i = %c0 to %c128 step %c1 {
      memref.store %c1_f32, %A[%i] : memref<128xf32>
      memref.store %c2_f32, %B[%i] : memref<128xf32>
    }
    %C = call @test(%A, %B) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %C_cast = memref.cast %C : memref<128xf32> to memref<*xf32>
    call @printMemrefF32(%C_cast) : (memref<*xf32>) -> ()

    // CHECK: Unranked Memref base@ = 0x{{[0-9a-f]+}}
    // CHECK-COUNT-128: 3

    memref.dealloc %A : memref<128xf32>
    memref.dealloc %B : memref<128xf32>
    memref.dealloc %C : memref<128xf32>
    return
  }

  func.func private @printMemrefF32(%ptr: memref<*xf32>) attributes {llvm.emit_c_interface}
}
