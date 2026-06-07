# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# WIP: tiledCopy on Intel PVC via xegpu — KNOWN BROKEN.
#
# Status: compiles (COMPILE_ONLY=1 succeeds), but at runtime only one row of
# each N-block is copied, leaving rows 1..7 of every M-block at zero. Result:
# `torch.allclose(A, B) == False`, with B having ~120/2880 elements correct.
#
# Diagnosis (unfinished, see xegpu-tiled-copy-debug branch history):
#   - vectorAdd (01-vectorAdd-xegpu.py) works end-to-end on the same SYCL
#     runtime, so the host-side launch / Stream / data-marshalling path is
#     correct.
#   - The kernel-side IR addressing math is correct: bid maps to
#     (m_block, n_block) via `bid % 3` / `bid / 3`, and per-thread offset
#     uses `tid/2 * stride[0]` for the row component.
#   - But on PVC, when block_dim=16 (matching the kernel's
#     thr_layout=(8,2)=16 threads), only the m_block=0 slice of each N-block
#     gets written -- as if `tid/2 * stride[0]` was zero for all lanes.
#   - Tried swapping the (X,Y,Z) <-> (dim0,dim1,dim2) ordering inside a
#     local copy of upstream SyclRuntimeWrappers.cpp. That broke vectorAdd
#     (which currently runs correctly on the upstream order), so the fix
#     is not a simple dim swap. Reverted.
#   - The interaction between SYCL's nd_range dimension ordering, MLIR's
#     gpu.thread_id->get_local_id(0) lowering, and FlyDSL's tile-block
#     bid math needs deeper investigation.
#
# Reproducer for future debugging:
#
#   PYTHONPATH=$PWD/build-fly/python_packages:$PWD \
#   FLYDSL_COMPILE_BACKEND=xegpu FLYDSL_RUNTIME_KIND=xegpu \
#   FLYDSL_RUNTIME_ENABLE_CACHE=0 \
#   python3 examples/02-tiledCopy-xegpu.py
#   # Result correct: False
#
# Backend-neutral changes from 02-tiledCopy.py (all correct on rocdl too):
#   - drops the rocdl-specific fx.rocdl.make_buffer_tensor() wraps on A, B
#   - uses fx.UniversalCopy128b() (was fx.rocdl.BufferCopy128b())
#   - .xpu() instead of .cuda(); torch.xpu.Stream instead of torch.cuda.Stream
# The kernel layout was reshaped from rocdl's (4-thread, val=(1,8)) to
# (16-thread thr=(8,2), val=(1,12)) to match PVC's SIMD16 subgroup, since
# lane mode does not redistribute work across sub-block-sized launches.

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def copy_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    # Tile sized so 16 PVC SIMD lanes each handle one row chunk:
    #   8 rows x 24 cols = 192 elements per tile.
    #   thr_layout = (8, 2): 16 threads laid out as 8 rows x 2 col-groups.
    #   val_layout = (1, 12): each thread copies 1 row x 12 cols = 12 elems.
    #   16 threads x 12 vals = 192 elements -- exactly covers the tile.
    # The Universal copy atom emits vector<4xf32> loads (128b each), so
    # each thread issues 3 vector loads (12 / 4 = 3).
    block_m = 8
    block_n = 24

    bA = fx.zipped_divide(A, (block_m, block_n))
    bB = fx.zipped_divide(B, (block_m, block_n))
    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))

    thr_layout = fx.make_layout((8, 2), (2, 1))
    val_layout = fx.make_layout((1, 12), (1, 1))
    copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    partition_src = thr_copy.partition_S(bA)
    partition_dst = thr_copy.partition_D(bB)

    frag = fx.make_fragment_like(partition_src)

    fx.copy(copy_atom, partition_src, frag)
    fx.copy(copy_atom, frag, partition_dst)


@flyc.jit
def tiledCopy(
    A: fx.Tensor,
    B: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    # M=24/8=3 M-blocks, N=120/24=5 N-blocks -> 15 blocks.
    # block=(16,1,1) matches the kernel's thr_layout=(8,2)=16 threads.
    copy_kernel(A, B).launch(grid=(15, 1, 1), block=(16, 1, 1), stream=stream)


if __name__ == "__main__":
    assert torch.xpu.is_available(), "torch.xpu unavailable; install torch with --index-url https://download.pytorch.org/whl/xpu"
    M, N = 8 * 3, 24 * 5
    A = torch.arange(M * N, dtype=torch.float32).reshape(M, N).xpu()
    B = torch.zeros(M, N, dtype=torch.float32).xpu()

    stream = fx.Stream(torch.xpu.current_stream())
    tiledCopy(A, B, stream=stream)
    torch.xpu.synchronize()

    is_correct = torch.allclose(A, B)
    print("Result correct:", is_correct)
    if not is_correct:
        print("A:", A)
        print("B:", B)
